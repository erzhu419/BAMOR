import copy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import torch

import collect_matched_scale_normals as collection
import run_matched_residual_scale_probe as probe
import run_conditional_change_probe as original
import run_headwise_change_probe as plain
import run_affine_residual_change_probe as affine
from affine_residual_change_evidence import AffineResidualHeadwiseEvidence
from headwise_change_evidence import CONFIG
from test_affine_residual_change_probe import make_source
from test_headwise_change_probe import targets


def make_fresh_fixture(root, env):
    for seed in probe.SEEDS:
        rng = np.random.default_rng(seed + 50000)
        data, cases = {}, {}
        for case, steps in collection.COLLECTION_STEPS.items():
            inputs = rng.normal(size=(steps, 5)).astype(np.float32)
            data.update({f"{case}_inputs": inputs, f"{case}_targets": targets(inputs)})
            cases[case] = dict(ground_truth_mode_history_audit_only=[[0, 0]], intervention_active=True)
        folder = collection.source_folder(root, env, seed)
        collection.write(folder / "readout.json", dict(protocol=collection.PROTOCOL, env=env,
            seed=seed, complete=True, collection_steps=collection.COLLECTION_STEPS,
            collection_interactions=5000, initialization_probes=3, additional_interactions=5003,
            source_step=75000, policy_optimizer_steps=0, critic_optimizer_steps=0,
            encoder_optimizer_steps=0, cases=cases))
        np.savez_compressed(folder / "transitions.npz", **data)


class MatchedResidualScaleTests(unittest.TestCase):
    def test_real_collection_frozen_models_normal_routes_and_physical_budget(self):
        training = original.training
        with tempfile.TemporaryDirectory() as tmp, patch.object(original, "SOURCE_HIDDEN", 8), \
             patch.object(original, "CHANGE_AT", 10), patch.object(collection, "CHANGE_AT", 10), \
             patch.dict(collection.COLLECTION_STEPS, stationary=40, sampler_change=20, actor_change=20):
            root, source = Path(tmp) / "fresh", Path(tmp) / "policies"
            for env_name in probe.ENVIRONMENTS:
                env = training.make_piecewise_mo_env(env_name, switch_interval=10**9, seed=999)
                try:
                    obs_dim, act_dim, reward_dim = env.observation_space.shape[0], env.action_space.shape[0], env.reward_dim
                    context_dim = training.context_dim_for_method(original.METHODS["fixed"],
                        obs_dim, act_dim, reward_dim, env.num_modes)
                    for index, (name, method) in enumerate(original.METHODS.items()):
                        torch.manual_seed(999 + index)
                        policy = training.PolicyNet(obs_dim, act_dim, context_dim, hidden=8)
                        encoder = training.LatentTransitionEncoder(obs_dim, act_dim, reward_dim, hidden=8,
                                                                   latent_dim=training.LATENT_CONTEXT_DIM)
                        folder = source / env_name.replace("-", "_") / name
                        collection.write(folder / f"{method}_seed2101.json",
                            dict(env=env_name, method=method, seed=2101, total_steps=75000))
                        torch.save(dict(policy=policy.state_dict(), encoder=encoder.state_dict()),
                                   folder / f"{method}_seed2101_model.pt")
                finally:
                    env.close()
                models, states = [], []
                loader = collection.load_models
                def record_load(env, seed, source):
                    policies, encoder = loader(env, seed, source)
                    models.extend([*policies.values(), encoder])
                    states.extend(copy.deepcopy(model.state_dict()) for model in models)
                    return policies, encoder
                with patch("builtins.print"), patch.object(collection, "load_models", record_load):
                    report = collection.run(env_name, 2101, root, source)
                self.assertTrue(report["complete"])
                self.assertEqual((report["collection_interactions"], report["initialization_probes"],
                                  report["additional_interactions"]), (80, 3, 83))
                self.assertTrue(all(not p.requires_grad for model in models for p in model.parameters()))
                self.assertTrue(all(torch.equal(state[key], model.state_dict()[key])
                                    for model, state in zip(models, states) for key in state))
                inputs, outputs, saved = collection.normal_data(root, env_name, 2101)
                self.assertEqual(len(inputs), 80)
                self.assertEqual(outputs.shape[1], obs_dim + reward_dim)
                self.assertTrue(all(r["ground_truth_mode_history_audit_only"] == [[0, 0]]
                                    for r in report["cases"].values()))
                self.assertTrue(report["cases"]["actor_change"]["intervention_active"])
                self.assertTrue(report["cases"]["sampler_change"]["intervention_active"])
                self.assertEqual(len({r["seeds"]["reset_seed"] for r in report["cases"].values()}), 3)
                with patch.object(training, "make_piecewise_mo_env", side_effect=AssertionError("re-collected")):
                    self.assertEqual(collection.run(env_name, 2101, root, source), saved)
                report["complete"] = False
                collection.write(collection.source_folder(root, env_name, 2101) / "readout.json", report)
                with self.assertRaises(RuntimeError):
                    collection.run(env_name, 2101, root, source)

    def test_offline_fit_uses_exact_deployed_residuals_and_excludes_held_sources(self):
        with tempfile.TemporaryDirectory() as tmp, patch.dict(CONFIG, hidden=8, updates=3, scale_updates=3):
            source, root, old, control, fresh = (Path(tmp) / name for name in ("source", "matched", "affine", "plain", "fresh"))
            env = probe.ENVIRONMENTS[0]
            make_source(source, env)
            make_fresh_fixture(fresh, env)
            recorded = []
            fit_scales = AffineResidualHeadwiseEvidence.fit_scale_losses
            def record_scale(evidence, inputs, losses, seed):
                recorded.append((inputs.copy(), losses.copy(), evidence.raw_losses(inputs, expected_y)))
                return fit_scales(evidence, inputs, losses, seed)
            roles = probe.fold_roles(2101)
            rows = [collection.normal_data(fresh, env, seed) for seed in roles["fit_seeds"]]
            rows.append(probe.normal_data(source, env, roles["scale_seed"], "expanded_normal"))
            expected_x, expected_y = (np.concatenate([r[index] for r in rows]) for index in (0, 1))
            with patch("builtins.print"), patch.object(original.training, "make_piecewise_mo_env",
                    side_effect=AssertionError("offline interaction")):
                plain.run(env, 2101, source, control)
                baseline = affine.run(env, 2101, source, old, control)
                with patch.object(AffineResidualHeadwiseEvidence, "fit_scale_losses", record_scale):
                    report = probe.run(env, 2101, source, root, old, fresh)
            np.testing.assert_array_equal(recorded[0][0], expected_x)
            np.testing.assert_array_equal(recorded[0][1], recorded[0][2])
            self.assertEqual(report["predictor_work"], baseline["predictor_work"])
            self.assertEqual((report["predictor_optimizer_steps"], report["residual_scale_optimizer_steps"]), (6, 9))
            self.assertEqual(report["additional_interactions"], 0)
            self.assertEqual(report["scale_fit_steps"], 15000)
            self.assertNotIn(roles["calibration_seed"], report["scale_fit_sources"])
            self.assertNotIn(roles["test_seed"], report["scale_fit_sources"])
            self.assertEqual([r["fresh"] for r in report["residual_blocks"]], [True, True, False])
            with patch.object(AffineResidualHeadwiseEvidence, "fit", side_effect=AssertionError("repeated fit")):
                self.assertEqual(probe.run(env, 2101, source, root, old, fresh), report)
            for seed in probe.SEEDS:
                copied = copy.deepcopy(report)
                copied["roles"] = probe.fold_roles(seed)
                copied["scale_fit_sources"] = copied["roles"]["fit_seeds"] + [copied["roles"]["scale_seed"]]
                copied["fresh_fit_sources"] = [dict(seed=s, steps=5000, source_step=75000) for s in copied["roles"]["fit_seeds"]]
                probe.write(root / env.replace("-", "_") / f"test_seed{seed}" / "readout.json", copied)
            with patch.object(probe, "ENVIRONMENTS", (env,)):
                summary = probe.summarize(root, fresh)
            self.assertEqual(summary["collection_interactions"], 25000)
            self.assertEqual(summary["initialization_probes"], 15)
            self.assertEqual(summary["additional_interactions"], 25015)
            self.assertEqual(summary["predictor_optimizer_steps"], 30)
            self.assertFalse(summary["evidence_integrated"])
            compact = probe.compact_report(summary)
            self.assertNotIn("windows", compact["environments"][env]["runs"][0]["arms"]["conditional_scale"]["calibration"])


if __name__ == "__main__":
    torch.set_num_threads(2)
    unittest.main()
