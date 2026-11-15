import copy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import torch

import run_robust_startup_confirmation as probe
import collect_matched_scale_normals as old_collection
import run_conditional_change_probe as original
from headwise_change_evidence import CONFIG
from test_affine_residual_change_probe import make_source
from test_matched_residual_scale_probe import make_fresh_fixture
from test_headwise_change_probe import targets
from test_startup_confirmation import fixture_collection


def metadata(env, seed, group):
    row = fixture_collection(env, seed, group)
    for i, case in enumerate(probe.acquisition.steps(group)):
        row["cases"][case]["seeds"] = probe.seeds(seed, group, i)
    return row


class RobustStartupConfirmationTests(unittest.TestCase):
    def test_disjoint_new_streams_and_physical_budget(self):
        for seed in probe.SEEDS:
            evaluation = probe.seeds(seed, "evaluation")
            self.assertEqual(evaluation, probe.seeds(seed, "evaluation", 5))
            old_eval = probe.acquisition.seeds(seed, "evaluation")
            for i in range(3):
                new = probe.seeds(seed, "startup", i)
                old = probe.acquisition.seeds(seed, "startup", i)
                matched = old_collection.seeds(seed, i)
                self.assertTrue(all(new[k] != old[k] and new[k] != matched[k]
                    and new[k] != evaluation[k] and evaluation[k] != old_eval[k] for k in new))
                for group in ("startup", "evaluation"):
                    probe.acquisition.validate(metadata(probe.ENVIRONMENTS[0], seed, group),
                        probe.ENVIRONMENTS[0], seed, group, seed_fn=probe.seeds)
        self.assertEqual(10 * sum(probe.acquisition.budget(g)["additional_interactions"]
                                 for g in ("startup", "evaluation")), 110090)

    def test_real_new_collection_keeps_pairing_and_frozen_policy_and_rejects_old_stream(self):
        acquisition, training = probe.acquisition, original.training
        with tempfile.TemporaryDirectory() as tmp, patch.object(original, "SOURCE_HIDDEN", 8), \
             patch.object(original, "CHANGE_AT", 10), patch.object(acquisition, "CHANGE_AT", 10), \
             patch.object(acquisition, "CASE_STEPS", 20), \
             patch.dict(acquisition.STARTUP_STEPS, stationary=40, sampler_change=20, actor_change=20):
            root, source = Path(tmp)/"collection", Path(tmp)/"policies"
            for env_name in probe.ENVIRONMENTS:
                env = training.make_piecewise_mo_env(env_name, switch_interval=10**9, seed=999)
                try:
                    obs_dim, act_dim, reward_dim = env.observation_space.shape[0], env.action_space.shape[0], env.reward_dim
                    context_dim = training.context_dim_for_method(original.METHODS["fixed"], obs_dim, act_dim, reward_dim, env.num_modes)
                    for i, (name, method) in enumerate(original.METHODS.items()):
                        torch.manual_seed(999 + i)
                        policy = training.PolicyNet(obs_dim, act_dim, context_dim, hidden=8)
                        encoder = training.LatentTransitionEncoder(obs_dim, act_dim, reward_dim, hidden=8, latent_dim=training.LATENT_CONTEXT_DIM)
                        path = source/env_name.replace("-", "_")/name
                        acquisition.write(path/f"{method}_seed2101.json", dict(env=env_name, method=method, seed=2101, total_steps=75000))
                        torch.save(dict(policy=policy.state_dict(), encoder=encoder.state_dict()), path/f"{method}_seed2101_model.pt")
                finally:
                    env.close()
                models, states, loader = [], [], acquisition.load_models
                def record_load(env, seed, source):
                    policies, encoder = loader(env, seed, source)
                    loaded = [*policies.values(), encoder]
                    models.extend(loaded)
                    states.extend(copy.deepcopy(m.state_dict()) for m in loaded)
                    return policies, encoder
                with patch("builtins.print"), patch.object(acquisition, "load_models", record_load):
                    a = acquisition.run_group(env_name, 2101, "startup", root, source, seed_fn=probe.seeds)
                    b = acquisition.run_group(env_name, 2101, "evaluation", root, source, seed_fn=probe.seeds)
                self.assertEqual((a["additional_interactions"], b["additional_interactions"]), (83, 126))
                self.assertEqual(b["paired_prefixes_verified"], 5)
                self.assertTrue(all(not p.requires_grad for m in models for p in m.parameters()))
                self.assertTrue(all(torch.equal(state[k], m.state_dict()[k]) for state, m in zip(states, models) for k in state))
                for group in ("startup", "evaluation"):
                    rows, report = acquisition.data(root, env_name, 2101, group, seed_fn=probe.seeds)
                    self.assertEqual(set(rows), set(acquisition.steps(group)))
                    with self.assertRaises(AssertionError):
                        acquisition.data(root, env_name, 2101, group)
                    with patch.object(training, "make_piecewise_mo_env", side_effect=AssertionError("Recollection")):
                        self.assertEqual(acquisition.run_group(env_name, 2101, group, root, source, seed_fn=probe.seeds), report)
        print("REAL_ROBUST_CONFIRMATION_COLLECTION_TEST interactions=420 including2 fixture initialization probes")

    def test_pipeline_freezes_before_collection_accounts_failure_and_never_retries(self):
        development = probe.development
        with tempfile.TemporaryDirectory() as tmp, patch.dict(CONFIG, hidden=8, updates=3, scale_updates=3):
            source, fresh, parent, root = (Path(tmp)/n for n in ("source", "fresh", "development", "confirmation"))
            env, seed = probe.ENVIRONMENTS[0], 2101
            make_source(source, env)
            make_fresh_fixture(fresh, env)
            roles = probe.fold_roles(seed)
            fit = [probe.normal_data(source, env, s, "expanded_normal") for s in roles["fit_seeds"]]
            scale = [probe.fresh_data(fresh, env, s) for s in roles["fit_seeds"]]
            scale.append(probe.normal_data(source, env, roles["scale_seed"], "expanded_normal"))
            fit_x, fit_y = (np.concatenate([r[i] for r in fit]) for i in (0, 1))
            scale_x, scale_y = (np.concatenate([r[i] for r in scale]) for i in (0, 1))
            evidence = probe.GuardedEvidence(5, 5, 2)
            evidence.weights = {probe.GUARDED: .5}
            work = evidence.fit(fit_x, fit_y, 5000, 80000000 + 10*seed)
            scale_work = evidence.fit_scale_losses(scale_x, evidence.raw_losses(scale_x, scale_y), 81000000 + 10*seed)
            evidence.radii = np.full(3, 1e10)
            rng = np.random.default_rng(7654)
            startup = {case: (x, targets(x)) for case, count in probe.acquisition.STARTUP_STEPS.items()
                       for x in [rng.normal(size=(count, 5)).astype(np.float32)]}
            with np.load(source/env.replace("-", "_")/f"seed{seed}"/"transitions.npz") as stored:
                evaluation = {c: (stored[f"{c}_inputs"].copy(), stored[f"{c}_targets"].copy()) for c in probe.CASES}
            x, y = (np.concatenate([r[i] for r in startup.values()]) for i in (0, 1))
            startup_x, startup_y = x.copy(), y.copy()
            old_x = rng.normal(size=(5000, 5)).astype(np.float32)
            cal = development.calibrate(evidence, old_x, targets(old_x))
            arms = {arm: dict(calibration=cal[arm], cases={}) for arm in probe.ARMS}
            for case, (x, y) in evaluation.items():
                windows, _ = development.paired_windows(evidence, x, y)
                for arm in probe.ARMS:
                    development.mark_alarms(windows[arm], cal[arm]["error_threshold"])
                    arms[arm]["cases"][case] = dict(windows=windows[arm])
            prior = dict(protocol=development.PROTOCOL, env=env, roles=roles, config=CONFIG, complete=True,
                aggregation=development.AGGREGATION, predictor_optimizer_steps=6, residual_scale_optimizer_steps=9,
                reward_dim=2, scale_fit_sources=roles["fit_seeds"]+[roles["scale_seed"]],
                mean_control_reproduced=True, thresholds_frozen_before_evaluation_read=True, arms=arms,
                predictor_work=work, scale_work=scale_work, frozen_selection=dict(global_weight=.5), **dict.fromkeys(development.ZERO_WORK, 0))
            parent_path = parent/env.replace("-", "_")/f"test_seed{seed}"/"readout.json"
            probe.write(parent_path, prior)
            events, original_fit, original_calibrate = [], probe.GuardedEvidence.fit, development.calibrate
            def record_fit(model, x, y, split, random_seed):
                np.testing.assert_array_equal(x, fit_x)
                np.testing.assert_array_equal(y, fit_y)
                return original_fit(model, x, y, split, random_seed)
            def collect_group(e, s, group, collect_root, seed_fn):
                self.assertIs(seed_fn, probe.seeds)
                events.append(group)
                if group == "evaluation":
                    saved = json.loads((root/env.replace("-", "_")/f"test_seed{seed}"/"readout.json").read_text())
                    self.assertTrue(saved["thresholds_frozen_before_evaluation_collection"])
                    self.assertEqual(set(saved["arms"]), set(probe.ARMS))
                row = metadata(e, s, group)
                probe.write(probe.acquisition.folder(collect_root, e, s, group)/"readout.json", row)
                return row
            def read_group(collect_root, e, s, group, seed_fn):
                self.assertIs(seed_fn, probe.seeds)
                return (startup if group == "startup" else evaluation), metadata(e, s, group)
            def freeze(model, x, y):
                events.append("freeze")
                np.testing.assert_array_equal(x, startup_x)
                np.testing.assert_array_equal(y, startup_y)
                models = [model.reward_predictor.model, model.dynamics_predictor.model, *model.scale_models]
                states = [copy.deepcopy(m.state_dict()) for m in models]
                result = original_calibrate(model, x, y)
                self.assertTrue(all(torch.equal(state[k], m.state_dict()[k]) for state, m in zip(states, models) for k in state))
                return result
            with patch("builtins.print"), patch.object(probe.acquisition, "run_group", collect_group), \
                 patch.object(probe.acquisition, "data", read_group), patch.object(development, "calibrate", freeze), \
                 patch.object(probe.GuardedEvidence, "fit", record_fit):
                result = probe.run(env, seed, source, root, parent, fresh)
            self.assertEqual(events, ["startup", "freeze", "evaluation"])
            self.assertEqual(result["additional_interactions"], 11009)
            self.assertEqual(result["initialization_probes"], 9)
            self.assertEqual(result["frozen_selection"], prior["frozen_selection"])
            for arm in probe.ARMS:
                self.assertNotEqual(result["arms"][arm]["calibration"]["error_threshold"], cal[arm]["error_threshold"])
            self.assertEqual(json.loads(parent_path.read_text()), prior)
            with patch.object(probe.GuardedEvidence, "fit", side_effect=AssertionError("Repeated fit")):
                self.assertEqual(probe.run(env, seed, source, root, parent, fresh), result)
            with patch.object(probe, "ENVIRONMENTS", (env,)), patch.object(probe, "SEEDS", (seed,)):
                summary = probe.summarize(root)
                self.assertEqual(summary["cumulative_acquisition_interactions"], 261199)
                self.assertFalse(summary["evidence_integrated"])
            failed_root = Path(tmp)/"failed"
            def fail(e, s, group, collect_root, seed_fn):
                row = metadata(e, s, group)
                row.update(complete=False, collection_interactions=3, initialization_probes=1, additional_interactions=4)
                probe.write(probe.acquisition.folder(collect_root, e, s, group)/"readout.json", row)
                raise RuntimeError("Interrupted fixture collection")
            with patch.object(probe.acquisition, "run_group", fail):
                with self.assertRaisesRegex(RuntimeError, "Interrupted fixture"):
                    probe.run(env, seed, source, failed_root, parent, fresh)
            partial = json.loads((failed_root/env.replace("-", "_")/f"test_seed{seed}"/"readout.json").read_text())
            self.assertFalse(partial["complete"])
            self.assertEqual(partial["additional_interactions"], 4)
            with patch.object(probe.GuardedEvidence, "fit", side_effect=AssertionError("Retry")):
                with self.assertRaisesRegex(RuntimeError, "Incomplete confirmation"):
                    probe.run(env, seed, source, failed_root, parent, fresh)
        print("ROBUST_CONFIRMATION_PIPELINE_TEST actual_interactions=0 simulated_new_interactions=11009 simulated_failed_interactions=4 mean_updates=18 scale_updates=27 affine_fits=6")


if __name__ == "__main__":
    torch.set_num_threads(2)
    unittest.main()
