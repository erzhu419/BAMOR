import copy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import torch

import collect_startup_confirmation as acquisition
import collect_matched_scale_normals as old_collection
from headwise_change_evidence import CONFIG
import run_startup_confirmation as probe
import run_deployment_startup_calibration as startup
import run_cross_policy_calibration as cross
import run_sensitivity_guarded_selection as guarded
import run_window_risk_selection as risk
import run_matched_residual_scale_probe as matched
import run_affine_residual_change_probe as affine
import run_headwise_change_probe as plain
import run_conditional_change_probe as original
from test_affine_residual_change_probe import make_source
from test_matched_residual_scale_probe import make_fresh_fixture
from test_headwise_change_probe import targets


def fixture_collection(env, seed, group):
    return dict(protocol=acquisition.PROTOCOL, env=env, seed=seed, group=group, complete=True,
        collection_steps=acquisition.steps(group), source_methods=original.METHODS, source_step=75000,
        policy_optimizer_steps=0, critic_optimizer_steps=0, encoder_optimizer_steps=0,
        paired_prefixes_verified=5 if group == "evaluation" else 0,
        cases={case: dict(seeds=acquisition.seeds(seed, group, i), intervention_active=True,
            ground_truth_mode_history_audit_only=[[0, 0], [500, int(case[-1])]] if case.startswith("mode") else [[0, 0]])
            for i, case in enumerate(acquisition.steps(group))}, **acquisition.budget(group))


class StartupConfirmationTests(unittest.TestCase):
    def test_new_seed_blocks_and_full_physical_budget(self):
        self.assertEqual(acquisition.budget("startup")["additional_interactions"], 5003)
        self.assertEqual(acquisition.budget("evaluation")["additional_interactions"], 6006)
        for seed in probe.SEEDS:
            evaluation = acquisition.seeds(seed, "evaluation")
            for i in range(3):
                new = acquisition.seeds(seed, "startup", i)
                old = old_collection.seeds(seed, i)
                self.assertTrue(all(new[k] != old[k] and new[k] != evaluation[k] for k in new))
                self.assertNotEqual(evaluation["reset_seed"], 50000000 + 1000 * (seed + 100))
        self.assertEqual(10 * sum(acquisition.budget(g)["additional_interactions"] for g in ("startup", "evaluation")), 110090)

    def test_real_acquisition_normal_routes_paired_prefixes_and_frozen_policies(self):
        training = original.training
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
                models, states = [], []
                loader = acquisition.load_models
                def record_load(env, seed, source):
                    policies, encoder = loader(env, seed, source)
                    loaded = [*policies.values(), encoder]
                    models.extend(loaded)
                    states.extend(copy.deepcopy(m.state_dict()) for m in loaded)
                    return policies, encoder
                with patch("builtins.print"), patch.object(acquisition, "load_models", record_load):
                    a = acquisition.run_group(env_name, 2101, "startup", root, source)
                    b = acquisition.run_group(env_name, 2101, "evaluation", root, source)
                self.assertEqual((a["additional_interactions"], b["additional_interactions"]), (83, 126))
                self.assertEqual(b["paired_prefixes_verified"], 5)
                self.assertTrue(all(not p.requires_grad for m in models for p in m.parameters()))
                self.assertTrue(all(torch.equal(state[k], model.state_dict()[k]) for state, model in zip(states, models) for k in state))
                for group in ("startup", "evaluation"):
                    data, report = acquisition.data(root, env_name, 2101, group)
                    self.assertEqual(set(data), set(acquisition.steps(group)))
                    self.assertTrue(report["cases"]["actor_change"]["intervention_active"])
                    self.assertTrue(report["cases"]["sampler_change"]["intervention_active"])
                    with patch.object(training, "make_piecewise_mo_env", side_effect=AssertionError("Recollection")):
                        self.assertEqual(acquisition.run_group(env_name, 2101, group, root, source), report)
        print("REAL_CONFIRMATION_COLLECTION_TEST interactions=420 including2 fixture initialization probes")

    def test_failed_acquisition_records_partial_interactions_and_does_not_retry(self):
        with tempfile.TemporaryDirectory() as tmp:
            root, env_name = Path(tmp), probe.ENVIRONMENTS[0]
            def fail(env, policies, encoder, steps, reset_seed, weight_seed, case):
                env.reset(seed=reset_seed)
                for _ in range(3):
                    env.step(np.zeros(env.action_space.shape, np.float32))
                raise RuntimeError("Interrupted probe")
            with patch.object(acquisition, "load_models", return_value=({}, torch.nn.Identity())), patch.object(acquisition, "collect", fail):
                with self.assertRaisesRegex(RuntimeError, "Interrupted probe"):
                    acquisition.run_group(env_name, 2101, "startup", root)
            r = json.loads((acquisition.folder(root, env_name, 2101, "startup")/"readout.json").read_text())
            self.assertEqual((r["collection_interactions"], r["initialization_probes"], r["additional_interactions"]), (3, 1, 4))
            self.assertFalse(r["complete"])
            with patch.object(original.training, "make_piecewise_mo_env", side_effect=AssertionError("Recollection")):
                with self.assertRaisesRegex(RuntimeError, "Incomplete acquisition"):
                    acquisition.run_group(env_name, 2101, "startup", root)
        print("FAILED_CONFIRMATION_COLLECTION_TEST interactions=4 retained")

    def test_pipeline_freezes_threshold_before_new_evaluation_and_shares_all_costs(self):
        with tempfile.TemporaryDirectory() as tmp, patch.dict(CONFIG, hidden=8, updates=3, scale_updates=3):
            source, fresh, base, old, parent, risk_root, guard_root, cross_root, prior, root = (Path(tmp)/n for n in
                ("source", "fresh", "plain", "affine", "matched", "risk", "guarded", "cross", "startup", "confirmation"))
            env, seed = probe.ENVIRONMENTS[0], 2101
            make_source(source, env)
            make_fresh_fixture(fresh, env)
            for s in probe.SEEDS:
                path = plain.source_folder(fresh, env, s)/"readout.json"
                meta = json.loads(path.read_text())
                for i, case in enumerate(old_collection.COLLECTION_STEPS):
                    meta["cases"][case]["seeds"] = old_collection.seeds(s, i)
                acquisition.write(path, meta)
            events, calibrate = [], probe.calibrate_fixed_support
            rng = np.random.default_rng(777)
            new_startup = {c:(rng.normal(size=(n,5)).astype(np.float32),None) for c,n in acquisition.STARTUP_STEPS.items()}
            new_startup = {c:(x,targets(x)) for c,(x,_) in new_startup.items()}
            with np.load(plain.source_folder(source, env, seed)/"transitions.npz") as d:
                new_eval = {c:(d[f"{c}_inputs"].copy(),d[f"{c}_targets"].copy()) for c in probe.CASES}
            def collect_group(env, seed, group, root):
                events.append(group)
                return fixture_collection(env,seed,group)
            def read_group(root,env,seed,group):
                return (new_startup if group=="startup" else new_eval),fixture_collection(env,seed,group)
            def freeze(evidence,x,y):
                events.append("freeze")
                expected_x,expected_y=(np.concatenate([r[i] for r in new_startup.values()]) for i in (0,1))
                np.testing.assert_array_equal(x,expected_x)
                np.testing.assert_array_equal(y,expected_y)
                radii,weights=evidence.radii.copy(),evidence.weights.copy()
                models=[evidence.reward_predictor.model,evidence.dynamics_predictor.model,*evidence.scale_models]
                states=[copy.deepcopy(m.state_dict()) for m in models]
                result=calibrate(evidence,x,y)
                np.testing.assert_array_equal(evidence.radii,radii)
                self.assertEqual(evidence.weights,weights)
                self.assertTrue(all(torch.equal(state[k],m.state_dict()[k]) for state,m in zip(states,models) for k in state))
                return result
            with patch("builtins.print"), patch.object(original.training,"make_piecewise_mo_env",side_effect=AssertionError("Unexpected environment")):
                plain.run(env,seed,source,base)
                affine.run(env,seed,source,old,base)
                matched.run(env,seed,source,parent,old,fresh)
                risk.run(env,seed,source,risk_root,parent,fresh)
                guarded.run(env,seed,source,guard_root,parent,fresh,risk_root)
                cross.run(env,seed,source,cross_root,guard_root,fresh)
                before=startup.run(env,seed,source,prior,cross_root,fresh)
                with patch.object(acquisition,"run_group",collect_group),patch.object(acquisition,"data",read_group),patch.object(probe,"calibrate_fixed_support",freeze):
                    result=probe.run(env,seed,source,root,prior,fresh)
            self.assertEqual(events,["startup","freeze","evaluation"])
            self.assertEqual(result["additional_interactions"],11009)
            self.assertEqual(result["initialization_probes"],9)
            self.assertEqual(result["frozen_selection"],before["frozen_selection"])
            self.assertEqual((result["predictor_optimizer_steps"],result["residual_scale_optimizer_steps"]),(6,9))
            self.assertTrue(result["shared_evaluation_scores_verified"])
            for arm in probe.CONTROLS:
                self.assertEqual(result["arms"][arm]["calibration"],before["arms"][startup.PRIMARY if arm=="frozen_old_startup" else arm]["calibration"])
            with patch.object(acquisition,"run_group",side_effect=AssertionError("Recollection")):
                self.assertEqual(probe.run(env,seed,source,root,prior,fresh),result)
            with patch.object(probe,"ENVIRONMENTS",(env,)),patch.object(probe,"SEEDS",(seed,)):
                summary=probe.summarize(root)
                with patch.object(probe,"screen",side_effect=lambda runs,arm:dict(development_screen_passed=arm!=probe.PRIMARY)):
                    self.assertEqual(probe.summarize(root)["decision"],"hold_conditional_evidence_integration")
            self.assertEqual(summary["additional_interactions"],11009)
            self.assertEqual(summary["cumulative_acquisition_interactions"],151109)
        print("CONFIRMATION_PIPELINE_TEST simulated_new_interactions=11009 actual_new_interactions=0 mean_updates=48 scale_updates=126")


if __name__ == "__main__":
    torch.set_num_threads(2)
    unittest.main()
