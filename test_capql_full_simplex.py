import copy
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import torch

import run_capql_full_simplex as experiment


class FullSimplexCAPQLTests(unittest.TestCase):
    def test_dirichlet_moments_support_and_seed(self):
        for dim in (2, 3):
            torch.manual_seed(713)
            sampler = experiment.previous.training.UniformSimplexWeightSampler(dim, np.pi / 8)
            weights = sampler.sample(20000)
            self.assertEqual(weights.shape, (20000, dim))
            self.assertEqual(weights.dtype, torch.float32)
            self.assertTrue((weights >= 0).all())
            torch.testing.assert_close(weights.sum(1), torch.ones(20000))
            torch.testing.assert_close(weights.mean(0), torch.full((dim,), 1 / dim), atol=.006, rtol=0)
            torch.testing.assert_close(weights.var(0), torch.full((dim,), (dim - 1) / (dim**2 * (dim + 1))),
                                       atol=.003, rtol=0)
            self.assertTrue((weights.max(0).values > .97).all())
            self.assertTrue((weights.min(0).values < .01).all())
            torch.manual_seed(713)
            self.assertTrue(torch.equal(weights, sampler.sample(20000)))

    def test_new_method_does_not_expand_frozen_rosters(self):
        training = experiment.previous.training
        self.assertEqual(training.capql_replica_index(experiment.METHOD), 0)
        self.assertTrue(training.is_capql_method(experiment.METHOD))
        self.assertIn(experiment.METHOD, training.METHODS)
        self.assertEqual(experiment.controls.METHODS, {"v25": experiment.previous.extension.METHODS["adaptive"],
                                                      "capql": "capql"})

    def test_summary_pairs_old_controls_without_charging_reused_work_as_new(self):
        with tempfile.TemporaryDirectory() as folder:
            root, old = Path(folder) / "new", Path(folder) / "old"
            for env in experiment.ENVIRONMENTS:
                spec = experiment.previous.extension.ENVIRONMENTS[env]
                for seed in experiment.SEEDS:
                    for method, value in (("v25", 14.), ("capql", 10.), (experiment.METHOD, 12.)):
                        is_new = method == experiment.METHOD
                        row = dict(protocol=experiment.PROTOCOL if is_new else experiment.controls.PROTOCOL,
                            env=env, method=method, seed=seed, complete=True,
                            evaluation=dict(
                                evaluation_weights=experiment.previous.training.full_simplex_grid(spec["reward_dim"]).tolist(),
                                episode_seeds_per_mode=[[100000 + seed * 1000 + mode * 10 + i for i in range(3)]
                                                        for mode in range(spec["num_modes"])],
                                ref_point=experiment.previous.training.reference_point_for_env(env, spec["reward_dim"]).tolist()),
                            mode_history=[[0, 0]], completed_budget=experiment.controls.budget(env),
                            metrics=dict(direct_utility=dict(mean=value, per_mode=[value] * spec["num_modes"]),
                                final_hv=value * 100, mode_final_hv=[value * 100] * spec["num_modes"],
                                training_seconds=3., evaluation_seconds=4.))
                        if is_new:
                            row["training_preference_sampler"] = experiment.sampler_spec(env)
                        path = (root if is_new else old) / env.replace("-", "_") / method / f"seed{seed}.json"
                        path.parent.mkdir(parents=True, exist_ok=True)
                        experiment.write_report(path, row)
            result = experiment.summarize(root, old)
            self.assertEqual(result["new_completed_budget"], dict(training_interactions=3000000,
                evaluation_interactions=10080000, environment_initialization_probe_steps=80))
            self.assertEqual(result["reused_control_budget"]["training_interactions"], 6000000)
            for row in result["environments"].values():
                self.assertEqual(row["paired"]["v25_minus_full_simplex"]["direct_utility"]["mean_delta"], 2.)
                self.assertEqual(row["paired"]["full_simplex_minus_native"]["direct_utility"]["mean_delta"], 2.)
                self.assertEqual(row["per_mode_utility_mean_delta"], [2.] * 4)
            path = root / "mo_hopper_v5" / experiment.METHOD / "seed1620.json"
            original = json.loads(path.read_text())
            for field in ("mode_history", "completed_budget", "training_preference_sampler"):
                changed = copy.deepcopy(original)
                if field == "mode_history":
                    changed[field].append([30000, 1])
                elif field == "completed_budget":
                    changed[field]["training_interactions"] -= 1
                else:
                    changed[field]["concentration"][0] = 2.
                experiment.write_report(path, changed)
                with self.subTest(field=field), self.assertRaises(AssertionError):
                    experiment.summarize(root, old)
            path.unlink()
            with self.assertRaises(FileNotFoundError):
                experiment.summarize(root, old)

    @unittest.skipUnless(importlib.util.find_spec("morl_baselines"), "CAPQL checkout is on the server")
    def test_real_short_runs_restore_native_sampler_and_final_evaluation(self):
        previous, training = experiment.previous, experiment.previous.training
        training.ensure_wandb_stub()
        from morl_baselines.multi_policy.capql import capql as module

        original_env, capql_init, native_sampler = training.make_piecewise_mo_env, module.CAPQL.__init__, module.WeightSamplerAngle
        uniform_sample = training.UniformSimplexWeightSampler.sample
        sampled = []

        def short_env(*args, **kw):
            kw.update(switch_interval=12, mode_order="sequential")
            env = original_env(*args, **kw)
            env.env._max_episode_steps = 4
            return env

        def small_capql(agent, *args, **kw):
            kw.update(buffer_size=128, net_arch=[8, 8], batch_size=4, learning_starts=5)
            capql_init(agent, *args, **kw)

        def track_sample(sampler, count):
            result = uniform_sample(sampler, count)
            sampled.append(result.clone())
            return result

        with tempfile.TemporaryDirectory() as folder, \
             patch.object(previous, "TOTAL_STEPS", 80), patch.object(experiment, "SEEDS", (912345,)), \
             patch.object(training, "make_piecewise_mo_env", side_effect=short_env), \
             patch.object(module.CAPQL, "__init__", new=small_capql), \
             patch.object(training.UniformSimplexWeightSampler, "sample", new=track_sample), patch("builtins.print"):
            for env in experiment.ENVIRONMENTS:
                reference = []
                for suffix in ("before", "after"):
                    if suffix == "after":
                        new = experiment.run(env, 912345, Path(folder) / "full")
                        self.assertEqual(new["completed_budget"], dict(training_interactions=80,
                            evaluation_interactions=1008, environment_initialization_probe_steps=2))
                        self.assertEqual(new["training_preference_sampler"], experiment.sampler_spec(env))
                        with patch.object(training, "train_capql", side_effect=AssertionError("retrained")):
                            self.assertEqual(experiment.run(env, 912345, Path(folder) / "full"), new)
                        self.assertIs(module.WeightSamplerAngle, native_sampler)
                    target = Path(folder) / env / suffix
                    training.train_capql(env, 80, 25000, 912345, str(target),
                                         eval_freq=100000, device="cpu", method="capql")
                    data = json.loads((target / "capql_seed912345.json").read_text())
                    reference.append(previous.validate(data, env, "capql", 912345,
                                                       target / "capql_seed912345_model.pt"))
                for metric in ("direct_utility", "final_hv", "mode_final_hv"):
                    self.assertEqual(reference[0]["metrics"][metric], reference[1]["metrics"][metric])
                self.assertEqual(new["evaluation"], reference[0]["evaluation"])
                self.assertEqual(new["mode_history"], reference[0]["mode_history"])
                self.assertEqual(new["completed_budget"], reference[0]["completed_budget"])
        self.assertEqual(len(sampled), 160)
        self.assertIs(module.WeightSamplerAngle, native_sampler)
        print("FULL_SIMPLEX_CAPQL_SMOKE training=480 evaluation=6048 probes=12")


if __name__ == "__main__":
    unittest.main()
