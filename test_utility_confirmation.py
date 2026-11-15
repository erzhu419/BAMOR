import copy
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

import run_utility_confirmation as experiment


class UtilityConfirmationTests(unittest.TestCase):
    def test_paired_bootstrap_uses_seed_pairs_not_pooled_groups(self):
        indices = np.random.default_rng(0).integers(0, 20, (20000, 20))
        controls = list(np.arange(20, dtype=float) + 10)
        values = [x + 2. for x in controls]
        result = experiment.paired_summary(values, controls, indices)
        self.assertEqual(result["paired_mean_ci95"], [2., 2.])
        self.assertEqual(result["paired_mean_ci97_5"], [2., 2.])
        self.assertEqual(result["wins"], 20)
        self.assertEqual(result["paired_deltas"], [2.] * 20)
        mixed = experiment.paired_summary([b + (-1.) ** i * 5 for i, b in enumerate(controls)], controls, indices)
        self.assertLess(mixed["paired_mean_ci97_5"][0], 0)
        self.assertGreater(mixed["paired_mean_ci97_5"][1], 0)

    def test_full_matrix_and_mismatched_budget_or_schedule(self):
        old, training = experiment.previous, experiment.previous.training
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            for env in experiment.ENVIRONMENTS:
                spec = old.extension.ENVIRONMENTS[env]
                for seed in experiment.SEEDS:
                    for method in experiment.METHODS:
                        value = 12. if method == "v25" else 10.
                        row = dict(protocol=experiment.PROTOCOL, env=env, method=method, seed=seed, complete=True,
                            evaluation=dict(evaluation_weights=training.full_simplex_grid(spec["reward_dim"]).tolist(),
                                episode_seeds_per_mode=[[100000 + seed * 1000 + mode * 10 + i for i in range(3)]
                                                        for mode in range(spec["num_modes"])],
                                ref_point=training.reference_point_for_env(env, spec["reward_dim"]).tolist()),
                            mode_history=[[0, 0]], completed_budget=experiment.budget(env),
                            metrics=dict(direct_utility=dict(mean=value, per_mode=[value] * spec["num_modes"]),
                                final_hv=value * 100, mode_final_hv=[value * 100] * spec["num_modes"],
                                training_seconds=3., evaluation_seconds=4.))
                        path = root / env.replace("-", "_") / method / f"seed{seed}.json"
                        path.parent.mkdir(parents=True, exist_ok=True)
                        experiment.write_report(path, row)
            result = experiment.summarize(root)
            self.assertEqual(result["completed_budget"], dict(training_interactions=6000000,
                evaluation_interactions=20160000, environment_initialization_probe_steps=160))
            for row in result["environments"].values():
                self.assertEqual(row["paired"]["direct_utility"]["mean_delta"], 2.)
                self.assertEqual(row["per_mode_utility_mean_delta"], [2.] * 4)
            path = root / "mo_hopper_v5" / "capql" / "seed1620.json"
            original = json.loads(path.read_text())
            for field in ("mode_history", "completed_budget", "evaluation", "complete"):
                changed = copy.deepcopy(original)
                if field == "mode_history":
                    changed[field].append([30000, 1])
                elif field == "completed_budget":
                    changed[field]["training_interactions"] -= 1
                elif field == "evaluation":
                    changed[field]["episode_seeds_per_mode"][0][0] += 1
                else:
                    changed[field] = False
                experiment.write_report(path, changed)
                with self.subTest(field=field), self.assertRaises(AssertionError):
                    experiment.summarize(root)
            path.unlink()
            with self.assertRaises(FileNotFoundError):
                experiment.summarize(root)

    @unittest.skipUnless(importlib.util.find_spec("morl_baselines"), "CAPQL checkout is on the server")
    def test_real_short_runs_match_previous_entrypoint(self):
        old, training = experiment.previous, experiment.previous.training
        training.ensure_wandb_stub()
        from morl_baselines.multi_policy.capql.capql import CAPQL

        original_env, capql_init = training.make_piecewise_mo_env, CAPQL.__init__
        kwargs = dict(hidden=8, lr=3e-4, gamma=.99, batch_size=4, learning_starts=5,
                      eval_freq=100000, device="cpu")

        def short_env(*args, **kw):
            kw.update(switch_interval=12, mode_order="sequential")
            env = original_env(*args, **kw)
            env.env._max_episode_steps = 4
            return env

        def small_capql(agent, *args, **kw):
            kw.update(buffer_size=128, net_arch=[8, 8], batch_size=4, learning_starts=5)
            capql_init(agent, *args, **kw)

        with tempfile.TemporaryDirectory() as folder, \
             patch.multiple(old, SEEDS=(17008,), TOTAL_STEPS=80), \
             patch.object(experiment, "SEEDS", (17008,)), \
             patch.object(old.pilot, "TRAIN_KWARGS", kwargs), \
             patch.object(training, "make_piecewise_mo_env", side_effect=short_env), \
             patch.object(CAPQL, "__init__", new=small_capql), patch("builtins.print"):
            for env in experiment.ENVIRONMENTS:
                for method in experiment.METHODS:
                    reference = old.run(env, method, 17008, Path(folder) / "previous")
                    new = experiment.run(env, method, 17008, Path(folder) / "confirmation")
                    self.assertTrue(new["complete"])
                    self.assertEqual(new["protocol"], experiment.PROTOCOL)
                    self.assertEqual(new["evaluation"], reference["evaluation"])
                    self.assertEqual(new["mode_history"], reference["mode_history"])
                    self.assertEqual(new["completed_budget"], dict(training_interactions=80,
                        evaluation_interactions=1008, environment_initialization_probe_steps=2))
                    for key in ("direct_utility", "final_hv", "mode_final_hv"):
                        self.assertEqual(new["metrics"][key], reference["metrics"][key])
                    with patch.object(training, "train_method", side_effect=AssertionError("retrained")), \
                         patch.object(training, "train_capql", side_effect=AssertionError("retrained")):
                        self.assertEqual(experiment.run(env, method, 17008, Path(folder) / "confirmation"), new)
            path = Path(folder) / "incomplete" / "mo_hopper_v5" / "v25" / "seed17008.json"
            path.parent.mkdir(parents=True)
            experiment.write_report(path, dict(protocol=experiment.PROTOCOL, env="mo-hopper-v5",
                                               method="v25", seed=17008, complete=False))
            with self.assertRaisesRegex(RuntimeError, "Incomplete attempt preserved"):
                experiment.run("mo-hopper-v5", "v25", 17008, Path(folder) / "incomplete")
        print("UTILITY_CONFIRMATION_SMOKE training=640 evaluation=8064 probes=16")


if __name__ == "__main__":
    unittest.main()
