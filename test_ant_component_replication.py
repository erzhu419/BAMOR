import copy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

import run_ant_component_replication as experiment


class ComponentReplicationTests(unittest.TestCase):
    def test_window_override_retains_native_prefix_tail_and_draws(self):
        for count in (12, 40, 41):
            selector = experiment.WindowSelector(3)
            selector.counts = dict.fromkeys(selector.operators, 2)
            selector.step_counts = dict.fromkeys(selector.operators, 1500)
            selector.block_history = [{}] * count
            ordinary = experiment.NativeSelector(3)
            ordinary.__dict__.update(copy.deepcopy(selector.__dict__))
            np.random.seed(34)
            drawn = ordinary.select()
            expected = np.random.get_state()
            np.random.seed(34)
            actual = selector.select()
            self.assertEqual(actual, experiment.base.B if count == 40 else drawn)
            np.testing.assert_array_equal(np.random.get_state()[1], expected[1])
            self.assertEqual(np.random.get_state()[2:], expected[2:])
            self.assertEqual(len(selector.draws), int(count >= 40))
            self.assertNotIn("score_credit_rule", selector.state_dict())

    def test_short_four_cell_driver_and_accounting(self):
        base, training = experiment.base, experiment.base.training
        make_env, trainer_cls = training.make_piecewise_mo_env, training.ConsistentTargetMOSACTrainer
        init = experiment.NativeSelector.__init__
        kwargs = dict(hidden=8, batch_size=4, learning_starts=5, eval_freq=1000,
                      log_freq=1000, device="cpu")

        def short_env(*args, **kw):
            kw["switch_interval"] = 12
            kw["mode_order"] = "sequential"
            env = make_env(*args, **kw)
            env.env._max_episode_steps = 4
            return env

        def short_selector(selector, *args, **kw):
            kw["block_steps"] = 4
            init(selector, *args, **kw)

        with tempfile.TemporaryDirectory() as folder, \
             patch.multiple(experiment, SEEDS=(17004,), CUT=53, BLOCK_END=57, END=61, KWARGS=kwargs), \
             patch.object(training, "make_piecewise_mo_env", side_effect=short_env), \
             patch.object(training, "ConsistentTargetMOSACTrainer",
                          side_effect=lambda *a, **kw: trainer_cls(*a, buffer_capacity=64, **kw)), \
             patch.object(experiment.NativeSelector, "__init__", new=short_selector), \
             patch("builtins.print"):
            root = Path(folder)
            report = experiment.run(17004, root)
            self.assertTrue(report["complete"])
            self.assertEqual(report["completed_budget"], dict(training_interactions=85,
                learner_updates=81, evaluation_interactions=6804, environment_initialization_probe_steps=8))
            full = report["branches"]["full"]
            self.assertGreater(len(full["mode_history"]), 1)
            for name, branch in report["branches"].items():
                self.assertEqual(branch["block_history"][:12], full["block_history"][:12])
                self.assertEqual(branch["selector_draws"][0]["executed"], base.B)
                if name != "full":
                    self.assertEqual(branch["routing"]["rng_matched_steps"], 8)
            result = json.loads((root / "seed17004_readout.json").read_text())
            for i in range(2):
                rows = [result["measurements"][name][-2:][i][experiment.PRIMARY] for name in experiment.BRANCHES]
                f, c, b, j = rows
                self.assertAlmostEqual(result["contrasts"]["interaction"][i][experiment.PRIMARY], j - c - b + f)
            self.assertTrue(experiment.summarize(root)["complete"])
            with patch.object(training, "train_method", side_effect=AssertionError("reran completed seed")):
                self.assertTrue(experiment.run(17004, root)["complete"])
            incomplete = root / "incomplete"
            incomplete.mkdir()
            base.write_report(incomplete / "seed17004.json", dict(protocol=experiment.PROTOCOL, seed=17004, complete=False))
            with self.assertRaisesRegex(RuntimeError, "Incomplete attempt preserved"):
                experiment.run(17004, incomplete)
            corrupted = copy.deepcopy(report)
            corrupted["branches"]["no_bc"]["block_history"][-1]["operator"] = "wrong"
            with self.assertRaises(AssertionError):
                experiment.summarize_seed(corrupted)
        print("ANT_REPLICATION_SMOKE training=85 updates=81 evaluation=6804 probes=8")

    def test_sign_test_uses_independent_seeds_not_preferences(self):
        metrics = (experiment.PRIMARY, "all_mode_direct_utility", "hv_mean_mode")
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            for seed in experiment.SEEDS:
                row = dict(protocol=experiment.PROTOCOL, seed=seed, complete=True, validation_passed=True,
                    completed_budget=dict.fromkeys(experiment.planned_budget(), 0), scope="test",
                    contrasts={"interaction": [{"step": step, **dict.fromkeys(metrics, -1.)}
                                               for step in (experiment.BLOCK_END, experiment.END)]})
                experiment.base.write_report(root / f"seed{seed}_readout.json", row)
            result = experiment.summarize(root)
            self.assertEqual(result["primary"]["one_sided_sign_p"], 1 / 32)
            self.assertTrue(result["primary"]["negative_direction_replicated"])
            row["contrasts"]["interaction"][0][experiment.PRIMARY] = 1.
            experiment.base.write_report(root / f"seed{seed}_readout.json", row)
            result = experiment.summarize(root)
            self.assertEqual(result["primary"]["one_sided_sign_p"], 6 / 32)
            self.assertFalse(result["primary"]["negative_direction_replicated"])


if __name__ == "__main__":
    unittest.main()
