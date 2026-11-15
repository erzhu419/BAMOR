import copy
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
import torch

import run_ant_cost_drift as experiment


class CostDriftTests(unittest.TestCase):
    def test_fixed_banks_signal_direction_and_rng_isolation(self):
        trainer = SimpleNamespace(device="cpu", critic_select_policy=False)
        trainer.buffer = SimpleNamespace(size=12, ptr=12,
            states=np.arange(24, dtype=np.float32).reshape(12, 2) / 100,
            contexts=np.ones((12, 5), dtype=np.float32), timesteps=np.arange(1, 13)[:, None])
        trainer.replace_context_preferences = lambda c, w: torch.cat((w, c[:, 3:]), dim=1)
        strength = [1.]

        def action(s, c, deterministic):
            self.assertTrue(deterministic)
            np.random.rand()
            torch.rand(1)
            return strength[0] * (s + c[:, :2])

        trainer.select_action = action
        with patch.multiple(experiment, CUT=12, WIDTH=4, SAMPLES=3):
            banks = experiment.capture_banks(trainer)
        self.assertEqual(banks["early"]["timesteps"], [1, 2, 4])
        self.assertEqual(banks["recent"]["timesteps"], [9, 10, 12])
        original = copy.deepcopy(banks)
        trainer.buffer.states[:] = -999
        trainer.buffer.contexts[:] = -999
        numpy_rng, torch_rng = np.random.get_state(), torch.get_rng_state()
        first = experiment.measure_cost(trainer, banks)
        strength[0] = 2.
        second = experiment.measure_cost(trainer, banks)
        np.testing.assert_array_equal(np.random.get_state()[1], numpy_rng[1])
        self.assertEqual(np.random.get_state()[2:], numpy_rng[2:])
        self.assertTrue(torch.equal(torch.get_rng_state(), torch_rng))
        for bank in banks:
            np.testing.assert_array_equal(banks[bank]["states"], original[bank]["states"])
            np.testing.assert_array_equal(banks[bank]["contexts"], original[bank]["contexts"])
            self.assertLess(first[bank]["mixed_score"], 0)
            self.assertAlmostEqual(second[bank]["mixed_score"], 4 * first[bank]["mixed_score"])
        weights = experiment.training.full_simplex_grid(3)
        mixed = (weights[:, 2] > 0) & (weights[:, 2] < 1)
        expected = np.asarray(first["early"]["negative_cost_by_weight"]) * weights[:, 2]
        self.assertAlmostEqual(first["early"]["mixed_score"], expected[mixed].mean())

    def test_readout_uses_same_intervals_and_preserves_ties(self):
        report, reference = dict(branches={}), dict(branches={})
        times = (10, 20, 30)
        for name, scores in (("full", [4., 7., 6.]), ("no_bc", [5., 4., 6.])):
            obs = [dict(step=0, banks={k: dict(mixed_score=-999.) for k in ("early", "recent")})]
            obs += [dict(step=t, banks=dict(early=dict(mixed_score=s), recent=dict(mixed_score=-s)))
                    for t, s in zip(times, scores)]
            report["branches"][name] = dict(observations=obs)
            reference["branches"][name] = dict(measurements=[
                dict(step=t, returns=np.full((1, 21, 3), s).tolist()) for t, s in zip(times, scores)])
        with patch.multiple(experiment, CUT=0, TIMES=times):
            result = experiment.compare_history(report, reference)
        self.assertEqual([(x["branch"], x["start"], x["end"]) for x in result["intervals"]],
                         [("full", 10, 20), ("full", 20, 30), ("no_bc", 10, 20), ("no_bc", 20, 30)])
        self.assertAlmostEqual(result["tracking"]["early"]["spearman"], 1.)
        self.assertAlmostEqual(result["tracking"]["recent"]["spearman"], -1.)
        self.assertEqual(result["tracking"]["early"]["sign_agreements"], 4)
        stats = experiment.tracking_stats([0., 0.], [-1., 1.])
        self.assertIsNone(stats["spearman"])
        self.assertEqual(stats["ties"], 2)

    def test_driver_reproduces_training_and_final_grids_with_no_middle_rollouts(self):
        temporal, previous, training = experiment.temporal, experiment.previous, experiment.training
        make_env, trainer_cls = training.make_piecewise_mo_env, training.ConsistentTargetMOSACTrainer
        init = previous.NativeSelector.__init__
        kwargs = dict(hidden=8, batch_size=4, learning_starts=5, eval_freq=1000, log_freq=1000, device="cpu")

        def short_env(*args, **kw):
            kw.update(switch_interval=12, mode_order="sequential")
            env = make_env(*args, **kw)
            env.env._max_episode_steps = 4
            return env

        def short_selector(selector, *args, **kw):
            kw["block_steps"] = 4
            init(selector, *args, **kw)

        with tempfile.TemporaryDirectory() as folder, \
             patch.multiple(previous, SEEDS=(17007,), CUT=53, BLOCK_END=57, END=80, KWARGS=kwargs), \
             patch.multiple(temporal, SEEDS=(17007,), CUT=53, BLOCK_END=57, END=80,
                            MIDDLE=(61, 65, 69, 73, 77), KWARGS=kwargs), \
             patch.multiple(experiment, SEEDS=(17007,), CUT=53, BLOCK_END=57, END=80, KWARGS=kwargs,
                            TIMES=(57, 61, 65, 69, 73, 77, 80), WIDTH=8, SAMPLES=4), \
             patch.object(training, "make_piecewise_mo_env", side_effect=short_env), \
             patch.object(training, "ConsistentTargetMOSACTrainer",
                          side_effect=lambda *a, **kw: trainer_cls(*a, buffer_capacity=64, **kw)), \
             patch.object(previous.NativeSelector, "__init__", new=short_selector), patch("builtins.print"):
            old = previous.run(17007, Path(folder) / "factorial")
            with patch.object(temporal, "load_reference", return_value=old):
                reference = temporal.run(17007, Path(folder) / "temporal")
            root = Path(folder) / "cost"
            with patch.object(experiment, "load_reference", return_value=reference):
                report = experiment.run(17007, root)
                self.assertTrue(experiment.run(17007, root)["complete"])
                incomplete = Path(folder) / "incomplete"
                incomplete.mkdir()
                experiment.base.write_report(incomplete / "seed17007.json",
                    dict(protocol=experiment.PROTOCOL, seed=17007, complete=False))
                with self.assertRaisesRegex(RuntimeError, "Incomplete attempt preserved"):
                    experiment.run(17007, incomplete)
            self.assertEqual(report["completed_budget"], dict(training_interactions=107, learner_updates=103,
                evaluation_interactions=1512, environment_initialization_probe_steps=4))
            self.assertEqual(report["banks"]["early"]["timesteps"], [1, 3, 5, 8])
            self.assertEqual(report["banks"]["recent"]["timesteps"], [46, 48, 50, 53])
            for name, branch in report["branches"].items():
                self.assertTrue(branch["complete"] and branch["final_grid_reproduced"])
                self.assertEqual(branch["training_reproduced_through"], 80)
                self.assertEqual(len(branch["observations"]), 8)
            self.assertEqual(report["branches"]["no_bc"]["routing"]["rng_matched_steps"], 27)
            result = experiment.summarize(root)
            self.assertEqual(result["runs"][0]["tracking"]["early"]["n"], 12)
            corrupted = copy.deepcopy(report)
            corrupted["branches"]["full"]["final_grid"]["returns"][0][0][0] += 1
            with self.assertRaises(AssertionError):
                experiment.summarize_seed(corrupted, reference)
        print("ANT_COST_DRIFT_SMOKE training=375 updates=363 evaluation=13860 probes=16")


if __name__ == "__main__":
    unittest.main()
