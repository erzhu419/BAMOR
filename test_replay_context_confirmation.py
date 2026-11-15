import copy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import run_replay_context_confirmation as experiment
import summarize_replay_context_confirmation as summary
from test_replay_refresh_summary import result as fixture


class ConfirmationTests(unittest.TestCase):
    def test_frozen_runner_arms_can_finish_without_each_other(self):
        for condition in experiment.CONDITIONS:
            with self.subTest(condition=condition), tempfile.TemporaryDirectory() as directory:
                def train(*args, **kwargs):
                    self.assertEqual(args[:5], ("mo-halfcheetah-v5", experiment.pilot.METHOD, 50000, 25000, 29))
                    self.assertEqual({k: v for k, v in kwargs.items() if k != "training_probe"},
                                     dict(hidden=64, learning_starts=5000, eval_freq=100000, device="cpu"))
                    hook = kwargs["training_probe"]
                    if condition == "refresh":
                        self.assertIsInstance(hook, experiment.pilot.ReplayContextRefresh)
                        self.assertEqual(hook.steps, experiment.pilot.REFRESH_STEPS)
                        hook.cases = [{"step": s, "transitions_reencoded": s, "seconds": 1.}
                                      for s in hook.steps]
                    else:
                        self.assertIsNone(hook)
                    Path(args[5]).mkdir(parents=True, exist_ok=True)
                    data = fixture()
                    data["seed"] = 29
                    data["episode_seeds_per_mode"] = [[129000 + 10*m + r for r in range(3)] for m in range(4)]
                    return data

                with patch.object(experiment.pilot.training, "train_method", side_effect=train):
                    data = experiment.run(condition, 29, directory)
                saved = json.loads((Path(directory) / condition / f"{experiment.pilot.METHOD}_seed29.json").read_text())
                self.assertEqual(saved, data)
                metrics = summary.validate_run(saved, condition, 29)
                self.assertNotIn("posthoc_late_mean_hv", metrics)
                self.assertEqual(metrics["late_min_hv"], metrics["final_hv"])
                if condition == "refresh":
                    self.assertTrue(json.loads((Path(directory) / condition / "seed29_refresh.json").read_text())["complete"])
                    bad = copy.deepcopy(saved)
                    bad["replay_context_refresh"]["complete"] = False
                    with self.assertRaises(AssertionError):
                        summary.validate_run(bad, condition, 29)
                bad = copy.deepcopy(saved)
                bad["replay_context_confirmation"]["late_snapshot_steps"] = [50000]
                with self.assertRaises(AssertionError):
                    summary.validate_run(bad, condition, 29)

    def test_final_gain_does_not_hide_late_regression_or_missing_seed(self):
        conditions = {c: {"per_seed": {s: {"final_hv": 10., "archive_hv": 12.,
                       "late_mean_hv": 9., "late_min_hv": 8., "final_mode_hv": [10.] * 4}
                       for s in experiment.SEEDS}} for c in experiment.CONDITIONS}
        for r in conditions["refresh"]["per_seed"].values():
            r["final_hv"] = 11.
            r["late_mean_hv"] = 7.
        report = summary.aggregate(conditions)
        self.assertTrue(all(report["primary_checks"].values()))
        self.assertFalse(report["secondary_stability_checks"]["late_mean_hv_min_not_worse"])
        self.assertFalse(report["ready_for_larger_confirmation"])
        self.assertEqual(report["final_wins"], 5)
        self.assertEqual(report["paired_refresh_minus_stored"][29]["late_mean_hv"], -2.)
        conditions["stored"]["per_seed"].pop(33)
        with self.assertRaises(AssertionError):
            summary.aggregate(conditions)


if __name__ == "__main__":
    unittest.main()
