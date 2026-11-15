"""Focused checks for the matched long-budget comparison."""

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import torch

import run_long_budget_match as experiment


class LongBudgetMatchTests(unittest.TestCase):
    def test_last_twenty_snapshots_are_fixed(self):
        self.assertEqual(len(experiment.STEPS), 20)
        self.assertEqual((experiment.STEPS[0], experiment.STEPS[-1]), (27500, 75000))

    def test_final_utility_uses_matching_preference(self):
        data = {"evaluation_weights": [[1.0, 0.0], [0.0, 1.0]],
                "policy_archive_snapshots": [{"step": 50000}],
                "pareto_per_mode": [[
                    {"archive_step": 50000, "weight": [1.0, 0.0], "return": [2.0, 3.0]},
                    {"archive_step": 50000, "weight": [0.0, 1.0], "return": [4.0, 5.0]},
                ]]}
        self.assertEqual(experiment.final_utility(data), {"mean": 3.5, "per_mode": [3.5]})
        data["pareto_per_mode"][0].reverse()
        with self.assertRaises(AssertionError):
            experiment.final_utility(data)

    def test_exact_50k_actor_prefix(self):
        with tempfile.TemporaryDirectory() as folder:
            old, new = Path(folder) / "old.pt", Path(folder) / "new.pt"
            torch.save({"policy": {"w": torch.tensor([1.0, 2.0])}}, old)
            torch.save({"policy_archive": [
                {"step": 50000, "policy": {"w": torch.tensor([1.0, 2.0])}}]}, new)
            self.assertTrue(experiment.policy_prefix_equal(old, new))
            torch.save({"policy_archive": [
                {"step": 50000, "policy": {"w": torch.tensor([1.0, 2.1])}}]}, new)
            self.assertFalse(experiment.policy_prefix_equal(old, new))

    def test_in_memory_mode_history_and_noncontinuation_report(self):
        env_name = "mo-halfcheetah-v5"
        num_modes = experiment.extension.ENVIRONMENTS[env_name]["num_modes"]
        def result(total_steps):
            return {
                "method": experiment.METHODS["v25"], "env": env_name,
                "seed": 1011, "total_steps": total_steps,
                "switch_interval": 25000, "evaluation_protocol": "locked_full_simplex_v1",
                "training_interactions": total_steps, "evaluation_interactions": 1,
                "environment_initialization_probe_steps": 2,
                "policy_archive_snapshots": [{"step": step} for step in (
                    experiment.STEPS if total_steps == 75000 else range(5000, 50001, 2500))],
                "evaluation_weights": [[1.0, 0.0, 0.0]],
                "episode_seeds_per_mode": [], "ref_point": [],
                "pareto_per_mode": [[{"archive_step": total_steps,
                                      "weight": [1.0, 0.0, 0.0],
                                      "return": [1.0, 2.0, 3.0]}]
                                    for _ in range(num_modes)],
                "ground_truth": {"mode_history": [(0, 0), (25000, 1)]},
                "hv_mean_mode": 1.0, "final_hv_mean_mode": 1.0,
                "hv_per_mode": [1.0] * num_modes,
                "final_hv_per_mode": [1.0] * num_modes,
                "training_seconds": 1.0, "evaluation_seconds": 1.0,
            }
        old, new = result(50000), result(75000)
        old["ground_truth"]["mode_history"] = [[0, 0], [25000, 1]]
        with patch.object(experiment, "policy_prefix_equal", return_value=True):
            valid = experiment.validate(new, old, env_name, "v25", 1011, None, None)
        self.assertTrue(valid["complete"])
        with patch.object(experiment, "policy_prefix_equal", return_value=False):
            invalid = experiment.validate(new, old, env_name, "v25", 1011, None, None)
        self.assertFalse(invalid["complete"])
        self.assertEqual(invalid["validation_issue"], "50k_policy_prefix_mismatch")


if __name__ == "__main__":
    unittest.main()
