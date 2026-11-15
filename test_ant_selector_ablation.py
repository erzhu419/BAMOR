"""Focused protocol check for the Ant selector ablation."""

import tempfile
import unittest
from pathlib import Path

import run_ant_selector_ablation as experiment
import run_final_policy_fresh as reference


class AntSelectorAblationTests(unittest.TestCase):
    def test_validate_matched_final_only_run(self):
        weights = [[i / 5, j / 5, (5 - i - j) / 5]
                   for i in range(6) for j in range(6 - i)]
        front = [{"weight": weight, "return": [1.0, 2.0, 3.0],
                  "archive_step": 75000} for weight in weights]
        data = {"method": experiment.METHOD, "env": experiment.ENV,
                "seed": 1201, "total_steps": 75000, "switch_interval": 25000,
                "evaluation_protocol": "locked_full_simplex_v1",
                "training_interactions": 75000, "evaluation_interactions": 1,
                "environment_initialization_probe_steps": 2,
                "policy_archive_snapshots": [{"step": 75000}],
                "pareto_per_mode": [front] * 3,
                "evaluation_weights": weights,
                "episode_seeds_per_mode": [[1, 2, 3]] * 3,
                "ref_point": [-1.0, -1.0, -1.0],
                "ground_truth": {"mode_history": [[0, 0]]},
                "hv_mean_mode": 1.0, "final_hv_mean_mode": 1.0,
                "final_hv_per_mode": [1.0] * 3,
                "training_seconds": 1.0, "evaluation_seconds": 1.0}
        adaptive = {"evaluation": {key: data[key] for key in (
            "evaluation_weights", "episode_seeds_per_mode", "ref_point")},
            "mode_history": [[0, 0]]}
        with tempfile.TemporaryDirectory() as folder:
            model = Path(folder) / "model.pt"
            model.touch()
            report = experiment.validate(data, 1201, model, adaptive)
            self.assertTrue(report["complete"])
            self.assertEqual(report["metrics"]["direct_utility"],
                             reference.direct_utility(data))
            adaptive["mode_history"] = [[0, 1]]
            with self.assertRaises(AssertionError):
                experiment.validate(data, 1201, model, adaptive)


if __name__ == "__main__":
    unittest.main()
