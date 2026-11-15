"""Focused checks for fresh final-policy comparison."""

import tempfile
import unittest
from pathlib import Path

import run_final_policy_fresh as experiment


class FinalPolicyFreshTests(unittest.TestCase):
    def test_seed_roster_and_direct_utility(self):
        self.assertEqual(experiment.SEEDS, (1201, 1202, 1203, 1204, 1205))
        data = {"evaluation_weights": [[1.0, 0.0], [0.0, 1.0]],
                "pareto_per_mode": [[
                    {"weight": [1.0, 0.0], "return": [2.0, 3.0]},
                    {"weight": [0.0, 1.0], "return": [4.0, 5.0]}]]}
        self.assertEqual(experiment.direct_utility(data),
                         {"mean": 3.5, "per_mode": [3.5]})
        data["pareto_per_mode"][0].reverse()
        with self.assertRaises(AssertionError):
            experiment.direct_utility(data)

    def test_single_snapshot_and_final_only_baseline(self):
        env_name = "mo-halfcheetah-v5"
        num_modes = experiment.extension.ENVIRONMENTS[env_name]["num_modes"]
        weights = [[i / 20, 1 - i / 20] for i in range(21)]
        with tempfile.TemporaryDirectory() as folder:
            model = Path(folder) / "model.pt"
            model.touch()
            data = {
                "method": experiment.METHODS["v25"], "env": env_name,
                "seed": 1201, "total_steps": 75000, "switch_interval": 25000,
                "evaluation_protocol": "locked_full_simplex_v1",
                "training_interactions": 75000, "evaluation_interactions": 1,
                "environment_initialization_probe_steps": 2,
                "policy_archive_snapshots": [{"step": 75000}],
                "evaluation_weights": weights,
                "episode_seeds_per_mode": [[1, 2, 3]] * num_modes,
                "ref_point": [-1.0, -1.0],
                "pareto_per_mode": [[{"weight": weight,
                                      "return": [1.0, 2.0], "archive_step": 75000}
                                     for weight in weights] for _ in range(num_modes)],
                "hv_mean_mode": 2.0, "final_hv_mean_mode": 2.0,
                "final_hv_per_mode": [2.0] * num_modes,
                "training_seconds": 1.0, "evaluation_seconds": 1.0,
                "ground_truth": {"mode_history": [[0, 0]]},
            }
            self.assertTrue(experiment.validate(data, env_name, "v25", 1201, model)["complete"])
            data["policy_archive_snapshots"].append({"step": 72500})
            with self.assertRaises(AssertionError):
                experiment.validate(data, env_name, "v25", 1201, model)
            data["policy_archive_snapshots"] = None
            data["method"] = "capql"
            data["train_seed"] = 1201
            for front in data["pareto_per_mode"]:
                for point in front:
                    del point["archive_step"]
            self.assertTrue(experiment.validate(data, env_name, "capql", 1201, model)["complete"])


if __name__ == "__main__":
    unittest.main()
