import json
from pathlib import Path
import tempfile
import unittest

from summarize_matched_pilot import METHODS, seed_diagnostics, summarize


def result(archive=5, final=3):
    return {
        "evaluation_protocol": "locked_full_simplex_v1",
        "policy_archive_snapshots": [{"step": s} for s in range(5000, 50001, 2500)],
        "total_steps": 50000,
        "ground_truth": {"num_modes": 3, "mode_history": [[0, 0], [20000, 2]]},
        "hv_mean_mode": archive, "final_hv_mean_mode": final,
        "hv_per_mode": [archive] * 3, "final_hv_per_mode": [final] * 3,
        "eum_per_mode": [1, 2, 3], "training_seconds": 10, "evaluation_seconds": 20,
        "evaluation_weights": [[1, 0, 0]], "episode_seeds_per_mode": [[1, 2, 3]],
        "ref_point": [0, 0, 0],
    }


class SummaryTests(unittest.TestCase):
    def test_custom_reference_for_target_repair_comparison(self):
        methods = ["bamor_v24_target_consistent_fixed_archive_mix", METHODS[1]]
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            env = root / "mo_halfcheetah_v5"
            env.mkdir()
            for method, value in zip(methods, [7, 4]):
                (env / f"{method}_seed23.json").write_text(json.dumps(result(value, 3)))
            data = summarize(root, [23], methods, ["mo-halfcheetah-v5"])
            self.assertEqual(data["completed"], 2)
            self.assertEqual(data["expected"], 2)
            self.assertEqual(data["reference_method"], methods[0])
            paired = data["environments"]["mo-halfcheetah-v5"]["paired"][methods[1]]
            self.assertEqual(paired["mean_difference"], 3)

    def test_mode_exposure_and_archive_gain(self):
        data = seed_diagnostics(result())
        self.assertEqual(data["training_steps_per_mode"], [20000, 0, 30000])
        self.assertEqual(data["training_switches"], 1)
        self.assertEqual(data["archive_gain"], 2)
        self.assertEqual(data["seen_mode_hv"], 5)
        self.assertEqual(data["unseen_mode_hv"], 5)
        complete = result()
        complete["ground_truth"]["mode_history"] = [[0, 0], [10000, 1], [30000, 2]]
        self.assertIsNone(seed_diagnostics(complete)["unseen_mode_hv"])

    def test_pair_only_common_completed_seeds_and_reject_different_schedule(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            env = root / "mo_ant_v5"
            env.mkdir()
            (env / f"{METHODS[0]}_seed20.json").write_text(json.dumps(result(5, 2)))
            (env / f"{METHODS[0]}_seed21.json").write_text(json.dumps(result(100, 100)))
            baseline = env / f"{METHODS[2]}_seed20.json"
            baseline.write_text(json.dumps(result(4, 3)))
            data = summarize(root, [20, 21])
            paired = data["environments"]["mo-ant-v5"]["paired"][METHODS[2]]
            self.assertEqual(paired["seeds"], [20])
            self.assertEqual(paired["mean_difference"], 1)
            self.assertEqual(paired["final_actor_differences"], [-1])
            self.assertEqual(paired["mode_hv_differences"], [[1, 1, 1]])
            self.assertEqual(paired["seen_mode_differences"], [1])
            self.assertEqual(paired["unseen_mode_differences"], [1])
            changed = result(4, 3)
            changed["ground_truth"]["mode_history"] = [[0, 0], [25000, 2]]
            baseline.write_text(json.dumps(changed))
            with self.assertRaisesRegex(ValueError, "Unmatched training mode schedule"):
                summarize(root, [20, 21])


if __name__ == "__main__":
    unittest.main()
