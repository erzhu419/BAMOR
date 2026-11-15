import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import summarize_capql_archive_extension as summary


class CapqlArchiveExtensionTest(unittest.TestCase):
    def test_paired_readout_and_mismatch(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary) / "capql"
            reference = Path(temporary) / "bamor"
            env = "mo-halfcheetah-v5"
            folder = env.replace("-", "_")
            common = dict(
                evaluation_protocol="locked_full_simplex_v1",
                evaluation_weights=[[0.0, 1.0], [1.0, 0.0]],
                episode_seeds_per_mode=[[11, 12, 13], [21, 22, 23]],
                ref_point=[-1.0, -1.0],
                ground_truth={"mode_history": [[0, 0], [30000, 1]]},
                policy_archive_snapshots=[{"step": step} for step in range(5000, 50001, 2500)],
                training_seconds=4.0, evaluation_seconds=8.0,
            )
            capql = {**common, "method": "capql_archive", "env": env, "seed": 1011,
                     "total_steps": 50000, "switch_interval": 25000,
                     "training_interactions": 50000, "evaluation_interactions": 1000,
                     "environment_initialization_probe_steps": 2,
                     "hv_mean_mode": 10.0, "final_hv_mean_mode": 8.0,
                     "hv_per_mode": [9.0, 11.0]}
            path = root / folder / "capql_archive_seed1011.json"
            path.parent.mkdir(parents=True)
            path.write_text(json.dumps(capql))
            for condition, archive, final in (
                ("legacy_adaptive", 12.0, 9.0),
                ("legacy_blind", 9.0, 7.0),
                ("scoped_adaptive", 11.0, 10.0),
            ):
                method = summary.extension.METHODS[condition.split("_")[1]]
                bamor = {**common, "hv_mean_mode": archive,
                         "final_hv_mean_mode": final, "hv_per_mode": [archive, archive]}
                file = reference / folder / condition / f"{method}_seed1011.json"
                file.parent.mkdir(parents=True)
                file.write_text(json.dumps(bamor))
            with patch.dict(summary.extension.ENVIRONMENTS, {
                env: {"seeds": (1011,), "num_modes": 2}}, clear=True):
                result = summary.summarize(root, reference)
                self.assertEqual(result["completed_budget"]["training_interactions"], 50000)
                primary = result["environments"][env]["paired"]["legacy_adaptive"]
                self.assertEqual(primary["archive_hv"]["mean_delta"], 2.0)
                self.assertEqual(primary["final_hv"]["mean_delta"], 1.0)
                self.assertEqual(primary["archive_hv"]["wins"], 1)
                capql["episode_seeds_per_mode"][0][0] = 99
                path.write_text(json.dumps(capql))
                with self.assertRaises(AssertionError):
                    summary.summarize(root, reference)


if __name__ == "__main__":
    unittest.main()
