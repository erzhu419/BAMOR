import json
import unittest
from pathlib import Path
from unittest.mock import patch

import summarize_ant_mechanism_alignment as alignment


class AlignmentTests(unittest.TestCase):
    def test_signed_scores_and_ties(self):
        result = alignment.paired([-3, 0, 4], [-5, 0, 5])
        self.assertEqual(result["paired_deltas"], [2, 0, -1])
        self.assertEqual((result["wins"], result["losses"], result["ties"]), (1, 1, 1))
        self.assertNotIn("relative_mean_delta", result)

    def test_existing_cohorts_reconcile_without_pooling(self):
        result = alignment.summarize(Path(__file__).parent / "results")
        self.assertEqual(result["development"]["seeds"], list(range(1201, 1206)))
        self.assertEqual(result["independent_both_searches_off"]["seeds"], list(range(1301, 1306)))
        primary = result["development"]["comparisons"]["both_off_blind_minus_actor_off_blind"][alignment.METRICS[0]]
        self.assertEqual((primary["wins"], primary["losses"]), (4, 1))
        self.assertTrue(all(value == 0 for value in result["new_budget"].values()))

    def test_reordered_seed_contrasts_are_rejected(self):
        original = Path.read_text

        def reordered(path, *args, **kwargs):
            text = original(path, *args, **kwargs)
            if str(path).endswith(alignment.SOURCES["actor"]):
                data = json.loads(text)
                data["runs"].reverse()
                return json.dumps(data)
            return text

        with patch.object(Path, "read_text", reordered), self.assertRaises(AssertionError):
            alignment.summarize(Path(__file__).parent / "results")


if __name__ == "__main__":
    unittest.main()
