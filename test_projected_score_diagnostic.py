import copy
import unittest

from diagnose_projected_allocation_scores import diagnose_run, episode_half, summarize


class ScoreDiagnosticTest(unittest.TestCase):
    def test_full_episode_window_boundaries(self):
        self.assertEqual(episode_half(dict(step=750, length=750), 1, 750, 1500), 0)
        self.assertEqual(episode_half(dict(step=1500, length=750), 1, 750, 1500), 1)
        self.assertIsNone(episode_half(dict(step=1000, length=1000), 1, 750, 1500))
        self.assertIsNone(episode_half(dict(step=900, length=200), 1, 750, 1500))

    def test_score_reconstruction_and_mode_crossing_exclusion(self):
        data = dict(mode_switches=[], episode_rewards=[
            dict(step=750, length=100, **{"return": [100, 100]}),
            dict(step=1500, length=50, **{"return": [50, 50]})],
            adaptive_operator_final=dict(block_history=[dict(
                end_step=1500, steps=1500, half_counts=[750, 750], complete=True,
                half_mean_rewards=[[0, 0], [1, 1]], scale_before_block=[1, 1],
                progress_score=0.7615941559557649, operator="uniform")]))
        result = summarize(diagnose_run(data))
        self.assertEqual(result["positive_score_episode_utility_declines"], 1)
        self.assertEqual(result["positive_score_episode_length_declines"], 1)
        crossing = copy.deepcopy(data)
        crossing["mode_switches"] = [dict(step=750, kind="env_truth")]
        result = summarize(diagnose_run(crossing))
        self.assertEqual(result["mode_crossing_blocks"], 1)
        self.assertEqual(result["stationary_blocks_with_two_full_episode_halves"], 0)
        data["adaptive_operator_final"]["block_history"][0]["progress_score"] = 0
        with self.assertRaises(ValueError):
            diagnose_run(data)


if __name__ == "__main__":
    unittest.main()
