import copy
import unittest

import numpy as np

import summarize_scoped_bc_credit as summary


class CreditSummaryTests(unittest.TestCase):
    def test_better_matched_utility_can_hide_worse_front(self):
        control = [[[1., 10., 1.], [1., 1., 10.], [10., 1., 1.]]]
        candidate = [[[3., 3., 3.]] * 3]
        result = summary.preference_decomposition(candidate, control, np.eye(3))
        self.assertAlmostEqual(result["matched_utility"]["relative_pct"], 200.)
        self.assertAlmostEqual(result["front_eum"]["relative_pct"], -70.)
        self.assertEqual(result["front_eum"]["losses_per_mode"], [3])
        hv = summary.experiment.training.compute_hv_nd
        self.assertLess(hv(candidate[0], np.array([-1., -1., -1.])),
                        hv(control[0], np.array([-1., -1., -1.])))

    def test_ties_and_zero_reference_are_not_wins(self):
        zeros = np.zeros((2, 3, 3))
        result = summary.preference_decomposition(zeros, zeros, np.eye(3))
        for metric in result.values():
            self.assertIsNone(metric["relative_pct"])
            self.assertEqual(metric["ties_per_mode"], [3, 3])
            self.assertEqual(metric["wins_per_mode"], [0, 0])

    def test_level_and_progress_are_distinct(self):
        native = dict(operator=summary.experiment.B, half_counts=[750, 750],
            scale_before_block=[1., 1., 1.], half_mean_rewards=[[0., 0., 0.], [1., 1., 1.]])
        pulse = {**native, "half_mean_rewards": [[2., 2., 2.], [2., 2., 2.]]}
        result = summary.block_comparison(native, pulse)
        self.assertEqual(result["mean_reward_delta"], [1.5, 1.5, 1.5])
        self.assertEqual(result["half_mean_reward_delta"], [[2., 2., 2.], [1., 1., 1.]])
        changed = copy.deepcopy(pulse)
        changed["scale_before_block"][0] = 2.
        with self.assertRaises(AssertionError):
            summary.block_comparison(native, changed)


if __name__ == "__main__":
    unittest.main()
