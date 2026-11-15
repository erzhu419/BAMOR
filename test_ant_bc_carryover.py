import unittest

import numpy as np

import diagnose_ant_bc_carryover as diagnosis


class CarryoverTests(unittest.TestCase):
    def test_reward_switch_applies_after_logged_step(self):
        history = [[0, 0], [68568, 1]]
        self.assertEqual(diagnosis.reward_mode_counts(history, 68001, 68750), {"0": 568, "1": 182})
        self.assertEqual(diagnosis.reward_mode_counts(history, 68568, 68569), {"0": 1, "1": 1})
        self.assertEqual(diagnosis.reward_mode_counts(history, 68751, 69500), {"1": 750})

    def test_partial_tail_does_not_invent_feedback(self):
        self.assertIsNone(diagnosis.feedback(dict(complete=False)))
        with self.assertRaises(AssertionError):
            diagnosis.feedback(dict(complete=False, progress_score=0.))
        rewards = [[1., 2., -3.], [2., 1., -2.]]
        gain = np.array([.5, -.5, .5])
        result = diagnosis.feedback(dict(complete=True, half_mean_rewards=rewards,
            scale_before_block=[2., 2., 2.], progress_score=float(np.tanh(gain).mean())))
        np.testing.assert_allclose(result["objective_score_terms"], np.tanh(gain) / 3)

    def test_weighted_objective_decomposition_keeps_pure_endpoint_separate(self):
        weights = diagnosis.experiment.base.training.full_simplex_grid(3).astype(float)
        mixed = (weights[:, 2] > 0) & (weights[:, 2] < 1)
        pure = weights[:, 2] == 1
        base = dict(step=75000, returns=np.zeros((3, 21, 3)).tolist(), mode0_mixed_energy_utility=0.)
        returns = np.zeros((3, 21, 3))
        returns[0, mixed, 2] = -10.
        returns[0, pure, 2] = 50.
        primary = float((-10 * weights[mixed, 2]).mean())
        changed = dict(step=75000, returns=returns.tolist(), mode0_mixed_energy_utility=primary)
        result = diagnosis.endpoint(base, changed, weights)
        self.assertAlmostEqual(result["primary_delta"], primary)
        self.assertEqual(result["mixed_mode0_negative_weights"], 14)
        self.assertEqual(result["group_utility_delta_per_mode"]["pure_energy"][0], 50.)
        np.testing.assert_allclose(result["mixed_weighted_objective_contributions_per_mode"][0], [0., 0., primary])
        changed["mode0_mixed_energy_utility"] += 1.
        with self.assertRaises(AssertionError):
            diagnosis.endpoint(base, changed, weights)


if __name__ == "__main__":
    unittest.main()
