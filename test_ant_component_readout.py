import unittest

import numpy as np

import summarize_ant_component_forks as summary


class ComponentReadoutTests(unittest.TestCase):
    def row(self, energy_change):
        weights = summary.experiment.training.full_simplex_grid(3)
        returns = np.zeros((3, 21, 3))
        returns[:, :, 2] = energy_change
        utility = (returns * weights).sum(-1)
        row = dict(step=66500, returns=returns.tolist(),
                   mode0_mixed_energy_utility=float(utility[0, (weights[:, 2] > 0) & (weights[:, 2] < 1)].mean()),
                   direct_utility_per_mode=utility.mean(1).tolist(),
                   objective_means_per_mode=returns.mean(1).tolist(),
                   hv_per_mode=[0., 0., 0.], hv_mean_mode=0., evaluation_interactions=189000)
        return row, weights

    def test_groups_reconstruct_all_weight_mean(self):
        row, weights = self.row(-100)
        result = summary.measurement(row, weights)
        groups = result["preference_group_utility_per_mode"]
        self.assertEqual(groups["zero_energy"], [0., 0., 0.])
        self.assertEqual(groups["pure_energy"], [-100., -100., -100.])
        np.testing.assert_allclose((14 * np.array(groups["mixed_energy"]) +
            6 * np.array(groups["zero_energy"]) + groups["pure_energy"]) / 21,
            result["direct_utility_per_mode"])
        row["mode0_mixed_energy_utility"] += 1
        with self.assertRaises(AssertionError):
            summary.measurement(row, weights)

    def test_relative_deficit_is_not_absolute_forgetting(self):
        a, w = self.row(-70)
        b, _ = self.row(-20)
        initial, _ = self.row(-100)
        a, b, initial = [summary.measurement(row, w) for row in (a, b, initial)]
        self.assertLess(summary.difference(a, b)["mode0_mixed_energy_utility"], 0)
        self.assertGreater(summary.difference(a, initial)["mode0_mixed_energy_utility"], 0)


if __name__ == "__main__":
    unittest.main()
