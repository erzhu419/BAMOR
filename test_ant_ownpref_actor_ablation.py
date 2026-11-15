import unittest

import numpy as np

import run_ant_ownpref_actor_ablation as ablation
import train_bamor_mujoco as training


class AntOwnPreferenceAblationTests(unittest.TestCase):
    def test_primary_metric_excludes_vertices_and_no_energy(self):
        weights = training.full_simplex_grid(3)
        front = []
        selected = []
        for index, weight in enumerate(weights):
            if 0 < weight[2] < 1:
                reward = [float(index), 0., 0.]
                selected.append(float(weight @ reward))
            else:
                reward = [1e6, 1e6, 1e6]
            front.append({"return": reward})
        self.assertEqual(len(selected), 14)
        data = {"evaluation_weights": weights.tolist(),
                "pareto_per_mode": [front, front, front]}
        self.assertAlmostEqual(ablation.mixed_energy_utility(data),
                               float(np.mean(selected)))


if __name__ == "__main__":
    unittest.main()
