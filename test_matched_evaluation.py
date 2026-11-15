import inspect
import unittest

import numpy as np

import train_bamor_mujoco as training


ADAPTIVE = "bamor_v23_extreme_hier_archive_mix"
FIXED = "bamor_v23_fixed_selector_archive_mix"


class MatchedEvaluationTests(unittest.TestCase):
    def test_feature_parity(self):
        for name, function in vars(training).items():
            if name.startswith("uses_") and inspect.isfunction(function):
                self.assertEqual(function(ADAPTIVE), function(FIXED), name)
        self.assertEqual(training.context_dim_for_method(ADAPTIVE, 17, 6, 2, 4),
                         training.context_dim_for_method(FIXED, 17, 6, 2, 4))

    def test_fixed_selector_ignores_scores_but_keeps_configs(self):
        a = training.ExtremeHierarchicalArchiveRiskAdaptiveOperatorSelector(3)
        b = training.FixedExtremeHierarchicalOperatorSelector(3)
        for arm in a.operators:
            self.assertEqual(a.trainer_config(arm), b.trainer_config(arm))
        b.counts = dict.fromkeys(b.operators, 2)
        b.scores = {arm: float(i) for i, arm in enumerate(b.operators)}
        np.random.seed(19)
        first = [b.select() for _ in range(50)]
        b.scores = {arm: -100 * value for arm, value in b.scores.items()}
        np.random.seed(19)
        self.assertEqual(first, [b.select() for _ in range(50)])
        self.assertEqual(a.min_family_fraction, b.min_family_fraction)

    def test_grids_cover_vertices(self):
        for dimension in [2, 3]:
            weights = training.full_simplex_grid(dimension)
            self.assertEqual(weights.shape, (21, dimension))
            np.testing.assert_allclose(weights.sum(axis=1), 1)
            for vertex in np.eye(dimension):
                self.assertTrue(np.any(np.all(weights == vertex, axis=1)))

    def test_no_cross_mode_union_and_order_independence(self):
        class Environment:
            current_mode_idx = 1
            num_modes = 2
            reward_dim = 2
            _lock_mode = False

            def _apply_mode(self, mode):
                self.current_mode_idx = mode

        env = Environment()

        def front(weights, seeds):
            self.assertTrue(env._lock_mode)
            self.assertEqual(seeds, [102000 + env.current_mode_idx * 10 + i
                                     for i in range(3)])
            returns = [[10, 1], [1, 10]][env.current_mode_idx]
            return [{"weight": w.tolist(), "return": returns,
                     "archive_step": step} for step in [5000, 7500] for w in weights]

        result = training.evaluate_locked_modes(env, front, 2, np.zeros(2))
        reversed_result = training.evaluate_locked_modes(
            env, lambda w, seeds: list(reversed(front(w, seeds))), 2, np.zeros(2))
        self.assertEqual(result["hv_per_mode"], [10, 10])
        self.assertEqual(result["hv_mean_mode"], 10)
        self.assertNotIn("hv_overall", result)
        self.assertEqual(result["hv_per_mode"], reversed_result["hv_per_mode"])
        self.assertFalse(env._lock_mode)
        self.assertEqual(env.current_mode_idx, 1)

    def test_evaluation_reuses_reset_seeds(self):
        class Environment:
            resets = []

            def reset(self, seed=None):
                self.resets.append(seed)
                return np.array([seed]), {}

            def step(self, action):
                return np.zeros(1), np.ones(2), True, False, {}

        class Agent:
            def eval(self, obs, weight):
                return np.zeros(1)

        env = Environment()
        training.evaluate_morl_agent_with_weights(
            env, Agent(), 2, [[0, 1], [1, 0]], num_episodes=3, episode_seeds=[7, 8, 9])
        self.assertEqual(env.resets, [7, 8, 9, 7, 8, 9])


if __name__ == "__main__":
    unittest.main()
