import unittest
from unittest.mock import patch

import numpy as np

import run_archive_actor_probe as probe


class ArchiveActorProbeTest(unittest.TestCase):
    def test_frozen_pilot_uses_five_seeds_per_environment(self):
        self.assertEqual(probe.PILOT_SEEDS, (1011, 1012, 1013, 1014, 1015))

    def test_weights_and_cached_returns(self):
        for dim in (2, 3):
            weights = probe.weights_for_dim(dim)
            grid = probe.training.full_simplex_grid(dim)
            self.assertTrue(all(any(np.allclose(weight, point) for point in grid)
                                for weight in weights))
        front = [
            {"archive_step": 10000, "weight": [1., 0.], "return": [3., 4.]},
            {"archive_step": 50000, "weight": [1., 0.], "return": [5., 6.]},
        ]
        np.testing.assert_array_equal(probe.cached_return(front, 10000, [1., 0.]), [3., 4.])

    def test_probe_uses_actual_steps_and_does_not_read_mode_before_choice(self):
        class FakeEnv:
            _total_steps = 0
            reward_dim = 2

            def get_ground_truth(self):
                return {"mode_history": [[0, 0]]}

            def close(self):
                pass

        env = FakeEnv()
        reset_seeds = []

        def one_episode(_env, _actor, _method, _encoder, _weights,
                        num_episodes, device, episode_seeds):
            self.assertEqual(num_episodes, 1)
            self.assertEqual(device, "cpu")
            reset_seeds.append(episode_seeds[0])
            _env._total_steps += 1000
            return [{"return": [2., 4.]}]

        with patch.object(probe.training, "make_piecewise_mo_env", return_value=env), \
             patch.object(probe.training, "evaluate_pareto_v2", side_effect=one_episode):
            cell = probe.probe_cell("mo-halfcheetah-v5", 1011, 0,
                                    np.asarray([.5, .5]), object(), object())
        self.assertEqual(cell["interactions"], 10000)
        self.assertEqual(cell["episodes"], 10)
        self.assertEqual(cell["score_per_step"], 0.003)
        self.assertEqual(reset_seeds, [15110000 + i for i in range(10)])


if __name__ == "__main__":
    unittest.main()
