import unittest
from unittest.mock import patch

import numpy as np
import torch

import diagnose_ant_mixed_critic as diagnostic
import train_bamor_mujoco as training


class AntMixedCriticTests(unittest.TestCase):
    def test_candidate_actions_and_paired_replay(self):
        torch.manual_seed(12)
        env = training.make_piecewise_mo_env("mo-ant-v5", seed=1201)
        env._lock_mode = True
        try:
            obs_dim = env.observation_space.shape[0]
            act_dim = env.action_space.shape[0]
            dim = training.context_dim_for_method(
                diagnostic.METHOD, obs_dim, act_dim, 3, 3)
            model = {
                "policy": training.PolicyNet(obs_dim, act_dim, dim, hidden=8),
                "q1": training.SoftQNet(obs_dim, act_dim, dim, 3, hidden=8),
                "q2": training.SoftQNet(obs_dim, act_dim, dim, 3, hidden=8),
                "encoder": training.LatentTransitionEncoder(
                    obs_dim, act_dim, 3, hidden=8),
            }
            for module in model.values():
                module.eval().requires_grad_(False)
            weight = diagnostic.WEIGHTS[1]
            history, obs, pair = diagnostic.prefix(env, model, weight, 778801, 4)
            self.assertIsNotNone(obs)
            labels, actions, scores = diagnostic.candidate_actions(
                model, obs, weight, pair)
            self.assertEqual(labels, ["own", "x_velocity", "y_velocity", "control_cost"])
            np.testing.assert_array_equal(actions[0], diagnostic.action(
                model, obs, diagnostic.context(weight, pair)))
            self.assertEqual(set(scores), {"target", "source"})
            self.assertEqual(scores["target"][0], scores["source"][0])
            first, used = diagnostic.branch(env, model, weight, 778801,
                                            history, actions[0], (1, 4))
            repeat, _ = diagnostic.branch(env, model, weight, 778801,
                                          history, actions[0], (1, 4))
            self.assertEqual(first, repeat)
            self.assertEqual(used, 4 + first["steps"])
        finally:
            env.close()

    def test_terminal_branch_has_absorbing_return(self):
        class TerminalEnv:
            def reset(self, *, seed):
                return np.zeros(2), {}

            def step(self, action):
                return np.zeros(2), np.array([1., 2., 3.]), True, False, {}

        with patch.object(diagnostic, "trackers", return_value=None), \
             patch.object(diagnostic, "observe"):
            result, used = diagnostic.branch(
                TerminalEnv(), {}, np.array([0., 0., 1.]), 1, [],
                np.zeros(1), (1, 4))
        self.assertEqual(result["returns"], {"1": 3., "4": 3.})
        self.assertTrue(result["ended"])
        self.assertEqual(used, 1)

    def test_summary_keeps_q_selected_losses(self):
        case = {"labels": ["own", "health", "speed"],
                "q": {"target": [0., 2., 1.], "source": [0., 1., 3.]},
                "branches": [{"returns": {"4": 10.}},
                             {"returns": {"4": 8.}},
                             {"returns": {"4": 12.}}]}
        effect = diagnostic.effect(case, 4)
        self.assertEqual(effect["target_gain"], -2.)
        self.assertEqual(effect["source_gain"], 2.)
        self.assertEqual(effect["oracle_gain"], 2.)
        summary = diagnostic.summarize_cases([case, {"prefix_ended": True}], 4)
        self.assertEqual(summary["target_gain_negative"], 1)
        self.assertEqual(summary["prefix_ended"], 1)


if __name__ == "__main__":
    unittest.main()
