import unittest
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import torch

import diagnose_candidate_calibration as diagnostic
import train_bamor_mujoco as training


def make_model():
    torch.manual_seed(9)
    trainer = training.ConsistentTargetMOSACTrainer(
        17, 6, 2, context_dim=13, hidden=8, buffer_capacity=1, device="cpu")
    encoder = training.LatentTransitionEncoder(17, 6, 2, hidden=8)
    return diagnostic.FrozenModel({"policy": trainer.policy.state_dict(),
                                   "q1": trainer.q1.state_dict(),
                                   "q2": trainer.q2.state_dict(),
                                   "encoder": encoder.state_dict()})


class CandidateCalibrationTests(unittest.TestCase):
    def test_real_prefix_and_paired_branch_replay(self):
        model = make_model()
        env = training.make_piecewise_mo_env("mo-halfcheetah-v5", seed=26)
        try:
            for mode in range(4):
                weight = np.array([.3, .7], dtype=np.float32)
                history = diagnostic.collect_prefix(env, model, mode, 802600, weight, 8)
                trackers = model.trackers(history)
                np.testing.assert_array_equal(diagnostic.replay_prefix(
                    env, mode, 802600, history), history[-1][3])
                rebuilt = model.trackers()
                obs = diagnostic.reset_locked(env, mode, 802600)
                for _, action, expected_reward, expected_obs in history:
                    next_obs, reward, _, _, _ = env.step(action)
                    np.testing.assert_array_equal(reward, expected_reward)
                    np.testing.assert_array_equal(next_obs, expected_obs)
                    model.observe(rebuilt, obs, action, reward, next_obs)
                    obs = next_obs
                torch.testing.assert_close(model.context(weight, trackers),
                                           model.context(weight, rebuilt), rtol=0, atol=0)
                actions, _, _ = diagnostic.probe_actions(model, obs, weight, trackers, 27)
                for rule in diagnostic.CONTINUATIONS:
                    args = (env, model, mode, 802600, history, weight, actions[0], rule)
                    first = diagnostic.rollout_branch(*args, 45, horizons=(1, 4))
                    diagnostic.rollout_branch(*args[:-2], actions[1], rule, 46, horizons=(1, 4))
                    second = diagnostic.rollout_branch(*args, 45, horizons=(1, 4))
                    self.assertEqual(first, second)
                    self.assertEqual(first["1"]["raw"], first["1"]["bonus_adjusted"])
        finally:
            env.close()

    def test_candidates_share_noise_and_keep_evaluation_context(self):
        model = make_model()
        trackers = model.trackers()
        obs, weight = np.zeros(17), np.array([.25, .75], dtype=np.float32)
        rng = torch.get_rng_state()
        actions, q, meta = diagnostic.probe_actions(model, obs, weight, trackers, 10)
        self.assertTrue(torch.equal(rng, torch.get_rng_state()))
        self.assertEqual(meta["candidate_q"][0], meta["own_resample_q"][0])
        self.assertGreaterEqual(q[1], q[0])
        self.assertGreaterEqual(q[2], q[0])
        c = model.context(weight, trackers)
        with torch.no_grad():
            actual = diagnostic.scores(model.trainer, torch.zeros(1, 17),
                                       torch.as_tensor(actions), c.expand(3, -1),
                                       torch.as_tensor(weight))
        np.testing.assert_allclose(actual.numpy(), q, rtol=1e-6)
        self.assertFalse(any(p.requires_grad for p in model.trainer.policy.parameters()))
        # With identical generating preferences, the same draws must give the same winner.
        with patch.object(model.trainer, "sample_preference_batch",
                          return_value=torch.as_tensor(weight).unsqueeze(0)):
            actions, q, _ = diagnostic.probe_actions(model, obs, weight, trackers, 10)
            np.testing.assert_array_equal(actions[1], actions[2])
            self.assertEqual(q[1], q[2])

    def test_discount_and_initial_entropy_exclusion(self):
        class Environment:
            def step(self, action):
                return np.zeros(17), np.array([2., 0.]), False, False, {}

        model = SimpleNamespace(trainer=SimpleNamespace(gamma=.5),
                                trackers=lambda history: None, observe=lambda *args: None)
        with patch.object(diagnostic, "replay_prefix", return_value=np.zeros(17)), \
             patch.object(diagnostic, "continuation_action", return_value=(np.zeros(6), 4.)):
            result = diagnostic.rollout_branch(Environment(), model, 0, 1, [],
                                                np.array([1., 0.]), np.zeros(6),
                                                "own", 5, horizons=(1, 3))
        self.assertEqual(result["1"]["raw"], 2.)
        self.assertEqual(result["1"]["bonus_adjusted"], 2.)
        self.assertEqual(result["3"]["raw"], 3.5)
        self.assertEqual(result["3"]["bonus_adjusted"], 6.5)

    def test_summary_conditions_on_strict_q_gain_and_pairs_repeats(self):
        returns = np.array([[[1., 1.], [3., 3.]],
                            [[0., 2.], [2., 4.]],
                            [[1., 1.], [3., 3.]]])
        entry = diagnostic.paired_statistics(np.array([0., 2., 0.]), returns)
        self.assertEqual(entry["mean_gain"], [[-1., 1.], [0., 0.]])
        case = {"effects": {rule: {"1": entry} for rule in diagnostic.CONTINUATIONS}}
        summary = diagnostic.summarize([case], (1,))["own"]["1"]
        self.assertEqual(summary["cross_preference_best"]["raw_positive_fraction_given_q_positive"], 0.)
        self.assertEqual(summary["cross_preference_best"]["bonus_positive_fraction_given_q_positive"], 1.)
        self.assertIsNone(summary["own_resample_best"]["raw_positive_fraction_given_q_positive"])


if __name__ == "__main__":
    unittest.main()
