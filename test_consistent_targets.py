import inspect
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import torch

import train_bamor_mujoco as training


class ConsistentTargetTests(unittest.TestCase):
    def make_trainer(self, cls, dimension):
        return cls(3, 2, dimension, hidden=8, buffer_capacity=16,
                   batch_size=4, device="cpu")

    def test_targets_and_terminal_masks(self):
        for dimension in [2, 3]:
            old = self.make_trainer(training.MOSACTrainer, dimension)
            new = self.make_trainer(training.ConsistentTargetMOSACTrainer, dimension)
            weights = torch.cat([torch.eye(dimension),
                                 torch.full((1, dimension), 1 / dimension)])
            reward = torch.arange(weights.numel()).reshape_as(weights).float()
            next_vector = reward + 2
            bonus = torch.linspace(-0.5, 0.5, len(weights)).reshape(-1, 1)
            next_scalar = (weights * next_vector).sum(-1, keepdim=True) + bonus
            for terminated, truncated in [(False, False), (False, True),
                                          (True, False), (True, True)]:
                self.assertEqual(new.replay_terminal(terminated, truncated), terminated)
                self.assertEqual(old.replay_terminal(terminated, truncated),
                                 terminated or truncated)
                done = torch.full((len(weights), 1), float(terminated))
                scalar, vector = new.critic_targets(reward, weights, done,
                                                    next_scalar, next_vector)
                expected = reward + new.gamma * (1 - done) * (next_vector + bonus)
                torch.testing.assert_close(vector, expected)
                torch.testing.assert_close(scalar, (weights * vector).sum(-1, keepdim=True))
                legacy_scalar, legacy_vector = old.critic_targets(
                    reward, weights, done, next_scalar, next_vector)
                torch.testing.assert_close(legacy_vector,
                                           reward + old.gamma * (1 - done) * next_vector)
                torch.testing.assert_close(legacy_scalar,
                    (weights * reward).sum(-1, keepdim=True) + old.gamma * (1 - done) * next_scalar)
                if not terminated:
                    self.assertFalse(torch.allclose(
                        legacy_scalar, (weights * legacy_vector).sum(-1, keepdim=True)))

    def test_actual_update_uses_consistent_targets_for_every_arm(self):
        torch.manual_seed(7)
        np.random.seed(7)
        for dimension in [2, 3]:
            trainer = self.make_trainer(training.ConsistentTargetMOSACTrainer, dimension)
            trainer.preference_relabeling = True
            selector = training.ExtremeHierarchicalArchiveRiskAdaptiveOperatorSelector(dimension)
            for index in range(8):
                state = np.full(3, index / 10, dtype=np.float32)
                weight = np.full(dimension, 1 / dimension, dtype=np.float32)
                trainer.buffer.push(state, np.zeros(2), weight, weight,
                                    np.ones(dimension), state + 0.1, weight, index % 3 == 0)
            original = trainer.critic_targets
            for arm in selector.operators:
                for key, value in selector.trainer_config(arm).items():
                    setattr(trainer, key, value)
                def check_targets(reward, weights, done, scalar, vector):
                    result = original(reward, weights, done, scalar, vector)
                    torch.testing.assert_close(result[0], (weights * result[1]).sum(-1, keepdim=True))
                    return result
                with patch.object(trainer, "critic_targets", side_effect=check_targets) as target:
                    q, vector = trainer.update()
                target.assert_called_once()
                self.assertTrue(np.isfinite(q), arm)
                self.assertTrue(np.isfinite(vector).all(), arm)

    def test_v24_feature_parity_with_v23(self):
        for method in training.V24_MUJOCO_METHODS:
            self.assertIn(method, training.METHODS)
            for name, function in vars(training).items():
                if name.startswith("uses_") and inspect.isfunction(function):
                    self.assertEqual(function(method),
                                     function("bamor_v23_extreme_hier_archive_mix"), name)
            self.assertEqual(training.policy_archive_interval_for_method(method), 2500)
            self.assertEqual(training.policy_archive_limit_for_method(method), 20)
            self.assertEqual(training.context_dim_for_method(method, 17, 6, 2, 4),
                training.context_dim_for_method("bamor_v23_extreme_hier_archive_mix", 17, 6, 2, 4))

    def test_vector_reselection_matches_lean_counterexample(self):
        trainer = self.make_trainer(training.ConsistentTargetMOSACTrainer, 2)
        weights = torch.tensor([[0.5, 0.5]])
        state = torch.zeros(1, 3)
        selector = training.ProgressExtremeHierarchicalOperatorSelector(2)
        counts = {selector.trainer_config(op)["envelope_target_candidates"]
                  for op in selector.operators}
        for count in sorted({2} | (counts - {1})):
            with self.subTest(candidate_count=count):
                trainer.envelope_target_candidates = count
                original = torch.zeros(count, 2)
                original[:2] = torch.tensor([[1.0, 0.0], [0.0, 0.9]])
                perturbed = original.clone()
                perturbed[1, 1] = 1.1
                selected = []
                for candidates in (original, perturbed):
                    scores = (candidates * weights).sum(-1)
                    self.assertEqual((scores == scores.max()).sum().item(), 1)
                    with patch.object(trainer, "sample_preference_batch",
                                      return_value=torch.tensor([[0.0, 1.0]])), \
                         patch.object(trainer.policy, "get_action",
                                      return_value=(torch.zeros(1, 2), torch.zeros(1, 1))), \
                         patch.object(trainer.q1_target, "forward",
                                      side_effect=[row[None] for row in candidates]), \
                         patch.object(trainer.q2_target, "forward",
                                      side_effect=[row[None] for row in candidates]):
                        selected.append(trainer.envelope_next_values(state, weights, weights))
                self.assertAlmostEqual((original - perturbed).abs().max().item(), 0.2)
                torch.testing.assert_close(selected[0][1], original[:1])
                torch.testing.assert_close(selected[1][1], perturbed[1:2])
                self.assertAlmostEqual(abs((selected[0][0] - selected[1][0]).item()), 0.05)
                self.assertAlmostEqual(abs((selected[0][1] - selected[1][1])[0, 0].item()), 1.0)

    def test_short_real_training_routes_v24_variants(self):
        make_env = training.make_piecewise_mo_env
        trainers = []
        trainer_cls = training.ConsistentTargetMOSACTrainer

        def short_env(*args, **kwargs):
            env = make_env(*args, **kwargs)
            env.env._max_episode_steps = 3
            return env

        def small_trainer(*args, **kwargs):
            trainer = trainer_cls(*args, buffer_capacity=128, **kwargs)
            trainers.append(trainer)
            return trainer

        with tempfile.TemporaryDirectory() as directory:
            for method in sorted(training.V24_MUJOCO_METHODS):
                with patch.object(training, "make_piecewise_mo_env", side_effect=short_env), \
                     patch.object(training, "ConsistentTargetMOSACTrainer", side_effect=small_trainer), \
                     patch("builtins.print"):
                    result = training.train_method(
                        "mo-halfcheetah-v5", method, 64, 25, 7, directory,
                        hidden=8, batch_size=4, learning_starts=5,
                        eval_freq=1000, log_freq=1000, device="cpu")
                self.assertEqual(result["bootstrap_terminal_rule"], "terminated_only")
                self.assertEqual(result["critic_target_rule"], "scalar_projection_of_soft_vector")
                self.assertEqual(result["evaluation_protocol"], "locked_full_simplex_v1")
                selector_state = result["adaptive_operator_final"]
                if method in {"bamor_v24_ownpref_actor_fixed_archive_mix",
                              "bamor_v24_ownpref_both_fixed_archive_mix"}:
                    self.assertEqual(trainers[-1].envelope_policy_candidates, 1)
                    self.assertEqual(selector_state["actor_update_rule"],
                                     "own_preference_with_existing_bc")
                if method in {"bamor_v24_ownpref_critic_fixed_archive_mix",
                              "bamor_v24_ownpref_both_fixed_archive_mix"}:
                    self.assertEqual(trainers[-1].envelope_target_candidates, 1)
                    self.assertEqual(selector_state["critic_candidate_rule"],
                                     "own_preference_only")
                self.assertEqual([s["step"] for s in result["policy_archive_snapshots"]], [64])
                self.assertTrue(result["episode_rewards"])
                self.assertTrue(all(e["length"] == 3 for e in result["episode_rewards"]))
                buffer = trainers[-1].buffer
                self.assertEqual(float(buffer.dones[:buffer.size].sum()), 0)


if __name__ == "__main__":
    torch.set_num_threads(1)
    unittest.main()
