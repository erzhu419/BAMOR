import unittest

import numpy as np
import torch

import train_bamor_mujoco as training


class PairwiseResidualTests(unittest.TestCase):
    def test_initial_policy_and_rng_match_base(self):
        torch.manual_seed(37)
        base = training.PolicyNet(4, 2, 5, hidden=8)
        rng = torch.get_rng_state()
        torch.manual_seed(37)
        pairwise = training.PairwiseResidualPolicyNet(4, 2, 5, 3, hidden=8)
        self.assertTrue(torch.equal(rng, torch.get_rng_state()))
        for name, value in base.state_dict().items():
            self.assertTrue(torch.equal(value, pairwise.state_dict()[name]), name)
        state = torch.randn(3, 4)
        context = torch.tensor([[1., 0., 0., .2, -.1],
                                [.2, .3, .5, .2, -.1],
                                [0., 0., 1., .2, -.1]])
        for left, right in zip(base(state, context), pairwise(state, context)):
            self.assertTrue(torch.equal(left, right))

    def test_residual_is_zero_at_vertices_and_learns_at_mixtures(self):
        torch.manual_seed(41)
        policy = training.PairwiseResidualPolicyNet(4, 2, 5, 3, hidden=8)
        state = torch.randn(4, 4)
        context = torch.tensor([[1., 0., 0., 0., 0.],
                                [0., 1., 0., 0., 0.],
                                [0., 0., 1., 0., 0.],
                                [.2, .3, .5, 0., 0.]])
        with torch.no_grad():
            policy.pair_mean_bias.fill_(1.0)
        mean, _ = policy(state, context)
        base_mean = policy.mean_head(policy.trunk(torch.cat([state, context], dim=-1)))
        self.assertTrue(torch.equal(mean[:3], base_mean[:3]))
        self.assertGreater((mean[3] - base_mean[3]).abs().sum().item(), 0)
        mean[3].sum().backward()
        self.assertGreater(policy.pair_mean_weight.grad.abs().sum().item(), 0)

    def test_adaptive_and_blind_both_use_own_preference_targets(self):
        adaptive = training.OwnPreferenceBothProgressExtremeOperatorSelector(3)
        blind = training.OwnPreferenceBothFixedProgressExtremeOperatorSelector(3)
        self.assertEqual(adaptive.operators, blind.operators)
        self.assertEqual(adaptive.warmup_sequence, blind.warmup_sequence)
        for arm in adaptive.operators:
            self.assertEqual(adaptive.trainer_config(arm), blind.trainer_config(arm))
            self.assertEqual(adaptive.trainer_config(arm)["envelope_policy_candidates"], 1)
            self.assertEqual(adaptive.trainer_config(arm)["envelope_target_candidates"], 1)
        state = adaptive.state_dict()
        self.assertEqual(state["actor_update_rule"], "own_preference_with_existing_bc")
        self.assertEqual(state["critic_candidate_rule"], "own_preference_only")
        self.assertEqual(state["score_signal"], "within_block_normalized_vector_progress")

    def test_trainer_updates_pairwise_parameters(self):
        torch.manual_seed(43)
        trainer = training.ConsistentTargetMOSACTrainer(
            4, 2, 3, context_dim=5, hidden=8, batch_size=4,
            buffer_capacity=16, device="cpu", pairwise_residual_policy=True)
        rng = np.random.default_rng(43)
        for _ in range(8):
            state = rng.normal(size=4).astype(np.float32)
            action = rng.uniform(-1, 1, size=2).astype(np.float32)
            weight = np.array([.2, .3, .5], dtype=np.float32)
            context = np.r_[weight, [0., 0.]].astype(np.float32)
            reward = rng.normal(size=3).astype(np.float32)
            trainer.buffer.push(state, action, weight, context, reward,
                                state, context, False)
        before = trainer.policy.pair_mean_weight.detach().clone()
        trainer.update()
        self.assertFalse(torch.equal(before, trainer.policy.pair_mean_weight))


if __name__ == "__main__":
    unittest.main()
