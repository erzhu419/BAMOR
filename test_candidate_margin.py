import unittest
from unittest.mock import patch

import torch

import train_bamor_mujoco as training


class CandidateMarginTests(unittest.TestCase):
    def make_trainer(self, dimension, count):
        trainer = training.ConsistentTargetMOSACTrainer(
            3, 2, dimension, hidden=8, buffer_capacity=16,
            batch_size=4, alpha=0.25, device="cpu")
        trainer.envelope_target_candidates = count
        return trainer

    def candidate_counts(self, dimension):
        selector = training.ProgressExtremeHierarchicalOperatorSelector(dimension)
        return sorted({selector.trainer_config(op)["envelope_target_candidates"]
                       for op in selector.operators})

    def evaluate(self, trainer, weights, candidates, bonuses, done, swap_twins=False):
        count, batch, dimension = candidates.shape
        mask = torch.arange(candidates.numel()).reshape_as(candidates) % 2 == 0
        if swap_twins:
            mask = ~mask
        q1 = candidates + mask * 0.5
        q2 = candidates + (~mask) * 0.25
        actions = [(torch.full((batch, 2), j / count, dtype=weights.dtype),
                    -bonuses[j] / trainer.alpha) for j in range(count)]
        with patch.object(trainer, "sample_preference_batch", return_value=weights), \
             patch.object(trainer.policy, "get_action", side_effect=actions), \
             patch.object(trainer.q1_target, "forward", side_effect=list(q1)), \
             patch.object(trainer.q2_target, "forward", side_effect=list(q2)):
            next_scalar, next_vector = trainer.envelope_next_values(
                torch.zeros(batch, 3, dtype=weights.dtype), weights, weights)
        reward = torch.arange(batch * dimension, dtype=weights.dtype).reshape_as(weights) / 8
        targets = trainer.critic_targets(reward, weights, done, next_scalar, next_vector)
        return (next_scalar, next_vector), targets

    def test_margin_certifies_soft_targets_for_v25_candidate_counts(self):
        epsilon, entropy_error = 1 / 16, 1 / 32
        delta = epsilon + entropy_error
        for dimension in (2, 3):
            weights = torch.cat((torch.eye(dimension, dtype=torch.float64),
                                 torch.full((1, dimension), 1 / dimension,
                                            dtype=torch.float64)))
            batch = len(weights)
            done = (torch.arange(batch) % 2).to(weights.dtype).reshape(-1, 1)
            for count in self.candidate_counts(dimension):
                with self.subTest(dimension=dimension, candidate_count=count):
                    trainer = self.make_trainer(dimension, count)
                    candidates = (2 * torch.arange(count, dtype=weights.dtype)[:, None, None]
                                  + torch.arange(dimension, dtype=weights.dtype)[None, None] / 8)
                    candidates = candidates.expand(count, batch, dimension).clone()
                    bonuses = (-torch.arange(count, dtype=weights.dtype)[:, None, None] / 4
                               ).expand(count, batch, 1).clone()
                    signs = 2 * (torch.arange(candidates.numel()).reshape_as(candidates) % 2) - 1
                    changed = candidates + epsilon * signs
                    changed_bonus = bonuses + entropy_error * signs[:, :, :1]
                    scores = (weights * candidates).sum(-1, keepdim=True) + bonuses
                    changed_scores = (weights * changed).sum(-1, keepdim=True) + changed_bonus
                    self.assertLessEqual((scores - changed_scores).abs().max().item(), delta + 1e-12)
                    if count > 1:
                        margin = scores[-1] - scores[:-1].amax(dim=0)
                        self.assertTrue((margin > 2 * delta).all().item())
                    old_next, old_targets = self.evaluate(
                        trainer, weights, candidates, bonuses, done)
                    new_next, new_targets = self.evaluate(
                        trainer, weights, changed, changed_bonus, done, swap_twins=True)
                    torch.testing.assert_close(old_next[1], candidates[-1])
                    torch.testing.assert_close(new_next[1], changed[-1])
                    torch.testing.assert_close(old_next[0], scores[-1])
                    torch.testing.assert_close(new_next[0], changed_scores[-1])
                    bound = trainer.gamma * (1 - done) * delta
                    self.assertTrue(((old_targets[1] - new_targets[1]).abs()
                                     <= bound + 1e-12).all().item())
                    self.assertTrue(((old_targets[0] - new_targets[0]).abs()
                                     <= bound + 1e-12).all().item())
                    for scalar, vector in (old_targets, new_targets):
                        torch.testing.assert_close(scalar, (weights * vector).sum(-1, keepdim=True))

    def test_twice_error_threshold_requires_strict_inequality(self):
        epsilon = 1 / 8
        weights = torch.tensor([[0.5, 0.5]], dtype=torch.float64)
        for count in self.candidate_counts(2):
            if count == 1:
                continue
            for gap in (3 / 16, 1 / 4, 5 / 16):
                with self.subTest(candidate_count=count, gap=gap):
                    trainer = self.make_trainer(2, count)
                    candidates = torch.full((count, 1, 2), -2.0, dtype=weights.dtype)
                    candidates[0] = torch.tensor([1.0, -1.0], dtype=weights.dtype)
                    candidates[1] = torch.tensor([gap - 1, gap + 1], dtype=weights.dtype)
                    changed = candidates.clone()
                    changed[0] += epsilon
                    changed[1] -= epsilon
                    bonuses = torch.zeros(count, 1, 1, dtype=weights.dtype)
                    done = torch.zeros(1, 1, dtype=weights.dtype)
                    old_next, _ = self.evaluate(trainer, weights, candidates, bonuses, done)
                    new_next, _ = self.evaluate(trainer, weights, changed, bonuses, done)
                    torch.testing.assert_close(old_next[1], candidates[1])
                    winner = 1 if gap > 2 * epsilon else 0
                    torch.testing.assert_close(new_next[1], changed[winner])
                    if gap == 2 * epsilon:
                        changed_scores = (weights * changed).sum(-1)
                        self.assertEqual((changed_scores == changed_scores.max()).sum().item(), 2)

    def test_entropy_error_cannot_be_omitted_from_margin(self):
        weights = torch.tensor([[0.75, 0.25]], dtype=torch.float64)
        for count in self.candidate_counts(2):
            if count == 1:
                continue
            with self.subTest(candidate_count=count):
                trainer = self.make_trainer(2, count)
                candidates = torch.full((count, 1, 2), -2.0, dtype=weights.dtype)
                candidates[:2, 0] = torch.eye(2, dtype=weights.dtype)
                bonuses = torch.zeros(count, 1, 1, dtype=weights.dtype)
                changed_bonus = bonuses.clone()
                changed_bonus[0] = -3 / 8
                changed_bonus[1] = 3 / 8
                done = torch.zeros(1, 1, dtype=weights.dtype)
                old_next, _ = self.evaluate(trainer, weights, candidates, bonuses, done)
                new_next, _ = self.evaluate(trainer, weights, candidates, changed_bonus, done)
                torch.testing.assert_close(old_next[1], candidates[0])
                torch.testing.assert_close(new_next[1], candidates[1])
                self.assertAlmostEqual(old_next[0].item(), 0.75)
                self.assertAlmostEqual(new_next[0].item(), 0.625)


if __name__ == "__main__":
    torch.set_num_threads(1)
    unittest.main()
