import copy
import inspect
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import torch
from torch import nn

import train_bamor_mujoco as training


METHOD = "bamor_v26_aligned_actor_fixed_archive_mix"


def legacy_actor_reference(trainer, s, w, c):
    contexts = [c]
    for _ in range(trainer.envelope_policy_candidates - 1):
        prefs = trainer.sample_preference_batch(len(w), trainer.preference_relabeling_strategy)
        contexts.append(trainer.replace_context_preferences(c, prefs))
    actions, log_probs, scores = [], [], []
    for context in contexts:
        action, log_prob = trainer.policy.get_action(s, context)
        q = torch.min(trainer.q1(s, action, c), trainer.q2(s, action, c))
        actions.append(action)
        log_probs.append(log_prob)
        scores.append((w * q).sum(-1, keepdim=True))
    scores = torch.stack(scores)
    best = scores.detach().squeeze(-1).argmax(0)
    batch = torch.arange(len(w))
    action = torch.stack(actions)[best, batch]
    log_prob = torch.stack(log_probs)[best, batch]
    scalar = scores[best, batch]
    return action, scalar, (trainer.alpha * log_prob - scalar).mean()


class SeparatePreferencePolicy(nn.Module):
    def __init__(self, candidate):
        super().__init__()
        self.own = nn.Parameter(torch.tensor(0.0))
        self.candidate = nn.Parameter(torch.tensor(float(candidate)))

    def get_action(self, s, c, deterministic=False):
        value = c[:, :1] * self.own + c[:, 1:2] * self.candidate
        if deterministic:
            return value
        return value, value * 0.0


class LinearQ(nn.Module):
    def forward(self, s, action, c):
        return action.expand(-1, 2)


class AlignedActorTests(unittest.TestCase):
    def make_trainer(self, cls, dimension):
        torch.manual_seed(7)
        trainer = cls(3, 2, dimension, context_dim=dimension + 2, hidden=8,
                      buffer_capacity=16, batch_size=4, device="cpu")
        trainer.preference_relabeling = True
        for index in range(8):
            state = np.full(3, index / 10, dtype=np.float32)
            weight = np.full(dimension, 1 / dimension, dtype=np.float32)
            context = np.r_[weight, np.array([0.4, -0.7], dtype=np.float32)]
            trainer.buffer.push(state, np.zeros(2), weight, context,
                                 np.ones(dimension), state + 0.1, context, index % 3 == 0)
        return trainer

    def test_legacy_extraction_preserves_values_gradients_and_rng(self):
        for dimension in [2, 3]:
            selector = training.FixedExtremeHierarchicalOperatorSelector(dimension)
            for arm in selector.operators:
                with self.subTest(dimension=dimension, arm=arm):
                    base = self.make_trainer(training.ConsistentTargetMOSACTrainer, dimension)
                    for key, value in selector.trainer_config(arm).items():
                        setattr(base, key, value)
                    s = torch.arange(12).float().reshape(4, 3) / 10
                    w = torch.full((4, dimension), 1 / dimension)
                    c = torch.cat([w, torch.ones(4, 2)], -1)
                    results = []
                    for function in [legacy_actor_reference,
                                     training.MOSACTrainer.envelope_actor_loss]:
                        trainer = copy.deepcopy(base)
                        torch.manual_seed(17)
                        values = function(trainer, s, w, c)
                        values[-1].backward()
                        gradients = [p.grad.clone() for module in [trainer.policy, trainer.q1, trainer.q2]
                                     for p in module.parameters()]
                        results.append((values, gradients, torch.get_rng_state()))
                    for before, after in zip(results[0][0] + tuple(results[0][1]),
                                             results[1][0] + tuple(results[1][1])):
                        torch.testing.assert_close(before, after, rtol=0, atol=0)
                    self.assertTrue(torch.equal(results[0][2], results[1][2]))

    def test_projection_gate_loss_and_no_candidate_gradient(self):
        for proposal, improves in [(0.8, True), (-0.8, False), (0.0, False)]:
            trainer = self.make_trainer(training.AlignedEnvelopeMOSACTrainer, 2)
            trainer.policy = SeparatePreferencePolicy(proposal)
            trainer.q1 = trainer.q2 = LinearQ()
            trainer.envelope_policy_candidates = 2
            s = torch.zeros(4, 3)
            w = torch.tensor([[1.0, 0.0]]).repeat(4, 1)
            c = torch.cat([w, torch.ones(4, 2)], -1)
            other = w.flip(-1)
            calls = []
            original = trainer.policy.get_action

            def record(state, context, **kwargs):
                calls.append((context.detach().clone(), torch.is_grad_enabled()))
                return original(state, context, **kwargs)

            with patch.object(trainer, "sample_preference_batch", return_value=other), \
                 patch.object(trainer.policy, "get_action", side_effect=record):
                action, q, loss = trainer.envelope_actor_loss(s, w, c)
            torch.testing.assert_close(action, torch.zeros(4, 1))
            torch.testing.assert_close(q, torch.zeros(4, 1))
            self.assertAlmostEqual(float(loss.detach()), 0.1 * proposal ** 2 if improves else 0.0, places=6)
            loss.backward()
            self.assertAlmostEqual(float(trainer.policy.own.grad),
                                   -1.0 - (0.2 * proposal if improves else 0.0), places=6)
            self.assertEqual(float(trainer.policy.candidate.grad), 0.0)
            self.assertEqual([grad for _, grad in calls], [True, False, True])
            for context, grad in calls:
                torch.testing.assert_close(context[:, 2:], c[:, 2:])
                if grad:
                    torch.testing.assert_close(context, c)
            stats = trainer.actor_update_diagnostics()
            self.assertEqual(stats["proposal_win_fraction"], float(improves))

    def test_actual_updates_keep_critic_and_single_candidate_arms_unchanged(self):
        for dimension in [2, 3]:
            selector = training.FixedExtremeHierarchicalOperatorSelector(dimension)
            for arm in selector.operators:
                with self.subTest(dimension=dimension, arm=arm):
                    trainers, targets, rngs = [], [], []
                    config = selector.trainer_config(arm)
                    for cls in [training.ConsistentTargetMOSACTrainer, training.AlignedEnvelopeMOSACTrainer]:
                        trainer = self.make_trainer(cls, dimension)
                        for key, value in config.items():
                            setattr(trainer, key, value)
                        np.random.seed(17)
                        torch.manual_seed(17)
                        q, target = trainer.update()
                        self.assertTrue(np.isfinite(q))
                        self.assertTrue(np.isfinite(target).all())
                        self.assertTrue(all(torch.isfinite(p).all() for p in trainer.policy.parameters()))
                        trainers.append(trainer)
                        targets.append(target)
                        rngs.append(torch.get_rng_state())
                    np.testing.assert_array_equal(*targets)
                    self.assertTrue(torch.equal(*rngs))
                    for name in ["q1", "q2", "q1_target", "q2_target"]:
                        for a, b in zip(getattr(trainers[0], name).parameters(),
                                        getattr(trainers[1], name).parameters()):
                            torch.testing.assert_close(a, b, rtol=0, atol=0)
                    changed = any(not torch.equal(a, b) for a, b in
                                  zip(trainers[0].policy.parameters(), trainers[1].policy.parameters()))
                    multiple = config["envelope_policy_candidates"] > 1
                    self.assertEqual(changed, multiple)
                    self.assertEqual(trainers[1].projection_updates, int(multiple))

    def test_feature_parity(self):
        self.assertIn(METHOD, training.METHODS)
        for name, function in vars(training).items():
            if name.startswith("uses_") and inspect.isfunction(function):
                self.assertEqual(function(METHOD),
                                 function("bamor_v24_target_consistent_fixed_archive_mix"), name)
        self.assertEqual(training.policy_archive_interval_for_method(METHOD), 2500)
        self.assertEqual(training.policy_archive_limit_for_method(METHOD), 20)

    def test_real_training_route_and_recorded_config(self):
        make_env = training.make_piecewise_mo_env
        trainer_cls = training.AlignedEnvelopeMOSACTrainer
        selector_cls = training.FixedExtremeHierarchicalOperatorSelector

        def short_env(*args, **kwargs):
            env = make_env(*args, **kwargs)
            env.env._max_episode_steps = 3
            return env

        def small_trainer(*args, **kwargs):
            return trainer_cls(*args, buffer_capacity=128, **kwargs)

        with tempfile.TemporaryDirectory() as directory, \
             patch.object(training, "make_piecewise_mo_env", side_effect=short_env), \
             patch.object(training, "AlignedEnvelopeMOSACTrainer", side_effect=small_trainer), \
             patch.object(training, "FixedExtremeHierarchicalOperatorSelector",
                          side_effect=lambda dim: selector_cls(dim, block_steps=4)), \
             patch("builtins.print"):
            result = training.train_method("mo-halfcheetah-v5", METHOD, 64, 25, 7, directory,
                hidden=8, batch_size=4, learning_starts=5, eval_freq=1000, log_freq=1000, device="cpu")
        self.assertEqual(result["critic_target_rule"], "scalar_projection_of_soft_vector")
        self.assertEqual(result["bootstrap_terminal_rule"], "terminated_only")
        self.assertEqual(result["evaluation_protocol"], "locked_full_simplex_v1")
        stats = result["actor_update"]
        self.assertEqual(stats["rule"], trainer_cls.actor_update_rule)
        self.assertEqual(stats["projection_coef"], 0.1)
        self.assertGreater(stats["projection_updates"], 0)
        self.assertGreater(stats["proposal_samples"], 0)
        expected = selector_cls(2).state_dict()["operator_trainer_configs"]
        self.assertEqual(result["adaptive_operator_final"]["operator_trainer_configs"], expected)


if __name__ == "__main__":
    torch.set_num_threads(1)
    unittest.main()
