import inspect
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import torch
from torch import nn

import train_bamor_mujoco as training


METHOD = "bamor_v27_aligned_target_fixed_archive_mix"
OWN_ACTOR_METHOD = "bamor_v27_aligned_target_ownpref_actor_fixed_archive_mix"
BASE = "bamor_v24_target_consistent_fixed_archive_mix"


@torch.no_grad()
def legacy_next_values(trainer, state, weights, context):
    contexts = [context]
    for _ in range(max(1, trainer.envelope_target_candidates) - 1):
        prefs = trainer.sample_preference_batch(len(weights), trainer.preference_relabeling_strategy)
        contexts.append(trainer.replace_context_preferences(context, prefs))
    vectors, scalars = [], []
    for candidate in contexts:
        action, log_prob = trainer.policy.get_action(state, candidate)
        vector = torch.min(trainer.q1_target(state, action, candidate),
                           trainer.q2_target(state, action, candidate))
        vectors.append(vector)
        scalars.append((weights * vector).sum(-1, keepdim=True) - trainer.alpha * log_prob)
    vectors, scalars = torch.stack(vectors), torch.stack(scalars)
    best = scalars.squeeze(-1).argmax(0)
    batch = torch.arange(len(weights))
    return scalars[best, batch], vectors[best, batch]


class PreferencePolicy(nn.Module):
    def __init__(self, dimension):
        super().__init__()
        self.dimension = dimension

    def get_action(self, state, context):
        edge = context[:, self.dimension - 1:self.dimension]
        return torch.cat([context[:, :1], edge], -1), -.5 - edge


class ContextQ(nn.Module):
    def __init__(self, dimension):
        super().__init__()
        self.dimension = dimension

    def forward(self, state, action, context):
        value = 2 * action[:, :1] + 4 * context[:, self.dimension - 1:self.dimension]
        return value.expand(-1, self.dimension)


class AlignedTargetTests(unittest.TestCase):
    def make_trainer(self, cls, dimension):
        torch.manual_seed(7)
        trainer = cls(3, 2, dimension, context_dim=dimension + 2, hidden=8,
                      buffer_capacity=16, batch_size=4, device="cpu")
        trainer.preference_relabeling = True
        for index in range(8):
            state = np.full(3, index / 10, dtype=np.float32)
            weight = np.full(dimension, 1 / dimension, dtype=np.float32)
            context = np.r_[weight, np.array([.3, -.7], dtype=np.float32)]
            trainer.buffer.push(state, np.zeros(2), weight, context, np.ones(dimension),
                                state + .1, context, index % 3 == 0)
        return trainer

    def test_legacy_extraction_preserves_values_and_rng_for_all_arms(self):
        for dim in (2, 3):
            trainer = self.make_trainer(training.ConsistentTargetMOSACTrainer, dim)
            selector = training.FixedExtremeHierarchicalOperatorSelector(dim)
            state = torch.arange(12).float().reshape(4, 3) / 10
            w = torch.full((4, dim), 1 / dim)
            c = torch.cat([w, torch.ones(4, 2)], -1)
            for arm in selector.operators:
                for k, v in selector.trainer_config(arm).items():
                    setattr(trainer, k, v)
                torch.manual_seed(17)
                before = legacy_next_values(trainer, state, w, c)
                rng = torch.get_rng_state()
                torch.manual_seed(17)
                after = trainer.envelope_next_values(state, w, c)
                self.assertTrue(torch.equal(rng, torch.get_rng_state()))
                for a, b in zip(before, after):
                    torch.testing.assert_close(a, b, rtol=0, atol=0)
                    self.assertFalse(b.requires_grad)

    def test_only_target_context_changes_winner_and_keeps_proposal_entropy(self):
        for dim in (2, 3):
            w = torch.zeros(4, dim)
            w[:, 0] = 1
            other = w.flip(-1)
            context = torch.cat([w, torch.tensor([[.3, -.7]]).repeat(4, 1)], -1)
            for cls, expected_value, bonus in (
                (training.ConsistentTargetMOSACTrainer, 4., .3),
                (training.PreferenceAlignedTargetMOSACTrainer, 2., .1),
            ):
                trainer = self.make_trainer(cls, dim)
                trainer.policy = PreferencePolicy(dim)
                trainer.q1_target, trainer.q2_target = ContextQ(dim), ContextQ(dim)
                trainer.envelope_target_candidates = 2
                with patch.object(trainer, "sample_preference_batch", return_value=other), \
                     patch.object(trainer.policy, "get_action", wraps=trainer.policy.get_action) as policy, \
                     patch.object(trainer.q1_target, "forward", wraps=trainer.q1_target.forward) as q1, \
                     patch.object(trainer.q2_target, "forward", wraps=trainer.q2_target.forward) as q2:
                    scalar, vector = trainer.envelope_next_values(torch.zeros(4, 3), w, context)
                torch.testing.assert_close(vector, torch.full((4, dim), expected_value))
                torch.testing.assert_close(scalar, torch.full((4, 1), expected_value + bonus))
                for i, call in enumerate(policy.call_args_list):
                    expected = context if i == 0 else trainer.replace_context_preferences(context, other)
                    torch.testing.assert_close(call.args[1], expected, rtol=0, atol=0)
                    for critic in (q1, q2):
                        target_context = (context if cls is training.PreferenceAlignedTargetMOSACTrainer
                                          else expected)
                        torch.testing.assert_close(critic.call_args_list[i].args[2], target_context,
                                                   rtol=0, atol=0)
                done = torch.tensor([[0.], [1.], [0.], [1.]])
                target_scalar, target_vector = trainer.critic_targets(
                    torch.ones(4, dim), w, done, scalar, vector)
                torch.testing.assert_close(target_vector,
                    torch.ones(4, dim) + trainer.gamma * (1 - done) * (vector + bonus))
                torch.testing.assert_close(target_scalar, (w * target_vector).sum(-1, keepdim=True))

    def test_actual_updates_preserve_single_candidate_and_rng(self):
        changed = False
        for dim in (2, 3):
            selector = training.FixedExtremeHierarchicalOperatorSelector(dim)
            for arm in selector.operators:
                config = selector.trainer_config(arm)
                trainers, values, rngs, calls = [], [], [], []
                for cls in (training.ConsistentTargetMOSACTrainer,
                            training.PreferenceAlignedTargetMOSACTrainer):
                    trainer = self.make_trainer(cls, dim)
                    for key, value in config.items():
                        setattr(trainer, key, value)
                    get_action = trainer.policy.get_action
                    contexts = []

                    def record_action(state, context, **kwargs):
                        contexts.append((context.detach().clone(), torch.is_grad_enabled()))
                        return get_action(state, context, **kwargs)

                    np.random.seed(17)
                    torch.manual_seed(17)
                    with patch.object(trainer.policy, "get_action", side_effect=record_action):
                        value = trainer.update()
                    self.assertTrue(np.isfinite(value[0]))
                    self.assertTrue(np.isfinite(value[1]).all())
                    trainers.append(trainer)
                    values.append(value)
                    rngs.append(torch.get_rng_state())
                    calls.append(contexts)
                self.assertTrue(torch.equal(*rngs))
                self.assertEqual(len(calls[0]), len(calls[1]))
                for (c1, g1), (c2, g2) in zip(*calls):
                    torch.testing.assert_close(c1, c2, rtol=0, atol=0)
                    self.assertEqual(g1, g2)
                self.assertIs(trainers[0].envelope_actor_loss.__func__,
                              trainers[1].envelope_actor_loss.__func__)
                if config["envelope_target_candidates"] == 1:
                    self.assertEqual(values[0][0], values[1][0])
                    np.testing.assert_array_equal(values[0][1], values[1][1])
                    for name in ("q1", "q2", "q1_target", "q2_target", "policy"):
                        for a, b in zip(getattr(trainers[0], name).parameters(),
                                        getattr(trainers[1], name).parameters()):
                            torch.testing.assert_close(a, b, rtol=0, atol=0)
                else:
                    changed |= not np.array_equal(values[0][1], values[1][1])
        self.assertTrue(changed)

    def test_feature_parity_with_v24_fixed(self):
        for method in (METHOD, OWN_ACTOR_METHOD):
            self.assertIn(method, training.METHODS)
            for name, function in vars(training).items():
                if name.startswith("uses_") and inspect.isfunction(function):
                    self.assertEqual(function(method), function(BASE), name)
            self.assertEqual(training.policy_archive_interval_for_method(method), 2500)
            self.assertEqual(training.policy_archive_limit_for_method(method), 20)
            self.assertEqual(training.context_dim_for_method(method, 17, 6, 2, 4), 13)

    def test_real_training_routes_fixed_selector_and_aligned_targets(self):
        make_env = training.make_piecewise_mo_env
        trainer_cls = training.PreferenceAlignedTargetMOSACTrainer
        exposed = []
        trainers = []

        def short_env(*args, **kwargs):
            env = make_env(*args, **kwargs)
            env.env._max_episode_steps = 3
            return env

        def small_trainer(*args, **kwargs):
            trainer = trainer_cls(*args, buffer_capacity=128, **kwargs)
            original = trainer.target_critic_context

            def record(update_context, candidate_context):
                result = original(update_context, candidate_context)
                torch.testing.assert_close(result, update_context, rtol=0, atol=0)
                exposed.append(not torch.equal(update_context, candidate_context))
                return result

            trainer.target_critic_context = record
            trainers.append(trainer)
            return trainer

        for method, selector_cls in (
            (METHOD, training.FixedExtremeHierarchicalOperatorSelector),
            (OWN_ACTOR_METHOD, training.OwnPreferenceFixedExtremeOperatorSelector),
        ):
            exposed.clear()
            with tempfile.TemporaryDirectory() as directory, \
                 patch.object(training, "make_piecewise_mo_env", side_effect=short_env), \
                 patch.object(training, "PreferenceAlignedTargetMOSACTrainer", side_effect=small_trainer), \
                 patch.object(training, selector_cls.__name__,
                              side_effect=lambda dim: selector_cls(dim, block_steps=4)), \
                 patch("builtins.print"):
                result = training.train_method("mo-halfcheetah-v5", method, 64, 25, 7, directory,
                    hidden=8, batch_size=4, learning_starts=5, eval_freq=1000, log_freq=1000, device="cpu")
            self.assertTrue(any(exposed))
            self.assertEqual(result["critic_candidate_context_rule"], "update_preference")
            self.assertEqual(result["critic_target_rule"], "scalar_projection_of_soft_vector")
            self.assertEqual(result["bootstrap_terminal_rule"], "terminated_only")
            self.assertEqual(result["evaluation_protocol"], "locked_full_simplex_v1")
            self.assertNotIn("actor_update", result)
            state = result["adaptive_operator_final"]
            self.assertEqual(state["operator_trainer_configs"],
                             selector_cls(2).state_dict()["operator_trainer_configs"])
            if method == OWN_ACTOR_METHOD:
                self.assertEqual(state["actor_update_rule"], "own_preference_with_existing_bc")
                self.assertTrue(all(c["envelope_policy_candidates"] == 1
                                    for c in state["operator_trainer_configs"].values()))
                self.assertTrue(any(c["envelope_target_candidates"] > 1
                                    for c in state["operator_trainer_configs"].values()))
            self.assertEqual([s["step"] for s in result["policy_archive_snapshots"]], [64])
            buffer = trainers[-1].buffer
            self.assertEqual(float(buffer.dones[:buffer.size].sum()), 0.)


if __name__ == "__main__":
    torch.set_num_threads(1)
    unittest.main()
