import copy
import unittest
from unittest.mock import patch

import numpy as np
import torch

import train_bamor_mujoco as training


PAIRS = [
    (training.FixedExtremeHierarchicalOperatorSelector,
     training.OwnPreferenceCriticFixedExtremeOperatorSelector),
    (training.OwnPreferenceFixedExtremeOperatorSelector,
     training.OwnPreferenceBothFixedExtremeOperatorSelector),
]


class EnvelopeFactorialTests(unittest.TestCase):
    def test_only_critic_candidates_change_for_every_arm(self):
        for dimension in [2, 3]:
            for control_cls, ablation_cls in PAIRS:
                control, ablation = control_cls(dimension), ablation_cls(dimension)
                self.assertEqual(control.operators, ablation.operators)
                self.assertEqual(len(ablation.operators), 18)
                state = ablation.state_dict()
                self.assertEqual(state["critic_candidate_rule"], "own_preference_only")
                for arm in control.operators:
                    expected = control.trainer_config(arm)
                    expected["envelope_target_candidates"] = 1
                    self.assertEqual(ablation.trainer_config(arm), expected)
                    self.assertEqual(state["operator_trainer_configs"][arm], expected)

    def test_fixed_schedule_is_unchanged_in_both_actor_settings(self):
        def trajectory(cls, dimension, sign):
            np.random.seed(27)
            selector = cls(dimension, block_steps=4)
            sequence = []
            for step in range(1, 401):
                arm = selector.select_for_step(step)
                sequence.append(arm)
                selector.observe_step(arm, sign * (step % 13 + 1))
                selector.maybe_finish_block(step)
            return sequence, selector

        for dimension in [2, 3]:
            for control_cls, ablation_cls in PAIRS:
                before, control = trajectory(control_cls, dimension, 1)
                after, ablation = trajectory(ablation_cls, dimension, -10)
                self.assertEqual(before, after)
                self.assertEqual(control.counts, ablation.counts)
                self.assertEqual(control.step_counts, ablation.step_counts)
                self.assertEqual(control.warmup_sequence, ablation.warmup_sequence)
                self.assertEqual(control.min_family_fraction, ablation.min_family_fraction)

    def test_single_context_critic_target_and_actor_isolation(self):
        for dimension in [2, 3]:
            torch.manual_seed(7)
            base = training.ConsistentTargetMOSACTrainer(
                3, 2, dimension, hidden=8, buffer_capacity=16, batch_size=4, device="cpu")
            base.preference_relabeling = True
            for index in range(8):
                state = np.full(3, index / 10, dtype=np.float32)
                weight = np.full(dimension, 1 / dimension, dtype=np.float32)
                base.buffer.push(state, np.zeros(2), weight, weight,
                                 np.ones(dimension), state + 0.1, weight, index % 3 == 0)
            selectors = [ablation_cls(dimension) for _, ablation_cls in PAIRS]
            for arm in selectors[0].operators:
                with self.subTest(dimension=dimension, arm=arm):
                    trainers, targets = [], []
                    for selector in selectors:
                        trainer = copy.deepcopy(base)
                        config = selector.trainer_config(arm)
                        for key, value in config.items():
                            setattr(trainer, key, value)
                        calls, critic_samples = [], []
                        get_action = trainer.policy.get_action
                        critic_targets = trainer.critic_targets

                        def record_action(state, context, **kwargs):
                            result = get_action(state, context, **kwargs)
                            calls.append(torch.is_grad_enabled())
                            if not torch.is_grad_enabled():
                                critic_samples.append((context.detach().clone(), result[1].detach().clone()))
                            return result

                        def check_target(reward, weights, done, scalar, vector):
                            self.assertEqual(len(critic_samples), 1)
                            context, log_prob = critic_samples[0]
                            torch.testing.assert_close(context, weights)
                            torch.testing.assert_close(scalar,
                                (weights * vector).sum(-1, keepdim=True) - trainer.alpha * log_prob)
                            result = critic_targets(reward, weights, done, scalar, vector)
                            torch.testing.assert_close(result[1], reward + trainer.gamma *
                                (1 - done) * (vector - trainer.alpha * log_prob))
                            torch.testing.assert_close(result[0],
                                (weights * result[1]).sum(-1, keepdim=True))
                            return result

                        np.random.seed(17)
                        torch.manual_seed(17)
                        with patch.object(trainer.policy, "get_action", side_effect=record_action), \
                             patch.object(trainer, "critic_targets", side_effect=check_target) as target_call:
                            q, target = trainer.update()
                        target_call.assert_called_once()
                        self.assertEqual(calls.count(False), 1)
                        self.assertEqual(calls.count(True), config["envelope_policy_candidates"])
                        self.assertTrue(np.isfinite(q))
                        self.assertTrue(np.isfinite(target).all())
                        self.assertTrue(all(torch.isfinite(p).all() for p in trainer.policy.parameters()))
                        trainers.append(trainer)
                        targets.append(target)
                    np.testing.assert_array_equal(*targets)
                    for name in ["q1", "q2", "q1_target", "q2_target"]:
                        for before, after in zip(getattr(trainers[0], name).parameters(),
                                                 getattr(trainers[1], name).parameters()):
                            torch.testing.assert_close(before, after, rtol=0, atol=0)
                    actor_changed = any(not torch.equal(before, after)
                        for before, after in zip(trainers[0].policy.parameters(),
                                                 trainers[1].policy.parameters()))
                    self.assertEqual(actor_changed,
                                     selectors[0].trainer_config(arm)["envelope_policy_candidates"] > 1)


if __name__ == "__main__":
    torch.set_num_threads(1)
    unittest.main()
