import copy
from itertools import product
import unittest
from unittest.mock import patch

import numpy as np
import torch

import train_bamor_mujoco as training


class OwnPreferenceActorTests(unittest.TestCase):
    def test_only_actor_candidates_change_for_every_arm(self):
        for dimension in [2, 3]:
            control = training.FixedExtremeHierarchicalOperatorSelector(dimension)
            ablation = training.OwnPreferenceFixedExtremeOperatorSelector(dimension)
            self.assertEqual(control.operators, ablation.operators)
            self.assertEqual(len(ablation.operators), 18)
            state = ablation.state_dict()
            self.assertEqual(state["actor_update_rule"], "own_preference_with_existing_bc")
            for arm in control.operators:
                expected = control.trainer_config(arm)
                expected["envelope_policy_candidates"] = 1
                self.assertEqual(ablation.trainer_config(arm), expected)
                self.assertEqual(state["operator_trainer_configs"][arm], expected)

    def test_score_blind_schedule_is_unchanged(self):
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
            before, control = trajectory(training.FixedExtremeHierarchicalOperatorSelector,
                                         dimension, 1)
            after, ablation = trajectory(training.OwnPreferenceFixedExtremeOperatorSelector,
                                         dimension, -10)
            self.assertEqual(before, after)
            self.assertEqual(control.counts, ablation.counts)
            self.assertEqual(control.step_counts, ablation.step_counts)
            self.assertEqual(control.warmup_sequence, ablation.warmup_sequence)
            self.assertEqual(control.min_family_fraction, ablation.min_family_fraction)

    def test_first_critic_update_matches_and_actor_search_is_removed(self):
        for trainer_cls, dimension in product(
            (training.ConsistentTargetMOSACTrainer, training.PreferenceAlignedTargetMOSACTrainer),
            (2, 3),
        ):
            torch.manual_seed(7)
            base = trainer_cls(
                3, 2, dimension, hidden=8, buffer_capacity=16, batch_size=4, device="cpu")
            base.preference_relabeling = True
            for index in range(8):
                state = np.full(3, index / 10, dtype=np.float32)
                weight = np.full(dimension, 1 / dimension, dtype=np.float32)
                base.buffer.push(state, np.zeros(2), weight, weight,
                                 np.ones(dimension), state + 0.1, weight, index % 3 == 0)
            control = training.FixedExtremeHierarchicalOperatorSelector(dimension)
            ablation = training.OwnPreferenceFixedExtremeOperatorSelector(dimension)
            for arm in control.operators:
                with self.subTest(trainer=trainer_cls.__name__, dimension=dimension, arm=arm):
                    trainers, targets = [], []
                    for selector in [control, ablation]:
                        trainer = copy.deepcopy(base)
                        config = selector.trainer_config(arm)
                        for key, value in config.items():
                            setattr(trainer, key, value)
                        calls = []
                        get_action = trainer.policy.get_action

                        def record_action(*args, **kwargs):
                            calls.append(torch.is_grad_enabled())
                            return get_action(*args, **kwargs)

                        np.random.seed(17)
                        torch.manual_seed(17)
                        with patch.object(trainer.policy, "get_action", side_effect=record_action):
                            q, target = trainer.update()
                        self.assertEqual(calls.count(False), config["envelope_target_candidates"])
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
                                     control.trainer_config(arm)["envelope_policy_candidates"] > 1)


if __name__ == "__main__":
    torch.set_num_threads(1)
    unittest.main()
