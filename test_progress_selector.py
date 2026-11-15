import inspect
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

import train_bamor_mujoco as training


Progress = training.ProgressExtremeHierarchicalOperatorSelector
Fixed = training.FixedProgressExtremeHierarchicalOperatorSelector
OwnPreference = training.OwnPreferenceFixedProgressExtremeOperatorSelector
OwnPreferenceAdaptive = training.OwnPreferenceProgressExtremeOperatorSelector
OwnPreferenceBoth = training.OwnPreferenceBothFixedProgressExtremeOperatorSelector


def observe(selector, rewards, start=1):
    for step, reward in enumerate(rewards, start):
        operator = selector.select_for_step(step)
        selector.observe_transition(operator, reward, step)
        selector.maybe_finish_block(step)


class ProgressSelectorTests(unittest.TestCase):
    def test_identical_rewards_do_not_reward_preference_distribution(self):
        for dimension in [2, 3]:
            selector = Progress(dimension, block_steps=12, learning_starts=12)
            rewards = np.tile(-np.arange(dimension), (252, 1))
            observe(selector, rewards)
            self.assertEqual(sum(selector.counts.values()), 20)
            self.assertEqual(sum(selector.step_counts.values()), 240)
            self.assertTrue(all(score == 0 for score in selector.scores.values()))
            self.assertTrue(all(b["progress_score"] == 0 for b in selector.block_history))

    def test_progress_direction_and_affine_units(self):
        for direction in [1, -1]:
            rewards = np.array([[1, -1], [-1, 1]] * 4 +
                               [[0, 0]] * 2 + [[direction, 2 * direction]] * 2)
            base, scaled = Progress(2, 4, 8), Progress(2, 4, 8)
            observe(base, rewards)
            observe(scaled, rewards * np.array([100, 0.02]) + np.array([300, -10]))
            a, b = base.block_history[-1], scaled.block_history[-1]
            self.assertGreater(direction * a["progress_score"], 0)
            np.testing.assert_allclose(a["normalized_vector_gain"], b["normalized_vector_gain"])
            self.assertAlmostEqual(a["progress_score"], b["progress_score"])

    def test_warmup_partial_blocks_and_prefix_only_scaling(self):
        selector = Progress(2, 4, 8)
        observe(selector, np.array([[1, -1], [-1, 1]] * 4))
        self.assertEqual(sum(selector.counts.values()), 0)
        self.assertEqual(sum(selector.step_counts.values()), 0)
        self.assertIsNone(selector.current_operator)
        before = np.sqrt(selector.vector_var)
        observe(selector, [[100, 100], [100, 100]], start=9)
        np.testing.assert_allclose(selector.block_scale, before)
        selector.maybe_finish_block(10, force=True)
        self.assertEqual(sum(selector.counts.values()), 0)
        self.assertEqual(selector.block_history[-1]["half_counts"], [2, 0])
        self.assertFalse(selector.block_history[-1]["complete"])

    def test_frozen_budget_has_thirty_complete_learning_blocks(self):
        selector = Progress(2)
        observe(selector, np.zeros((50000, 2)))
        selector.maybe_finish_block(50000, force=True)
        self.assertEqual(sum(selector.counts.values()), 30)
        self.assertEqual(sum(selector.step_counts.values()), 45000)
        self.assertEqual(selector.block_history[0]["end_step"], 6500)
        self.assertEqual(selector.block_history[-1]["end_step"], 50000)
        self.assertTrue(all(b["complete"] for b in selector.block_history))

    def test_controls_keep_configs_and_fixed_ignores_scores(self):
        a, b = Progress(3), Fixed(3)
        old = training.ExtremeHierarchicalArchiveRiskAdaptiveOperatorSelector(3)
        self.assertEqual(a.warmup_sequence, old.warmup_sequence)
        self.assertEqual(a.min_family_fraction, old.min_family_fraction)
        for arm in a.operators:
            self.assertEqual(a.trainer_config(arm), old.trainer_config(arm))
            self.assertEqual(a.trainer_config(arm), b.trainer_config(arm))
        b.counts = dict.fromkeys(b.operators, 2)
        b.scores = {arm: float(i) for i, arm in enumerate(b.operators)}
        np.random.seed(19)
        first = [b.select() for _ in range(50)]
        b.scores = {arm: -100 * value for arm, value in b.scores.items()}
        stored = b.scores.copy()
        np.random.seed(19)
        self.assertEqual(first, [b.select() for _ in range(50)])
        self.assertEqual(b.scores, stored)
        for method in training.V25_MUJOCO_METHODS:
            self.assertIn(method, training.METHODS)
            for name, function in vars(training).items():
                if name.startswith("uses_") and inspect.isfunction(function):
                    self.assertEqual(function(method),
                        function("bamor_v24_target_consistent_archive_mix"), name)
            self.assertEqual(training.policy_archive_interval_for_method(method), 2500)

    def test_own_preference_actor_changes_only_candidate_count(self):
        control, ablation = Fixed(3), OwnPreference(3)
        self.assertEqual(control.operators, ablation.operators)
        self.assertEqual(control.warmup_sequence, ablation.warmup_sequence)
        self.assertEqual(control.min_family_fraction, ablation.min_family_fraction)
        for arm in control.operators:
            expected = control.trainer_config(arm)
            actual = ablation.trainer_config(arm)
            self.assertEqual(actual.pop("envelope_policy_candidates"), 1)
            expected.pop("envelope_policy_candidates")
            self.assertEqual(actual, expected)
        self.assertEqual(ablation.state_dict()["actor_update_rule"],
                         "own_preference_with_existing_bc")

    def test_own_preference_adaptive_keeps_progress_selection(self):
        control, ablation = Progress(3), OwnPreferenceAdaptive(3)
        self.assertEqual(control.operators, ablation.operators)
        self.assertEqual(control.warmup_sequence, ablation.warmup_sequence)
        self.assertEqual(control.min_family_fraction, ablation.min_family_fraction)
        for arm in control.operators:
            expected = control.trainer_config(arm)
            actual = ablation.trainer_config(arm)
            self.assertEqual(actual.pop("envelope_policy_candidates"), 1)
            expected.pop("envelope_policy_candidates")
            self.assertEqual(actual, expected)
        self.assertEqual(ablation.state_dict()["score_signal"],
                         "within_block_normalized_vector_progress")
        self.assertEqual(ablation.state_dict()["actor_update_rule"],
                         "own_preference_with_existing_bc")

    def test_own_preference_both_changes_only_critic_candidate_count(self):
        control, ablation = OwnPreference(3), OwnPreferenceBoth(3)
        self.assertEqual(control.operators, ablation.operators)
        self.assertEqual(control.warmup_sequence, ablation.warmup_sequence)
        for arm in control.operators:
            expected = control.trainer_config(arm)
            actual = ablation.trainer_config(arm)
            self.assertEqual(actual.pop("envelope_target_candidates"), 1)
            expected.pop("envelope_target_candidates")
            self.assertEqual(actual, expected)
        self.assertEqual(ablation.state_dict()["critic_candidate_rule"],
                         "own_preference_only")

    def test_real_training_routes_both_variants(self):
        make_env = training.make_piecewise_mo_env
        trainer_cls = training.ConsistentTargetMOSACTrainer

        def short_env(*args, **kwargs):
            env = make_env(*args, **kwargs)
            env.env._max_episode_steps = 3
            return env

        def small_trainer(*args, **kwargs):
            return trainer_cls(*args, buffer_capacity=128, **kwargs)

        with tempfile.TemporaryDirectory() as directory:
            for method in sorted(training.V25_MUJOCO_METHODS):
                with patch.object(training, "make_piecewise_mo_env", side_effect=short_env), \
                     patch.object(training, "ConsistentTargetMOSACTrainer", side_effect=small_trainer), \
                     patch("builtins.print"):
                    result = training.train_method(
                        "mo-halfcheetah-v5", method, 64, 25, 7, directory,
                        hidden=8, batch_size=4, learning_starts=5,
                        eval_freq=1000, log_freq=1000, device="cpu")
                selector = result["adaptive_operator_final"]
                self.assertEqual(selector["score_signal"], "within_block_normalized_vector_progress")
                self.assertEqual(sum(selector["family_step_counts"].values()), 59)
                self.assertEqual(sum(selector["counts"].values()), 0)
                self.assertEqual(len(selector["block_history"]), 1)
                self.assertFalse(selector["block_history"][0]["complete"])
                self.assertEqual(result["bootstrap_terminal_rule"], "terminated_only")
                self.assertEqual(result["critic_target_rule"], "scalar_projection_of_soft_vector")
                self.assertEqual(result["evaluation_protocol"], "locked_full_simplex_v1")


if __name__ == "__main__":
    unittest.main()
