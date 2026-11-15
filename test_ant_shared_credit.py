import copy
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

import ant_shared_credit as algorithm
import diagnose_ant_allocation as diagnostic
import run_ant_shared_credit as experiment
from test_ant_allocation import fixture


def finish(selector, op, start, gain, complete=True):
    selector.current_operator = op
    selector.block_start_step = start
    selector.block_count = selector.block_steps if complete else 3
    selector.half_counts[:] = [2, 2] if complete else [2, 1]
    selector.half_sums[:] = [np.zeros(3), np.asarray(gain) * selector.half_counts[1]]
    selector.step_counts[op] += selector.block_count
    selector.maybe_finish_block(start + selector.block_count - 1, force=True)


class SharedCreditTests(unittest.TestCase):
    def test_configuration_groups_preserve_portfolio_and_rng(self):
        before = np.random.get_state()
        selector = algorithm.SharedCreditOwnPreferenceSelector(3)
        base = diagnostic.CLASSES["base_adaptive"](3)
        self.assertEqual(len(selector.credit_groups), 11)
        self.assertEqual(selector.operators, base.operators)
        self.assertEqual(selector.warmup_sequence, base.warmup_sequence)
        for group in selector.credit_groups:
            for op in group:
                self.assertEqual(diagnostic.effective_config(selector, op),
                                 diagnostic.effective_config(base, group[0]))
        self.assertEqual(diagnostic.replay.selection_distribution(selector),
                         diagnostic.replay.selection_distribution(base))
        after = np.random.get_state()
        np.testing.assert_array_equal(before[1], after[1])
        self.assertEqual(before[2:], after[2:])

    def test_alias_updates_once_without_inflating_counts(self):
        selector = algorithm.SharedCreditOwnPreferenceSelector(3, block_steps=4, learning_starts=0)
        left, right = "extreme:edge_relabel_envelope_policy", "extreme:edge_relabel"
        gain = [.1, .2, -.1]
        score = float(np.tanh(gain).mean())
        finish(selector, left, 1, gain)
        expected = selector.lr * score
        self.assertAlmostEqual(selector.scores[left], expected)
        self.assertEqual(selector.scores[left], selector.scores[right])
        self.assertEqual(sum(selector.counts.values()), 1)
        self.assertEqual(selector.counts[right], 0)
        finish(selector, right, 5, gain)
        expected = (1 - selector.lr) * expected + selector.lr * score
        self.assertAlmostEqual(selector.scores[left], expected)
        self.assertEqual(selector.scores[left], selector.scores[right])
        self.assertEqual(sum(selector.counts.values()), 2)
        self.assertEqual(sum(selector.step_counts.values()), 8)
        before = selector.scores.copy()
        finish(selector, right, 9, gain, complete=False)
        self.assertEqual(selector.scores, before)
        self.assertEqual(sum(selector.step_counts.values()), 11)
        self.assertEqual(sum(selector.counts.values()), 2)

    def test_saved_credit_reconstruction(self):
        raw = fixture()
        experiment.credit_check(raw, "per_label")
        selector = algorithm.SharedCreditOwnPreferenceSelector(3)
        state = raw["adaptive_operator_final"]
        state.update(score_credit_rule="effective_configuration_shared_ema", credit_groups=selector.credit_groups)
        for block in state["block_history"]:
            op = block["operator"]
            if block["complete"]:
                value = (1 - selector.lr) * selector.scores[op] + selector.lr * block["progress_score"]
                for alias in selector.credit_aliases[op]:
                    selector.scores[alias] = value
            block["score_after"] = selector.scores[op]
        state["scores"] = selector.scores
        experiment.credit_check(raw, "shared")
        broken = copy.deepcopy(raw)
        broken["adaptive_operator_final"]["scores"][selector.operators[0]] += 1
        with self.assertRaises(AssertionError):
            experiment.credit_check(broken, "shared")

    def test_short_ant_training_preserves_warmup(self):
        training = experiment.training
        make_env = training.make_piecewise_mo_env
        trainer_cls = training.ConsistentTargetMOSACTrainer
        native_cls = training.OwnPreferenceBothProgressExtremeOperatorSelector

        def short_env(*args, **kwargs):
            env = make_env(*args, **kwargs)
            env.env._max_episode_steps = 3
            return env

        def small_trainer(*args, **kwargs):
            return trainer_cls(*args, buffer_capacity=128, **kwargs)

        results = []
        with tempfile.TemporaryDirectory() as folder:
            for cls in (native_cls, algorithm.SharedCreditOwnPreferenceSelector):
                with patch.object(training, "make_piecewise_mo_env", side_effect=short_env), \
                     patch.object(training, "ConsistentTargetMOSACTrainer", side_effect=small_trainer), \
                     patch.object(training, "OwnPreferenceBothProgressExtremeOperatorSelector",
                                  side_effect=lambda *a, **kw: cls(*a, block_steps=4, **kw)), \
                     patch("builtins.print"):
                    results.append(training.train_method(
                        "mo-ant-v5", experiment.previous.METHODS["base_adaptive"], 64, 25000,
                        17001, folder, hidden=8, batch_size=4, learning_starts=5,
                        eval_freq=1000, log_freq=1000, archive_steps=(64,), device="cpu"))
        keys = ("operator", "half_mean_rewards", "normalized_vector_gain", "progress_score")
        for a, b in zip(*(r["adaptive_operator_final"]["block_history"][:10] for r in results)):
            self.assertEqual({k: a[k] for k in keys}, {k: b[k] for k in keys})
        episodes = [[r for r in result["episode_rewards"] if r["step"] <= 45] for result in results]
        self.assertEqual(episodes[0], episodes[1])
        self.assertEqual(results[1]["adaptive_operator_final"]["score_credit_rule"],
                         "effective_configuration_shared_ema")
        self.assertEqual(results[1]["training_interactions"], 64)
        for result in results:
            self.assertEqual(result["evaluation_interactions"], 567)
            self.assertEqual(result["environment_initialization_probe_steps"], 2)


if __name__ == "__main__":
    unittest.main()
