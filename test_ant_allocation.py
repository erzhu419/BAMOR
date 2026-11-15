import copy
import unittest

import numpy as np

import diagnose_ant_allocation as diagnostic


def fixture():
    selector = diagnostic.CLASSES["base_adaptive"](3)
    blocks = []
    for i in range(47):
        op = selector.warmup_sequence[i % 10]
        complete = i < 46
        gain = np.array([.1, -.2, .3]) * ((i % 3) - 1)
        score = float(np.tanh(gain).mean())
        if complete:
            selector.scores[op] = (1 - selector.lr) * selector.scores[op] + selector.lr * score
            selector.counts[op] += 1
        steps = 1500 if complete else 1000
        selector.step_counts[op] += steps
        block = {"operator": op, "end_step": 6500 + i * 1500 if complete else 75000,
                 "steps": steps, "complete": complete, "score_after": selector.scores[op],
                 "half_counts": [750, 750] if complete else [750, 250],
                 "scale_before_block": [1., 1., 1.]}
        if complete:
            block.update(half_mean_rewards=[[0., 0., 0.], gain.tolist()],
                         normalized_vector_gain=gain.tolist(), progress_score=score)
        blocks.append(block)
    selector.block_history = blocks
    return {"env": diagnostic.experiment.ENV, "seed": 1301, "total_steps": 75000,
            "training_interactions": 75000, "method": diagnostic.experiment.METHODS["base_adaptive"],
            "adaptive_operator_final": selector.state_dict(),
            "ground_truth": {"mode_history": [[0, 0], [20750, 1]]}}


class AntAllocationTests(unittest.TestCase):
    def test_native_score_reconstruction_and_unscored_tail(self):
        raw = fixture()
        diagnostic.validate_raw(raw, "base_adaptive", 1301)
        changed = copy.deepcopy(raw)
        changed["adaptive_operator_final"]["block_history"][12]["progress_score"] += .1
        with self.assertRaises(AssertionError):
            diagnostic.validate_raw(changed, "base_adaptive", 1301)
        changed = copy.deepcopy(raw)
        changed["adaptive_operator_final"]["block_history"][-1]["progress_score"] = 0.
        with self.assertRaises(AssertionError):
            diagnostic.validate_raw(changed, "base_adaptive", 1301)

    def test_credit_decomposition_and_switch_boundary(self):
        raw = fixture()
        blocks = raw["adaptive_operator_final"]["block_history"]
        history = [[0, 0], [6500, 1], [20001, 2]]
        selector = diagnostic.selector_prefix(blocks, 20)
        self.assertEqual(diagnostic.switch_crossings(blocks[0], history), [])
        self.assertEqual(diagnostic.switch_crossings(blocks[10], history), [20001])
        for op in selector.operators:
            credit = diagnostic.credit_sources(blocks, 20, op, selector.lr, history)
            self.assertAlmostEqual(sum(r["ema_contribution"] for r in credit), selector.scores[op])

    def test_prefix_probabilities_ignore_future_and_mode_annotations(self):
        raw = fixture()
        blocks = raw["adaptive_operator_final"]["block_history"]
        before = np.random.get_state()
        record = diagnostic.decision_record(blocks, 15, [[0, 0]])
        changed = copy.deepcopy(blocks)
        for row in changed[15:]:
            row["score_after"] = .9
        other = diagnostic.decision_record(changed, 15, [[0, 0], [20750, 1]])
        for key in ("adaptive_probabilities", "blind_probabilities", "score_before"):
            self.assertEqual(record[key], other[key])
        after = np.random.get_state()
        np.testing.assert_array_equal(before[1], after[1])
        self.assertEqual(before[2:], after[2:])
        self.assertAlmostEqual(sum(record["adaptive_probabilities"].values()), 1.)
        self.assertGreater(other["pre_switch_latest_probability_mass"], 0.)

    def test_work_includes_first_update_and_tail(self):
        blocks = fixture()["adaptive_operator_final"]["block_history"]
        counts = diagnostic.work_counts(blocks)
        self.assertEqual(counts["learner_updates"], 70001)
        self.assertEqual(counts["base_batch_rows"], 70001 * 256)
        selector = diagnostic.CLASSES["base_adaptive"](3)
        expected = 3 + sum(b["steps"] * selector.trainer_config(b["operator"])["critic_relabel_copies"]
                           for b in blocks)
        self.assertEqual(counts["actor_and_critic_expanded_rows_each"], expected * 256)
        self.assertEqual(len(diagnostic.run_feedback(fixture())[0]), 37)

    def test_equivalent_labels_and_relabel_rate_difference(self):
        selector = diagnostic.CLASSES["base_adaptive"](3)
        config = lambda op: diagnostic.effective_config(selector, op)
        self.assertEqual(config("recovery:stratified_relabel_envelope"),
                         config("recovery:stratified_relabel_envelope_policy"))
        self.assertNotEqual(config("extreme:edge_envelope_policy"),
                            config("extreme:edge_relabel_envelope_policy"))


if __name__ == "__main__":
    unittest.main()
