import copy
import unittest

import numpy as np

import diagnose_scoped_arm_feedback as diagnostic


def blocks(dimension):
    selector = diagnostic.replay.training.ProgressExtremeHierarchicalOperatorSelector(dimension)
    scores = selector.scores.copy()
    rows = []
    for i, op in enumerate(selector.warmup_sequence * 3):
        gain = np.linspace(-.2, .4, dimension) * np.sin(i + 1)
        progress = float(np.tanh(gain).mean())
        scores[op] = (1 - selector.lr) * scores[op] + selector.lr * progress
        rows.append(dict(complete=True, steps=1500, end_step=6500 + i * 1500, operator=op,
                         normalized_vector_gain=gain.tolist(), progress_score=progress,
                         score_after=scores[op]))
    return rows


class ScopedArmFeedbackTests(unittest.TestCase):
    def test_dimensions_native_ema_and_random_state(self):
        for dimension in (2, 3):
            rows = blocks(dimension)
            before = np.random.get_state()
            result = diagnostic.native_feedback(rows, [[0, 0]], dimension)
            after = np.random.get_state()
            self.assertEqual(before[0], after[0])
            np.testing.assert_array_equal(before[1], after[1])
            self.assertEqual(before[2:], after[2:])
            self.assertEqual(len(result["blocks"]), 20)
            for op in (diagnostic.E, diagnostic.B):
                uses = [r for r in rows[10:] if r["operator"] == op]
                self.assertEqual(result["arms"][op]["terminal_score"], uses[-1]["score_after"])
                self.assertEqual(result["arms"][op]["first_target_progress"], uses[0]["progress_score"])
            for row in result["blocks"]:
                self.assertEqual(len(row["objective_score_terms"]), dimension)
                self.assertAlmostEqual(sum(row["objective_score_terms"]), row["progress_score"])
                self.assertAlmostEqual(sum(row["probabilities"].values()), 1.)

    def test_choice_replay_has_no_current_or_future_feedback(self):
        rows = blocks(3)
        a = diagnostic.native_feedback(rows, [[0, 0]], 3)
        changed = copy.deepcopy(rows)
        for r in changed[16:]:
            r["score_after"] += .9
        b = diagnostic.native_feedback(changed, [[0, 0]], 3)
        for i in range(7):
            self.assertEqual(a["blocks"][i]["probabilities"], b["blocks"][i]["probabilities"])
            self.assertEqual(a["blocks"][i]["score_before"], b["blocks"][i]["score_before"])

    def test_modes_are_annotations_not_selection_inputs(self):
        rows = blocks(2)
        a = diagnostic.native_feedback(rows, [[0, 0]], 2)
        b = diagnostic.native_feedback(rows, [[0, 0], [20750, 3], [25200, 1]], 2)
        self.assertFalse(a["blocks"][0]["switches_inside_block"])
        self.assertEqual(b["blocks"][0]["switches_inside_block"], [[20750, 3]])
        for x, y in zip(a["blocks"], b["blocks"]):
            self.assertEqual(x["probabilities"], y["probabilities"])
            self.assertEqual(x["progress_score"], y["progress_score"])

    def test_conditional_contrasts_and_ties(self):
        values = dict(legacy_adaptive=100., envelope_only=110., bc_only=95., scoped_adaptive=90.)
        e_scores = dict(legacy_adaptive=0., envelope_only=2., bc_only=0., scoped_adaptive=3.)
        b_scores = dict(legacy_adaptive=0., envelope_only=0., bc_only=-1., scoped_adaptive=-2.)
        feedback, metrics = {}, {}
        for c in diagnostic.CONDITIONS:
            feedback[c] = dict(extreme_family_score=e_scores[c],
                arms={op: dict.fromkeys(diagnostic.SIGNALS, scores[c])
                      for op, scores in ((diagnostic.E, e_scores), (diagnostic.B, b_scores))},
                blocks=[dict(end_step=21500, operator=diagnostic.E,
                             probabilities={diagnostic.E: .5, diagnostic.B: .5})])
            metrics[c] = dict.fromkeys(diagnostic.METRICS, values[c])
        result = diagnostic.case_comparison(feedback, metrics)
        self.assertEqual(result["interaction"]["final_hv"], -15.)
        self.assertFalse(result["contrasts"]["envelope_with_bc"]["direction_agreement"]["terminal_score"]["final_hv"])
        counts = diagnostic.agreement_counts({"env": {1: result}})
        self.assertEqual(counts["terminal_score"]["final_hv"], dict(agree=3, disagree=1, ties=0))
        feedback["envelope_only"]["arms"][diagnostic.E]["terminal_score"] = 0.
        tied = diagnostic.case_comparison(feedback, metrics)
        self.assertIsNone(tied["contrasts"]["envelope_without_bc"]["direction_agreement"]["terminal_score"]["final_hv"])
        feedback["bc_only"]["blocks"][0]["operator"] = diagnostic.B
        with self.assertRaises(AssertionError):
            diagnostic.case_comparison(feedback, metrics)


if __name__ == "__main__":
    unittest.main()
