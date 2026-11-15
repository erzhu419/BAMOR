import unittest

import numpy as np

import analyze_selector_loss as analysis


class SelectorLossTests(unittest.TestCase):
    def test_decomposition_matches_live_selector_in_all_branches(self):
        seen = set()
        for name, selector in analysis.examples().items():
            with self.subTest(state=name):
                scores_before = dict(selector.scores)
                counts_before = dict(selector.counts)
                steps_before = dict(selector.step_counts)
                report = analysis.loss_decomposition(selector)
                seen.add(report["route"])
                self.assertAlmostEqual(report["score_loss"], sum(report["components"].values()))
                self.assertTrue(all(value >= -1e-12 for value in report["components"].values()))
                self.assertAlmostEqual(sum(report["operator_probabilities"].values()), 1.0)
                self.assertEqual(scores_before, selector.scores)
                self.assertEqual(counts_before, selector.counts)
                self.assertEqual(steps_before, selector.step_counts)
        self.assertEqual(seen, {"warmup", "family_quota", "free_family_sampling"})

    def test_conditional_transfer_and_finite_sum(self):
        for eta in (0.0, 0.02, 0.2, 1.0, 2.0):
            losses, bounds = [], []
            for selector in analysis.examples().values():
                report = analysis.loss_decomposition(selector)
                scores = np.array([selector.scores[op] for op in selector.operators])
                p = np.array([report["operator_probabilities"][op] for op in selector.operators])
                # Include truth rankings that disagree with the stored scores.
                for best in range(len(scores)):
                    truth = np.maximum(-1, scores - eta)
                    truth[best] = min(1, scores[best] + eta)
                    loss = float(truth.max() - p @ truth)
                    bound = min(report["bounded_progress_worst_case"],
                                report["score_loss"] + 2 * eta)
                    self.assertLessEqual(loss, bound + 1e-12)
                    losses.append(loss)
                    bounds.append(bound)
            self.assertLessEqual(sum(losses), sum(bounds) + 1e-10)

    def test_coarse_v25_bound_is_worse_than_bounded_score_range(self):
        report = analysis.analyze()
        self.assertGreater(report["coarse_free_sampling_bound"], 2.0)
        self.assertAlmostEqual(report["coarse_free_sampling_bound"],
                               sum(report["coarse_components"].values()))

    def test_zero_score_loss_does_not_identify_good_true_progress(self):
        example = analysis.analyze()["zero_scores_bounded_mean_example"]
        self.assertEqual(example["score_loss"], 0.0)
        self.assertAlmostEqual(example["conditional_mean_loss"], 40 / 21)
        self.assertAlmostEqual(example["transfer_bound"], example["conditional_mean_loss"])

    def test_quota_cost_is_not_hidden_in_temperature_terms(self):
        selector = analysis.examples()["quota_routes_away"]
        report = analysis.loss_decomposition(selector)
        self.assertEqual(report["route"], "family_quota")
        self.assertAlmostEqual(report["components"]["family_routing"], 0.1)
        self.assertAlmostEqual(report["components"]["within_softmax"], 0.0)
        self.assertAlmostEqual(report["components"]["within_exploration"], 0.0)


if __name__ == "__main__":
    unittest.main()
