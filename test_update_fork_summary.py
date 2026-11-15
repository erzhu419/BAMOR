import unittest

from summarize_update_forks import correlation, grouped, metrics


def row(weight, gain, signal):
    return {"weight_index": weight, "surrogate": signal,
            "on_minus_off": [gain] * 3, "on_minus_base": [gain + 1] * 3,
            "off_minus_base": [1.] * 3,
            "paired_on_minus_off": [[gain] * 3] * 3,
            "gradient_alignment_advantage": signal, "own_min_other_cosine": -.5}


class ForkSummaryTests(unittest.TestCase):
    def test_choice_gains_do_not_confuse_sign_accuracy_with_return(self):
        rows = [row(0, 2., 1.), row(1, -8., 1.), row(2, -1., -1.)]
        result = metrics(rows, 2)
        self.assertEqual(result["surrogate_sign_correct"], 2)
        self.assertEqual(result["sign_denominator"], 3)
        self.assertEqual(result["surrogate_choose_on"], 2)
        self.assertEqual(result["mean_selected_minus_off"], -2.)
        self.assertAlmostEqual(result["mean_selected_minus_on"], 1 / 3)
        self.assertEqual(result["mean_selected_minus_base"], -1.)

    def test_ties_and_paired_repeat_disagreement(self):
        rows = [row(0, 0., 0.), row(1, 1., 0.), row(2, -1., -2.)]
        rows[1]["paired_on_minus_off"] = [[-1., 2., 2.]] * 3
        result = metrics(rows, 2)
        self.assertEqual(result["outcome_ties"], 1)
        self.assertEqual(result["sign_denominator"], 2)
        self.assertEqual(result["surrogate_sign_correct"], 1)
        self.assertEqual(result["on_all_repeats_better"], 0)
        self.assertEqual(result["on_all_repeats_worse"], 1)

    def test_grouping_keeps_preferences_and_horizons_separate(self):
        rows = [row(i, i + 1., i + 1.) for i in range(3)]
        rows[1]["on_minus_off"][0] = -20.
        result = grouped(rows)
        self.assertEqual(result["32"]["by_preference"]["1"]["mean_on_minus_off"], -20.)
        self.assertEqual(result["512"]["by_preference"]["1"]["mean_on_minus_off"], 2.)
        self.assertEqual(result["512"]["all_preferences"]["n"], 3)

    def test_constant_and_missing_gradient_signals_are_not_correlations(self):
        self.assertIsNone(correlation([1, 1, 1], [1, 2, 3]))
        rows = [row(i, i, i) for i in range(3)]
        for r in rows:
            r["gradient_alignment_advantage"] = None
        result = metrics(rows, 2)
        self.assertIsNone(result["correlations"]["gradient_alignment_advantage"]["spearman"])
        self.assertAlmostEqual(result["correlations"]["surrogate"]["spearman"], 1.)


if __name__ == "__main__":
    unittest.main()
