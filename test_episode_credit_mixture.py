import json
import unittest

from diagnose_episode_credit_mixture import loss_budget, paired_moments


def rows(ema, trend):
    return [dict(ema_error=a, trend_error=b) for a, b in zip(ema, trend)]


def budget_row(step, ema_error, selected_error):
    return dict(seed=2101, start_step=step, end_step=step + 9, length=10,
        epoch_start=0, operator="mixed", detected_mixed=False, true_mixed=False,
        true_modes=[0], selected="trend", utility=0., ema_prediction=-ema_error,
        trend_prediction=-selected_error, ema_error=ema_error, trend_error=selected_error,
        selected_error=selected_error, past_comparisons=3,
        past_squared_errors=dict(ema=5., trend=1.), fit_samples=4)


class EpisodeCreditMixtureTests(unittest.TestCase):
    def test_opposite_errors_optimal_mixture_and_orientation(self):
        measured = paired_moments(rows([2., -2.], [-2., 2.]))
        self.assertEqual(measured["hindsight_trend_weight"], .5)
        self.assertEqual(measured["error_covariance"], -4.)
        self.assertEqual(measured["mse"]["hindsight_mixture"], 0.)
        self.assertEqual(measured["hindsight_mse_improvement_over_best_endpoint"], 4.)

    def test_endpoint_optima_and_identical_forecasts(self):
        for ema, trend, weight in (([1., 2.], [2., 4.], 0.),
                                   ([2., 4.], [1., 2.], 1.),
                                   ([1., 2.], [1., 2.], 0.)):
            measured = paired_moments(rows(ema, trend))
            self.assertEqual(measured["hindsight_trend_weight"], weight)
            self.assertEqual(measured["hindsight_mse_improvement_over_best_endpoint"], 0.)
            json.dumps(measured, allow_nan=False)

    def test_bias_not_only_centered_covariance_and_swap_symmetry(self):
        measured = paired_moments(rows([2., 2.], [-1., -1.]))
        self.assertEqual(measured["error_covariance"], 0.)
        self.assertAlmostEqual(measured["hindsight_trend_weight"], 2 / 3)
        self.assertAlmostEqual(measured["mse"]["hindsight_mixture"], 0.)
        swapped = paired_moments(rows([-1., -1.], [2., 2.]))
        self.assertAlmostEqual(swapped["hindsight_trend_weight"], 1 - measured["hindsight_trend_weight"])

    def test_loss_budget_retains_helpful_cases_and_uses_gross_share(self):
        measured = loss_budget([budget_row(1, 1., 3.), budget_row(11, 3., 1.),
                                budget_row(21, 2., 3.), budget_row(31, 2., 2.)])
        self.assertEqual(measured["net_excess_squared_error"], 5.)
        self.assertEqual(measured["gross_positive_excess_squared_error"], 13.)
        self.assertEqual(measured["helpful_squared_error_delta"], -8.)
        self.assertEqual(measured["harmful_episodes"], 2)
        self.assertEqual(measured["helpful_episodes"], 1)
        self.assertEqual(measured["top_positive_excess_fraction"], 1.)
        self.assertEqual(measured["top_episodes"][0]["start_step"], 1)
        self.assertEqual(measured["by_detected_mixed"]["False"]["net_excess_squared_error"], 5.)
        json.dumps(measured, allow_nan=False)

    def test_no_harmful_cases_and_reset_crossing_not_filtered(self):
        record = budget_row(1, 2., 1.)
        record.update(detected_mixed=True, true_mixed=True)
        measured = loss_budget([record])
        self.assertIsNone(measured["top_positive_excess_fraction"])
        self.assertEqual(measured["top_episodes"], [])
        self.assertEqual(measured["by_detected_mixed"]["True"]["net_excess_squared_error"], -3.)
        self.assertEqual(measured["count"], 1)


if __name__ == "__main__":
    unittest.main()
