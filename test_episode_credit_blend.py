import copy
import json
import unittest

import diagnose_episode_credit_selector as selector
from diagnose_episode_credit_blend import controls, metrics, rescore, validate_reference
from episode_credit_blend import EpisodeCreditBlend
from test_episode_credit_trend import template


class EpisodeCreditBlendTests(unittest.TestCase):
    def test_zero_evidence_and_no_forecast_fallback(self):
        blend = EpisodeCreditBlend(.13)
        before = blend.predict(1)
        self.assertEqual(before["trend_weight"], 0.)
        self.assertIsNone(before["blend_prediction"])
        blend.observe(1, 2., before, False)
        before = blend.predict(1001)
        self.assertEqual(before["trend_weight"], 0.)
        self.assertEqual(before["blend_prediction"], before["ema_prediction"])
        self.assertEqual(before["past_comparisons"], 0)

    def test_sufficient_statistics_weight_direction_and_bias(self):
        blend = EpisodeCreditBlend(.13)
        before = dict(ema_prediction=0., trend_prediction=2.)
        blend.observe(1, 1., before, False)
        chosen = blend.predict(1001)
        self.assertEqual(chosen["trend_weight"], .5)
        self.assertEqual(chosen["past_cross_product"], 2.)
        self.assertEqual(chosen["past_disagreement_energy"], 4.)
        self.assertEqual(chosen["past_comparisons"], 1)
        blend.observe(1001, 3., before, False)
        self.assertEqual(blend.predict(2001)["trend_weight"], 1.)
        self.assertEqual(chosen["past_cross_product"], 2.)

    def test_endpoint_clipping_and_identical_forecasts(self):
        for utility, expected in ((-1., 0.), (3., 1.)):
            blend = EpisodeCreditBlend(.13)
            blend.observe(1, utility, dict(ema_prediction=0., trend_prediction=2.), False)
            self.assertEqual(blend.predict(1001)["trend_weight"], expected)
        blend = EpisodeCreditBlend(.13)
        blend.observe(1, 1., dict(ema_prediction=0., trend_prediction=0.), False)
        before = blend.predict(1001)
        self.assertEqual(before["trend_weight"], 0.)
        self.assertEqual(before["past_disagreement_energy"], 0.)

    def test_weight_matches_direct_past_least_squares(self):
        blend = EpisodeCreditBlend(.13)
        errors = [(3., -1.), (-2., 1.), (.5, 1.5)]
        for index, (ema, trend) in enumerate(errors):
            blend.observe(index * 1000 + 1, 0.,
                dict(ema_prediction=-ema, trend_prediction=-trend), False)
        numerator = sum(ema * (ema - trend) for ema, trend in errors)
        denominator = sum((ema - trend) ** 2 for ema, trend in errors)
        self.assertEqual(blend.predict(3001)["trend_weight"], numerator / denominator)

    def test_reset_clears_both_forecasters_and_weight_evidence(self):
        blend = EpisodeCreditBlend(.13)
        for index in range(5):
            before = blend.predict(index * 1000 + 1)
            blend.observe(index * 1000 + 1, float(index), before, False)
        self.assertGreater(blend.comparisons, 0)
        blend.observe(5001, 1e6, blend.predict(5001), True)
        before = blend.predict(6001)
        self.assertIsNone(before["ema_prediction"])
        self.assertIsNone(before["trend_prediction"])
        for name in ("trend_weight", "past_cross_product", "past_disagreement_energy", "past_comparisons"):
            self.assertEqual(before[name], 0.)

    def test_reference_forecasts_and_cohort_unchanged_and_drift_rejected(self):
        episodes = selector.trend.constructed_control(template(40), 2101, 0, selector.trend.KNOWN_EFFECT, 1.)
        old, _ = selector.rescore(episodes, .13)
        new, _ = rescore(episodes, .13)
        validate_reference(new, old, selector.metrics(old))
        self.assertEqual(selector.trend.metrics(old), selector.trend.metrics(new))
        changed = copy.deepcopy(new)
        changed[0]["trend_prediction"] += 1.
        with self.assertRaisesRegex(ValueError, "reference forecasts"):
            validate_reference(changed, old, selector.metrics(old))
        with self.assertRaisesRegex(ValueError, "evaluation cohort"):
            validate_reference(new[:-1], old, selector.metrics(old))

    def test_prefix_current_and_future_outcomes_do_not_change_current_prediction(self):
        episodes = selector.trend.constructed_control(template(20), 2101, 0, selector.trend.KNOWN_EFFECT, 1.)
        full, _ = rescore(episodes, .13)
        prefix, _ = rescore(episodes[:10], .13)
        self.assertEqual(full[:5], prefix)
        changed = copy.deepcopy(episodes)
        changed[10]["utility"] = 1e6
        changed[-1]["utility"] = -1e6
        future, _ = rescore(changed, .13)
        self.assertEqual(full[:5], future[:5])
        for key in ("trend_weight", "blend_prediction", "past_cross_product",
                    "past_disagreement_energy", "past_comparisons"):
            self.assertEqual(full[5][key], future[5][key])

    def test_audit_labels_and_current_length_are_not_weight_inputs(self):
        episodes = selector.trend.constructed_control(template(20), 2101, 0, selector.trend.KNOWN_EFFECT, 1.)
        full, _ = rescore(episodes, .13)
        changed = [dict(ep, operator=selector.trend.OPERATORS[index % 3], env="other", true_modes=[99],
                        length=999999) for index, ep in enumerate(episodes)]
        relabeled, _ = rescore(changed, .13)
        for before, after in zip(full, relabeled):
            for key in ("trend_weight", "blend_prediction", "past_cross_product", "blend_error"):
                self.assertEqual(before[key], after[key])

    def test_reset_crossing_outcome_retained_in_prediction_comparison(self):
        episodes = selector.trend.constructed_control(template(10), 2101, 0, selector.trend.KNOWN_EFFECT, 1.)
        episodes[6]["detected_mixed"] = True
        episodes[6]["utility"] = 1e6
        rows, counts = rescore(episodes, .13)
        self.assertEqual(counts["retained_reset_crossing_forecast_episodes"], 1)
        self.assertEqual(counts["unavailable_forecast_episodes"], 3)
        self.assertEqual(metrics(rows)["count"], 2)

    def test_convex_prediction_and_fixed_equal_comparator(self):
        episodes = selector.trend.constructed_control(template(30), 2101, 0, selector.trend.KNOWN_EFFECT, 1.)
        rows, _ = rescore(episodes, .13)
        for row in rows:
            a, b = row["ema_prediction"], row["trend_prediction"]
            self.assertGreaterEqual(row["blend_prediction"], min(a, b) - 1e-12)
            self.assertLessEqual(row["blend_prediction"], max(a, b) + 1e-12)
            self.assertEqual(row["equal_mixture_prediction"], (a + b) / 2.)

    def test_constructed_controls_and_readout_are_native_json(self):
        readout = controls([(2101, template(80), .13)])
        self.assertTrue(readout["deterministic_growth_null_passed"])
        self.assertTrue(readout["known_effect_order_recovered"])
        self.assertEqual(json.loads(json.dumps(readout, allow_nan=False)), readout)


if __name__ == "__main__":
    unittest.main()
