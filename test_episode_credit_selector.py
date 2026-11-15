import copy
import json
import unittest

import diagnose_episode_credit_trend as trend
from diagnose_episode_credit_selector import controls, metrics, rescore
from episode_credit_selector import EpisodeCreditSelector
from test_episode_credit_trend import template


class EpisodeCreditSelectorTests(unittest.TestCase):
    def test_past_error_choice_initial_tie_and_causal_fallback(self):
        selector = EpisodeCreditSelector(.13)
        for index, utility in enumerate((0., 10., 20.)):
            before = selector.predict(index * 1000 + 1)
            self.assertEqual(before["selected"], "ema")
            selector.observe(index * 1000 + 1, utility, before, False)
        before = selector.predict(3001)
        self.assertEqual(before["past_comparisons"], 0)
        self.assertEqual(before["selected"], "ema")
        selector.observe(3001, 30., before, False)
        chosen = selector.predict(4001)
        self.assertEqual(chosen["selected"], "trend")
        self.assertEqual(chosen["past_comparisons"], 1)
        previous_losses = copy.deepcopy(chosen["past_squared_errors"])
        selector.observe(4001, 0., chosen, False)
        self.assertEqual(chosen["selected"], "trend")
        self.assertEqual(chosen["past_squared_errors"], previous_losses)
        self.assertEqual(selector.predict(5001)["selected"], "ema")
        selector.squared_errors = dict(ema=1., trend=1.)
        self.assertEqual(selector.predict(5001)["selected"], "ema")

    def test_reset_clears_forecasters_and_comparison_evidence(self):
        selector = EpisodeCreditSelector(.13)
        for index in range(5):
            before = selector.predict(index * 1000 + 1)
            selector.observe(index * 1000 + 1, float(index), before, False)
        self.assertGreater(selector.comparisons, 0)
        selector.observe(5001, 1e6, selector.predict(5001), True)
        before = selector.predict(6001)
        self.assertEqual(before["selected"], "ema")
        self.assertIsNone(before["trend_prediction"])
        self.assertEqual(before["past_squared_errors"], dict(ema=0., trend=0.))
        self.assertEqual(before["past_comparisons"], 0)

    def test_reference_forecasts_and_cohort_unchanged(self):
        episodes = trend.constructed_control(template(40), 2101, 0, trend.KNOWN_EFFECT, 1.)
        old, _ = trend.rescore(episodes, .13)
        new, counts = rescore(episodes, .13)
        self.assertEqual(trend.metrics(old), trend.metrics(new))
        self.assertEqual(counts["paired_forecast_episodes"], len(old))
        for old_row, new_row in zip(old, new):
            for key in old_row:
                self.assertEqual(old_row[key], new_row[key])

    def test_prefix_and_current_or_future_outcome_do_not_select_current_model(self):
        episodes = trend.constructed_control(template(20), 2101, 0, trend.KNOWN_EFFECT, 1.)
        full, _ = rescore(episodes, .13)
        prefix, _ = rescore(episodes[:10], .13)
        self.assertEqual(full[:5], prefix)
        changed = copy.deepcopy(episodes)
        changed[10]["utility"] = 1e6
        future, _ = rescore(changed, .13)
        self.assertEqual(full[:5], future[:5])
        for key in ("selected", "selected_prediction", "past_squared_errors", "past_comparisons"):
            self.assertEqual(full[5][key], future[5][key])

    def test_labels_and_current_length_are_not_selector_inputs(self):
        episodes = trend.constructed_control(template(20), 2101, 0, trend.KNOWN_EFFECT, 1.)
        full, _ = rescore(episodes, .13)
        changed = [dict(ep, operator=trend.OPERATORS[index % 3], env="other", true_modes=[99],
                        length=999999) for index, ep in enumerate(episodes)]
        relabeled, _ = rescore(changed, .13)
        for before, after in zip(full, relabeled):
            for key in ("selected", "selected_prediction", "past_squared_errors", "selected_error"):
                self.assertEqual(before[key], after[key])

    def test_reset_crossing_outcome_retained_in_prediction_comparison(self):
        episodes = trend.constructed_control(template(10), 2101, 0, trend.KNOWN_EFFECT, 1.)
        episodes[6]["detected_mixed"] = True
        episodes[6]["utility"] = 1e6
        rows, counts = rescore(episodes, .13)
        self.assertEqual(counts["retained_reset_crossing_forecast_episodes"], 1)
        self.assertEqual(counts["unavailable_forecast_episodes"], 3)
        self.assertEqual(metrics(rows)["count"], 2)

    def test_constructed_controls_and_readout_are_native_json(self):
        readout = controls([(2101, template(80), .13)])
        self.assertTrue(readout["deterministic_growth_null_passed"])
        self.assertTrue(readout["known_effect_order_recovered"])
        self.assertEqual(json.loads(json.dumps(readout, allow_nan=False)), readout)


if __name__ == "__main__":
    unittest.main()
