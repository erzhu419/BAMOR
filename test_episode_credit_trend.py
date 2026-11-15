import copy
import json
import unittest

import numpy as np

from diagnose_episode_credit_trend import OPERATORS, constructed_control, controls, diagnose, metrics, rescore
from episode_credit_trend import EpisodeCreditTrend, TREND_WINDOW
from test_episode_feedback_diagnostic import fixture


def template(count=600):
    rng = np.random.default_rng(0)
    return [dict(start_step=index * 1000 + 1, utility=0.,
        phase="warmup" if index < 5 else "free", operator=str(rng.choice(OPERATORS)),
        detected_mixed=False) for index in range(count)]


class EpisodeCreditTrendTests(unittest.TestCase):
    def test_irregular_clock_linear_forecast_and_fixed_window(self):
        baseline = EpisodeCreditTrend(.13)
        for step in (1, 200, 1050):
            baseline.observe(step, 3 + step / 1000)
        self.assertAlmostEqual(baseline.predict(2050)["trend_prediction"], 5.05)
        for step in range(3000, 15000, 1000):
            baseline.observe(step, 3 + step / 1000)
        self.assertEqual(baseline.predict(16000)["fit_samples"], TREND_WINDOW)
        self.assertAlmostEqual(baseline.predict(16000)["trend_prediction"], 19.)
        baseline.reset()
        self.assertIsNone(baseline.predict(17000)["trend_prediction"])

    def test_growth_null_not_operator_reward_and_known_effect_retained(self):
        episodes = template()
        null = dict.fromkeys(OPERATORS, 0.)
        rows, _ = rescore(constructed_control(episodes, 2101, 0, null, 0.), .13)
        stats = metrics(rows)
        self.assertLess(stats["trend"]["rmse"], 1e-9)
        self.assertGreater(stats["ema"]["mean_score"], .1)
        effect = dict(zip(OPERATORS, (-1., 0., 1.)))
        rows, _ = rescore(constructed_control(episodes, 2101, 0, effect, 1.), .13)
        contrasts = metrics(rows)["trend"]["centered_raw_contrasts"]
        self.assertLess(contrasts["uniform"], contrasts["stratified"])
        self.assertLess(contrasts["stratified"], contrasts["mixed"])

    def test_recorded_baseline_round_trip_and_adaptive_route_rejected(self):
        data = fixture()
        with self.assertRaisesRegex(ValueError, "fixed three-arm random route"):
            diagnose(data)
        data["adaptive_operator_final"]["selection_uses_scores"] = False
        original = copy.deepcopy(data)
        _, rows, counts = diagnose(data)
        self.assertEqual(data, original)
        self.assertEqual(counts["paired_forecast_episodes"], 1)
        self.assertAlmostEqual(rows[0]["trend_error"], 0.)

    def test_constructed_control_readout_is_native_json(self):
        readout = controls([(2101, template(40), .13)])
        self.assertTrue(readout["deterministic_growth_null_passed"])
        self.assertEqual(json.loads(json.dumps(readout, allow_nan=False)), readout)

    def test_reset_crossing_evaluated_then_excluded_from_future_fit(self):
        episodes = template(10)
        episodes = constructed_control(episodes, 2101, 0, dict.fromkeys(OPERATORS, 0.), 0.)
        episodes[6]["detected_mixed"] = True
        episodes[6]["utility"] = 1e6
        rows, counts = rescore(episodes, .13)
        self.assertEqual(counts["retained_reset_crossing_forecast_episodes"], 1)
        self.assertEqual(counts["unavailable_forecast_episodes"], 3)
        self.assertEqual(len(rows), 2)
        baseline = EpisodeCreditTrend(.13)
        baseline.observe(7001, episodes[7]["utility"])
        baseline.observe(8001, episodes[8]["utility"])
        baseline.observe(9001, episodes[9]["utility"])
        self.assertAlmostEqual(baseline.predict(10001)["trend_prediction"], 13.001)

    def test_prefix_and_audit_labels_do_not_enter_forecast(self):
        episodes = constructed_control(template(20), 2101, 0, dict.fromkeys(OPERATORS, 0.), 1.)
        original = copy.deepcopy(episodes)
        full, _ = rescore(episodes, .13)
        prefix, _ = rescore(episodes[:10], .13)
        self.assertEqual(full[:5], prefix)
        relabeled = [dict(ep, operator=OPERATORS[index % 3], true_mixed=True,
                          true_modes=[99], length=999999) for index, ep in enumerate(episodes)]
        changed, _ = rescore(relabeled, .13)
        for before, after in zip(full, changed):
            for key in ("ema_prediction", "trend_prediction", "scale", "ema_error", "trend_error"):
                self.assertEqual(before[key], after[key])
        episodes[-1]["utility"] = 1e6
        future, _ = rescore(episodes, .13)
        self.assertEqual(full[:-1], future[:-1])
        self.assertEqual(original[-1]["utility"], relabeled[-1]["utility"])


if __name__ == "__main__":
    unittest.main()
