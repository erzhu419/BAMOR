import copy
import unittest

import numpy as np

import diagnose_headwise_scales as diagnostic


def window(index, speed, cost=.1, eligible=True, alarm=False):
    heads = {name: dict(eligible=eligible, supported_steps=100 if eligible else 0,
        domain="state_action", statistic=value if eligible else None,
        alarm=bool(eligible and alarm and name == "reward_0"))
        for name, value in (("reward_0", speed), ("reward_1", cost), ("dynamics", .1))}
    return dict(start_step=index * 100 + 1, end_step=(index + 1) * 100,
        eligible=eligible, supported_steps=100 if eligible else 0,
        statistic=max(speed, cost, .1) if eligible else None, heads=heads, alarm=bool(eligible and alarm))


class HeadwiseScaleTests(unittest.TestCase):
    def test_effective_scale_is_residual_weighted_harmonic_mean(self):
        residuals = np.array([1., 9.])
        scales = np.array([1., 3.])
        mean_scale = 2.
        left = window(0, float(np.mean(residuals / mean_scale)))
        right = window(0, float(np.mean(residuals / scales)))
        row = diagnostic.pair_window(left, right, mean_scale, (4., 2.))
        self.assertEqual(row["raw_loss_mean"], 5.)
        self.assertEqual(row["effective_conditional_scale"], 2.5)
        self.assertNotEqual(row["effective_conditional_scale"], float(scales.mean()))
        self.assertAlmostEqual(row["scale_amplification"], .8)
        self.assertAlmostEqual(row["conditional_ratio"], row["global_ratio"] * .8 * 2.)

    def test_zero_and_unsupported_scales_remain_unidentified(self):
        row = diagnostic.pair_window(window(0, 0), window(0, 0), 2., (4., 2.))
        self.assertEqual(row["raw_loss_mean"], 0.)
        self.assertIsNone(row["effective_conditional_scale"])
        missing = diagnostic.pair_window(window(0, 0, eligible=False), window(0, 0, eligible=False), 2., (4., 2.))
        self.assertFalse(missing["eligible"])
        self.assertIsNone(missing["raw_loss_mean"])
        self.assertIsNone(missing["conditional_ratio"])

    def test_pairing_rejects_changed_support_or_window_coordinates(self):
        left = window(0, 2.)
        for key, value in (("end_step", 200), ("supported_steps", 99)):
            right = copy.deepcopy(left)
            right[key] = value
            with self.assertRaises(AssertionError):
                diagnostic.pair_window(left, right, 2., (4., 2.))
        right = copy.deepcopy(left)
        right["heads"]["reward_0"]["supported_steps"] = 99
        with self.assertRaises(AssertionError):
            diagnostic.pair_window(left, right, 2., (4., 2.))

    def test_quantile_operands_preserve_joint_winning_head_and_ties(self):
        windows = [window(0, 1.), window(1, 2., cost=4.), window(2, 3.)]
        threshold = float(np.quantile([row["statistic"] for row in windows], .75))
        paired = diagnostic.pair_windows(windows, windows, 2., (threshold, threshold))
        result = diagnostic.tail(dict(calibration=dict(windows=windows, error_threshold=threshold)), paired, .75)
        self.assertEqual([row["weight"] for row in result["quantile_operands"]], [.5, .5])
        self.assertEqual(result["quantile_operands"][1]["windows"][0]["joint_winners"], ["reward_1"])
        self.assertEqual(sum(row["weight"] * row["joint_score"] for row in result["quantile_operands"]), threshold)

    def test_normal_groups_count_prefix_once_and_keep_suppressed_alarms(self):
        calibration = [window(index, 1. if index < 2 else 2.) for index in range(3)]
        threshold = float(np.quantile([row["statistic"] for row in calibration], .99))
        arms = {}
        for arm in diagnostic.ARMS:
            cases = {}
            for case in diagnostic.CASES:
                windows = [window(index, 2., alarm=index == 0) for index in range(10)]
                if arm == "global_scale":
                    windows[5]["alarm"] = True
                    windows[5]["heads"]["reward_0"]["alarm"] = True
                cases[case] = dict(windows=windows, available=True, any_post_alarm=arm == "global_scale",
                                   first_post_alarm_delay=100 if arm == "global_scale" else None, timely_alarm=arm == "global_scale")
            arms[arm] = dict(calibration=dict(windows=calibration, error_threshold=threshold, support_radii={}), cases=cases)
        row = dict(protocol=diagnostic.INPUT_PROTOCOL, env=diagnostic.ENV, complete=True,
            roles=dict(test_seed=2101), scale_work=dict(global_scales=[2., 2., 2.]), config=dict(quantile=.99), arms=arms)
        before = copy.deepcopy(row)
        result = diagnostic.fold(row)
        groups = result["normal_groups"]
        self.assertEqual(sum(len(group["windows"]) for group in groups.values()), 20)
        self.assertEqual(len(groups["shared_alarm"]["windows"]), 1)
        self.assertEqual(len(groups["global_only_alarm"]["windows"]), 3)
        self.assertEqual(row, before)
        self.assertEqual(result["cases"]["mode1"]["outcomes"]["global_scale"]["first_post_alarm_delay"], 100)


if __name__ == "__main__":
    unittest.main()
