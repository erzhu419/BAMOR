import copy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

import diagnose_episode_blocked_calibration as diagnostic


def windows(scores, eligible=None):
    if eligible is None:
        eligible = [True] * len(scores)
    return [dict(start_step=100 * index + 1, end_step=100 * (index + 1),
        supported_steps=100 if supported else 0, eligible=supported,
        statistic=float(score) if supported else None, alarm=False)
        for index, (score, supported) in enumerate(zip(scores, eligible))]


def arm_fixture(expanded):
    scores = [.1] * (50 if expanded else 40)
    scores[0], scores[30] = 2., 3.
    if expanded:
        scores[-1] = 4.
    calibration = windows(scores)
    threshold = float(np.quantile(scores, .99))
    cases = {}
    for case in diagnostic.CASES:
        case_scores = [threshold + 1.] + [.1] * 9
        eligible = [True] * 10
        if case == "mode1":
            case_scores[5:] = [threshold + 1.] * 5
        elif case == "mode2":
            case_scores[5] = threshold + 1.
            eligible[7:] = [False] * 3
        elif case == "mode3":
            case_scores[8] = threshold + 1.
        case_windows = windows(case_scores, eligible)
        for window in case_windows:
            window["alarm"] = bool(window["eligible"] and window["statistic"] > threshold)
        cases[case] = dict(windows=case_windows, available=case != "mode2",
            any_post_alarm=case.startswith("mode"), first_post_alarm_delay=(
                400 if case == "mode3" else 100 if case.startswith("mode") else None),
            timely_alarm=case == "mode1")
    return dict(calibration_steps=len(scores) * 100,
        calibration=dict(windows=calibration, error_threshold=threshold), cases=cases)


class BlockedCalibrationTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.source = Path(self.tmp.name) / "source"
        self.env, self.seed = diagnostic.ENVIRONMENTS[0], 2102
        folder = self.source / self.env.replace("-", "_") / "seed2102"
        diagnostic.write(folder / "readout.json", dict(probe_seed=2202))
        terminals = {}
        for name, _, _, _ in diagnostic.SEGMENTS:
            data = np.zeros((3000 if name == "prefix" else 1000, 2), bool)
            data[999::1000, 1] = True
            if name == "actor_change":
                data[299, 0] = True
            terminals[f"{name}_terminals"] = data
        np.savez_compressed(folder / "transitions.npz", **terminals)
        self.context = diagnostic.terminal_context(self.source, self.env, self.seed)

    def test_exact_quantile_interpolation_and_source_segments(self):
        arm = arm_fixture(True)
        result = diagnostic.calibration_tail(arm, .99, self.context)
        operands = result["quantile_operands"]
        np.testing.assert_allclose([item["weight"] for item in operands], [.49, .51])
        np.testing.assert_allclose(sum(item["weight"] * item["value"] for item in operands), result["threshold"])
        self.assertEqual([item["windows"][0]["segment"] for item in operands], ["stationary", "actor_change"])
        self.assertEqual(operands[1]["windows"][0]["source_end_step"], 1000)
        broken = copy.deepcopy(arm)
        broken["calibration"]["error_threshold"] += 1.
        with self.assertRaises(AssertionError):
            diagnostic.calibration_tail(broken, .99, self.context)

    def test_ineligible_calibration_window_is_not_a_quantile_operand(self):
        arm = arm_fixture(True)
        arm["calibration"]["windows"][-1].update(eligible=False, statistic=None, supported_steps=0)
        values = [window["statistic"] for window in arm["calibration"]["windows"] if window["eligible"]]
        arm["calibration"]["error_threshold"] = float(np.quantile(values, .99))
        result = diagnostic.calibration_tail(arm, .99, self.context)
        self.assertEqual(result["eligible_windows"], 49)
        self.assertTrue(all(window["calibration_end_step"] != 5000
                            for item in result["quantile_operands"] for window in item["windows"]))

    def test_suffix_coordinates_and_terminal_episode_provenance(self):
        window = dict(start_step=4501, end_step=4600, statistic=1., supported_steps=90)
        result = diagnostic.provenance(window, self.context)
        self.assertEqual((result["source_start_step"], result["source_end_step"]), (501, 600))
        self.assertEqual(result["episode_ids"], [1])
        self.assertEqual(result["reset_seeds"], [50000000 + 1000 * 2202 + 1])
        self.assertEqual(result["all_window_steps_early_fraction"], 0.)
        prefix = diagnostic.provenance(dict(start_step=1001, end_step=1100,
                                             statistic=1., supported_steps=80), self.context)
        self.assertEqual(prefix["episode_ids"], [1])
        self.assertEqual(prefix["all_window_steps_early_fraction"], 1.)

    def test_unavailable_timely_alarm_and_late_alarm_stay_distinct(self):
        arm = arm_fixture(True)
        threshold = arm["calibration"]["error_threshold"]
        for case, expected in (("mode1", "timely_alarm"), ("mode2", "coverage_unavailable"),
                               ("mode3", "late_alarm"), ("stationary", "no_alarm")):
            result = diagnostic.test_phase(arm["cases"][case]["windows"][5:], threshold)
            self.assertEqual(result["outcome"], expected)
            if case == "mode2":
                self.assertEqual(result["alarm_windows"], 1)
                self.assertIsNone(result["windows"][-1]["ratio"])
            if case == "mode3":
                self.assertLess(result["early_maximum_ratio"], 1.)
                self.assertGreater(result["maximum_ratio"], 1.)

    def test_monotone_quantile_obstruction_and_unpenalized_rescue(self):
        rows = []
        for seed in range(2101, 2106):
            missed = seed >= 2104
            rows.append(dict(roles=dict(test_seed=seed), arms=dict(expanded_normal=dict(
                calibration=dict(quantile=.99), cases=dict(
                    stationary=dict(alarm_windows=int(seed == 2102), maximum_ratio=.4 if missed else 2.),
                    mode1=dict(outcome="no_alarm" if missed else "timely_alarm", available=True,
                               early_maximum_ratio=.2 if missed else 2.))))))
        result = diagnostic.threshold_obstruction(rows, "expanded_normal")
        self.assertTrue(result["no_shared_score_quantile_can_pass"])
        self.assertEqual(result["existing_negative_sources"], [2102])
        self.assertEqual(result["lower_quantile_timely_upper_bound"], 3)
        self.assertEqual([item["seed"] for item in result["dominated_missed_sources"]], [2104, 2105])
        rows[3]["arms"]["expanded_normal"]["cases"]["stationary"]["maximum_ratio"] = .1
        result = diagnostic.threshold_obstruction(rows, "expanded_normal")
        self.assertFalse(result["no_shared_score_quantile_can_pass"])
        self.assertEqual(result["unpenalized_rescuable_sources"], [2104])
        self.assertEqual(result["lower_quantile_timely_upper_bound"], 4)

    def test_file_route_uses_only_stored_windows_and_terminal_arrays(self):
        root = Path(self.tmp.name) / "blocked"
        original = dict(protocol=diagnostic.INPUT_PROTOCOL, complete=True, env=self.env,
            roles=dict(test_seed=2101, calibration_seed=2102, fit_seeds=[2103, 2104, 2105]),
            predictor_config=dict(quantile=.99), arms={name: arm_fixture(name == "expanded_normal") for name in diagnostic.ARMS})
        path = root / self.env.replace("-", "_") / "test_seed2101" / "readout.json"
        diagnostic.write(path, original)
        before = path.read_text()
        with patch.object(diagnostic, "ENVIRONMENTS", (self.env,)), patch.object(diagnostic, "SEEDS", (2101,)):
            result = diagnostic.run(root, self.source)
        self.assertEqual(path.read_text(), before)
        self.assertEqual(result["predictor_optimizer_steps"], 0)
        self.assertEqual(result["additional_interactions"], 0)
        self.assertEqual(result["decision"], "hold_conditional_evidence_integration")
        self.assertEqual(result["environments"][self.env]["runs"][0]["arms"]["expanded_normal"]["shared_prefix"]["alarm_windows"], 1)
        json.dumps(result, allow_nan=False)


if __name__ == "__main__":
    unittest.main()
