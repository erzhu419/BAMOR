import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

import diagnose_offset_reset_events as diagnostic


def windows(alarms):
    return [dict(start_step=end - 99, end_step=end, eligible=True, supported_steps=100,
        alarm=end in alarms, heads={})
        for end in range(100, 1001, 100)]


def selector_trace(events):
    selector = diagnostic.confirmation.acquisition.training.EpisodeReturnSamplingSelector(2)
    rng_before = np.random.get_state()
    operators = []
    try:
        np.random.seed(4819)
        for relative in range(1, 1001):
            step = 5000 + relative
            operator = selector.select_for_step(step)
            operators.append(operator)
            with patch.object(selector.belief, "update", return_value=dict(mode_switched=relative in events)):
                selector.observe_transition(operator, [1. + relative / 1000, -.25], step, relative % 100 == 0)
    finally:
        np.random.set_state(rng_before)
    return dict(operators=operators, state=selector.state_dict())


class OffsetResetEventTests(unittest.TestCase):
    def test_physical_clock_causality_and_persistent_alarm_suppression(self):
        first, persistent = windows({600}), windows({600, 700, 800, 900, 1000})
        self.assertEqual(diagnostic.reset_events(first), [600])
        self.assertEqual(diagnostic.reset_events(persistent), [600])
        self.assertEqual(diagnostic.reset_events(windows({700})), [700])
        self.assertEqual(diagnostic.reset_events(windows(set())), [])
        for end in range(100, 1001, 100):
            self.assertEqual(diagnostic.reset_events(persistent[:end // 100]),
                [step for step in diagnostic.reset_events(persistent) if step <= end])

    def test_equal_events_preserve_actual_selector_choices_and_credit_state(self):
        a = selector_trace(diagnostic.reset_events(windows({600})))
        b = selector_trace(diagnostic.reset_events(windows({600, 700, 800, 900, 1000})))
        self.assertEqual(a, b)
        self.assertEqual(a["state"]["reset_history"], [dict(step=5600, epoch=1)])
        mixed = [e for e in a["state"]["episode_history"] if e["mixed_epoch"]]
        self.assertEqual(len(mixed), 1)
        self.assertFalse(mixed[0]["credited"])
        self.assertGreater(len(set(a["operators"])), 1)
        self.assertNotEqual(selector_trace([700]), a)

    def test_stored_only_report_detects_real_event_differences(self):
        confirmation = diagnostic.confirmation
        env, seed = confirmation.PARENT.ENVIRONMENTS[0], 2101
        with tempfile.TemporaryDirectory() as tmp:
            parent = Path(tmp)
            cases = {case: {arm: dict(windows=windows({600}), first_post_alarm_delay=100)
                for arm in (diagnostic.CONTROL, diagnostic.PRIMARY)} for case in confirmation.PARENT.CASES}
            cases["mode2"][diagnostic.PRIMARY]["windows"] = windows({600, 700, 800, 900, 1000})
            prior = dict(protocol=confirmation.PROTOCOL, complete=True, env=env,
                roles=confirmation.PARENT.fold_roles(seed), calibration_steps=5000,
                arms={arm: dict(cases={case: row[arm] for case, row in cases.items()})
                    for arm in (diagnostic.CONTROL, diagnostic.PRIMARY)})
            path = parent / env.replace("-", "_") / f"test_seed{seed}" / "readout.json"
            confirmation.PARENT.write(path, prior)
            with patch.object(confirmation.PARENT, "ENVIRONMENTS", (env,)), \
                 patch.object(confirmation.PARENT, "SEEDS", (seed,)), \
                 patch.object(confirmation.acquisition, "data", side_effect=AssertionError("Trajectory read")), \
                 patch.object(confirmation.acquisition, "run_group", side_effect=AssertionError("Acquisition")):
                result = diagnostic.summarize(parent)
                self.assertEqual(result["case_pairs"], 6)
                self.assertEqual(result["changed_alarm_window_count"], 4)
                self.assertEqual(result["changed_reset_cases"], [])
                self.assertEqual(result["decision"], "close_offset_direct_reset_route")
                self.assertTrue(all(result[key] == 0 for key in diagnostic.ZERO_WORK))
                prior["arms"][diagnostic.PRIMARY]["cases"]["mode2"]["windows"] = windows({700})
                prior["arms"][diagnostic.PRIMARY]["cases"]["mode2"]["first_post_alarm_delay"] = 200
                confirmation.PARENT.write(path, prior)
                changed = diagnostic.summarize(parent)
                self.assertEqual(changed["decision"], "inspect_offline_reset_differences")
                self.assertEqual(changed["changed_reset_cases"][0]["case"], "mode2")
                self.assertEqual(json.loads(path.read_text()), prior)


if __name__ == "__main__":
    unittest.main()
