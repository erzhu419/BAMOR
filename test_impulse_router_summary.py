import copy
import unittest

import numpy as np

import summarize_budgeted_router as common
from summarize_impulse_router import REPRODUCTION_FIELDS, reproduction_check
from test_budgeted_router_summary import result


def impulse_result(condition="route"):
    data = result(condition)
    fields = {"protocol": common.IMPULSE_PROTOCOL, "selection_scope": "fork_updates_only",
              "normal_training_gate": "default_portfolio"}
    data["budgeted_routing_experiment"].update(fields)
    data["update_routing"].update(fields)
    del data["update_routing"]["enabled"]
    for case in data["update_routing"]["cases"]:
        case["previous_on"] = True
    return data


class ImpulseSummaryTests(unittest.TestCase):
    def test_all_impulse_conditions_expire_without_changing_budget(self):
        for condition in common.pilot.CONDITIONS[:3]:
            with self.subTest(condition=condition):
                data = common.summarize_run(impulse_result(condition), condition, 26, common.IMPULSE_PROTOCOL)
                self.assertEqual(data["budget"]["total_training_interactions"], 86864)
                self.assertTrue(all(c["held_until_main_step"] == c["main_step"] for c in data["decisions"]))
                self.assertTrue(all(not c["mode_switches_while_held"] for c in data["decisions"]))

    def test_off_choice_is_followed_by_fresh_default_not_a_persistent_gate(self):
        data = impulse_result()
        case = data["update_routing"]["cases"][0]
        case.update(on_candidates=5, chosen="off", preference_advantages=[-1.] * 3,
                    mean_utility_advantage=-1.)
        case["score_returns"]["off"] = np.ones((3, 3, 3, 2)).tolist()
        common.summarize_run(data, "route", 26, common.IMPULSE_PROTOCOL)
        data["update_routing"]["cases"][1]["previous_on"] = False
        with self.assertRaises(AssertionError):
            common.summarize_run(data, "route", 26, common.IMPULSE_PROTOCOL)

    def test_protocols_cannot_be_silently_mixed(self):
        with self.assertRaisesRegex(AssertionError, "Mismatched budget: protocol"):
            common.summarize_run(impulse_result(), "route", 26)
        with self.assertRaisesRegex(AssertionError, "Mismatched budget: protocol"):
            common.summarize_run(result("route"), "route", 26, common.IMPULSE_PROTOCOL)

    def test_reproduction_requires_both_evaluated_front_and_training_choices(self):
        a = {k: [] for k in REPRODUCTION_FIELDS}
        a["update_routing"] = {"cases": [{"chosen": "on"}]}
        b = copy.deepcopy(a)
        self.assertTrue(all(reproduction_check(a, b).values()))
        b["update_routing"]["cases"][0]["chosen"] = "off"
        with self.assertRaisesRegex(AssertionError, "Fixed-on reproduction differs"):
            reproduction_check(a, b)
        b = copy.deepcopy(a)
        b["pareto_per_mode"] = [1]
        with self.assertRaises(AssertionError):
            reproduction_check(a, b)


if __name__ == "__main__":
    unittest.main()
