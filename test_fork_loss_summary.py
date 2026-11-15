import copy
import unittest

from summarize_fork_loss_components import COMPONENTS, SIGNALS, group_summary, validate_case


class LossSummaryTests(unittest.TestCase):
    def test_signs_ties_and_gain_are_distinct(self):
        rows = []
        for signal, gain in ((1., 2.), (1., -10.), (0., 1.)):
            rows.append({"signals": {k: signal for k in SIGNALS},
                         "on_minus_off": [gain] * 3, "on_minus_base": [gain] * 3,
                         "off_minus_base": [0.] * 3,
                         "paired_on_minus_off": [[gain] * 3] * 3,
                         "gradient_alignment_advantage": None, "own_min_other_cosine": None})
        result = group_summary(rows)
        m = result["decisions"]["total"]["512"]
        self.assertEqual((m["choose_on"], m["ties"], m["sign_correct"]), (2, 1, 1))
        self.assertAlmostEqual(m["selected_minus_off"], -8. / 3.)
        self.assertAlmostEqual(m["selected_minus_on"], -1. / 3.)
        self.assertEqual(result["sign_flips_if_removed"]["entropy_loss"], 2)

    @staticmethod
    def cases():
        measured = {k: [0.] * 3 for k in SIGNALS}
        measured["deterministic_reset_q"] = [[0.] * 3] * 3
        values = {"base": copy.deepcopy(measured), "on": copy.deepcopy(measured),
                  "off": copy.deepcopy(measured)}
        saved = {k: [0.] * 3 for k in SIGNALS}
        saved["deterministic_reset_q_paired"] = [[0.] * 3] * 3
        effects = {"on_minus_off": {"mean_scalar_by_weight_horizon": [[0.] * 3] * 3}}
        original = {"config": {"envelope_policy_candidates": 1}, "step": 10000,
                    "operator": "identity", "effects": effects,
                    "own_soft_loss_with_bc_frozen_critic": {k: [0.] * 3 for k in values},
                    "surrogate_on_advantage_by_weight": [0.] * 3}
        case = {"config": original["config"], "step": 10000, "operator": "identity",
                "matched_original_losses": True, "measurements": values,
                "on_advantages": saved, "cached_return_effects": copy.deepcopy(effects),
                "extra_environment_steps": 0, "diagnostic_resets": 3}
        return case, original

    def test_identity_and_cached_alignment(self):
        case, original = self.cases()
        validate_case(case, original)
        case["cached_return_effects"]["on_minus_off"]["mean_scalar_by_weight_horizon"][0][0] = 1.
        with self.assertRaises(AssertionError):
            validate_case(case, original)

    def test_component_arithmetic_and_saved_direction(self):
        for key in COMPONENTS:
            case, original = self.cases()
            case["measurements"]["on"][key][0] = 1.
            with self.assertRaises(AssertionError):
                validate_case(case, original)
        case, original = self.cases()
        case["on_advantages"]["deterministic_reset_q"][0] = 1.
        with self.assertRaises(AssertionError):
            validate_case(case, original)


if __name__ == "__main__":
    unittest.main()
