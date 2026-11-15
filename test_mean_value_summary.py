import copy
import unittest

import summarize_mean_value_pilot as summary
from test_replay_refresh_summary import result as fixture


def variant():
    data = fixture()
    data.update(method=summary.pilot.METHOD, seed=29,
                episode_seeds_per_mode=[[129000 + 10*m + r for r in range(3)] for m in range(4)],
                training_interactions=50000, evaluation_interactions=4788000,
                environment_initialization_probe_steps=2)
    data["mean_value_pilot"] = {
        "protocol": summary.pilot.PROTOCOL, "complete": True, "seeds": list(summary.pilot.SEEDS),
        "mean_value_fraction": .5, "extra_training_updates": 0,
        "late_snapshot_steps": list(summary.pilot.controls.LATE_STEPS)}
    data["actor_update"] = {"rule": "half_selected_half_own_preference_mean_value",
        "mean_value_fraction": .5, "single_candidate_updates": 19001,
        "envelope_updates": 26000, "updates": 45001}
    return data


class MeanValueSummaryTests(unittest.TestCase):
    def test_explicit_method_identity_and_returns(self):
        data = variant()
        result = summary.variant_metrics(data, 29)
        self.assertEqual(result["late_min_hv"], data["final_hv_mean_mode"])
        self.assertNotIn("posthoc_late_mean_hv", result)
        with self.assertRaises(AssertionError):
            summary.common.metrics(data, 29)
        data["method"] = summary.pilot.BASE
        with self.assertRaises(AssertionError):
            summary.variant_metrics(data, 29)
        modes = summary.return_diagnostics(data)
        self.assertEqual(len(modes), 4)
        self.assertEqual(modes[0]["preference_grid_mean_return"], [2., -1.])
        self.assertAlmostEqual(modes[0]["own_preference_utility_mean"], .5)

    def test_incomplete_intervention_or_misreported_results_rejected(self):
        for mutation in ("complete", "fraction", "updates", "path", "budget", "grid", "hv", "refresh"):
            data = variant()
            if mutation == "complete":
                data["mean_value_pilot"]["complete"] = False
            elif mutation == "fraction":
                data["actor_update"]["mean_value_fraction"] = .1
            elif mutation == "updates":
                data["actor_update"]["single_candidate_updates"] -= 1
            elif mutation == "path":
                data["actor_update"].update(single_candidate_updates=0, envelope_updates=45001)
            elif mutation == "budget":
                data["evaluation_interactions"] -= 1000
            elif mutation == "grid":
                data["pareto_per_mode"][0].pop()
            elif mutation == "hv":
                data["final_hv_mean_mode"] += 100
            else:
                data["replay_context_refresh"] = {"complete": True}
            with self.subTest(mutation=mutation), self.assertRaises(AssertionError):
                summary.variant_metrics(data, 29)

    def test_gate_keeps_final_and_all_four_stability_aggregates(self):
        conditions = {c: {"per_seed": {s: {"final_hv": 10., "archive_hv": 12.,
            "late_mean_hv": 9., "late_min_hv": 8., "final_mode_hv": [10.] * 4}
            for s in summary.pilot.SEEDS}} for c in ("v27", "v28")}
        for r in conditions["v28"]["per_seed"].values():
            r["final_hv"] = 11.
        self.assertTrue(summary.aggregate(copy.deepcopy(conditions))["advance_to_fresh_confirmation"])
        conditions["v28"]["per_seed"][33]["late_min_hv"] = 1.
        result = summary.aggregate(copy.deepcopy(conditions))
        self.assertTrue(all(result["primary_checks"].values()))
        self.assertFalse(result["secondary_stability_checks"]["late_min_hv_min_not_worse"])
        self.assertFalse(result["secondary_stability_checks"]["late_min_hv_mean_not_worse"])
        self.assertFalse(result["advance_to_fresh_confirmation"])
        self.assertEqual(result["final_wins"], 5)
        self.assertEqual(result["paired_v28_minus_v27"][33]["late_min_hv"], -7.)
        conditions["v27"]["per_seed"].pop(33)
        with self.assertRaises(AssertionError):
            summary.aggregate(conditions)


if __name__ == "__main__":
    unittest.main()
