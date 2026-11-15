import copy
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

import summarize_budgeted_router as summary


def result(condition="train_on", seed=26):
    p = summary.pilot
    trials = condition in p.CONDITIONS[:3]
    steps = p.MAIN_STEPS if trials else p.TOTAL_BUDGET
    archives = p.ARCHIVE_MAIN_STEPS if trials else p.ARCHIVE_BUDGET_STEPS
    weights = p.training.full_simplex_grid(2).tolist()
    fronts = [[{"weight": w, "return": [2., -1.], "archive_index": i,
                "archive_step": step, "archive_interaction_step": budget}
               for i, (step, budget) in enumerate(zip(archives, p.ARCHIVE_BUDGET_STEPS))
               for w in weights] for _ in range(4)]
    hv = p.training.compute_hv_nd([[2., -1.]], [-50., -3000.])
    data = {"env": "mo-halfcheetah-v5", "seed": seed, "total_steps": steps,
            "switch_interval": 25000, "evaluation_protocol": "locked_full_simplex_v1",
            "archive_context": "final_encoder", "evaluation_weights": weights,
            "episode_seeds_per_mode": [[100000 + seed * 1000 + m * 10 + r for r in range(3)]
                                       for m in range(4)], "ref_point": [-50., -3000.],
            "policy_archive_snapshots": [{"step": s, "interaction_step": b}
                                         for s, b in zip(archives, p.ARCHIVE_BUDGET_STEPS)],
            "pareto_per_mode": fronts, "hv_per_mode": [hv] * 4, "final_hv_per_mode": [hv] * 4,
            "hv_mean_mode": hv, "final_hv_mean_mode": hv, "eum_per_mode": [1.] * 4,
            "ground_truth": {"num_modes": 4, "mode_history": [[0, 0], [20000, 2]] +
                             ([] if trials else [[60000, 1]])},
            "training_seconds": 1., "evaluation_seconds": 2.,
            "budgeted_routing_experiment": {
                "protocol": "budgeted_raw_return_router_v1", "condition": condition, "complete": True,
                "main_environment_steps": steps, "trial_environment_steps": 4 * p.TRIAL_STEPS if trials else 0,
                "total_training_interactions": p.TOTAL_BUDGET, "normal_updates": steps - 5000 + 1,
                "fork_updates": 8 * p.UPDATES if trials else 0,
                "committed_fork_updates": 4 * p.UPDATES if trials else 0,
                "archive_interaction_steps": list(p.ARCHIVE_BUDGET_STEPS)}}
    if trials:
        enabled = condition != "forced_off"
        cases = []
        for i, step in enumerate(p.DECISION_STEPS):
            cases.append({"main_step": step, "interaction_step": step + (i + 1) * p.TRIAL_STEPS,
                          "extra_environment_steps": p.TRIAL_STEPS, "fork_updates": 2 * p.UPDATES,
                          "committed_updates": p.UPDATES, "previous_on": enabled, "operator": "test",
                          "on_candidates": 1, "chosen": "on" if enabled else "off",
                          "preference_advantages": [0.] * 3, "mean_utility_advantage": 0.,
                          "score_reset_seeds": [30000000 + seed * 100000 + step * 100 + 20000 + r
                                                for r in range(3)],
                          "score_returns": {label: np.zeros((3, 3, 3, 2)).tolist()
                                            for label in ("on", "off")}})
        data["update_routing"] = {"condition": condition, "decision_steps": list(p.DECISION_STEPS),
                                  "horizons": list(p.HORIZONS), "repeats": p.REPEATS,
                                  "updates_per_branch": p.UPDATES, "enabled": enabled,
                                  "extra_environment_steps": 4 * p.TRIAL_STEPS, "cases": cases}
    return data


class BudgetedSummaryTests(unittest.TestCase):
    def test_all_conditions_cost_and_mode_exposure(self):
        for condition in summary.pilot.CONDITIONS:
            with self.subTest(condition=condition):
                d = summary.summarize_run(result(condition), condition, 26)
                self.assertEqual(d["budget"]["total_training_interactions"], 86864)
                self.assertEqual(d["training_steps_per_mode"],
                                 [20000, 0, 30000, 0] if condition in summary.pilot.CONDITIONS[:3]
                                 else [20000, 26864, 40000, 0])
                self.assertEqual(d["mode_final_to_best_snapshot"], [1.] * 4)

    def test_reject_wrong_budget_incomplete_grid_or_hv(self):
        data = result()
        bad = copy.deepcopy(data)
        bad["budgeted_routing_experiment"]["trial_environment_steps"] = 1
        with self.assertRaisesRegex(AssertionError, "Mismatched budget"):
            summary.summarize_run(bad, "train_on", 26)
        bad = copy.deepcopy(data)
        bad["pareto_per_mode"][0][0]["weight"] = [1., 0.]
        with self.assertRaisesRegex(AssertionError, "Incomplete snapshot grid"):
            summary.summarize_run(bad, "train_on", 26)
        data["final_hv_per_mode"][0] += 100
        with self.assertRaises(AssertionError):
            summary.summarize_run(data, "train_on", 26)

    def test_reject_choice_not_following_identity_rule(self):
        data = result("route")
        data["update_routing"]["cases"][0]["chosen"] = "off"
        with self.assertRaises(AssertionError):
            summary.summarize_run(data, "route", 26)

    def test_final_actor_is_not_archive_union(self):
        data = result()
        for front in data["pareto_per_mode"]:
            for point in front[-21:]:
                point["return"] = [-49., -2999.]
        data["final_hv_mean_mode"] = 1.
        data["final_hv_per_mode"] = [1.] * 4
        d = summary.summarize_run(data, "train_on", 26)
        self.assertEqual(d["final_hv"], 1.)
        self.assertGreater(d["archive_hv"], 100000)

    def test_paired_common_seed_and_schedule_prefix(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for condition in ("route", "train_on"):
                target = root / condition / f"{summary.pilot.forks.METHOD}_seed26.json"
                target.parent.mkdir()
                target.write_text(json.dumps(result(condition)))
            report = summary.summarize(root)
            self.assertEqual(report["completed"], 2)
            self.assertEqual(len(report["missing"]), 13)
            pair = report["paired_route_minus"]["train_on"]
            self.assertEqual(pair["seeds"], [26])
            self.assertEqual(pair["final_hv"]["mean_difference"], 0.)
            changed = result("train_on")
            changed["ground_truth"]["mode_history"][1][0] += 1
            target.write_text(json.dumps(changed))
            with self.assertRaisesRegex(AssertionError, "Unmatched main schedule prefix"):
                summary.summarize(root)


if __name__ == "__main__":
    unittest.main()
