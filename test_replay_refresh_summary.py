import copy
import unittest

import summarize_replay_context_refresh as summary


def result():
    weights = summary.pilot.training.full_simplex_grid(2).tolist()
    fronts = [[{"weight": w, "return": [2., -1.], "archive_index": i, "archive_step": step}
               for i, step in enumerate(summary.ARCHIVE_STEPS) for w in weights] for _ in range(4)]
    hv = summary.pilot.training.compute_hv_nd([[2., -1.]], [-50., -3000.])
    return {"method": summary.pilot.METHOD, "env": "mo-halfcheetah-v5", "seed": 26,
            "total_steps": 50000, "switch_interval": 25000,
            "evaluation_protocol": "locked_full_simplex_v1", "archive_context": "final_encoder",
            "evaluation_weights": weights,
            "episode_seeds_per_mode": [[126000 + m * 10 + r for r in range(3)] for m in range(4)],
            "ref_point": [-50., -3000.], "pareto_per_mode": fronts,
            "policy_archive_snapshots": [{"step": s} for s in summary.ARCHIVE_STEPS],
            "hv_per_mode": [hv] * 4, "final_hv_per_mode": [hv] * 4,
            "hv_mean_mode": hv, "final_hv_mean_mode": hv, "eum_per_mode": [1.] * 4,
            "episode_rewards": [{"step": 50000}],
            "ground_truth": {"num_modes": 4, "mode_history": [[0, 0], [20000, 2], [40000, 0]]},
            "training_seconds": 100., "evaluation_seconds": 200.}


class RefreshSummaryTests(unittest.TestCase):
    def test_final_actor_not_archive_union_and_old_mode_retention(self):
        data = result()
        for front in data["pareto_per_mode"]:
            for row in front[-21:]:
                row["return"] = [-49., -2999.]
        data["final_hv_mean_mode"] = 1.
        data["final_hv_per_mode"] = [1.] * 4
        metrics = summary.metrics(data, 26)
        self.assertEqual(metrics["final_hv"], 1.)
        self.assertEqual(metrics["old_modes"], [2])
        self.assertEqual(metrics["training_steps_per_mode"], [30000, 0, 20000, 0])
        self.assertAlmostEqual(metrics["mode_final_to_best_snapshot"][2], 1. / metrics["archive_hv"])
        self.assertGreater(metrics["archive_hv"], 100000)
        self.assertEqual(metrics["posthoc_late_snapshot_hv"], [metrics["archive_hv"]] * 2 + [1.])
        self.assertAlmostEqual(metrics["posthoc_late_mean_hv"], (2 * metrics["archive_hv"] + 1) / 3)

    def test_incomplete_evaluation_or_misreported_hv_rejected(self):
        for mutation in ("grid", "hv", "terminal"):
            data = result()
            if mutation == "grid":
                data["pareto_per_mode"][0][0]["weight"] = [1., 0.]
            elif mutation == "hv":
                data["final_hv_per_mode"][0] += 100
            else:
                data["episode_rewards"][-1]["step"] = 49000
            with self.subTest(mutation=mutation), self.assertRaises(AssertionError):
                summary.metrics(data, 26)

    def test_unfinalized_trace_requires_every_refresh_and_zero_extra_budget(self):
        steps = summary.pilot.REFRESH_STEPS
        trace = {"protocol": summary.pilot.PROTOCOL, "complete": False,
                 "scheduled_steps": list(steps), "transitions_reencoded": sum(steps),
                 "extra_environment_steps": 0, "extra_updates": 0, "refresh_seconds": 45.,
                 "cases": [{"step": s, "transitions_reencoded": s, "seconds": 1.} for s in steps]}
        self.assertFalse(summary.validate_trace(trace)["complete"])
        for mutation in ("missing", "budget"):
            bad = copy.deepcopy(trace)
            if mutation == "missing":
                bad["cases"].pop()
            else:
                bad["extra_updates"] = 1
            with self.subTest(mutation=mutation), self.assertRaises(AssertionError):
                summary.validate_trace(bad)


if __name__ == "__main__":
    unittest.main()
