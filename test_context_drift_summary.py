"""Checks for completion gating and the RMS reduction used by the diagnostic."""

import copy
import math
import unittest

import summarize_context_drift as summary


def fixture():
    cohorts = [{"start_transition_index": i, "end_transition_index": i + 10000,
                "anchor_step": i + 10000, "sample_timesteps": list(range(i + 1, i + 257)),
                "posthoc_sample_counts_per_mode": [256, 0, 0, 0]}
               for i in range(0, 50000, 10000)]
    weights = [[0., 1.], [.5, .5], [1., 0.]]
    prefs = [{"weight": w, "q_means": {"stored_context_stored_action": 10.},
              **{key: {"rms": rms, "mean": -1., "p95_abs": 5.}
                 for key in summary.ACTOR_METRICS + summary.Q_METRICS}}
             for w, rms in zip(weights, (1., 2., 3.))]
    return {
        "protocol": summary.PROTOCOL, "method": summary.METHOD, "seed": 26,
        "complete": True, "measurements_complete": True,
        "base_reproduction": {"matched": True, "matched_steps_and_schedule": True,
                              "max_episode_return_error": 0.},
        "main_environment_steps": 50000, "total_training_interactions": 50000,
        "extra_environment_steps": 0, "extra_updates": 0,
        "steps": list(summary.STEPS), "cohort_width": 10000, "samples_per_cohort": 256,
        "weights": weights, "cohorts": cohorts, "ground_truth_mode_history": [[0, 0]],
        "cases": [{"step": step, "panels": [
            {"start_transition_index": i, "anchor_step": i + 10000, "samples": 256,
             "preferences": copy.deepcopy(prefs),
             **{key: {"rms": .1, "mean": 0.} for key in summary.LATENT_METRICS}}
            for i in range(0, step // 10000 * 10000, 10000)]} for step in summary.STEPS],
    }


class SummaryTests(unittest.TestCase):
    def test_rms_reduction_preserves_all_time_cohorts(self):
        result = summary.summarize_run(fixture())
        self.assertEqual(len(result["rows"]), 30)
        row = result["rows"][-1]
        self.assertEqual((row["step"], row["anchor_step"]), (50000, 50000))
        self.assertAlmostEqual(row["actor_rms_all_preferences"]["actor_context_effect"],
                               math.sqrt(14. / 3.))
        self.assertNotIn("sample_timesteps", result["cohorts"][0])
        self.assertEqual(row["preferences"][0]["q_delta_mean"][summary.Q_METRICS[0]], -1.)

    def test_incomplete_or_different_trajectory_rejected(self):
        for key in ("complete", "measurements_complete"):
            data = fixture()
            data[key] = False
            with self.subTest(key=key), self.assertRaises(AssertionError):
                summary.summarize_run(data)
        data = fixture()
        data["base_reproduction"]["max_episode_return_error"] = 1.
        with self.assertRaises(AssertionError):
            summary.summarize_run(data)

    def test_wrong_budget_missing_panel_and_wrong_anchor_rejected(self):
        for change in ("budget", "panel", "anchor"):
            data = fixture()
            if change == "budget":
                data["extra_environment_steps"] = 1
            elif change == "panel":
                data["cases"][-1]["panels"].pop()
            else:
                data["cases"][-1]["panels"][-1]["anchor_step"] = 40000
            with self.subTest(change=change), self.assertRaises(AssertionError):
                summary.summarize_run(data)


if __name__ == "__main__":
    unittest.main()
