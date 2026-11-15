import copy
import math
import unittest

from diagnose_episode_credit_warmup import diagnose, summarize
from episode_credit_baseline import EpisodeCreditBaseline
from test_episode_feedback_diagnostic import fixture


class EpisodeCreditWarmupTests(unittest.TestCase):
    def test_three_past_complete_returns_seed_scale_keep_ema_mean(self):
        baseline = EpisodeCreditBaseline(.13)
        for utility in (0., 10., 20.):
            before = baseline.snapshot()
            self.assertIsNone(baseline.score(utility, before))
            baseline.observe(utility)
        before = baseline.snapshot()
        self.assertEqual(before["count"], 3)
        self.assertAlmostEqual(before["mean"], 3.731)
        self.assertEqual(before["scale"], 10.)
        self.assertEqual(baseline.score(30., before), math.tanh((30 - 3.731) / 10))
        baseline.observe(30.)
        self.assertAlmostEqual(baseline.variance, .87 * (100 + .13 * (30 - 3.731) ** 2))
        self.assertEqual(before, dict(count=3, mean=3.731, scale=10.))

    def test_constant_returns_abstain_until_empirical_dispersion(self):
        baseline = EpisodeCreditBaseline(.13)
        for _ in range(3):
            baseline.observe(7.)
        self.assertIsNone(baseline.score(8., baseline.snapshot()))
        baseline.observe(8.)
        self.assertEqual(baseline.snapshot()["scale"], .5)
        self.assertAlmostEqual(baseline.mean, 7.13)
        baseline.reset()
        self.assertEqual(baseline.snapshot(), dict(count=0, mean=None, scale=None))

    def test_recorded_route_frozen_common_cohort_and_source_unchanged(self):
        data = fixture()
        original = copy.deepcopy(data)
        episodes, segments = diagnose(data)
        summary = summarize(episodes, segments)
        self.assertEqual(data, original)
        self.assertEqual(summary["withheld_source_credited_episodes"], 2)
        self.assertEqual(summary["withheld_source_floor_credited_episodes"], 1)
        self.assertEqual(summary["newly_credited_episodes"], 0)
        self.assertEqual(summary["common_source"]["count"], 2)
        self.assertEqual(summary["common_candidate"]["count"], 2)
        self.assertEqual([ep["phase"] for ep in episodes], ["warmup", "forced", "forced", "forced", "free"])
        self.assertEqual([ep["operator"] for ep in episodes],
                         [ep["operator"] for ep in original["adaptive_operator_final"]["episode_history"]])
        self.assertEqual(segments[0]["scale_initialized_at_end_step"], 30)

    def test_prefix_and_ground_truth_do_not_change_candidate_feedback(self):
        data = fixture()
        full, _ = diagnose(data)
        prefix = copy.deepcopy(data)
        prefix["adaptive_operator_final"]["episode_history"] = prefix["adaptive_operator_final"]["episode_history"][:4]
        prefix["total_steps"] = 40
        short, _ = diagnose(prefix)
        self.assertEqual(full[:4], short)
        relabeled = copy.deepcopy(data)
        relabeled["ground_truth"]["mode_history"] = [[0, 2], [15, 3]]
        changed, _ = diagnose(relabeled)
        for before, after in zip(full, changed):
            for key in ("candidate_score", "candidate_credited", "candidate_scale_before", "initialization_count_before"):
                self.assertEqual(before[key], after[key])

    def test_mixed_episode_reset_excludes_return_and_delays_readiness(self):
        data = fixture()
        records = data["adaptive_operator_final"]["episode_history"]
        records[2].update(epoch_end=1, mixed_epoch=True, credited=False, score=None, score_after=0.)
        records[3].update(epoch_start=1, epoch_end=1, operator="uniform", baseline_before=None,
                          scale_before=None, credited=False, score=None, score_after=0.)
        records[4].update(epoch_start=1, epoch_end=1, operator="uniform", baseline_before=30.,
                          scale_before=1e-6, score=1., score_after=.13)
        data["adaptive_operator_final"]["reset_history"] = [dict(step=25, epoch=1)]
        episodes, segments = diagnose(data)
        self.assertTrue(all(not ep["candidate_credited"] for ep in episodes))
        self.assertIsNone(episodes[3]["candidate_scale_before"])
        self.assertEqual(episodes[4]["initialization_count_before"], 1)
        self.assertEqual(segments[1]["completed_eligible_baseline_episodes"], 2)
        self.assertEqual(summarize(episodes, segments)["added_zero_credit_segments"], 2)

    def test_corrupt_legacy_mean_or_scale_stops_analysis(self):
        for field in ("baseline_before", "scale_before"):
            data = fixture()
            data["adaptive_operator_final"]["episode_history"][2][field] += 1.
            with self.assertRaisesRegex(ValueError, "reconstructed past episodes"):
                diagnose(data)


if __name__ == "__main__":
    unittest.main()
