import copy
import unittest

import numpy as np

from diagnose_episode_allocation import diagnose_run, episode_modes, summarize


def fixture():
    utilities = [0., 10., 20., 30., 40.]
    operators = ["uniform", "uniform", "stratified", "mixed", "uniform"]
    mean, variance, ready = 0., 0., False
    scores = dict.fromkeys(operators, 0.)
    episodes = []
    for index, (utility, op) in enumerate(zip(utilities, operators)):
        baseline = mean if ready else None
        scale = max(np.sqrt(variance), 1e-6) if ready else None
        credit = index > 0
        score = float(np.tanh((utility - baseline) / scale)) if credit else None
        if credit:
            scores[op] = .87 * scores[op] + .13 * score
        episodes.append(dict(start_step=index * 10 + 1, end_step=(index + 1) * 10,
            length=10, operator=op, utility=utility, epoch_start=0, epoch_end=0,
            mixed_epoch=False, baseline_before=baseline, scale_before=scale,
            credited=credit, score=score, score_after=scores[op]))
        if ready:
            delta = utility - mean
            mean += .13 * delta
            variance = .87 * (variance + .13 * delta ** 2)
        else:
            mean, ready = utility, True
    return dict(total_steps=50, ground_truth=dict(mode_history=[[0, 0], [20, 1]]),
        adaptive_operator_final=dict(operators=["uniform", "stratified", "mixed"],
            baseline_rate=.13, score_rate=.13, epsilon=.15, temperature=.35,
            learning_starts=10, selection_uses_scores=True, reset_history=[],
            episode_history=episodes))


class EpisodeFeedbackDiagnosticTests(unittest.TestCase):
    def test_canonical_after_transition_boundary_and_round_trip(self):
        history = [[0, 0], [20, 1], [30, 0]]
        self.assertEqual(episode_modes(history, 11, 20), [0])
        self.assertEqual(episode_modes(history, 21, 30), [1])
        self.assertEqual(episode_modes(history, 20, 21), [0, 1])
        self.assertEqual(episode_modes(history, 11, 40), [0, 1, 0])

    def test_forced_free_and_baseline_provenance(self):
        episodes, segments = diagnose_run(fixture())
        self.assertEqual([ep["phase"] for ep in episodes], ["warmup", "forced", "forced", "forced", "free"])
        self.assertAlmostEqual(episodes[2]["baseline_nonmatching_or_mixed_mass"], 1.)
        self.assertAlmostEqual(episodes[3]["baseline_nonmatching_or_mixed_mass"], .87)
        self.assertAlmostEqual(episodes[4]["baseline_nonmatching_or_mixed_mass"], .87 ** 2)
        summary = summarize(episodes, segments)
        self.assertEqual(summary["free_credited_episodes"], 1)
        self.assertEqual(summary["scale_floor_credited_episodes"], 1)
        self.assertEqual(summary["selection_step_counts"], {"forced": 30, "free": 10})
        self.assertEqual(summary["segments_without_free_credited_episodes"], 0)

    def test_true_mixed_feedback_without_event_log(self):
        data = fixture()
        data["ground_truth"]["mode_history"] = [[0, 0], [15, 1]]
        episodes, segments = diagnose_run(data)
        self.assertTrue(episodes[1]["true_mixed"])
        self.assertAlmostEqual(episodes[2]["baseline_nonmatching_or_mixed_mass"], 1.)
        self.assertEqual(summarize(episodes, segments)["true_mixed_credited_episodes"], 1)

    def test_detected_reset_clears_provenance_scores_and_forcing(self):
        data = fixture()
        records = data["adaptive_operator_final"]["episode_history"]
        records[2].update(epoch_end=1, mixed_epoch=True, credited=False, score=None, score_after=0.)
        records[3].update(epoch_start=1, epoch_end=1, operator="uniform", baseline_before=None,
                          scale_before=None, credited=False, score=None, score_after=0.)
        records[4].update(epoch_start=1, epoch_end=1, operator="uniform", baseline_before=30.,
                          scale_before=1e-6, score=1., score_after=.13)
        data["adaptive_operator_final"]["reset_history"] = [dict(step=25, epoch=1)]
        episodes, segments = diagnose_run(data)
        self.assertEqual(episodes[4]["phase"], "forced")
        self.assertEqual(episodes[4]["baseline_nonmatching_or_mixed_mass"], 0.)
        self.assertEqual(summarize(episodes, segments)["segments_without_three_credited_episodes"], 2)

    def test_fixed_probability_and_bad_reconstruction(self):
        data = fixture()
        data["adaptive_operator_final"]["selection_uses_scores"] = False
        episodes, _ = diagnose_run(data)
        self.assertEqual(episodes[-1]["maximum_choice_probability"], 1 / 3)
        for field, value in (("score_after", 0.), ("operator", "mixed")):
            broken = copy.deepcopy(data)
            broken["adaptive_operator_final"]["episode_history"][2][field] = value
            with self.assertRaises(ValueError):
                diagnose_run(broken)


if __name__ == "__main__":
    unittest.main()
