import copy
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import torch

import diagnose_source_held_windows as probe
from headwise_change_evidence import CONFIG


def window(index, raw, scales, threshold, supported=100):
    heads = {name: dict(eligible=supported >= 80, supported_steps=supported,
        domain="state_action", statistic=value / scale if supported >= 80 else None)
        for name, value, scale in zip(("reward_0", "dynamics"), raw, scales)}
    values = [head["statistic"] for head in heads.values() if head["eligible"]]
    return dict(start_step=index * 100 + 1, end_step=(index + 1) * 100, eligible=bool(values),
        supported_steps=supported, statistic=max(values) if values else None, heads=heads,
        alarm=bool(values and max(values) > threshold))


def paired(old_score, new_score, old_threshold, new_threshold, supported=100):
    return probe.paired_window(window(0, [1., 0.], [1., 1.], 999, supported),
        window(0, [1., 0.], [1 / old_score, 1.], old_threshold, supported),
        window(0, [1., 0.], [1 / new_score, 1.], new_threshold, supported),
        [1., 1.], (old_threshold, new_threshold))


def saved_source(env, seed):
    old_cal = [window(i, [i + 1., 1.], [1., 1.], 999) for i in range(50)]
    new_cal = [window(i, [i + 1., 1.], [1.25, 1.25], 999) for i in range(50)]
    old_t, new_t = (float(np.quantile([row["statistic"] for row in rows], .99)) for rows in (old_cal, new_cal))
    arms = {}
    for arm, cal, threshold, scale in (("global_scale", old_cal, 999., 2.),
                                     ("conditional_scale", old_cal, old_t, 2.),
                                     (probe.SELECTED, new_cal, new_t, 2.)):
        if arm == "global_scale":
            cal = [window(i, [i + 1., 1.], [2., 2.], 999) for i in range(50)]
        for row in cal:
            row.pop("alarm", None)
        cases = {}
        for case in (*probe.NORMAL_CASES, *probe.MODE_CASES):
            raw = 100. if case.startswith("mode") or case == "stationary" else 80. if case == "sampler_change" else 0.
            rows = [window(i, [10. if i < 5 else raw, 1.], [scale, scale], threshold) for i in range(10)]
            cases[case] = dict(windows=rows)
        arms[arm] = dict(calibration=dict(error_threshold=threshold, support_radii=dict(reward_0=1., dynamics=1.), windows=cal), cases=cases)
    return dict(protocol=probe.INPUT_PROTOCOL, env=env, roles=probe.fold_roles(seed), complete=True,
        config=CONFIG, predictor_optimizer_steps=4000, residual_scale_optimizer_steps=4000,
        scale_source=dict(reward_dim=1), fit_steps=10000, scale_steps=5000, calibration_steps=5000,
        control_scores_reproduced=True, scale_work=dict(global_scales=[2., 2.]), arms=arms,
        trajectory_control=dict(selection=dict(global_weight=0), arms={probe.SELECTED: copy.deepcopy(arms["conditional_scale"])}),
        selection=dict(global_weight=.25, normal_nll=.8, candidates=[dict(global_weight=0., normal_nll=1.)]))


class SourceHeldWindowTests(unittest.TestCase):
    def test_raw_error_factorization_and_winning_head_change(self):
        row = probe.paired_window(window(0, [4., 3.], [1., 1.], 999),
            window(0, [4., 3.], [2., 3.], 2.5), window(0, [4., 3.], [4., 1.], 2.),
            [1., 1.], (2.5, 2.))
        self.assertEqual((row["old_dominant_head"], row["dominant_head"]), ("reward_0", "dynamics"))
        self.assertTrue(row["winner_changed"])
        for head in row["heads"].values():
            self.assertEqual(head["raw_error_factor"], 1.)
            self.assertAlmostEqual(head["new_ratio"], head["old_ratio"] * head["scale_factor"] * head["threshold_factor"])

    def test_four_algebraic_sufficiency_cases(self):
        rows = [paired(2., 5., 4., 3.), paired(3.5, 2.5, 4., 2.),
                paired(3.5, 5., 4., 3.), paired(2.5, 3.5, 4., 3.)]
        self.assertEqual([row["sufficient_change"] for row in rows],
            ["score_sufficient", "threshold_sufficient", "both_sufficient", "joint_required"])

    def test_falling_score_can_still_alarm_when_threshold_falls_further(self):
        row = paired(3.5, 2.5, 4., 2.)
        self.assertLess(row["new_joint_score"], row["old_joint_score"])
        self.assertEqual(row["sufficient_change"], "threshold_sufficient")
        boundary = paired(3., 2., 4., 2.)
        self.assertFalse(boundary["new_alarm"])

    def test_missing_support_zero_error_and_pairing_errors(self):
        missing = paired(1., 1., 2., 2., supported=0)
        self.assertIsNone(missing["dominant_head"])
        zero = window(0, [0., 0.], [1., 1.], 2.)
        row = probe.paired_window(zero, zero, zero, [1., 1.], (2., 2.))
        self.assertIsNone(row["heads"]["reward_0"]["scale_factor"])
        broken = copy.deepcopy(zero)
        broken["heads"]["reward_0"]["supported_steps"] -= 1
        with self.assertRaises(AssertionError):
            probe.paired_window(zero, zero, broken, [1., 1.], (2., 2.))
        broken = copy.deepcopy(zero)
        broken["heads"]["reward_0"]["statistic"] = 1.
        broken["statistic"] = 1.
        with self.assertRaises(AssertionError):
            probe.paired_window(zero, broken, zero, [1., 1.], (2., 2.))

    def test_both_quantile_tails_reproduce_thresholds(self):
        source = saved_source(probe.ENVIRONMENTS[0], 2101)
        result = probe.analyze_fold(source)
        for side in ("old", "new"):
            tail = result["calibration"][f"{side}_tail"]
            self.assertAlmostEqual(sum(row["weight"] * row["joint_score"] for row in tail["operands"]), tail["threshold"])
            self.assertEqual([row["end_step"] for row in tail["operands"]], [4900, 5000])

    def test_saved_pipeline_no_models_npz_interactions_and_unique_prefixes(self):
        with tempfile.TemporaryDirectory() as tmp:
            inputs, root = Path(tmp) / "input", Path(tmp) / "output"
            env = probe.ENVIRONMENTS[0]
            for seed in probe.SEEDS:
                probe.write(inputs / env.replace("-", "_") / f"test_seed{seed}" / "readout.json", saved_source(env, seed))
            with patch.object(np, "load", side_effect=AssertionError("No raw arrays")), \
                 patch.object(torch.optim, "Adam", side_effect=AssertionError("No model updates")):
                results = [probe.run(env, seed, root, inputs) for seed in probe.SEEDS]
                self.assertEqual(probe.run(env, 2101, root, inputs), results[0])
                with patch.object(probe, "ENVIRONMENTS", (env,)):
                    summary = probe.summarize(root)
            row = results[0]
            normal = [w for group in row["normal_groups"].values() for w in group["windows"]]
            self.assertEqual(len(normal), 20)
            self.assertEqual(sum(w["case"] == "shared_prefix" for w in normal), 5)
            self.assertEqual(sum(g["summary"]["windows"] for g in row["mode_groups"].values()), 15)
            self.assertEqual(row["normal_groups"]["new_only_alarm"]["summary"]["windows"], 5)
            self.assertTrue(all(row[key] == 0 for row in [summary, *results] for key in probe.WORK))
            compact = probe.compact_report(summary)["environments"][env]["runs"][0]
            self.assertNotIn("windows", compact["calibration"])
            self.assertNotIn("windows", compact["normal_groups"]["neither_alarm"])
            self.assertIn("windows", compact["normal_groups"]["new_only_alarm"])
            broken = saved_source(env, 2101)
            broken["arms"][probe.SELECTED]["cases"]["actor_change"]["windows"][0]["statistic"] += 1.
            with self.assertRaises(AssertionError):
                probe.analyze_fold(broken)


if __name__ == "__main__":
    unittest.main()
