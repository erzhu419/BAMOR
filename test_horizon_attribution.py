import copy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import torch

import diagnose_horizon_attribution as diagnostic
from test_margin_selection import cached_fixture
from test_paired_advantage_summary import terminal_fixture


def decisions_fixture(cases):
    m = diagnostic.margin
    old = cached_fixture(cases)
    old["protocol"] = m.PROTOCOL
    old["rows"] = [m.reference.join_row(c, r, {"margin": np.array([0., 2., 0., 1., 0.])}, selectors=m.SELECTORS)
                   for c, r in zip(cases, old["rows"])]
    old["summary"] = m.summarize_rows(old["rows"])
    return old


class HorizonAttributionTests(unittest.TestCase):
    def test_scalarization_and_discount_origin(self):
        d = diagnostic
        case = terminal_fixture()["train_cases"][0]
        case["weight"] = [.25, .75]
        for h, value in ((32, [10., 2.]), (128, [2., 6.]), (512, [6., 6.])):
            case["branches"]["base"]["discounted_returns"][str(h)] = [2., 2.]
            case["branches"]["soft_plus"]["discounted_returns"][str(h)] = value
        a, b, c = [d.paired_gains(case, h)[1] for h in d.HORIZONS]
        self.assertEqual((a, b, c), (2., 3., 4.))
        self.assertEqual((b - a, c - b), (1., 1.))

    def test_reversals_ties_and_hindsight_are_distinct(self):
        d = diagnostic
        before = np.array([[0., 2., -1., 1., 0.]])
        after = np.array([[0., -2., 1., .5, 0.]])
        cases = [{"mode": 0, "weight_index": 0}]
        result = d.stability(before, after, cases)
        self.assertEqual(result["nonbase_sign"], {"reversals": 2, "eligible": 3, "total": 4, "rate": 2 / 3})
        self.assertEqual(result["best_action_exact_matches"], 0)
        self.assertEqual(result["prefix_oracle_at_later_horizon"]["mean_gain"], -2.)
        self.assertEqual(result["mean_prefix_oracle_regret"], 3.)
        tiny = d.reversal_counts(np.array([1e-7, 0.]), np.array([-1e-7, 1.]))
        self.assertEqual(tiny["eligible"], 0)
        self.assertIsNone(tiny["rate"])
        tie = d.stability(np.zeros((1, 5)), np.zeros((1, 5)), cases)
        self.assertEqual(tie["best_action_within_tolerance"], 1)

    def test_tail_groups_and_leave_one_keep_equal_cell_weight(self):
        d = diagnostic
        cases = terminal_fixture()["train_cases"]
        values = np.array([16. if c["mode"] == 0 and c["repeat"] == 0 else 0. for c in cases])
        mass = d.reset_mass(values, cases)
        self.assertEqual((mass["reset_groups"], mass["top_groups"][0]["anchors"]), (64, 3))
        self.assertEqual((mass["total"], mass["top1_share"]), (48., 1.))
        result = d.influence(values, cases)
        self.assertEqual(result["all_sample_mean_gap"], .25)
        self.assertEqual(result["leave_one_reset_min"]["mean_gap"], 0.)
        self.assertAlmostEqual(result["leave_one_reset_max"]["mean_gap"], 16. / 60.)
        self.assertIsNone(result["adverse_gap_mass"]["top1_share"])

    def test_run_uses_cached_choices_not_prefix_oracles_or_validation(self):
        d = diagnostic
        source = terminal_fixture()
        source["validation_cases"] = source["summary"] = None
        cases = source["train_cases"]
        decisions = decisions_fixture(cases)
        for c in cases:
            c["branches"]["soft_plus"]["discounted_returns"]["32"] = [-2., -2.]
            c["branches"]["soft_plus"]["discounted_returns"]["128"] = [2., 2.]
        original = copy.deepcopy(decisions)
        with (patch.object(d.margin, "fit_predict", side_effect=AssertionError("No fitting")),
              patch.object(d.p.training, "make_piecewise_mo_env", side_effect=AssertionError("No rollouts")),
              patch.object(torch, "load", side_effect=AssertionError("No checkpoints"))):
            result = d.analyze_run(source, decisions, "v27", 31)
        self.assertEqual(result["horizons"]["32"]["selectors"]["margin"]["mean_gain"], -2.)
        self.assertEqual(result["horizons"]["512"]["selectors"]["margin"]["mean_gain"], 1.)
        gap = result["contrasts"]["margin_minus_ridge"]
        self.assertEqual(gap["segment_mean_gaps"], {"0:32": -2., "32:128": 4., "128:512": -1.})
        self.assertEqual(decisions, original)
        json.dumps(result, allow_nan=False)
        for kind in ("pairing", "labels", "choice", "primary"):
            changed = copy.deepcopy(decisions)
            if kind == "pairing":
                changed["rows"][0]["reset_seed"] += 1
            elif kind == "labels":
                changed["rows"][0]["candidate_gains"]["soft_plus"] += 1.
            elif kind == "choice":
                changed["rows"][0]["choices"]["margin"] = "base"
            else:
                changed["summary"]["aggregate"]["margin"]["mean_gain"] += 1.
            with self.subTest(kind=kind), self.assertRaises(AssertionError):
                d.analyze_run(source, changed, "v27", 31)

    def test_four_run_export_requires_all_groups_and_counts_no_new_work(self):
        d = diagnostic
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for condition in ("v27", "v28"):
                for seed in (31, 33):
                    source = terminal_fixture()
                    source.update(condition=condition, seed=seed, validation_cases=None, summary=None)
                    for c in source["train_cases"]:
                        c["reset_seed"] = d.p.reset_seed("train", seed, c["mode"], c["repeat"])
                    decisions = decisions_fixture(source["train_cases"])
                    decisions.update(condition=condition, seed=seed)
                    for name, report in (("source", source), ("decisions", decisions)):
                        path = root / name / condition / f"seed{seed}.json"
                        path.parent.mkdir(parents=True, exist_ok=True)
                        path.write_text(json.dumps(report))
            result = d.run(root / "source", root / "decisions")
            self.assertEqual((len(result["runs"]), result["reused_anchors"]), (4, 768))
            self.assertEqual((result["new_environment_interactions"], result["model_updates"]), (0, 0))
            self.assertEqual(result["primary_horizon"], 512)


if __name__ == "__main__":
    unittest.main()
