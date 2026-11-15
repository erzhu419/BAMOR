import copy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import torch

import diagnose_ridge_learning_curve as diagnostic
import summarize_ridge_learning_curve as summary
from test_paired_advantage_summary import terminal_fixture
from test_state_curvature import cached_fixture as weak_fixture


def cached_fixture(cases):
    d = diagnostic
    old = weak_fixture(cases)
    old["protocol"] = d.previous.PROTOCOL
    old["rows"] = [d.reference.join_row(c, r, {"state_curvature": np.zeros(5)}, selectors=d.previous.SELECTORS)
                   for c, r in zip(cases, old["rows"])]
    for fold in range(4):
        fit, held = d.cv.split_fold(cases, fold)
        model = d.cv.QuadraticRidge(d.p.training_data(fit))
        scores = model(torch.stack([d.cv.features(c) for c in held])).numpy()
        predictions = {d.reference.case_key(c): s for c, s in zip(held, scores)}
        for row in old["rows"]:
            if row["fold"] == fold:
                row["scores"]["ridge"] = dict(zip(d.p.CANDIDATES, map(float, predictions[d.reference.case_key(row)])))
                row["choices"]["ridge"] = d.p.choose(row["scores"]["ridge"])
                row["selected_gains"]["ridge"] = row["candidate_gains"][row["choices"]["ridge"]]
    return old


class RidgeLearningCurveTests(unittest.TestCase):
    def test_subsets_are_nested_balanced_complementary_and_label_independent(self):
        d = diagnostic
        cases = terminal_fixture()["train_cases"]
        for c in cases:
            c["branches"] = c["predictions"] = None
        for fold in range(4):
            fit, held = d.cv.split_fold(cases, fold)
            full_ids = {c["reset_seed"] for c in fit}
            small_ids = []
            for chain in range(3):
                subsets = [d.fitting_subset(fit, fold, size, chain) for size in d.SIZES]
                ids = [{c["reset_seed"] for c in subset} for subset in subsets]
                self.assertTrue(ids[0] < ids[1] < ids[2] == full_ids)
                self.assertTrue(ids[2].isdisjoint(c["reset_seed"] for c in held))
                self.assertEqual(subsets[2], fit)
                for size, subset in zip(d.SIZES, subsets):
                    self.assertEqual(len(subset), size * 3)
                    for mode in range(4):
                        self.assertEqual(len({c["reset_seed"] for c in subset if c["mode"] == mode}), size // 4)
                    for reset in {c["reset_seed"] for c in subset}:
                        self.assertEqual({c["weight_index"] for c in subset if c["reset_seed"] == reset}, {0, 1, 2})
                small_ids.append(ids[0])
            self.assertEqual(set.union(*small_ids), full_ids)
            self.assertFalse(any(small_ids[i] & small_ids[j] for i in range(3) for j in range(i)))

    def test_fit_normalizes_only_subset_and_ignores_held_labels(self):
        d = diagnostic
        fit, held = d.cv.split_fold(terminal_fixture()["train_cases"], 0)
        subset = d.fitting_subset(fit, 0, 16, 0)
        before, rng = copy.deepcopy(subset), torch.get_rng_state().clone()
        scores, meta = d.fit_predict(subset, torch.stack([d.cv.features(c) for c in held]))
        self.assertEqual(subset, before)
        for c in held:
            c["branches"] = c["predictions"] = None
        for c in fit:
            c["predictions"] = None
            c["branches"]["base"]["discounted_returns"]["128"] = None
            if c not in subset:
                c["anchor_observation"] = [1e9] * 17
                c["branches"] = None
        repeated, repeated_meta = d.fit_predict(subset, torch.stack([d.cv.features(c) for c in held]))
        np.testing.assert_array_equal(scores, repeated)
        self.assertEqual(meta, repeated_meta)
        self.assertEqual((meta["fit_anchors"], meta["fit_nonbase_rows"], meta["parameters"]), (48, 192, 243))
        self.assertTrue((scores[:, 0] == 0).all())
        torch.testing.assert_close(torch.get_rng_state(), rng, rtol=0, atol=0)

    def test_runner_reproduces_endpoint_and_summary_rejects_protocol_drift(self):
        d = diagnostic
        source = terminal_fixture()
        source["validation_cases"] = source["summary"] = None
        previous = cached_fixture(source["train_cases"])
        with (tempfile.TemporaryDirectory() as directory,
              patch.object(d.p.training, "make_piecewise_mo_env", side_effect=AssertionError("No rollouts")),
              patch.object(d.cv, "fit_fold", side_effect=AssertionError("No MLP/control refit")),
              patch.object(d.previous, "fit_predict", side_effect=AssertionError("No curvature refit")),
              patch.object(torch, "load", side_effect=AssertionError("No checkpoints"))):
            root = Path(directory)
            for name, data in (("source", source), ("previous", previous)):
                path = root / name / "v27/seed31.json"
                path.parent.mkdir(parents=True)
                path.write_text(json.dumps(data))
            result = d.run("v27", 31, root / "source", root / "previous", root / "out")
            self.assertEqual(summary.validate_run(result, previous, source, "v27", 31), result["summary"])
            self.assertEqual(result["ridge_solves"], 28)
            for kind in ("subset", "control", "endpoint", "grid", "budget", "summary"):
                changed = copy.deepcopy(result)
                if kind == "subset":
                    changed["folds"][0]["fits"][0]["fit_reset_ids"] = changed["folds"][0]["held_reset_ids"]
                elif kind == "control":
                    changed["rows"][0]["scores"]["state_curvature"]["raw_plus"] += 1.
                elif kind == "endpoint":
                    changed["rows"][0]["scores"]["ridge48"]["raw_minus"] += .001
                elif kind == "grid":
                    changed["rows"][0] = copy.deepcopy(changed["rows"][1])
                elif kind == "budget":
                    changed["ridge_solves"] += 1
                else:
                    changed["summary"]["recent_improvement"] = not result["summary"]["recent_improvement"]
                with self.subTest(kind=kind), self.assertRaises(AssertionError):
                    summary.validate_run(changed, previous, source, "v27", 31)

    def test_summary_pools_errors_without_ensembling_chain_predictions(self):
        d = diagnostic
        cases = terminal_fixture()["train_cases"]
        old = cached_fixture(cases)
        predictions = {d.selector(size, chain): np.array([0., *([2. * chain] * 4)])
                       for size, chain in d.FIT_SPECS}
        rows = [d.reference.join_row(c, r, predictions, selectors=d.SELECTORS)
                for c, r in zip(cases, old["rows"])]
        result = d.summarize_rows(rows)
        truth = np.array([1., 0., .5, 0.])
        expected = float(np.sqrt(np.mean([(2. * chain - truth) ** 2 for chain in range(3)])))
        self.assertAlmostEqual(result["curve"][0]["rmse512"], expected)
        self.assertGreater(expected, float(np.sqrt(np.mean((2. - truth) ** 2))))
        self.assertEqual(result["curve"][2]["chains"][0]["rmse512"],
                         result["curve"][2]["chains"][2]["rmse512"])

    def test_recent_trend_does_not_hide_downside_or_endpoint_selection(self):
        d = diagnostic
        points = []
        for size, gain, negative, rmse in ((16, 0., -.1, 3.), (32, 1., -.1, 2.), (48, 2., -.2, 1.)):
            point = {"fit_reset_groups": size, "mean_gain": gain, "mean_negative_part": negative, "rmse512": rmse}
            points.append({**point, "chains": [dict(point) for _ in range(3)]})
        result = d.curve_analysis(points)
        self.assertFalse(result["recent_improvement"])
        self.assertEqual(result["recent_checks"], {"mean_gain_increases": True, "downside_not_worse": False,
                                                  "rmse_decreases": True})
        self.assertEqual(result["transitions"][1]["favorable_chain_counts"],
                         {"mean_gain": 3, "mean_negative_part": 0, "rmse512": 3})


if __name__ == "__main__":
    unittest.main()
