import copy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import torch

import diagnose_ridge_batch_cross as diagnostic
import summarize_ridge_batch_cross as summary
from test_ridge_expansion import fixture
from test_ridge_learning_curve import cached_fixture


def cross_fixture():
    d = diagnostic
    source, labels = fixture()
    source["validation_cases"] = source["summary"] = None
    for c in labels["train_cases"]:
        c["branches"]["soft_plus"]["discounted_returns"]["512"] = [-1., -1.]
        c["branches"]["raw_plus"]["discounted_returns"]["512"] = [2., 2.]
    cached = cached_fixture(source["train_cases"])
    lookup = {d.reference.case_key(r): r for r in cached["rows"]}
    rows = []
    for fold in range(4):
        fit, held = d.previous.split_expanded(source["train_cases"], labels["train_cases"], fold)
        scores, _ = d.curve.fit_predict(fit, torch.stack([d.cv.features(c) for c in held]))
        rows.extend(d.reference.join_row(c, lookup[d.reference.case_key(c)], {"ridge96": scores[i]},
                                        selectors=d.previous.SELECTORS) for i, c in enumerate(held))
    expanded = {"protocol": d.previous.PROTOCOL, "condition": "v27", "seed": 31,
        "complete": True, "source_split": "train_cases", "rows": rows, "summary": {"group_passes": False}}
    return source, labels, expanded


class RidgeBatchCrossTests(unittest.TestCase):
    def test_three_fits_exclude_both_held_batches_and_preserve_all_preferences(self):
        d = diagnostic
        source, labels, _ = cross_fixture()
        for fold in range(4):
            fits, held = d.split_batches(source["train_cases"], labels["train_cases"], fold)
            self.assertEqual([len(fits[name]) for name in d.MODELS], [144, 144, 288])
            self.assertEqual([len(held[batch]) for batch in d.BATCHES], [48, 48])
            self.assertEqual(fits["combined96"], fits["old48"] + fits["new48"])
            for name, cases in fits.items():
                self.assertTrue(all(c["repeat"] % 4 != fold for c in cases))
                for reset in {c["reset_seed"] for c in cases}:
                    self.assertEqual({c["weight_index"] for c in cases if c["reset_seed"] == reset}, {0, 1, 2})
                if name == "old48":
                    self.assertTrue(all(c["repeat"] < 16 for c in cases))
                elif name == "new48":
                    self.assertTrue(all(c["repeat"] >= 16 for c in cases))

    def test_fits_never_receive_held_labels_or_critic_scores(self):
        d = diagnostic
        source, labels, _ = cross_fixture()
        fits, held = d.split_batches(source["train_cases"], labels["train_cases"], 0)
        held_cases = held["old"] + held["new"]
        features = torch.stack([d.cv.features(c) for c in held_cases])
        rng = torch.get_rng_state().clone()
        before = copy.deepcopy(fits)
        scores, metadata = d.fit_models(fits, features)
        self.assertEqual(fits, before)
        for c in held_cases:
            c["branches"] = c["predictions"] = None
        for c in fits["combined96"]:
            c["predictions"] = None
        repeated, repeated_meta = d.fit_models(fits, torch.stack([d.cv.features(c) for c in held_cases]))
        for name in d.MODELS:
            np.testing.assert_array_equal(scores[name], repeated[name])
            self.assertTrue((scores[name][:, 0] == 0.).all())
        self.assertEqual(metadata, repeated_meta)
        torch.testing.assert_close(torch.get_rng_state(), rng, rtol=0, atol=0)

    def test_runner_and_summary_preserve_both_grids_and_cached_endpoints(self):
        d = diagnostic
        source, labels, expanded = cross_fixture()
        with (tempfile.TemporaryDirectory() as directory,
              patch.object(d.previous, "collect_labels", side_effect=AssertionError("No collection")),
              patch.object(d.p.training, "make_piecewise_mo_env", side_effect=AssertionError("No environments")),
              patch.object(torch, "load", side_effect=AssertionError("No checkpoints"))):
            root = Path(directory)
            for name, data in (("source/v27/seed31.json", source), ("expansion/v27/seed31_labels.json", labels),
                               ("expansion/v27/seed31.json", expanded)):
                path = root / name
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text(json.dumps(data))
            result = d.run("v27", 31, root / "source", root / "expansion", root / "out")
            self.assertEqual(summary.validate_run(result, expanded, source, labels, "v27", 31), result["summary"])
            self.assertEqual((result["ridge_solves"], len(result["rows"])), (12, 384))
            self.assertFalse(result["previous_all_control_pass"])
            self.assertNotIn("group_passes", result["summary"])
            for batch in d.BATCHES:
                self.assertEqual(len(result["summary"]["batches"][batch]["cells"]), 12)
                for name, values in result["summary"]["batches"][batch]["aggregate"].items():
                    self.assertEqual(sum(values["selection_counts"].values()), 192)
            for kind in ("batch", "grid", "fold", "endpoint", "reference", "budget", "summary"):
                bad = copy.deepcopy(result)
                if kind == "batch":
                    bad["rows"][0]["batch"] = "new"
                elif kind == "grid":
                    bad["rows"][0] = copy.deepcopy(bad["rows"][1])
                elif kind == "fold":
                    bad["folds"][0]["fits"]["new48"]["fit_reset_ids"] = bad["folds"][0]["held_reset_ids"]["new"]
                elif kind == "endpoint":
                    bad["rows"][0]["scores"]["combined96"]["raw_minus"] += .001
                elif kind == "reference":
                    bad["rows"][0]["scores"]["soft"]["raw_minus"] += .001
                elif kind == "budget":
                    bad["ridge_solves"] += 1
                else:
                    bad["summary"]["batches"]["new"]["aggregate"]["old48"]["mean_gain"] += 1.
                with self.subTest(kind=kind), self.assertRaises(AssertionError):
                    summary.validate_run(bad, expanded, source, labels, "v27", 31)

    def test_metrics_keep_downside_and_use_nonbase_errors_without_ensembling(self):
        d = diagnostic
        source, _, _ = cross_fixture()
        cases = copy.deepcopy(source["train_cases"][:2])
        for c, value in zip(cases, (-3., 5.)):
            c["branches"]["soft_plus"]["discounted_returns"]["512"] = [value, value]
        predictions = {name: np.array([0., 2., 0., 1., 0.]) for name in d.MODELS}
        rows = [d.decision_row(c, "old", predictions) for c in cases]
        values = d.metrics(rows)["old48"]
        self.assertEqual(values["mean_gain"], 1.)
        self.assertEqual(values["mean_negative_part"], -1.5)
        self.assertAlmostEqual(values["rmse512"], np.sqrt((25 + 9 + .25 + .25) / 8))
        choices = [r["choices"] for r in rows]
        for c in cases:
            c["branches"]["soft_plus"]["discounted_returns"]["512"] = [100., 100.]
        self.assertEqual([d.decision_row(c, "old", predictions)["choices"] for c in cases], choices)


if __name__ == "__main__":
    unittest.main()
