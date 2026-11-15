import copy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import torch

import diagnose_ridge_transform as diagnostic
import summarize_ridge_transform as summary
from test_ridge_expansion import fixture


def transform_fixture():
    d = diagnostic
    source, labels = fixture()
    source["validation_cases"] = source["summary"] = None
    for batch, cases in enumerate((source["train_cases"], labels["train_cases"])):
        for c in cases:
            c["anchor_observation"][0] = .2 * c["repeat"] + 5. * batch
    rows = []
    for fold in range(4):
        fits, held = d.previous.split_batches(source["train_cases"], labels["train_cases"], fold)
        cases = [(batch, c) for batch in d.previous.BATCHES for c in held[batch]]
        scores, _ = d.previous.fit_models(fits, torch.stack([d.cv.features(c) for _, c in cases]))
        rows.extend(d.previous.decision_row(c, batch, {name: values[i] for name, values in scores.items()})
                    for i, (batch, c) in enumerate(cases))
    cached = {"protocol": d.previous.PROTOCOL, "condition": "v27", "seed": 31, "complete": True,
        "source_split": "train_cases", "previous_all_control_pass": False, "rows": rows,
        "summary": d.previous.summarize_rows(rows)}
    return source, labels, cached


class RidgeTransformTests(unittest.TestCase):
    def test_combined_transform_reproduces_original_solver(self):
        d = diagnostic
        source, labels = fixture()
        fits, _ = d.previous.split_batches(source["train_cases"], labels["train_cases"], 0)
        data = d.p.training_data(fits["combined96"])
        original = d.cv.QuadraticRidge(data)
        transplanted = d.FixedTransformRidge(data, data["inputs"])
        torch.testing.assert_close(transplanted.beta, original.beta, rtol=0, atol=0)
        torch.testing.assert_close(transplanted(data["inputs"]), original(data["inputs"]), rtol=0, atol=0)

    def test_donor_controls_only_feature_transform_not_combined_target_scale(self):
        d = diagnostic
        rng = torch.Generator().manual_seed(5)
        inputs = torch.randn(48, 8, generator=rng)
        targets = torch.randn(48, generator=rng) * 3.
        donor = inputs[:20].clone()
        donor[:, :-2] += 10.
        donor[:, -2:] *= .1
        data = {"inputs": inputs, "targets": targets, "action_dim": 2}
        before = copy.deepcopy(data)
        model = d.FixedTransformRidge(data, donor)
        expected = d.cv.QuadraticFeatures(donor, 2)
        for key in ("state_mean", "state_scale", "delta_scale", "feature_scale"):
            torch.testing.assert_close(getattr(model, key), getattr(expected, key), rtol=0, atol=0)
        torch.testing.assert_close(model.target_scale, targets.double().std(unbiased=False), rtol=0, atol=0)
        self.assertFalse(torch.equal(model.state_mean, inputs[:, :-2].double().mean(0)))
        base = inputs.clone()
        base[:, -2:] = 0.
        self.assertTrue((model(base) == 0.).all())
        for key in ("inputs", "targets"):
            torch.testing.assert_close(data[key], before[key], rtol=0, atol=0)

    def test_fit_ignores_both_held_labels_and_critic_scores(self):
        d = diagnostic
        source, labels, _ = transform_fixture()
        fits, held = d.previous.split_batches(source["train_cases"], labels["train_cases"], 0)
        held_cases = held["old"] + held["new"]
        before = copy.deepcopy(fits)
        rng = torch.get_rng_state().clone()
        scores, meta = d.fit_models(fits, torch.stack([d.cv.features(c) for c in held_cases]))
        self.assertEqual(before, fits)
        for c in held_cases:
            c["branches"] = c["predictions"] = None
        for c in fits["combined96"]:
            c["predictions"] = None
            c["branches"]["base"]["discounted_returns"]["128"] = None
        repeated, repeated_meta = d.fit_models(fits, torch.stack([d.cv.features(c) for c in held_cases]))
        for name in d.DONORS:
            np.testing.assert_array_equal(scores[name], repeated[name])
            self.assertTrue((scores[name][:, 0] == 0.).all())
        self.assertEqual(meta, repeated_meta)
        torch.testing.assert_close(torch.get_rng_state(), rng, rtol=0, atol=0)

    def test_runner_preserves_cached_controls_and_validates_donor_sets(self):
        d = diagnostic
        source, labels, cached = transform_fixture()
        cached_before = copy.deepcopy(cached)
        with (tempfile.TemporaryDirectory() as directory,
              patch.object(d.previous, "fit_models", side_effect=AssertionError("No cached-control refits")),
              patch.object(d.previous.previous, "collect_labels", side_effect=AssertionError("No collection")),
              patch.object(d.p.training, "make_piecewise_mo_env", side_effect=AssertionError("No environments")),
              patch.object(torch, "load", side_effect=AssertionError("No checkpoints"))):
            root = Path(directory)
            for name, data in (("source/v27/seed31.json", source), ("expansion/v27/seed31_labels.json", labels),
                               ("reference/v27/seed31.json", cached)):
                path = root / name
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text(json.dumps(data))
            result = d.run("v27", 31, root / "source", root / "expansion", root / "reference", root / "out")
            self.assertEqual(summary.validate_run(result, cached, source, labels, "v27", 31), result["summary"])
            self.assertEqual((result["ridge_solves"], len(result["rows"])), (8, 384))
            self.assertEqual(cached, cached_before)
            compact = summary.compact_result(result["summary"])
            self.assertNotIn("selection_counts", compact["batches"]["old"]["cells"][0]["metrics"]["combined96"])
            self.assertIn("selection_counts", result["summary"]["batches"]["old"]["cells"][0]["metrics"]["combined96"])
            for kind in ("donor", "target_scale", "control", "grid", "budget", "summary"):
                bad = copy.deepcopy(result)
                if kind == "donor":
                    bad["folds"][0]["fits"]["combined_old_transform"]["transform_reset_ids"] = bad["folds"][0]["held_reset_ids"]["old"]
                elif kind == "target_scale":
                    bad["folds"][0]["fits"]["combined_old_transform"]["target_scale"] *= 2.
                elif kind == "control":
                    bad["rows"][0]["scores"]["old48"]["raw_minus"] += .001
                elif kind == "grid":
                    bad["rows"][0] = copy.deepcopy(bad["rows"][1])
                elif kind == "budget":
                    bad["ridge_solves"] += 1
                else:
                    bad["summary"]["batches"]["new"]["aggregate"]["combined_old_transform"]["mean_gain"] += 1.
                with self.subTest(kind=kind), self.assertRaises(AssertionError):
                    summary.validate_run(bad, cached, source, labels, "v27", 31)


if __name__ == "__main__":
    unittest.main()
