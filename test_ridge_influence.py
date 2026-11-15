import copy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import torch

import diagnose_ridge_influence as d
import summarize_ridge_influence as summary
from test_ridge_transform import transform_fixture


class RidgeInfluenceTests(unittest.TestCase):
    def test_exact_identity_includes_penalty_and_label_derivatives(self):
        rng = torch.Generator().manual_seed(42)
        x = torch.randn(60, 7, generator=rng, dtype=torch.float64)
        y = torch.randn(60, generator=rng, dtype=torch.float64)
        old, combined, label, feedback, ridge = d.solve_decomposition(x[:36], y[:36], x[36:], y[36:])
        torch.testing.assert_close(label.sum(1) + feedback.sum(1) + ridge[:, 0], combined - old, rtol=1e-12, atol=1e-12)
        self.assertGreater(float(ridge.norm()), 1e-3)
        self.assertGreater(float((label.sum(1) + feedback.sum(1) - (combined - old)).norm()), 1e-3)
        changed = y.clone()
        changed[36:40] *= 1.001
        _, perturbed, *_ = d.solve_decomposition(x[:36], changed[:36], x[36:], changed[36:])
        torch.testing.assert_close(perturbed - combined, .001 * label[:, 0], rtol=1e-9, atol=1e-12)

    def test_same_batch_cancels_and_zero_response_has_no_ratio(self):
        rng = torch.Generator().manual_seed(9)
        x = torch.randn(24, 7, generator=rng, dtype=torch.float64)
        y = torch.randn(24, generator=rng, dtype=torch.float64)
        old, combined, label, feedback, ridge = d.solve_decomposition(x, y, x, y)
        torch.testing.assert_close(old, combined, rtol=1e-12, atol=1e-12)
        torch.testing.assert_close((label + feedback).sum(1), -ridge[:, 0], rtol=1e-12, atol=1e-12)
        self.assertIsNone(d.ratio(0., 0.))

    def test_reproduces_combined_and_never_reads_held_labels(self):
        source, labels, _ = transform_fixture()
        fits, held = d.previous.split_batches(source["train_cases"], labels["train_cases"], 0)
        cases = held["old"] + held["new"]
        features = torch.stack([d.cv.features(c) for c in cases])
        before = copy.deepcopy(fits)
        result = d.decompose(fits, features)
        model = d.cv.QuadraticRidge(d.p.training_data(fits["combined96"]))
        np.testing.assert_allclose(result["scores"]["combined96"], model(features).numpy(), rtol=1e-10, atol=1e-10)
        self.assertEqual(before, fits)
        for c in cases:
            c["branches"] = c["predictions"] = None
        for c in fits["combined96"]:
            c["predictions"] = None
        repeated = d.decompose(fits, torch.stack([d.cv.features(c) for c in cases]))
        for key in ("parts", "labels", "feedback", "ridge", "held"):
            torch.testing.assert_close(result[key], repeated[key], rtol=0, atol=0)

    def test_runner_preserves_controls_and_summary_checks_attribution(self):
        source, labels, cached = transform_fixture()
        with (tempfile.TemporaryDirectory() as directory,
              patch.object(d.previous.previous, "collect_labels", side_effect=AssertionError("No collection")),
              patch.object(d.p.training, "make_piecewise_mo_env", side_effect=AssertionError("No environment")),
              patch.object(torch, "load", side_effect=AssertionError("No checkpoint"))):
            root = Path(directory)
            for name, value in (("source/v27/seed31.json", source), ("expansion/v27/seed31_labels.json", labels),
                                ("reference/v27/seed31.json", cached)):
                path = root / name
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text(json.dumps(value))
            result = d.run("v27", 31, root / "source", root / "expansion", root / "reference", root / "out")
            compact = summary.validate_run(result, cached, source, labels, "v27", 31)
            self.assertEqual((result["model_solves"], result["attribution_solves"]), (8, 4))
            self.assertEqual(len(compact["folds"][0]["influence"]["old"]["top5_resets"]), 5)
            self.assertIn("resets", result["folds"][0])
            for kind in ("reset", "ridge", "control", "grid", "budget"):
                bad = copy.deepcopy(result)
                if kind == "reset":
                    bad["folds"][0]["resets"][0]["reset_seed"] = bad["folds"][0]["held_reset_ids"]["new"][0]
                elif kind == "ridge":
                    bad["folds"][0]["components"]["ridge"]["batches"]["old"]["mean_switch_margin"] += 1.
                elif kind == "control":
                    bad["rows"][0]["scores"]["combined96"]["raw_minus"] += .1
                elif kind == "grid":
                    bad["rows"][0] = copy.deepcopy(bad["rows"][1])
                else:
                    bad["attribution_solves"] += 1
                with self.subTest(kind=kind), self.assertRaises(AssertionError):
                    summary.validate_run(bad, cached, source, labels, "v27", 31)


if __name__ == "__main__":
    unittest.main()
