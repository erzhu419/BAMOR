import copy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import torch

import diagnose_ridge_spectrum as diagnostic
import summarize_ridge_spectrum as summary
from test_paired_advantage_summary import terminal_fixture
from test_residual_advantage import previous_fixture


def reference_fixture(cases):
    d = diagnostic
    old = previous_fixture(cases)
    by_key = {(r["mode"], r["weight_index"], r["repeat"]): r for r in old["rows"]}
    old["folds"] = []
    for fold in range(4):
        fit, held = d.cv.split_fold(cases, fold)
        data = d.p.training_data(fit)
        model = d.cv.QuadraticRidge(data)
        rmse = float((model(data["inputs"]) - data["targets"].double()).square().mean().sqrt())
        old["folds"].append({"fold": fold, "fit_reset_ids": sorted({c["reset_seed"] for c in fit}),
            "held_reset_ids": sorted({c["reset_seed"] for c in held}), "train_rmse": {"ridge": rmse}})
        for c in held:
            row = by_key[(c["mode"], c["weight_index"], c["repeat"])]
            row["scores"]["ridge"] = dict(zip(d.p.CANDIDATES, model(d.cv.features(c)).tolist()))
    return old


class RidgeSpectrumTests(unittest.TestCase):
    def test_orthogonal_and_shrinkage_components_with_known_solution(self):
        beta, stats = diagnostic.decompose(np.array([[2.], [0.]]), np.array([2., 3.]))
        np.testing.assert_allclose(beta, [2. / 3.])
        self.assertAlmostEqual(stats["projection_mse"], 4.5)
        self.assertAlmostEqual(stats["shrinkage_mse"], 2. / 9.)
        self.assertAlmostEqual(stats["ridge_mse"], 4.5 + 2. / 9.)
        self.assertAlmostEqual(stats["effective_df"], 2. / 3.)

    def test_rank_deficiency_and_zero_targets(self):
        phi = np.array([[1., 1.], [-1., -1.], [0., 0.]])
        y = np.array([1., -1., 2.])
        beta, stats = diagnostic.decompose(phi, y)
        direct = np.linalg.solve(phi.T @ phi / 3 + np.eye(2), phi.T @ y / 3)
        np.testing.assert_allclose(beta, direct)
        self.assertEqual(stats["rank"], 1)
        self.assertAlmostEqual(stats["projection_mse"], 4. / 3.)
        _, zero = diagnostic.decompose(phi, np.zeros(3))
        self.assertIsNone(zero["shrinkage_share"])
        self.assertEqual(zero["ridge_mse"], 0.)

    def test_reconstruction_matches_ridge_and_ignores_held_labels(self):
        d = diagnostic
        fit, held = d.cv.split_fold(terminal_fixture()["train_cases"], 0)
        before = copy.deepcopy(fit)
        inputs = torch.stack([d.cv.features(c) for c in held])
        predictions, stats = d.inspect_fold(fit, inputs)
        model = d.cv.QuadraticRidge(d.p.training_data(fit))
        np.testing.assert_allclose(predictions, model(inputs).numpy(), rtol=1e-8, atol=1e-8)
        for c in held:
            c["branches"] = c["predictions"] = None
        repeated, repeated_stats = d.inspect_fold(fit, torch.stack([d.cv.features(c) for c in held]))
        np.testing.assert_array_equal(predictions, repeated)
        self.assertEqual(stats, repeated_stats)
        self.assertEqual(fit, before)

    def test_runner_reconstructs_cached_model_and_preserves_protocol(self):
        d = diagnostic
        source = terminal_fixture()
        source["validation_cases"] = source["summary"] = None
        previous = reference_fixture(source["train_cases"])
        with (tempfile.TemporaryDirectory() as directory,
              patch.object(d.p.training, "make_piecewise_mo_env", side_effect=AssertionError("No rollouts")),
              patch.object(d.cv, "fit_fold", side_effect=AssertionError("No new selectors")),
              patch.object(torch, "load", side_effect=AssertionError("No checkpoints"))):
            root = Path(directory)
            for name, data in (("source", source), ("previous", previous)):
                path = root / name / "v27/seed31.json"
                path.parent.mkdir(parents=True)
                path.write_text(json.dumps(data))
            report = d.run("v27", 31, root / "source", root / "previous", root / "out")
            self.assertEqual(summary.validate_run(report, "v27", 31), report["summary"])
            self.assertEqual((report["svd_decompositions"], report["ridge_reconstructions"]), (4, 4))
            for kind in ("budget", "fold", "penalty", "split", "decomposition", "summary"):
                changed = copy.deepcopy(report)
                if kind == "budget":
                    changed["new_environment_interactions"] = 1
                elif kind == "fold":
                    changed["folds"][0] = copy.deepcopy(changed["folds"][1])
                elif kind == "penalty":
                    changed["folds"][0]["penalty"] = .1
                elif kind == "split":
                    changed["folds"][0]["fit_reset_ids"] = changed["folds"][0]["held_reset_ids"]
                elif kind == "decomposition":
                    changed["folds"][0]["projection_mse"] += 1.
                else:
                    changed["summary"]["projection_share"] += 1.
                with self.subTest(kind=kind), self.assertRaises(AssertionError):
                    summary.validate_run(changed, "v27", 31)
            previous["rows"][0]["scores"]["ridge"]["soft_plus"] += .01
            (root / "previous/v27/seed31.json").write_text(json.dumps(previous))
            with self.assertRaises(AssertionError):
                d.run("v27", 31, root / "source", root / "previous", root / "out2")


if __name__ == "__main__":
    unittest.main()
