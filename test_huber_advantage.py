import copy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import torch

import diagnose_huber_advantage as d
import summarize_huber_advantage as summary
from test_ridge_transform import transform_fixture


def fixture():
    source, labels, cached = transform_fixture()
    rows = []
    for r in cached["rows"]:
        if r["batch"] != "old":
            continue
        mapping = {name: {"ridge": "old48", "ridge96": "combined96"}.get(name, name if name in r["scores"] else "soft")
                   for name in d.expansion.SELECTORS}
        rows.append({**r, **{field: {name: copy.deepcopy(r[field][original]) for name, original in mapping.items()}
                            for field in ("scores", "choices", "selected_gains")}})
    legacy = {"complete": True, "protocol": d.expansion.PROTOCOL, "condition": "v27", "seed": 31,
        "source_split": "train_cases", "rows": rows, "summary": d.expansion.summarize_rows(rows)}
    return source, labels, cached, legacy


class HuberAdvantageTests(unittest.TestCase):
    def test_quadratic_region_has_exact_ridge_penalty(self):
        rng = torch.Generator().manual_seed(8)
        x = torch.randn(80, 7, generator=rng, dtype=torch.float64) * .2
        y = torch.randn(80, generator=rng, dtype=torch.float64) * .01
        beta, weights, meta = d.solve_huber(x, y)
        ridge = torch.linalg.solve(x.T @ x / len(x) + torch.eye(7), x.T @ y / len(x))
        torch.testing.assert_close(beta, ridge, rtol=1e-12, atol=1e-12)
        self.assertTrue((weights == 1.).all())
        self.assertEqual(meta["linear_solves"], 1)

    def test_outlier_fit_matches_autograd_stationarity_and_keeps_labels(self):
        rng = torch.Generator().manual_seed(17)
        x = torch.randn(100, 7, generator=rng, dtype=torch.float64)
        y = .2 * x[:, 0]
        y[0] = 100.
        before = y.clone()
        beta, weights, meta = d.solve_huber(x, y)
        b = beta.clone().requires_grad_()
        objective = torch.nn.functional.huber_loss(x @ b, y, delta=1.) + .5 * b.square().sum()
        grad = torch.autograd.grad(objective, b)[0]
        self.assertLessEqual(float(grad.abs().max()), d.GRAD_TOL)
        self.assertAlmostEqual(float(grad.abs().max()), meta["stationarity_inf"], places=14)
        self.assertTrue((weights > 0.).all())
        self.assertLess(float(weights[0]), .02)
        torch.testing.assert_close(y, before, rtol=0, atol=0)
        with patch.object(d, "MAX_ITERATIONS", 1), self.assertRaisesRegex(RuntimeError, "did not converge"):
            d.solve_huber(x, y)

    def test_same_features_scale_and_no_held_label_or_critic_input(self):
        source, labels, _, _ = fixture()
        fits, held = d.previous.split_batches(source["train_cases"], labels["train_cases"], 0)
        data = d.p.training_data(fits["combined96"])
        huber, ridge = d.HuberRidge(data), d.cv.QuadraticRidge(data)
        for key in ("state_mean", "state_scale", "delta_scale", "feature_scale", "target_scale"):
            torch.testing.assert_close(getattr(huber, key), getattr(ridge, key), rtol=0, atol=0)
        cases = held["old"] + held["new"]
        rng = torch.get_rng_state().clone()
        before = copy.deepcopy(fits)
        scores, meta = d.fit_predict(fits["combined96"], torch.stack([d.cv.features(c) for c in cases]))
        self.assertEqual(before, fits)
        for c in cases:
            c["branches"] = c["predictions"] = None
        for c in fits["combined96"]:
            c["predictions"] = None
            c["branches"]["base"]["discounted_returns"]["128"] = None
        repeated, repeated_meta = d.fit_predict(fits["combined96"], torch.stack([d.cv.features(c) for c in cases]))
        np.testing.assert_array_equal(scores, repeated)
        self.assertEqual(meta, repeated_meta)
        self.assertTrue((scores[:, 0] == 0.).all())
        torch.testing.assert_close(torch.get_rng_state(), rng, rtol=0, atol=0)

    def test_acceptance_keeps_downside_old_controls_and_both_batches(self):
        _, _, cached, legacy = fixture()
        rows = [d.join_row(r, np.array([0., 1., 0., 0., 0.])) for r in cached["rows"]]
        old = legacy["summary"]["aggregate"]
        result = d.summarize_rows(rows, old)
        harder = copy.deepcopy(old)
        harder[d.expansion.CONTROLS[-1]]["mean_gain"] = 1e5
        self.assertFalse(d.summarize_rows(rows, harder)["old_primary_checks"]["mean_not_worse_than_controls"])
        negative = {"mean_gain": 2., "mean_negative_part": -.5}
        self.assertFalse(d.gate(negative, [{"mean_gain": 1., "mean_negative_part": -.1}])["downside_not_worse_than_controls"])
        for r in rows:
            if r["batch"] == "new":
                r["selected_gains"][d.TARGET] = -1.
        changed = d.summarize_rows(rows, old)
        self.assertEqual(changed["old_primary_checks"], result["old_primary_checks"])
        self.assertFalse(changed["batch_checks"]["new"]["positive_mean_gain"])
        self.assertFalse(changed["group_passes"])

    def test_runner_preserves_controls_and_validates_solver_protocol(self):
        source, labels, cached, legacy = fixture()
        with (tempfile.TemporaryDirectory() as directory,
              patch.object(d.previous, "fit_models", side_effect=AssertionError("No control refits")),
              patch.object(d.expansion, "collect_labels", side_effect=AssertionError("No collection")),
              patch.object(d.p.training, "make_piecewise_mo_env", side_effect=AssertionError("No environments")),
              patch.object(torch, "load", side_effect=AssertionError("No checkpoints"))):
            root = Path(directory)
            for name, data in (("source/v27/seed31.json", source), ("expansion/v27/seed31_labels.json", labels),
                               ("expansion/v27/seed31.json", legacy), ("reference/v27/seed31.json", cached)):
                path = root / name
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text(json.dumps(data))
            result = d.run("v27", 31, root / "source", root / "expansion", root / "reference", root / "out")
            compact = summary.validate_run(result, cached, legacy, source, labels, "v27", 31)
            self.assertEqual(compact["summary"]["group_passes"], result["summary"]["group_passes"])
            for kind in ("fold", "scale", "convergence", "delta", "budget", "control", "grid", "gate"):
                bad = copy.deepcopy(result)
                if kind == "fold":
                    bad["folds"][0]["fit_reset_ids"] = bad["folds"][0]["held_reset_ids"]["old"]
                elif kind == "scale":
                    bad["folds"][0]["target_scale"] *= 2.
                elif kind == "convergence":
                    bad["folds"][0]["stationarity_inf"] = .1
                elif kind == "delta":
                    bad["delta"] = .5
                elif kind == "budget":
                    bad["linear_solves"] += 1
                elif kind == "control":
                    bad["rows"][0]["scores"]["old48"]["soft_plus"] += .001
                elif kind == "grid":
                    bad["rows"][0] = copy.deepcopy(bad["rows"][1])
                else:
                    bad["summary"]["group_passes"] = not bad["summary"]["group_passes"]
                with self.subTest(kind=kind), self.assertRaises(AssertionError):
                    summary.validate_run(bad, cached, legacy, source, labels, "v27", 31)


if __name__ == "__main__":
    unittest.main()
