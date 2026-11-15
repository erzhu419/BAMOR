import copy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import torch

import diagnose_weak_ridge as diagnostic
import summarize_weak_ridge as summary
from test_horizon_attribution import decisions_fixture
from test_paired_advantage_summary import terminal_fixture


def cached_fixture(cases):
    d = diagnostic
    previous = decisions_fixture(cases)
    previous["protocol"] = d.previous.PROTOCOL
    previous["rows"] = [d.previous.decision_row(c, r, np.zeros(5)) for c, r in zip(cases, previous["rows"])]
    return previous


class WeakRidgeTests(unittest.TestCase):
    def test_default_solver_is_unchanged_and_weaker_penalty_reduces_fit_error(self):
        d = diagnostic
        data = d.p.training_data(terminal_fixture()["train_cases"][:8])
        original = d.cv.QuadraticRidge(data)
        phi = original.transform(data["inputs"])
        y = data["targets"].double() / original.target_scale
        beta = torch.linalg.solve(phi.T @ phi / len(phi) + torch.eye(phi.shape[1], dtype=torch.float64),
                                  phi.T @ y / len(phi))
        torch.testing.assert_close(original.beta, beta, rtol=0, atol=0)
        weak = d.cv.QuadraticRidge(data, penalty=d.PENALTY)
        self.assertEqual(d.cv.RIDGE_PENALTY, 1.)
        torch.testing.assert_close(original.transform(data["inputs"]), weak.transform(data["inputs"]), rtol=0, atol=0)
        old_error = (original(data["inputs"]) - data["targets"]).square().mean()
        new_error = (weak(data["inputs"]) - data["targets"]).square().mean()
        self.assertLess(float(new_error), float(old_error))
        self.assertFalse(torch.equal(original.beta, weak.beta))

    def test_fit_uses_512_only_and_never_receives_held_labels(self):
        d = diagnostic
        fit, held = d.cv.split_fold(terminal_fixture()["train_cases"], 0)
        before = copy.deepcopy(fit)
        rng = torch.get_rng_state().clone()
        scores, meta = d.fit_predict(fit, torch.stack([d.cv.features(c) for c in held]))
        self.assertEqual(fit, before)
        for c in fit:
            c["predictions"] = None
            for branch in c["branches"].values():
                branch["discounted_returns"]["128"] = None
        for c in held:
            c["branches"] = c["predictions"] = None
        repeated, repeated_meta = d.fit_predict(fit, torch.stack([d.cv.features(c) for c in held]))
        np.testing.assert_array_equal(scores, repeated)
        self.assertTrue((scores[:, 0] == 0.).all())
        self.assertEqual(meta, repeated_meta)
        self.assertEqual((meta["train_horizon"], meta["penalty"]), (512, .1))
        torch.testing.assert_close(torch.get_rng_state(), rng, rtol=0, atol=0)

    def test_runner_retains_controls_budget_and_frozen_penalty(self):
        d = diagnostic
        source = terminal_fixture()
        source["validation_cases"] = source["summary"] = None
        previous = cached_fixture(source["train_cases"])
        with (tempfile.TemporaryDirectory() as directory,
              patch.object(d.p.training, "make_piecewise_mo_env", side_effect=AssertionError("No rollouts")),
              patch.object(d.cv, "fit_fold", side_effect=AssertionError("No control refit")),
              patch.object(d.previous, "fit_predict", side_effect=AssertionError("No prefix refit")),
              patch.object(torch, "load", side_effect=AssertionError("No checkpoints"))):
            root = Path(directory)
            for name, data in (("source", source), ("previous", previous)):
                path = root / name / "v27/seed31.json"
                path.parent.mkdir(parents=True)
                path.write_text(json.dumps(data))
            result = d.run("v27", 31, root / "source", root / "previous", root / "out")
            self.assertEqual(summary.validate_run(result, previous, source, "v27", 31), result["summary"])
            self.assertEqual((result["ridge_solves"], result["selector_updates"], result["new_environment_interactions"]), (4, 0, 0))
            for kind in ("grid", "budget", "control", "split", "penalty", "horizon", "gate"):
                changed = copy.deepcopy(result)
                if kind == "grid":
                    changed["rows"][0] = copy.deepcopy(changed["rows"][1])
                elif kind == "budget":
                    changed["ridge_solves"] += 1
                elif kind == "control":
                    changed["rows"][0]["scores"]["ridge128"]["soft_plus"] += 1.
                elif kind == "split":
                    changed["folds"][0]["fit_reset_ids"] = changed["folds"][0]["held_reset_ids"]
                elif kind == "penalty":
                    changed["folds"][0]["penalty"] = 1.
                elif kind == "horizon":
                    changed["folds"][0]["train_horizon"] = 128
                else:
                    changed["summary"]["group_passes"] = not result["summary"]["group_passes"]
                with self.subTest(kind=kind), self.assertRaises(AssertionError):
                    summary.validate_run(changed, previous, source, "v27", 31)

    def test_positive_gain_does_not_override_downside_or_mix_score_units(self):
        d = diagnostic
        cases = terminal_fixture()["train_cases"]
        for c in cases:
            value = -4. if c["repeat"] == 0 else 1.
            c["branches"]["soft_plus"]["discounted_returns"]["512"] = [value, value]
        old = cached_fixture(cases)
        rows = [d.reference.join_row(c, r, {"weak_ridge": np.array([0., 2., 0., 1., 0.])}, selectors=d.SELECTORS)
                for c, r in zip(cases, old["rows"])]
        result = d.summarize_rows(rows)
        self.assertTrue(result["checks"]["positive_mean_gain"])
        self.assertTrue(result["checks"]["mean_not_worse_than_controls"])
        self.assertFalse(result["checks"]["downside_not_worse_than_controls"])
        self.assertFalse(result["group_passes"])
        self.assertIn("weak_ridge", result["calibration"])
        self.assertFalse({"ridge128", "margin", "cost_sensitive"}.intersection(result["calibration"]))


if __name__ == "__main__":
    unittest.main()
