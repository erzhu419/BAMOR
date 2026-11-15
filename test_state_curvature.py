import copy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import torch

import diagnose_state_curvature as diagnostic
import summarize_state_curvature as summary
from test_weak_ridge import cached_fixture as prefix_fixture
from test_paired_advantage_summary import terminal_fixture


def cached_fixture(cases):
    d = diagnostic
    old = prefix_fixture(cases)
    old["protocol"] = d.previous.PROTOCOL
    old["rows"] = [d.reference.join_row(c, r, {"weak_ridge": np.zeros(5)}, selectors=d.previous.SELECTORS)
                   for c, r in zip(cases, old["rows"])]
    return old


class StateCurvatureTests(unittest.TestCase):
    def test_old_features_are_preserved_and_new_block_is_even_and_base_zero(self):
        d = diagnostic
        cases = terminal_fixture()["train_cases"][:8]
        for i, c in enumerate(cases):
            c["anchor_observation"] = [(i + 1.) * (j + 1.) for j in range(17)]
        data = d.p.training_data(cases)
        old = d.cv.QuadraticRidge(data)
        new = d.StateCurvatureRidge(data, penalty=1.)
        x = data["inputs"]
        torch.testing.assert_close(new.transform(x)[:, :243], old.transform(x), rtol=1e-12, atol=1e-12)
        self.assertEqual((len(old.beta), len(new.beta)), (243, 279))
        reversed_x = x.clone()
        reversed_x[:, -6:] *= -1
        torch.testing.assert_close(new.transform(x)[:, 243:], new.transform(reversed_x)[:, 243:], rtol=0, atol=0)
        base = x.clone()
        base[:, -6:] = 0.
        self.assertTrue((new.transform(base) == 0.).all())
        self.assertTrue((new(base) == 0.).all())

    def test_new_term_can_fit_state_dependent_even_response(self):
        d = diagnostic
        x = torch.tensor([[-1., 1.], [-1., -1.], [1., 1.], [1., -1.]], dtype=torch.float32)
        y = x[:, 0] * x[:, 1].square()
        data = {"inputs": x, "targets": y, "action_dim": 1}
        old, new = d.cv.QuadraticRidge(data), d.StateCurvatureRidge(data, penalty=1.)
        torch.testing.assert_close(old(x), torch.zeros(4, dtype=torch.float64), rtol=0, atol=1e-12)
        torch.testing.assert_close(new(x), y.double() / 2, rtol=0, atol=1e-12)

    def test_fit_uses_only_fitting_labels_and_preserves_inputs(self):
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
        self.assertEqual(meta, repeated_meta)
        self.assertEqual((meta["parameters"], meta["extra_parameters"], meta["penalty"]), (279, 36, 1.))
        torch.testing.assert_close(torch.get_rng_state(), rng, rtol=0, atol=0)

    def test_runner_and_summary_preserve_cached_controls_and_protocol(self):
        d = diagnostic
        source = terminal_fixture()
        source["validation_cases"] = source["summary"] = None
        previous = cached_fixture(source["train_cases"])
        with (tempfile.TemporaryDirectory() as directory,
              patch.object(d.p.training, "make_piecewise_mo_env", side_effect=AssertionError("No rollouts")),
              patch.object(d.cv, "fit_fold", side_effect=AssertionError("No control refit")),
              patch.object(d.previous, "fit_predict", side_effect=AssertionError("No weak ridge refit")),
              patch.object(torch, "load", side_effect=AssertionError("No checkpoints"))):
            root = Path(directory)
            for name, data in (("source", source), ("previous", previous)):
                path = root / name / "v27/seed31.json"
                path.parent.mkdir(parents=True)
                path.write_text(json.dumps(data))
            result = d.run("v27", 31, root / "source", root / "previous", root / "out")
            self.assertEqual(summary.validate_run(result, previous, source, "v27", 31), result["summary"])
            self.assertEqual((result["ridge_solves"], result["selector_updates"], result["new_environment_interactions"]), (4, 0, 0))
            for kind in ("grid", "budget", "control", "split", "penalty", "features", "gate"):
                changed = copy.deepcopy(result)
                if kind == "grid":
                    changed["rows"][0] = copy.deepcopy(changed["rows"][1])
                elif kind == "budget":
                    changed["ridge_solves"] += 1
                elif kind == "control":
                    changed["rows"][0]["scores"]["weak_ridge"]["soft_plus"] += 1.
                elif kind == "split":
                    changed["folds"][0]["fit_reset_ids"] = changed["folds"][0]["held_reset_ids"]
                elif kind == "penalty":
                    changed["folds"][0]["penalty"] = .1
                elif kind == "features":
                    changed["folds"][0]["extra_parameters"] = 0
                else:
                    changed["summary"]["group_passes"] = not result["summary"]["group_passes"]
                with self.subTest(kind=kind), self.assertRaises(AssertionError):
                    summary.validate_run(changed, previous, source, "v27", 31)

    def test_gain_does_not_override_downside_and_calibration_units_stay_separate(self):
        d = diagnostic
        cases = terminal_fixture()["train_cases"]
        for c in cases:
            value = -4. if c["repeat"] == 0 else 1.
            c["branches"]["soft_plus"]["discounted_returns"]["512"] = [value, value]
        old = cached_fixture(cases)
        rows = [d.reference.join_row(c, r, {"state_curvature": np.array([0., 2., 0., 1., 0.])}, selectors=d.SELECTORS)
                for c, r in zip(cases, old["rows"])]
        result = d.summarize_rows(rows)
        self.assertTrue(result["checks"]["positive_mean_gain"])
        self.assertTrue(result["checks"]["mean_not_worse_than_controls"])
        self.assertFalse(result["checks"]["downside_not_worse_than_controls"])
        self.assertFalse(result["group_passes"])
        self.assertIn("state_curvature", result["calibration"])
        self.assertFalse({"ridge128", "margin", "cost_sensitive"}.intersection(result["calibration"]))


if __name__ == "__main__":
    unittest.main()
