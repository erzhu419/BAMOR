import copy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import torch

import diagnose_prefix_ridge as diagnostic
import summarize_prefix_ridge as summary
from test_horizon_attribution import decisions_fixture
from test_paired_advantage_summary import terminal_fixture


class PrefixRidgeTests(unittest.TestCase):
    def test_same_features_solver_and_dtype_as_original_ridge(self):
        d = diagnostic
        cases = terminal_fixture()["train_cases"][:8]
        original, prefix = d.p.training_data(cases), d.training_data(cases)
        for key in ("inputs", "targets"):
            torch.testing.assert_close(original[key], prefix[key], rtol=0, atol=0)
        old, new = d.cv.QuadraticRidge(original), d.cv.QuadraticRidge(prefix)
        torch.testing.assert_close(old.beta, new.beta, rtol=0, atol=0)
        self.assertEqual((new.beta.numel(), new.beta.dtype), (243, torch.float64))
        self.assertTrue((new(torch.stack([d.cv.features(c) for c in cases]))[:, 0] == 0.).all())

    def test_fit_ignores_512_labels_critic_predictions_and_all_held_labels(self):
        d = diagnostic
        fit, held = d.cv.split_fold(terminal_fixture()["train_cases"], 0)
        rng = torch.get_rng_state().clone()
        before = copy.deepcopy(fit)
        inputs = torch.stack([d.cv.features(c) for c in held])
        scores, meta = d.fit_predict(fit, inputs)
        self.assertEqual(fit, before)
        for c in fit:
            c["predictions"] = None
            for branch in c["branches"].values():
                branch["discounted_returns"]["512"] = None
        for c in held:
            c["branches"] = c["predictions"] = None
        repeated, repeated_meta = d.fit_predict(fit, torch.stack([d.cv.features(c) for c in held]))
        np.testing.assert_array_equal(scores, repeated)
        self.assertEqual(meta, repeated_meta)
        torch.testing.assert_close(torch.get_rng_state(), rng, rtol=0, atol=0)
        self.assertEqual((meta["fit_anchors"], meta["fit_nonbase_rows"]), (144, 576))

    def test_runner_budget_controls_horizon_and_pairing(self):
        d = diagnostic
        source = terminal_fixture()
        source["validation_cases"] = source["summary"] = None
        for c in source["train_cases"]:
            c["branches"]["soft_plus"]["discounted_returns"]["128"] = [2., 2.]
        previous = decisions_fixture(source["train_cases"])
        with (tempfile.TemporaryDirectory() as directory,
              patch.object(d.p.training, "make_piecewise_mo_env", side_effect=AssertionError("No rollouts")),
              patch.object(d.cv, "fit_fold", side_effect=AssertionError("No control refit")),
              patch.object(d.previous, "fit_predict", side_effect=AssertionError("No margin refit")),
              patch.object(torch, "load", side_effect=AssertionError("No checkpoints"))):
            root = Path(directory)
            for name, data in (("source", source), ("previous", previous)):
                path = root / name / "v27/seed31.json"
                path.parent.mkdir(parents=True)
                path.write_text(json.dumps(data))
            result = d.run("v27", 31, root / "source", root / "previous", root / "out")
            self.assertEqual(summary.validate_run(result, previous, source, "v27", 31), result["summary"])
            self.assertEqual((result["ridge_solves"], result["selector_updates"], result["new_environment_interactions"]), (4, 0, 0))
            for kind in ("grid", "budget", "control", "split", "horizon", "prefix", "gate"):
                changed = copy.deepcopy(result)
                if kind == "grid":
                    changed["rows"][0] = copy.deepcopy(changed["rows"][1])
                elif kind == "budget":
                    changed["ridge_solves"] += 1
                elif kind == "control":
                    changed["rows"][0]["scores"]["margin"]["soft_plus"] += 1.
                elif kind == "split":
                    changed["folds"][0]["fit_reset_ids"] = changed["folds"][0]["held_reset_ids"]
                elif kind == "horizon":
                    changed["folds"][0]["train_horizon"] = 512
                elif kind == "prefix":
                    changed["rows"][0]["prefix_candidate_gains"] = changed["rows"][0]["candidate_gains"]
                else:
                    changed["summary"]["group_passes"] = not result["summary"]["group_passes"]
                with self.subTest(kind=kind), self.assertRaises(AssertionError):
                    summary.validate_run(changed, previous, source, "v27", 31)

    def test_short_horizon_success_does_not_override_long_horizon_downside(self):
        d = diagnostic
        cases = terminal_fixture()["train_cases"]
        for c in cases:
            value = -4. if c["repeat"] == 0 else 1.
            c["branches"]["soft_plus"]["discounted_returns"]["512"] = [value, value]
        previous = decisions_fixture(cases)
        rows = [d.decision_row(c, r, np.array([0., 2., 0., 1., 0.])) for c, r in zip(cases, previous["rows"])]
        result = d.summarize_rows(rows)
        self.assertEqual(result["prefix_diagnostic"]["selectors"]["ridge128"]["mean_negative_part"], 0.)
        self.assertTrue(result["checks"]["positive_mean_gain"])
        self.assertTrue(result["checks"]["mean_not_worse_than_controls"])
        self.assertFalse(result["checks"]["downside_not_worse_than_controls"])
        self.assertFalse(result["group_passes"])
        self.assertNotIn("ridge128", result["calibration"])
        self.assertEqual(result["prefix_diagnostic"]["horizon"], 128)


if __name__ == "__main__":
    unittest.main()
