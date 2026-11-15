import copy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import torch

import diagnose_advantage_cv as diagnostic
import summarize_advantage_cv as summary
from test_paired_advantage_summary import terminal_fixture


class AdvantageCVTests(unittest.TestCase):
    def test_grouping_keeps_preferences_and_candidates_together(self):
        d = diagnostic
        source = terminal_fixture()
        source["validation_cases"] = "not an input"
        source["summary"] = "not an input"
        cases = d.training_cases(source, "v27", 31)
        seen = []
        for fold in range(4):
            fit, held = d.split_fold(cases, fold)
            self.assertEqual((len(fit), len(held)), (144, 48))
            self.assertEqual(len({c["reset_seed"] for c in held}), 16)
            self.assertTrue({c["reset_seed"] for c in fit}.isdisjoint(c["reset_seed"] for c in held))
            for reset in {c["reset_seed"] for c in held}:
                self.assertEqual({c["weight_index"] for c in held if c["reset_seed"] == reset}, {0, 1, 2})
            seen.extend((c["mode"], c["weight_index"], c["repeat"]) for c in held)
        self.assertEqual(len(set(seen)), 192)

    def test_ridge_is_anchored_regularized_and_can_reject_both_signs(self):
        d = diagnostic
        x = torch.tensor([[s, a] for s in (-1., 1.) for a in (-2., -1., 1., 2.)])
        data = {"inputs": x, "targets": -x[:, -1].square(), "action_dim": 1}
        original = copy.deepcopy(data)
        model = d.QuadraticRidge(data)
        result = model(torch.tensor([[0., 0.], [0., 1.], [0., -1.]]))
        self.assertEqual(float(result[0]), 0.)
        self.assertTrue((result[1:] < 0).all())
        phi = model.basis(x) / model.feature_scale
        residual = phi.T @ (phi @ model.beta - data["targets"].double() / model.target_scale) / len(x)
        torch.testing.assert_close(residual + d.RIDGE_PENALTY * model.beta, torch.zeros_like(model.beta), atol=1e-12, rtol=0)
        for key in ("inputs", "targets"):
            torch.testing.assert_close(data[key], original[key], atol=0, rtol=0)
        full = d.QuadraticRidge(d.pilot.training_data(terminal_fixture()["train_cases"]))
        self.assertEqual(len(full.beta), 243)

    def test_fold_predictions_ignore_held_labels_and_fixed_rule_is_fold_local(self):
        d = diagnostic
        cases = terminal_fixture()["train_cases"]
        fit, held = d.split_fold(cases, 0)
        inputs = torch.cat([d.features(c) for c in held])
        with patch.object(d, "MLP_UPDATES", 5):
            predictions, meta = d.fit_fold(fit, inputs, 31, 0)
        self.assertEqual(meta["mlp"]["data_rows"], 576)
        self.assertEqual(meta["ridge_coefficients"], 243)
        fixed = d.pilot.fixed_preference_scores(fit)
        original = d.decision_row(held[0], 0, {k: v[0] for k, v in predictions.items()}, fixed)
        changed = copy.deepcopy(held[0])
        changed["branches"]["soft_plus"]["discounted_returns"]["512"] = [-100., -100.]
        poisoned = d.decision_row(changed, 0, {k: v[0] for k, v in predictions.items()}, fixed)
        self.assertEqual(original["scores"], poisoned["scores"])
        self.assertEqual(original["choices"], poisoned["choices"])
        self.assertNotEqual(original["candidate_gains"], poisoned["candidate_gains"])
        self.assertEqual(fixed, d.pilot.fixed_preference_scores(fit))

    def test_runner_accounts_oof_fits_without_env_or_checkpoint(self):
        d = diagnostic
        source = terminal_fixture()
        source["validation_cases"] = None
        source["summary"] = None
        with (tempfile.TemporaryDirectory() as directory, patch.object(d, "MLP_UPDATES", 2),
              patch.object(d.pilot.training, "make_piecewise_mo_env", side_effect=AssertionError("No rollouts")),
              patch.object(d.pilot.sources, "load_inputs", side_effect=AssertionError("No policy inputs")),
              patch.object(torch, "load", side_effect=AssertionError("No saved evaluators"))):
            root = Path(directory)
            (root / "v27").mkdir()
            (root / "v27/seed31.json").write_text(json.dumps(source))
            result = d.run("v27", 31, root, root / "out")
            self.assertTrue(result["complete"])
            self.assertEqual((result["mlp_updates"], result["ridge_solves"]), (8, 4))
            self.assertEqual(result["new_environment_interactions"], 0)
            self.assertEqual(len(result["rows"]), 192)
            self.assertEqual(result["summary"], d.summarize_rows(result["rows"]))
            self.assertEqual(summary.validate_run(result, "v27", 31), result["summary"])
            for fold in result["folds"]:
                fit, _ = d.split_fold(source["train_cases"], fold["fold"])
                self.assertEqual(fold["fixed_preference_scores"], d.pilot.fixed_preference_scores(fit))
            for kind in ("grid", "budget", "split", "choice", "gate"):
                changed = copy.deepcopy(result)
                if kind == "grid":
                    changed["rows"][0] = copy.deepcopy(changed["rows"][1])
                elif kind == "budget":
                    changed["mlp_updates"] += 1
                elif kind == "split":
                    changed["folds"][0]["fit_reset_ids"] = changed["folds"][0]["held_reset_ids"]
                elif kind == "choice":
                    changed["rows"][0]["choices"]["ridge"] = "base"
                else:
                    changed["summary"]["group_passes"] = not changed["summary"]["group_passes"]
                with self.subTest(kind=kind), self.assertRaises(AssertionError):
                    summary.validate_run(changed, "v27", 31)

    def test_downside_gate_cannot_be_overridden_by_positive_mean(self):
        d = diagnostic
        zero = np.zeros(5)
        predicted = np.asarray([0., 2., 0., 1., 0.])
        cases = terminal_fixture()["train_cases"]
        rows = []
        for case in cases:
            case["branches"]["soft_plus"]["discounted_returns"]["512"] = (
                [-4., -4.] if case["repeat"] == 0 else [1., 1.])
            fixed = {str(wi): dict.fromkeys(d.pilot.CANDIDATES, 0.) for wi in range(3)}
            rows.append(d.decision_row(case, case["repeat"] % 4, {"ridge": predicted, "mlp": zero}, fixed))
        result = d.summarize_rows(rows)
        self.assertTrue(result["checks"]["positive_mean_gain"])
        self.assertTrue(result["checks"]["mean_not_worse_than_controls"])
        self.assertFalse(result["checks"]["downside_not_worse_than_controls"])
        self.assertFalse(result["group_passes"])


if __name__ == "__main__":
    unittest.main()
