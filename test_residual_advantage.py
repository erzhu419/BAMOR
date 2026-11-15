import copy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import torch

import diagnose_residual_advantage as diagnostic
import summarize_residual_advantage as summary
from test_paired_advantage_summary import terminal_fixture


def previous_fixture(cases):
    d = diagnostic
    zero = np.zeros(5)
    fixed = {str(i): dict.fromkeys(d.p.CANDIDATES, 0.) for i in range(3)}
    rows = [d.cv.decision_row(c, c["repeat"] % 4, {"mlp": zero, "ridge": zero}, fixed) for c in cases]
    return {"complete": True, "protocol": d.cv.PROTOCOL, "condition": "v27", "seed": 31,
            "source_split": "train_cases", "fold_rule": "repeat % 4", "rows": rows}


class ResidualAdvantageTests(unittest.TestCase):
    def test_model_inputs_ignore_labels_and_candidates_share_the_base_state(self):
        cases = terminal_fixture()["train_cases"][:3]
        original = diagnostic.model_inputs(cases)
        poisoned = copy.deepcopy(cases)
        for c in poisoned:
            c["branches"] = None
            c["external_scores"] = {"invalid": 999}
        repeated = diagnostic.model_inputs(poisoned)
        for a, b in zip(original, repeated):
            np.testing.assert_array_equal(a, b)
        self.assertEqual(original[0].shape, (3, 36))
        self.assertEqual(original[1].shape, (3, 5, 2))
        self.assertTrue((original[1][:, 0] == 0).all())

    def test_exact_base_nonnegative_prior_and_negative_opposite_actions(self):
        d = diagnostic
        state = np.zeros((4, 36))
        q = np.column_stack(([-2., -1., 1., 2.], [0.] * 4))
        targets = -q[:, 0] ** 2
        model = d.CriticResidual(state, q, targets)
        self.assertEqual(len(model.beta), 77)
        self.assertTrue((model.prior >= 0).all())
        predictions = model.predict(np.zeros((3, 36)), np.array([[0., 0.], [1., 0.], [-1., 0.]]))
        self.assertEqual(float(predictions["residual"][0]), 0.)
        self.assertEqual(float(predictions["calibrated_prior"][0]), 0.)
        self.assertTrue((predictions["residual"][1:] < 0).all())
        phi = model.basis(state, q) / model.feature_scale
        residual = targets / model.target_scale - q / model.q_scale @ model.prior
        np.testing.assert_allclose(phi.T @ (phi @ model.beta - residual) / len(q) + d.PENALTY * model.beta,
                                   0., atol=1e-12, rtol=0)
        linear = d.CriticResidual(state, q, 3 * q[:, 0])
        self.assertGreater(linear.prior[0], 0)
        linear.beta[:] = 0.
        values = linear.predict(state, q)
        np.testing.assert_array_equal(values["residual"], values["calibrated_prior"])

    def test_fit_uses_only_training_rows_and_preserves_inputs(self):
        d = diagnostic
        fit, held = d.cv.split_fold(terminal_fixture()["train_cases"], 0)
        before = copy.deepcopy(fit)
        inputs = d.model_inputs(held)
        predicted, meta = d.fit_predict(fit, inputs)
        self.assertEqual(meta["fit_rows"], 576)
        self.assertEqual(fit, before)
        for c in held:
            c["branches"] = None
        again, _ = d.fit_predict(fit, d.model_inputs(held))
        for name in predicted:
            np.testing.assert_array_equal(predicted[name], again[name])
            self.assertEqual(predicted[name].shape, (48, 5))

    def test_runner_reuses_controls_and_never_loads_models_or_validation(self):
        d = diagnostic
        source = terminal_fixture()
        source["validation_cases"] = source["summary"] = None
        previous = previous_fixture(source["train_cases"])
        with (tempfile.TemporaryDirectory() as directory,
              patch.object(d.p.training, "make_piecewise_mo_env", side_effect=AssertionError("No environment")),
              patch.object(d.cv, "fit_fold", side_effect=AssertionError("Reuse controls")),
              patch.object(torch, "load", side_effect=AssertionError("No checkpoints"))):
            root = Path(directory)
            for name, data in (("source", source), ("previous", previous)):
                path = root / name / "v27/seed31.json"
                path.parent.mkdir(parents=True)
                path.write_text(json.dumps(data))
            result = d.run("v27", 31, root / "source", root / "previous", root / "out")
            self.assertEqual((result["nnls_fits"], result["ridge_solves"], result["mlp_updates"]), (4, 4, 0))
            self.assertEqual(result["summary"], summary.validate_run(result, previous, "v27", 31))
            for kind in ("grid", "budget", "control", "split", "choice", "gate"):
                changed = copy.deepcopy(result)
                if kind == "grid":
                    changed["rows"][0] = copy.deepcopy(changed["rows"][1])
                elif kind == "budget":
                    changed["new_environment_interactions"] = 1
                elif kind == "control":
                    changed["rows"][0]["scores"]["mlp"]["soft_plus"] = 99.
                elif kind == "split":
                    changed["folds"][0]["fit_reset_ids"] = changed["folds"][0]["held_reset_ids"]
                elif kind == "choice":
                    changed["rows"][0]["choices"]["residual"] = "base"
                else:
                    changed["summary"]["group_passes"] = not result["summary"]["group_passes"]
                with self.subTest(kind=kind), self.assertRaises(AssertionError):
                    summary.validate_run(changed, previous, "v27", 31)

    def test_positive_mean_does_not_override_downside(self):
        d = diagnostic
        cases = terminal_fixture()["train_cases"]
        for c in cases:
            v = -4. if c["repeat"] == 0 else 1.
            c["branches"]["soft_plus"]["discounted_returns"]["512"] = [v, v]
        previous = previous_fixture(cases)
        rows = [d.join_row(c, r, {"calibrated_prior": np.zeros(5), "residual": np.array([0., 2., 0., 1., 0.])})
                for c, r in zip(cases, previous["rows"])]
        result = d.summarize_rows(rows)
        self.assertTrue(result["checks"]["positive_mean_gain"])
        self.assertTrue(result["checks"]["mean_not_worse_than_controls"])
        self.assertFalse(result["checks"]["downside_not_worse_than_controls"])
        self.assertFalse(result["group_passes"])


if __name__ == "__main__":
    unittest.main()
