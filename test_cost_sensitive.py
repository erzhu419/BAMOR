import copy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import torch

import diagnose_cost_sensitive as diagnostic
import summarize_cost_sensitive as summary
from test_paired_advantage_summary import terminal_fixture
from test_residual_advantage import previous_fixture


def cached_fixture(cases):
    d = diagnostic
    old = previous_fixture(cases)
    old["protocol"] = d.reference.PROTOCOL
    old["rows"] = [d.reference.join_row(c, r, {"residual": np.zeros(5), "calibrated_prior": np.zeros(5)})
                   for c, r in zip(cases, old["rows"])]
    return old


class CostSensitiveTests(unittest.TestCase):
    def test_objective_rewards_gains_and_doubles_losses(self):
        d = diagnostic
        gains = torch.tensor([[0., 1., -2.]], dtype=torch.float64)
        torch.testing.assert_close(d.utility(gains), torch.tensor([[0., 1., -4.]], dtype=torch.float64))
        logits = torch.zeros_like(gains, requires_grad=True)
        loss = d.objective(logits, d.utility(gains), torch.tensor([2.], dtype=torch.float64))
        self.assertAlmostEqual(float(loss.detach()), 5.)
        loss.backward()
        self.assertLess(float(logits.grad[0, 1]), 0.)
        self.assertGreater(float(logits.grad[0, 2]), 0.)

    def test_shared_features_match_ridge_and_keep_base_exactly_zero(self):
        d = diagnostic
        cases = terminal_fixture()["train_cases"][:8]
        inputs = d.model_inputs(cases)
        model = d.CostSensitiveSelector(inputs, 6)
        ridge = d.cv.QuadraticRidge(d.p.training_data(cases))
        torch.testing.assert_close(model.features.transform(inputs), ridge.transform(inputs), rtol=0, atol=0)
        self.assertEqual(model.beta.numel(), 243)
        self.assertTrue((model(inputs) == 0).all())
        with torch.no_grad():
            model.beta.fill_(1.)
        self.assertTrue((model(inputs)[:, 0] == 0).all())

    def test_training_rejects_all_bad_actions_and_is_deterministic(self):
        d = diagnostic
        inputs = torch.tensor([[[s, a] for a in (0., 1., -1., 2., -2.)] for s in (-1., 1.)], dtype=torch.float64)
        gains = -inputs[:, :, -1].square()
        original = inputs.clone()
        rng = torch.get_rng_state().clone()
        model, meta = d.fit_selector(inputs, gains, 1, updates=150)
        repeated, _ = d.fit_selector(inputs, gains, 1, updates=150)
        torch.testing.assert_close(model.beta, repeated.beta, rtol=0, atol=0)
        torch.testing.assert_close(inputs, original, rtol=0, atol=0)
        torch.testing.assert_close(torch.get_rng_state(), rng, rtol=0, atol=0)
        self.assertEqual(meta["train_mean_gain"], 0.)
        self.assertTrue((model(inputs).argmax(-1) == 0).all())
        self.assertTrue(all(not p.requires_grad and p.grad is None for p in model.parameters()))

    def test_held_labels_are_not_inputs(self):
        d = diagnostic
        fit, held = d.cv.split_fold(terminal_fixture()["train_cases"], 0)
        before = copy.deepcopy(fit)
        with patch.object(d, "UPDATES", 2):
            scores, metadata = d.fit_predict(fit, d.model_inputs(held))
            for c in held:
                c["branches"] = c["predictions"] = None
            repeated, _ = d.fit_predict(fit, d.model_inputs(held))
        np.testing.assert_array_equal(scores, repeated)
        self.assertEqual(fit, before)
        self.assertEqual((metadata["fit_anchors"], metadata["fit_nonbase_rows"]), (144, 576))

    def test_runner_and_summary_preserve_cached_controls_and_budget(self):
        d = diagnostic
        source = terminal_fixture()
        source["validation_cases"] = source["summary"] = None
        previous = cached_fixture(source["train_cases"])
        with (tempfile.TemporaryDirectory() as directory, patch.object(d, "UPDATES", 2),
              patch.object(d.p.training, "make_piecewise_mo_env", side_effect=AssertionError("No rollouts")),
              patch.object(d.cv, "fit_fold", side_effect=AssertionError("No control refit")),
              patch.object(d.reference, "fit_predict", side_effect=AssertionError("No residual refit")),
              patch.object(torch, "load", side_effect=AssertionError("No checkpoints"))):
            root = Path(directory)
            for name, data in (("source", source), ("previous", previous)):
                path = root / name / "v27/seed31.json"
                path.parent.mkdir(parents=True)
                path.write_text(json.dumps(data))
            result = d.run("v27", 31, root / "source", root / "previous", root / "out")
            self.assertEqual((result["selector_updates"], result["mlp_updates"], result["new_environment_interactions"]), (8, 0, 0))
            self.assertEqual(summary.validate_run(result, previous, "v27", 31), result["summary"])
            self.assertNotIn("cost_sensitive", result["summary"]["calibration"])
            for kind in ("grid", "budget", "control", "split", "objective", "gate"):
                changed = copy.deepcopy(result)
                if kind == "grid":
                    changed["rows"][0] = copy.deepcopy(changed["rows"][1])
                elif kind == "budget":
                    changed["selector_updates"] += 1
                elif kind == "control":
                    changed["rows"][0]["scores"]["ridge"]["soft_plus"] += 1.
                elif kind == "split":
                    changed["folds"][0]["fit_reset_ids"] = changed["folds"][0]["held_reset_ids"]
                elif kind == "objective":
                    changed["folds"][0]["downside_weight"] = 0.
                else:
                    changed["summary"]["group_passes"] = not result["summary"]["group_passes"]
                with self.subTest(kind=kind), self.assertRaises(AssertionError):
                    summary.validate_run(changed, previous, "v27", 31)

    def test_gain_does_not_override_downside_and_logits_have_no_rmse(self):
        d = diagnostic
        cases = terminal_fixture()["train_cases"]
        for c in cases:
            v = -4. if c["repeat"] == 0 else 1.
            c["branches"]["soft_plus"]["discounted_returns"]["512"] = [v, v]
        cached = cached_fixture(cases)
        rows = [d.reference.join_row(c, r, {"cost_sensitive": np.array([0., 2., 0., 1., 0.])}, selectors=d.SELECTORS)
                for c, r in zip(cases, cached["rows"])]
        result = d.summarize_rows(rows)
        self.assertTrue(result["checks"]["positive_mean_gain"])
        self.assertTrue(result["checks"]["mean_not_worse_than_controls"])
        self.assertFalse(result["checks"]["downside_not_worse_than_controls"])
        self.assertFalse(result["group_passes"])
        self.assertNotIn("cost_sensitive", result["calibration"])


if __name__ == "__main__":
    unittest.main()
