import copy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import torch

import diagnose_margin_selection as diagnostic
import summarize_margin_selection as summary
from test_cost_sensitive import cached_fixture as residual_fixture
from test_paired_advantage_summary import terminal_fixture


def cached_fixture(cases):
    d = diagnostic
    old = residual_fixture(cases)
    old["protocol"] = d.previous.PROTOCOL
    old["rows"] = [d.reference.join_row(c, r, {"cost_sensitive": np.zeros(5)}, selectors=d.previous.SELECTORS)
                   for c, r in zip(cases, old["rows"])]
    return old


class MarginSelectionTests(unittest.TestCase):
    def test_hinge_bounds_deterministic_regret_including_ties_and_base(self):
        d = diagnostic
        utility = torch.tensor([[0., 2., -4.], [0., -1., -2.], [0., 1., 1.]], dtype=torch.float64)
        for scores in (torch.zeros_like(utility), utility, -utility, utility * .5):
            selected = scores.argmax(-1, keepdim=True)
            regret = utility.max(-1).values - utility.gather(-1, selected).squeeze(-1)
            self.assertTrue((d.margin_losses(scores, utility) >= regret).all())
        torch.testing.assert_close(d.margin_losses(utility, utility), torch.zeros(3, dtype=torch.float64))
        scores = torch.zeros_like(utility, requires_grad=True)
        loss = d.objective(scores, utility, torch.tensor([2.], dtype=torch.float64))
        self.assertAlmostEqual(float(loss.detach()), 4. + (6. + 2. + 1.) / 3.)
        loss.backward()
        self.assertLess(float(scores.grad[0, 1]), 0.)
        self.assertGreater(float(scores.grad[0, 2]), 0.)

    def test_training_learns_base_and_beneficial_action_deterministically(self):
        d = diagnostic
        inputs = torch.tensor([[[s, a] for a in (0., 1., -1., 2., -2.)] for s in (-1., 1.)], dtype=torch.float64)
        original = inputs.clone()
        rng = torch.get_rng_state().clone()
        for beneficial in (False, True):
            gains = inputs[:, :, -1] if beneficial else -inputs[:, :, -1].square()
            model, meta = d.fit_selector(inputs, gains, 1, updates=150)
            repeated, _ = d.fit_selector(inputs, gains, 1, updates=150)
            torch.testing.assert_close(model.beta, repeated.beta, rtol=0, atol=0)
            self.assertTrue((model(inputs)[:, 0] == 0.).all())
            self.assertTrue((model(inputs).argmax(-1) == (3 if beneficial else 0)).all())
            self.assertEqual(meta["train_mean_gain"], 2. if beneficial else 0.)
            self.assertTrue(all(not p.requires_grad and p.grad is None for p in model.parameters()))
        torch.testing.assert_close(inputs, original, rtol=0, atol=0)
        torch.testing.assert_close(torch.get_rng_state(), rng, rtol=0, atol=0)

    def test_held_labels_and_critic_predictions_are_not_inputs(self):
        d = diagnostic
        fit, held = d.cv.split_fold(terminal_fixture()["train_cases"], 0)
        original = copy.deepcopy(fit)
        with patch.object(d, "UPDATES", 2):
            scores, meta = d.fit_predict(fit, d.previous.model_inputs(held))
            for c in fit:
                c["predictions"] = None
            for c in held:
                c["branches"] = c["predictions"] = None
            repeated, meta2 = d.fit_predict(fit, d.previous.model_inputs(held))
        np.testing.assert_array_equal(scores, repeated)
        self.assertEqual(meta, meta2)
        self.assertEqual(meta["fit_anchors"], len(original))

    def test_runner_reuses_controls_and_rejects_invalid_aggregation(self):
        d = diagnostic
        source = terminal_fixture()
        source["validation_cases"] = source["summary"] = None
        previous = cached_fixture(source["train_cases"])
        with (tempfile.TemporaryDirectory() as directory, patch.object(d, "UPDATES", 2),
              patch.object(d.p.training, "make_piecewise_mo_env", side_effect=AssertionError("No rollouts")),
              patch.object(d.cv, "fit_fold", side_effect=AssertionError("No control refit")),
              patch.object(d.previous, "fit_predict", side_effect=AssertionError("No softmax refit")),
              patch.object(torch, "load", side_effect=AssertionError("No checkpoints"))):
            root = Path(directory)
            for name, data in (("source", source), ("previous", previous)):
                path = root / name / "v27/seed31.json"
                path.parent.mkdir(parents=True)
                path.write_text(json.dumps(data))
            result = d.run("v27", 31, root / "source", root / "previous", root / "out")
            self.assertEqual(summary.validate_run(result, previous, "v27", 31), result["summary"])
            self.assertEqual((result["selector_updates"], result["new_environment_interactions"]), (8, 0))
            for kind in ("grid", "budget", "control", "split", "objective", "gate"):
                changed = copy.deepcopy(result)
                if kind == "grid":
                    changed["rows"][0] = copy.deepcopy(changed["rows"][1])
                elif kind == "budget":
                    changed["selector_updates"] += 1
                elif kind == "control":
                    changed["rows"][0]["scores"]["cost_sensitive"]["soft_plus"] += 1.
                elif kind == "split":
                    changed["folds"][0]["fit_reset_ids"] = changed["folds"][0]["held_reset_ids"]
                elif kind == "objective":
                    changed["folds"][0]["objective"] = "softmax"
                else:
                    changed["summary"]["group_passes"] = not result["summary"]["group_passes"]
                with self.subTest(kind=kind), self.assertRaises(AssertionError):
                    summary.validate_run(changed, previous, "v27", 31)

    def test_positive_gain_cannot_override_downside_or_turn_logits_into_returns(self):
        d = diagnostic
        cases = terminal_fixture()["train_cases"]
        for c in cases:
            value = -4. if c["repeat"] == 0 else 1.
            c["branches"]["soft_plus"]["discounted_returns"]["512"] = [value, value]
        cached = cached_fixture(cases)
        rows = [d.reference.join_row(c, r, {"margin": np.array([0., 2., 0., 1., 0.])}, selectors=d.SELECTORS)
                for c, r in zip(cases, cached["rows"])]
        result = d.summarize_rows(rows)
        self.assertTrue(result["checks"]["positive_mean_gain"])
        self.assertTrue(result["checks"]["mean_not_worse_than_controls"])
        self.assertFalse(result["checks"]["downside_not_worse_than_controls"])
        self.assertFalse(result["group_passes"])
        self.assertFalse({"margin", "cost_sensitive"}.intersection(result["calibration"]))


if __name__ == "__main__":
    unittest.main()
