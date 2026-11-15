import copy
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import torch

from affine_residual_change_evidence import AffineResidualPredictor, AffineResidualHeadwiseEvidence
from cross_source_scale_evidence import cross_source_residuals
from headwise_change_evidence import CONFIG
import run_cross_source_scale_probe as probe
import run_affine_residual_change_probe as affine
import run_headwise_change_probe as plain
import run_conditional_change_probe as original
from test_affine_residual_change_probe import make_source
from test_headwise_change_probe import targets


def source_pair():
    rng = np.random.default_rng(7)
    return [(x, targets(x), dict(seed=seed)) for seed, x in (
        (1, rng.normal(size=(120, 5)).astype(np.float32) + .5),
        (2, rng.normal(size=(120, 5)).astype(np.float32) - .5))]


def fit_evidence(sources):
    x, y = (np.concatenate([row[index] for row in sources]) for index in (0, 1))
    evidence = AffineResidualHeadwiseEvidence(5, 5, 2, hidden=8, updates=3, scale_updates=3, window=10)
    evidence.fit(x, y, len(sources[0][0]), 13)
    return evidence


class CrossSourceScaleTests(unittest.TestCase):
    def test_whole_source_exclusion_fixed_normalization_and_action_heads(self):
        sources = source_pair()
        evidence = fit_evidence(sources)
        trained, predicted = [], []
        fit, predict = AffineResidualPredictor.fit, AffineResidualPredictor.predict
        def record_fit(model, inputs, outputs, seed):
            trained.append(inputs.copy())
            return fit(model, inputs, outputs, seed)
        def record_predict(model, inputs):
            prediction = predict(model, inputs)
            predicted.append(prediction.copy())
            return prediction
        with patch.object(AffineResidualPredictor, "fit", record_fit), \
             patch.object(AffineResidualPredictor, "predict", record_predict):
            inputs, losses, work = cross_source_residuals(evidence, sources, 91)
        np.testing.assert_array_equal(inputs, np.concatenate([row[0] for row in sources]))
        for actual, expected in zip(trained, [sources[1][0]] * 2 + [sources[0][0]] * 2):
            np.testing.assert_array_equal(actual, expected)
        self.assertEqual([(r["train_seed"], r["residual_seed"]) for r in work["sources"]], [(2, 1), (1, 2)])
        self.assertEqual(work["optimizer_steps"], 12)
        for index, source in enumerate(sources):
            prediction = np.concatenate(predicted[2 * index:2 * index + 2], axis=1)
            expected = ((prediction - source[1]) / evidence.y_scale) ** 2
            np.testing.assert_allclose(losses[index * 120:(index + 1) * 120, 0], expected[:, 0])
            np.testing.assert_allclose(losses[index * 120:(index + 1) * 120, 2], expected[:, 2:].mean(1))
        self.assertTrue(evidence.action_heads[1])
        self.assertLess(float(losses[:, 1].max()), 1e-10)

    def test_held_targets_change_errors_not_held_source_prediction(self):
        sources = source_pair()
        evidence = fit_evidence(sources)
        prediction = evidence.reward_predictor.predict(sources[0][0]).copy()
        _, losses, _ = cross_source_residuals(evidence, sources, 41)
        changed = copy.deepcopy(sources)
        changed[0][1][:, :2] += 7
        _, changed_losses, _ = cross_source_residuals(evidence, changed, 41)
        # The first residual block's predictors use only source2; action targets shift by7.
        np.testing.assert_allclose(changed_losses[:120, 1], (7 / evidence.y_scale[1]) ** 2, rtol=2e-6)
        self.assertGreater(float(changed_losses[:120, 0].mean()), float(losses[:120, 0].mean()))
        np.testing.assert_array_equal(evidence.reward_predictor.predict(sources[0][0]), prediction)

    def test_supplied_scale_losses_and_final_calibration_do_not_refit_mean(self):
        sources = source_pair()
        evidence = fit_evidence(sources)
        inputs, losses, _ = cross_source_residuals(evidence, sources, 71)
        mean = evidence.reward_predictor.predict(inputs).copy()
        work = evidence.fit_scale_losses(inputs, losses, 19)
        np.testing.assert_allclose(evidence.global_scales,
            np.maximum(losses.mean(0), evidence.config["residual_floor"]))
        self.assertEqual(work["optimizer_steps"], 9)
        scale = evidence.scales(inputs, "conditional_scale").copy()
        evidence.calibrate(inputs, np.concatenate([r[1] for r in sources]) + 100)
        np.testing.assert_array_equal(evidence.reward_predictor.predict(inputs), mean)
        np.testing.assert_array_equal(evidence.scales(inputs, "conditional_scale"), scale)

    def test_stored_run_excludes_calibration_test_and_counts_auxiliary_work(self):
        with tempfile.TemporaryDirectory() as tmp, patch.dict(CONFIG, hidden=8, updates=3, scale_updates=3):
            source, root, old, control = (Path(tmp) / name for name in ("source", "candidate", "affine", "plain"))
            env = probe.ENVIRONMENTS[0]
            make_source(source, env)
            recorded = []
            fit_scales = AffineResidualHeadwiseEvidence.fit_scale_losses
            def record_scale(evidence, inputs, losses, seed):
                recorded.append((inputs.copy(), losses.copy()))
                return fit_scales(evidence, inputs, losses, seed)
            with patch("builtins.print"), patch.object(original.training, "make_piecewise_mo_env",
                    side_effect=AssertionError("new interaction")):
                plain.run(env, 2101, source, control)
                baseline = affine.run(env, 2101, source, old, control)
                with patch.object(AffineResidualHeadwiseEvidence, "fit_scale_losses", record_scale):
                    report = probe.run(env, 2101, source, root, old)
            roles = report["roles"]
            expected = np.concatenate([probe.normal_data(source, env, seed, "expanded_normal")[0]
                                       for seed in roles["fit_seeds"] + [roles["scale_seed"]]])
            np.testing.assert_array_equal(recorded[0][0], expected)
            self.assertEqual(report["predictor_work"], baseline["predictor_work"])
            self.assertEqual(report["affine_residual_control"]["arms"], baseline["arms"])
            self.assertEqual((report["predictor_optimizer_steps"], report["auxiliary_predictor_optimizer_steps"],
                              report["residual_scale_optimizer_steps"]), (6, 12, 9))
            self.assertEqual(report["scale_fit_steps"], 15000)
            self.assertNotIn(roles["calibration_seed"], report["scale_fit_sources"])
            self.assertNotIn(roles["test_seed"], report["scale_fit_sources"])
            for direction in report["auxiliary_work"]["sources"]:
                self.assertEqual((direction["train_steps"], direction["residual_steps"]), (5000, 5000))
            with patch.object(AffineResidualHeadwiseEvidence, "fit", side_effect=AssertionError("repeated fit")):
                self.assertEqual(probe.run(env, 2101, source, root, old), report)
            with patch.object(probe, "ENVIRONMENTS", (env,)), patch.object(probe, "SEEDS", [2101]):
                summary = probe.summarize(root)
            compact = probe.compact_report(summary)
            self.assertEqual(summary["auxiliary_predictor_optimizer_steps"], 12)
            self.assertFalse(summary["evidence_integrated"])
            self.assertEqual(summary["decision"], "hold_conditional_evidence_integration")
            self.assertNotIn("windows", compact["environments"][env]["runs"][0]["arms"]["conditional_scale"]["calibration"])
            self.assertIn("windows", summary["environments"][env]["runs"][0]["arms"]["conditional_scale"]["calibration"])


if __name__ == "__main__":
    torch.set_num_threads(2)
    unittest.main()
