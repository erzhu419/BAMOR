import copy
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import torch

from headwise_change_evidence import CONFIG
import run_cross_policy_calibration as probe
import run_sensitivity_guarded_selection as guarded
import run_window_risk_selection as risk
import run_matched_residual_scale_probe as matched
import run_affine_residual_change_probe as affine
import run_headwise_change_probe as plain
import run_conditional_change_probe as original
from test_affine_residual_change_probe import make_source
from test_matched_residual_scale_probe import make_fresh_fixture
from test_calibration_expansion import fixed_evidence


class CrossPolicyCalibrationTests(unittest.TestCase):
    def test_pool_keeps_whole_windows_equal_sources_and_regimes(self):
        same_x = np.repeat(np.arange(50), 100)[:, None]
        train_x = same_x + 100
        same = (same_x, same_x * 2, dict(seed=21))
        train = (train_x, train_x * 2, dict(seed=22))
        x, y, sources = probe.balanced_windows(same, train, 100)
        np.testing.assert_array_equal(x[:2500], same_x.reshape(50, 100, 1)[::2].reshape(2500, 1))
        np.testing.assert_array_equal(x[2500:], train_x.reshape(50, 100, 1)[1::2].reshape(2500, 1))
        np.testing.assert_array_equal(y, x * 2)
        self.assertEqual([s["steps"] for s in sources], [2500, 2500])
        for source in sources:
            indices = np.array(source["window_indices"])
            self.assertEqual([int(((indices >= a) & (indices < b)).sum()) for a, b in
                ((0, 30), (30, 40), (40, 50))], [15, 5, 5])
        all_indices = sum([s["window_indices"] for s in sources], [])
        self.assertEqual([sum(a <= i < b for i in all_indices) for a, b in
            ((30, 35), (35, 40), (40, 45), (45, 50))], [5, 5, 5, 5])

    def test_calibration_uses_fixed_support_and_joint_eligible_scores(self):
        evidence = fixed_evidence([[1., .1], [3., .1], [1000., 3.], [1000., 5.]],
            [[.1, .1], [.2, .2], [5., .1], [5., .1]])
        radii = evidence.radii.copy()
        old_thresholds = evidence.thresholds.copy()
        result = probe.calibrate_fixed_support(evidence, np.zeros((4, 2)), np.zeros((4, 2)))
        np.testing.assert_array_equal(evidence.radii, radii)
        self.assertEqual(result["windows"][1]["statistic"], 4.)
        self.assertEqual(result["windows"][1]["heads"]["reward_0"]["supported_steps"], 0)
        self.assertAlmostEqual(result["error_threshold"], float(np.quantile([2., 4.], .99)))
        for arm, value in old_thresholds.items():
            self.assertEqual(evidence.thresholds[arm], value)

    def test_new_quantile_can_lower_threshold_without_a_floor(self):
        evidence = fixed_evidence([[1., 1.]] * 100, [[0., 0.]] * 100)
        evidence.thresholds[probe.GUARDED] = 20.
        result = probe.calibrate_fixed_support(evidence, np.zeros((100, 2)), np.zeros((100, 2)))
        self.assertEqual(result["error_threshold"], 1.)

    def test_pipeline_freezes_models_excludes_test_and_counts_only_final_fits(self):
        with tempfile.TemporaryDirectory() as tmp, patch.dict(CONFIG, hidden=8, updates=3, scale_updates=3):
            source, fresh, base, old, parent, risk_root, prior, root = (Path(tmp) / name for name in
                ("source", "fresh", "plain", "affine", "matched", "risk", "guarded", "calibration"))
            env, seed = probe.ENVIRONMENTS[0], 2101
            make_source(source, env)
            make_fresh_fixture(fresh, env)
            observed, read_seeds, fitted_rows = [], [], []
            calibrate, read_fresh, fit_scale = probe.calibrate_fixed_support, probe.fresh_data, probe.GuardedEvidence.fit_scale_losses
            def record_fresh(root, env, seed):
                read_seeds.append(seed)
                return read_fresh(root, env, seed)
            def record_fit(evidence, x, losses, seed):
                fitted_rows.append(x.copy())
                return fit_scale(evidence, x, losses, seed)
            def record_calibration(evidence, x, y):
                models = [evidence.reward_predictor.model, evidence.dynamics_predictor.model, *evidence.scale_models]
                states = [copy.deepcopy(model.state_dict()) for model in models]
                radii, weights = evidence.radii.copy(), evidence.weights.copy()
                result = calibrate(evidence, x, y)
                np.testing.assert_array_equal(evidence.radii, radii)
                self.assertEqual(evidence.weights, weights)
                self.assertTrue(all(not p.requires_grad for model in models for p in model.parameters()))
                self.assertTrue(all(torch.equal(state[k], model.state_dict()[k])
                    for state, model in zip(states, models) for k in state))
                observed.append((x.copy(), y.copy()))
                return result
            with patch("builtins.print"), patch.object(original.training, "make_piecewise_mo_env",
                    side_effect=AssertionError("No new environment")):
                plain.run(env, seed, source, base)
                affine.run(env, seed, source, old, base)
                matched.run(env, seed, source, parent, old, fresh)
                risk.run(env, seed, source, risk_root, parent, fresh)
                before = guarded.run(env, seed, source, prior, parent, fresh, risk_root)
                with patch.object(probe, "fresh_data", record_fresh), \
                     patch.object(probe, "calibrate_fixed_support", record_calibration), \
                     patch.object(probe.GuardedEvidence, "fit_scale_losses", record_fit):
                    result = probe.run(env, seed, source, root, prior, fresh)
            roles = result["roles"]
            self.assertEqual(read_seeds, roles["fit_seeds"] + [roles["calibration_seed"], roles["scale_seed"]])
            self.assertNotIn(seed, read_seeds)
            self.assertEqual(len(fitted_rows), 1)
            scale_rows = [read_fresh(fresh, env, s) for s in roles["fit_seeds"]]
            scale_rows.append(probe.normal_data(source, env, roles["scale_seed"], "expanded_normal"))
            np.testing.assert_array_equal(fitted_rows[0], np.concatenate([r[0] for r in scale_rows]))
            expected = [probe.normal_data(source, env, roles["calibration_seed"], "expanded_normal"),
                read_fresh(fresh, env, roles["calibration_seed"]), read_fresh(fresh, env, roles["scale_seed"])]
            expected.append(probe.balanced_windows(expected[1], expected[2], CONFIG["window"]))
            for actual, wanted in zip(observed, expected):
                for i in (0, 1):
                    np.testing.assert_array_equal(actual[i], wanted[i])
            self.assertEqual(result["frozen_selection"]["global_weight"], before["selections"][probe.GUARDED]["global_weight"])
            self.assertEqual(result["arms"]["original_calibration"]["cases"], before["arms"][probe.GUARDED]["cases"])
            self.assertEqual((result["predictor_optimizer_steps"], result["residual_scale_optimizer_steps"]), (6, 9))
            self.assertTrue(all(result[k] == 0 for k in probe.ZERO_WORK))
            with patch.object(probe.GuardedEvidence, "fit", side_effect=AssertionError("Repeated fit")):
                self.assertEqual(probe.run(env, seed, source, root, prior, fresh), result)
            with patch.object(probe, "ENVIRONMENTS", (env,)), patch.object(probe, "SEEDS", (seed,)):
                summary = probe.summarize(root)
                with patch.object(probe, "screen", side_effect=lambda runs, arm:
                        dict(development_screen_passed=arm != probe.PRIMARY)):
                    self.assertEqual(probe.summarize(root)["decision"], "hold_conditional_evidence_integration")
            self.assertEqual(summary["residual_scale_optimizer_steps"], 9)
            compact = probe.compact_report(summary)
            self.assertNotIn("windows", compact["environments"][env]["runs"][0]["arms"][probe.PRIMARY]["calibration"])
        print("CROSS_POLICY_CALIBRATION_TEST interactions=0 mean_updates=36 total_scale_updates=108")


if __name__ == "__main__":
    torch.set_num_threads(2)
    unittest.main()
