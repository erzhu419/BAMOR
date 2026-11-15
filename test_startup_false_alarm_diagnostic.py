import copy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import torch

import diagnose_startup_false_alarms as diagnostic
from headwise_change_evidence import CONFIG
from test_affine_residual_change_probe import make_source
from test_matched_residual_scale_probe import make_fresh_fixture
from test_headwise_change_probe import targets
from test_startup_confirmation import fixture_collection


class StartupFalseAlarmDiagnosticTests(unittest.TestCase):
    def test_support_only_decomposition_separates_numerator_and_denominator(self):
        support = np.array([True, True, False])
        raw = np.array([2., 4., 10000.])
        result = diagnostic.head_decomposition(raw, raw / 2, np.array([.1, .2, 1.]),
            np.array([.1, .2, 1.]), 2., support, True)
        self.assertEqual(result["raw_mean_loss"], 3.)
        self.assertEqual(result["global_score"], 1.5)
        self.assertEqual(result["conditional_score"], 20.)
        self.assertAlmostEqual(result["effective_conditional_scale"], .15)
        self.assertAlmostEqual(result["amplification_vs_global"], 20 / 1.5)
        self.assertEqual(result["normalized_top1_share"], .5)
        self.assertEqual(result["normalized_top5_share"], 1.)
        self.assertEqual(diagnostic.head_decomposition(raw, raw, raw, raw, 2., support, False),
            dict(eligible=False, supported_steps=2))
        zero = diagnostic.head_decomposition(np.zeros(3), np.zeros(3), np.ones(3),
            np.ones(3), 1., support, True)
        self.assertIsNone(zero["effective_conditional_scale"])
        self.assertIsNone(zero["amplification_vs_global"])
        self.assertEqual(zero["normalized_top1_share"], 0.)

    def test_quantile_tail_exactly_matches_frozen_linear_interpolation(self):
        windows = [dict(start_step=100*i+1, end_step=100*i+100, eligible=True,
            statistic=float(i), heads=dict(dynamics=dict(eligible=True, statistic=float(i)))) for i in range(38)]
        windows.append(dict(eligible=False, statistic=None))
        before = copy.deepcopy(windows)
        result = diagnostic.tail(windows, .99)
        self.assertEqual(result["eligible_windows"], 38)
        self.assertEqual((result["lower_index"], result["upper_index"]), (36, 37))
        self.assertAlmostEqual(result["threshold"], np.quantile(np.arange(38), .99))
        self.assertEqual(result["maximum"], 37.)
        self.assertEqual(windows, before)

    def test_top_step_marks_terminal_and_reset_without_changing_scores(self):
        evidence = diagnostic.GuardedEvidence(5, 5, 2, hidden=8, updates=2, scale_updates=2, window=10)
        x = np.random.default_rng(4).normal(size=(100, 5)).astype(np.float32)
        evidence.fit(x, targets(x), 50, 4)
        evidence.fit_scales(x, targets(x), 5)
        evidence.weights = {diagnostic.GUARDED: 0.}
        evidence.radii, evidence.thresholds = np.full(3, 1e10), {diagnostic.GUARDED: 2.}
        y = targets(x[:10])
        y[4:6, 0] += [1000, 500]
        arrays = diagnostic.scores(evidence, x[:10], y)
        terminals = np.zeros((10, 2), bool)
        terminals[4, 0] = True
        old = copy.deepcopy(arrays["windows"])
        detail = diagnostic.window_detail(evidence, x[:10], y, terminals, arrays, arrays["windows"][0])
        self.assertEqual(detail["dominant_head"], "reward_0")
        by_step = {r["step"]: r for r in detail["top_steps"]}
        self.assertTrue(by_step[5]["terminated"])
        self.assertTrue(by_step[6]["first_after_reset"])
        self.assertEqual(detail["terminated_steps"], [5])
        self.assertEqual(arrays["windows"], old)
        print("TOP_STEP_TEST interactions=0 mean_updates=4 scale_updates=6 affine_fits=2")

    def test_real_reconstruction_excludes_startup_and_evaluation_from_fitting(self):
        with tempfile.TemporaryDirectory() as tmp, patch.dict(CONFIG, hidden=8, updates=3, scale_updates=3):
            source, fresh, parent, root = (Path(tmp)/n for n in ("source", "fresh", "parent", "diagnostic"))
            env, seed = diagnostic.ENV, 2102
            make_source(source, env)
            make_fresh_fixture(fresh, env)
            roles = diagnostic.fold_roles(seed)
            fit = [diagnostic.normal_data(source, env, s, "expanded_normal") for s in roles["fit_seeds"]]
            scale = [diagnostic.fresh_data(fresh, env, s) for s in roles["fit_seeds"]]
            scale.append(diagnostic.normal_data(source, env, roles["scale_seed"], "expanded_normal"))
            fit_x, fit_y = (np.concatenate([r[i] for r in fit]) for i in (0, 1))
            scale_x, scale_y = (np.concatenate([r[i] for r in scale]) for i in (0, 1))
            evidence = diagnostic.GuardedEvidence(5, 5, 2)
            evidence.weights = {diagnostic.GUARDED: .5}
            work = evidence.fit(fit_x, fit_y, 5000, 80000000 + 10*seed)
            scale_work = evidence.fit_scale_losses(scale_x, evidence.raw_losses(scale_x, scale_y), 81000000 + 10*seed)
            evidence.radii = np.full(3, 1e10)
            evidence.thresholds = {}
            groups = {}
            rng = np.random.default_rng(4321)
            for group in ("startup", "evaluation"):
                stored, rows = {}, {}
                for case, count in diagnostic.acquisition.steps(group).items():
                    x = rng.normal(size=(count, 5)).astype(np.float32)
                    y = targets(x)
                    if group == "evaluation" and case.startswith("mode"):
                        y[500:, :2] += 500
                    rows[case] = (x, y)
                    stored.update({f"{case}_inputs": x, f"{case}_targets": y,
                        f"{case}_terminals": np.zeros((count, 2), bool)})
                folder = diagnostic.acquisition.folder(parent/"collection", env, seed, group)
                diagnostic.write(folder/"readout.json", fixture_collection(env, seed, group))
                np.savez_compressed(folder/"transitions.npz", **stored)
                groups[group] = rows
            cal_x, cal_y = (np.concatenate([r[i] for r in groups["startup"].values()]) for i in (0, 1))
            calibration = diagnostic.confirmation.calibrate_fixed_support(evidence, cal_x, cal_y)
            cases = {case: dict(windows=diagnostic.scores(evidence, x, y)["windows"])
                for case, (x, y) in groups["evaluation"].items()}
            prior = dict(protocol=diagnostic.confirmation.PROTOCOL, complete=True, config=CONFIG,
                roles=roles, frozen_selection=dict(global_weight=.5), predictor_work=work, scale_work=scale_work,
                arms={diagnostic.confirmation.PRIMARY: dict(calibration=calibration, cases=cases)})
            diagnostic.write(parent/env.replace("-", "_")/f"test_seed{seed}"/"readout.json", prior)
            observed = []
            original_fit = diagnostic.GuardedEvidence.fit
            def record_fit(model, x, y, split, random_seed):
                observed.append((x.copy(), y.copy()))
                return original_fit(model, x, y, split, random_seed)
            with patch.object(diagnostic.acquisition, "run_group", side_effect=AssertionError("Recollection")), \
                 patch.object(diagnostic.GuardedEvidence, "fit", record_fit), patch("builtins.print"):
                result = diagnostic.run(seed, root, parent, source, fresh)
            np.testing.assert_array_equal(observed[0][0], fit_x)
            np.testing.assert_array_equal(observed[0][1], fit_y)
            self.assertTrue(result["frozen_scores_and_alarms_reproduced"])
            self.assertEqual((result["predictor_optimizer_steps"], result["residual_scale_optimizer_steps"]), (6, 9))
            self.assertEqual(result["additional_interactions"], 0)
            self.assertEqual(result["decision"], "hold_conditional_evidence_integration")
            self.assertEqual(result["frozen_threshold"], calibration["error_threshold"])
            with patch.object(diagnostic.GuardedEvidence, "fit", side_effect=AssertionError("Repeated fit")):
                self.assertEqual(diagnostic.run(seed, root, parent, source, fresh), result)
            self.assertEqual(json.loads((parent/env.replace("-", "_")/f"test_seed{seed}"/"readout.json").read_text()), prior)
        print("RECONSTRUCTION_TEST interactions=0 mean_updates=12 scale_updates=18 affine_fits=4")


if __name__ == "__main__":
    torch.set_num_threads(2)
    unittest.main()
