import copy
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import torch

from affine_residual_change_evidence import AffineResidualHeadwiseEvidence
from headwise_change_evidence import HeadwiseChangeEvidence, CONFIG
import run_calibration_expansion as probe
import run_matched_residual_scale_probe as matched
import run_affine_residual_change_probe as affine
import run_headwise_change_probe as plain
import run_conditional_change_probe as original
from test_affine_residual_change_probe import make_source
from test_matched_residual_scale_probe import make_fresh_fixture


def fixed_evidence(losses, distances, window=2):
    evidence = HeadwiseChangeEvidence(2, 2, 1, window=window)
    evidence.radii = np.array([.5, .5])
    evidence.thresholds = dict(global_scale=20., conditional_scale=20.)
    evidence.domains = ["state_action", "state_action"]
    evidence.distances = lambda inputs: np.asarray(distances)
    evidence.raw_losses = lambda inputs, targets: np.asarray(losses)
    evidence.scales = lambda inputs, arm: np.ones((len(inputs), 2))
    return evidence


class CalibrationExpansionTests(unittest.TestCase):
    def test_support_radii_remain_fixed_and_joint_scores_use_eligible_heads(self):
        losses = np.array([[1., .1], [3., .1], [1000., 3.], [1000., 5.]])
        evidence = fixed_evidence(losses, [[.1, .1], [.2, .2], [5., .1], [5., .1]])
        before = evidence.radii.copy()
        report = probe.calibrate_fixed_support(evidence, np.zeros((4, 2)), np.zeros((4, 2)), 2)
        np.testing.assert_array_equal(evidence.radii, before)
        for arm in report.values():
            self.assertEqual(arm["windows"][1]["heads"]["reward_0"]["supported_steps"], 0)
            self.assertEqual(arm["windows"][1]["statistic"], 4.)
            self.assertEqual(arm["fresh_eligible_windows"], 1)
            self.assertAlmostEqual(arm["error_threshold"], float(np.quantile([2., 4.], .99)))

    def test_fixed_quantile_can_lower_threshold_without_a_monotone_floor(self):
        values = np.r_[np.ones(49), 20., np.zeros(50)]
        losses = np.column_stack([np.repeat(values, 2)] * 2)
        evidence = fixed_evidence(losses, np.zeros((200, 2)))
        previous = float(np.quantile(values[:50], .99))
        evidence.thresholds = {arm: previous for arm in probe.ARMS}
        result = probe.calibrate_fixed_support(evidence, np.zeros((200, 2)), np.zeros((200, 2)), 100)
        self.assertAlmostEqual(result["conditional_scale"]["error_threshold"], float(np.quantile(values, .99)))
        self.assertLess(result["conditional_scale"]["error_threshold"], previous)

    def test_score_reproduction_ignores_deliberate_alarm_changes_but_rejects_score_changes(self):
        evidence = fixed_evidence([[1., 2.]] * 4, [[0., 0.]] * 4)
        saved = probe.calibrate_fixed_support(evidence, np.zeros((4, 2)), np.zeros((4, 2)), 2)["global_scale"]["windows"]
        changed = copy.deepcopy(saved)
        for row in changed:
            row["alarm"] = True
            for head in row["heads"].values():
                head["alarm"] = True
        probe.verify_windows(changed, saved)
        changed[0]["heads"]["reward_0"]["statistic"] += .1
        with self.assertRaises(AssertionError):
            probe.verify_windows(changed, saved)

    def test_stored_pipeline_roles_frozen_models_controls_and_reconstruction_cost(self):
        with tempfile.TemporaryDirectory() as tmp, patch.dict(CONFIG, hidden=8, updates=3, scale_updates=3):
            source, fresh, parent, base, old, root = (Path(tmp) / name for name in
                ("source", "fresh", "matched", "plain", "affine", "expanded"))
            env = probe.ENVIRONMENTS[0]
            make_source(source, env)
            make_fresh_fixture(fresh, env)
            with patch("builtins.print"), patch.object(original.training, "make_piecewise_mo_env",
                    side_effect=AssertionError("No new environment")):
                plain.run(env, 2101, source, base)
                affine.run(env, 2101, source, old, base)
                control = matched.run(env, 2101, source, parent, old, fresh)
                observed, read_seeds = [], []
                calibrate = probe.calibrate_fixed_support
                fresh_data = probe.fresh_data
                def record_fresh(root, env, seed):
                    read_seeds.append(seed)
                    return fresh_data(root, env, seed)
                def record_calibration(evidence, inputs, targets, original_steps):
                    models = [evidence.reward_predictor.model, evidence.dynamics_predictor.model, *evidence.scale_models]
                    states = [copy.deepcopy(model.state_dict()) for model in models]
                    radii = evidence.radii.copy()
                    scales = evidence.scales(inputs, "conditional_scale").copy()
                    result = calibrate(evidence, inputs, targets, original_steps)
                    np.testing.assert_array_equal(evidence.radii, radii)
                    np.testing.assert_array_equal(evidence.scales(inputs, "conditional_scale"), scales)
                    self.assertTrue(all(not p.requires_grad for model in models for p in model.parameters()))
                    self.assertTrue(all(torch.equal(state[key], model.state_dict()[key])
                        for state, model in zip(states, models) for key in state))
                    observed.append((inputs.copy(), targets.copy(), original_steps))
                    return result
                with patch.object(probe, "fresh_data", record_fresh), \
                     patch.object(probe, "calibrate_fixed_support", record_calibration):
                    result = probe.run(env, 2101, source, root, parent, fresh)
            roles = result["roles"]
            self.assertEqual(read_seeds, roles["fit_seeds"] + [roles["calibration_seed"]])
            self.assertNotIn(roles["calibration_seed"], result["scale_fit_sources"])
            self.assertNotIn(roles["test_seed"], read_seeds)
            self.assertEqual((result["predictor_optimizer_steps"], result["residual_scale_optimizer_steps"]), (6, 9))
            self.assertEqual(result["matched_scale_control"]["arms"], control["arms"])
            self.assertTrue(result["original_calibration_reproduced"] and result["evaluation_scores_unchanged"])
            self.assertEqual(result["additional_interactions"], 0)
            expected = [probe.normal_data(source, env, roles["calibration_seed"], "expanded_normal"),
                        probe.fresh_data(fresh, env, roles["calibration_seed"])]
            for index in (0, 1):
                np.testing.assert_array_equal(observed[0][index], np.concatenate([row[index] for row in expected]))
            self.assertEqual(observed[0][2], 5000)
            with patch.object(AffineResidualHeadwiseEvidence, "fit", side_effect=AssertionError("Repeated fit")):
                self.assertEqual(probe.run(env, 2101, source, root, parent, fresh), result)
            for seed in probe.SEEDS:
                copied = copy.deepcopy(result)
                copied["roles"] = probe.fold_roles(seed)
                copied["scale_fit_sources"] = copied["roles"]["fit_seeds"] + [copied["roles"]["scale_seed"]]
                copied["fresh_calibration_source"]["seed"] = copied["roles"]["calibration_seed"]
                probe.write(root / env.replace("-", "_") / f"test_seed{seed}" / "readout.json", copied)
            with patch.object(probe, "ENVIRONMENTS", (env,)):
                report = probe.summarize(root)
            self.assertEqual(report["predictor_optimizer_steps"], 30)
            self.assertEqual(report["residual_scale_optimizer_steps"], 45)
            self.assertEqual(report["additional_interactions"], 0)
            compact = probe.compact_report(report)
            self.assertNotIn("windows", compact["environments"][env]["runs"][0]["arms"]["conditional_scale"]["calibration"])
        print("CALIBRATION_EXPANSION_TEST interactions=0 mean_updates=24 scale_updates=36")


if __name__ == "__main__":
    torch.set_num_threads(2)
    unittest.main()
