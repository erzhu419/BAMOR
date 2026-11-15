import copy
import tempfile
from pathlib import Path
import unittest
from unittest.mock import patch

import numpy as np
import torch

from headwise_change_evidence import HeadwiseChangeEvidence, ARMS, CONFIG
import run_conditional_change_probe as original_probe
import run_headwise_change_probe as probe


def targets(inputs):
    return np.column_stack([inputs[:, 0], -np.sum(inputs[:, 3:] ** 2, axis=1), inputs[:, :3]])


def fitted_evidence():
    rng = np.random.default_rng(4)
    fit_x = rng.normal(size=(240, 5)).astype(np.float32)
    scale_x = rng.normal(size=(120, 5)).astype(np.float32)
    evidence = HeadwiseChangeEvidence(5, 5, 2, hidden=8, updates=3, scale_updates=3, window=10)
    work = evidence.fit(fit_x, targets(fit_x), 120, 14)
    scale_work = evidence.fit_scales(scale_x, targets(scale_x), 17)
    return evidence, rng, work, scale_work


class HeadwiseChangeTests(unittest.TestCase):
    def test_four_source_roles_are_disjoint(self):
        for seed in probe.SEEDS:
            roles = probe.fold_roles(seed)
            self.assertEqual(len(roles["fit_seeds"]), 2)
            ordered = roles["fit_seeds"] + [roles[key] for key in ("scale_seed", "calibration_seed", "test_seed")]
            self.assertEqual(len(set(ordered)), 5)
            self.assertEqual(set(ordered), set(probe.SEEDS))

    def test_reduced_support_requires_cross_source_reward_prediction(self):
        evidence, _, work, scale_work = fitted_evidence()
        self.assertEqual(evidence.action_heads.tolist(), [False, True])
        self.assertEqual(evidence.domains, ["state_action", "action", "state_action"])
        self.assertGreater(min(row[0] for row in work["action_validation_mse"]), CONFIG["action_validation_mse"])
        self.assertLess(max(row[1] for row in work["action_validation_mse"]), CONFIG["action_validation_mse"])
        self.assertEqual((work["optimizer_steps"], scale_work["optimizer_steps"]), (6, 9))

    def test_state_ood_preserves_only_valid_action_evidence(self):
        evidence, rng, _, _ = fitted_evidence()
        cal_x = rng.normal(size=(100, 5)).astype(np.float32)
        evidence.calibrate(cal_x, targets(cal_x))
        changed_x = cal_x.copy()
        changed_x[:, :3] += 100
        changed_y = targets(changed_x)
        changed_y[:, 1] += 10
        evaluation = evidence.evaluate(changed_x, changed_y, "conditional_scale")
        unchanged = evidence.evaluate(changed_x, targets(changed_x), "conditional_scale")
        np.testing.assert_array_equal(evaluation["head_supported"], unchanged["head_supported"])
        self.assertFalse(evaluation["head_supported"][:, 0].any())
        self.assertFalse(evaluation["head_supported"][:, 2].any())
        self.assertGreater(evaluation["head_supported"][:, 1].mean(), .9)
        self.assertTrue(all(window["eligible"] and window["alarm"] for window in evaluation["windows"]))
        changed_x[:, 3:] += 100
        unavailable = evidence.evaluate(changed_x, changed_y, "conditional_scale")
        self.assertTrue(all(not row["eligible"] and not row["alarm"] for row in unavailable["windows"]))

    def test_input_only_scales_and_models_freeze_before_calibration(self):
        evidence, rng, _, _ = fitted_evidence()
        inputs = rng.normal(size=(100, 5)).astype(np.float32)
        models = [evidence.reward_predictor.model, evidence.dynamics_predictor.model] + evidence.scale_models
        states = [copy.deepcopy(model.state_dict()) for model in models]
        scales = evidence.scales(inputs, "conditional_scale").copy()
        means = evidence.global_scales.copy()
        evidence.calibrate(inputs, targets(inputs) + 20)
        evaluation = evidence.evaluate(inputs, targets(inputs) - 20, "conditional_scale")
        np.testing.assert_array_equal(evidence.scales(inputs, "conditional_scale"), scales)
        np.testing.assert_array_equal(evidence.global_scales, means)
        np.testing.assert_allclose(evaluation["errors"], evidence.raw_losses(inputs, targets(inputs) - 20) / scales)
        np.testing.assert_array_equal(evidence.scales(inputs, "global_scale"), np.broadcast_to(means, scales.shape))
        self.assertTrue(all(torch.equal(state[key], model.state_dict()[key])
                            for state, model in zip(states, models) for key in state))
        self.assertTrue(all(not parameter.requires_grad for model in models for parameter in model.parameters()))

    def test_joint_calibration_uses_maximum_of_eligible_head_means(self):
        evidence = HeadwiseChangeEvidence(5, 5, 2, window=10)
        evidence.domains = ["state_action", "action", "state_action"]
        errors = np.tile([2., 4., 1000.], (20, 1))
        supported = np.ones((20, 3), bool)
        supported[:, 2] = False
        supported[10:, :2] = False
        rows = evidence._windows(errors, supported)
        self.assertEqual(rows[0]["statistic"], 4.)
        self.assertTrue(rows[0]["eligible"])
        self.assertFalse(rows[0]["heads"]["dynamics"]["eligible"])
        self.assertIsNone(rows[0]["heads"]["dynamics"]["statistic"])
        self.assertFalse(rows[1]["eligible"])
        self.assertIsNone(rows[1]["statistic"])

    def test_missing_comparisons_still_fail_source_screen(self):
        cases = {case: dict(available=True, any_post_alarm=case.startswith("mode"),
            timely_alarm=case.startswith("mode"), pre_available=True, pre_alarm_windows=0,
            intervention_active=True) for case in probe.CASES}
        runs = [dict(arms={arm: dict(cases=copy.deepcopy(cases)) for arm in ARMS}) for _ in probe.SEEDS]
        self.assertTrue(probe.screen(runs, "conditional_scale")["development_screen_passed"])
        for row in runs[:2]:
            row["arms"]["conditional_scale"]["cases"]["mode2"].update(available=False, timely_alarm=False)
        self.assertFalse(probe.screen(runs, "conditional_scale")["development_screen_passed"])
        self.assertTrue(probe.screen(runs, "global_scale")["development_screen_passed"])

    def test_stored_run_respects_all_roles_and_has_no_interactions(self):
        with tempfile.TemporaryDirectory() as tmp, patch.dict(CONFIG, updates=3, scale_updates=3, hidden=8):
            source, root = Path(tmp) / "source", Path(tmp) / "output"
            env = probe.ENVIRONMENTS[0]
            for seed in probe.SEEDS:
                rng = np.random.default_rng(seed)
                prefix = rng.normal(size=(3000, 5)).astype(np.float32)
                stationary = rng.normal(size=(1000, 5)).astype(np.float32)
                prefix[:, :3] += .05 * (seed - 2101)
                stationary[:, :3] += .05 * (seed - 2101)
                terminal = np.zeros((3000, 2), bool)
                terminal[999::1000, 1] = True
                data = dict(prefix_inputs=prefix, prefix_targets=targets(prefix), prefix_terminals=terminal,
                            prefix_weights=np.zeros((3000, 2), np.float32))
                cases = {}
                for case in probe.CASES:
                    inputs = stationary.copy()
                    if case in ("actor_change", "sampler_change"):
                        inputs[500:, 3:] *= .8
                    output = targets(inputs)
                    if case.startswith("mode"):
                        output[500:, :2] += 500
                    end = np.zeros((1000, 2), bool)
                    end[-1, 1] = True
                    data.update({f"{case}_inputs": inputs, f"{case}_targets": output, f"{case}_terminals": end})
                    cases[case] = dict(ground_truth_mode_history_audit_only=(
                        [[0, 0], [500, int(case[-1])]] if case.startswith("mode") else [[0, 0]]), intervention_active=True)
                probe.write(probe.source_folder(source, env, seed) / "readout.json",
                    dict(protocol=original_probe.PROTOCOL, env=env, seed=seed, complete=True,
                         paired_prefixes_verified=5, cases=cases))
                np.savez_compressed(probe.source_folder(source, env, seed) / "transitions.npz", **data)
            observed = []
            fit, fit_scales, calibrate = (HeadwiseChangeEvidence.fit, HeadwiseChangeEvidence.fit_scales,
                                        HeadwiseChangeEvidence.calibrate)

            def record_fit(evidence, inputs, outputs, split_at, seed):
                observed.append(("fit", inputs.copy(), outputs.copy()))
                return fit(evidence, inputs, outputs, split_at, seed)

            def record_scales(evidence, inputs, outputs, seed):
                observed.append(("scale", inputs.copy(), outputs.copy()))
                return fit_scales(evidence, inputs, outputs, seed)

            def record_calibration(evidence, inputs, outputs):
                observed.append(("calibration", inputs.copy(), outputs.copy()))
                return calibrate(evidence, inputs, outputs)

            with patch.object(HeadwiseChangeEvidence, "fit", record_fit), \
                 patch.object(HeadwiseChangeEvidence, "fit_scales", record_scales), \
                 patch.object(HeadwiseChangeEvidence, "calibrate", record_calibration), \
                 patch.object(original_probe.training, "make_piecewise_mo_env", side_effect=AssertionError("new interaction")), \
                 patch("builtins.print"):
                report = probe.run(env, 2101, source, root)
            roles = probe.fold_roles(2101)
            expected_fit = np.concatenate([probe.normal_data(source, env, seed, "expanded_normal")[0]
                                           for seed in roles["fit_seeds"]])
            expected_scale = probe.normal_data(source, env, roles["scale_seed"], "expanded_normal")[0]
            expected_cal = probe.normal_data(source, env, roles["calibration_seed"], "expanded_normal")[0]
            for recorded, expected in zip(observed, (expected_fit, expected_scale, expected_cal)):
                np.testing.assert_array_equal(recorded[1], expected)
                self.assertLess(float(recorded[2].max()), 500.)
            self.assertEqual([row[0] for row in observed], ["fit", "scale", "calibration"])
            self.assertTrue(report["complete"] and report["development_only"])
            self.assertEqual(report["additional_interactions"], 0)
            self.assertEqual((report["predictor_optimizer_steps"], report["residual_scale_optimizer_steps"]), (6, 9))
            self.assertEqual((report["fit_steps"], report["scale_steps"], report["calibration_steps"]), (10000, 5000, 5000))
            for case in probe.CASES:
                for arm in ARMS:
                    self.assertIn("dynamics", report["arms"][arm]["cases"][case]["heads"])
            with patch.object(HeadwiseChangeEvidence, "fit", side_effect=AssertionError("repeated fit")):
                self.assertEqual(probe.run(env, 2101, source, root), report)
        print("HEADWISE_TEST interactions=0 predictor_updates=24 scale_updates=36")


if __name__ == "__main__":
    torch.set_num_threads(2)
    unittest.main()
