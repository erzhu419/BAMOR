import copy
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import torch

from affine_residual_change_evidence import AffineResidualPredictor, AffineResidualHeadwiseEvidence
from headwise_change_evidence import CONFIG
import run_affine_residual_change_probe as probe
import run_headwise_change_probe as control
import run_conditional_change_probe as original
from test_headwise_change_probe import targets


def make_source(source, env):
    for seed in probe.SEEDS:
        rng = np.random.default_rng(seed)
        prefix = rng.normal(size=(3000, 5)).astype(np.float32)
        stationary = rng.normal(size=(1000, 5)).astype(np.float32)
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
            dict(protocol=original.PROTOCOL, env=env, seed=seed, complete=True,
                 paired_prefixes_verified=5, cases=cases))
        np.savez_compressed(probe.source_folder(source, env, seed) / "transitions.npz", **data)


class AffineResidualTests(unittest.TestCase):
    def test_affine_fit_generalizes_and_residual_composes_in_original_units(self):
        rng = np.random.default_rng(7)
        x = rng.normal(size=(240, 5)).astype(np.float32)
        coefficients = rng.normal(size=(5, 2)).astype(np.float32)
        y = x @ coefficients + np.array([2., -3.], np.float32)
        evidence = AffineResidualPredictor(5, 2, 2, hidden=8, updates=3)
        work = evidence.fit(x, y, 19)
        heldout = rng.normal(size=(100, 5)).astype(np.float32) + 5
        np.testing.assert_allclose(evidence.affine_predict(heldout), heldout @ coefficients + [2., -3.], atol=1e-5)
        normalized = torch.as_tensor((heldout - evidence.x_mean) / evidence.x_scale)
        np.testing.assert_allclose(evidence.predict(heldout), evidence.affine_predict(heldout)
            + evidence.model(normalized).numpy() * evidence.y_scale, atol=5e-6)
        self.assertEqual((work["optimizer_steps"], work["affine_fit_rows"], work["affine_rank"]), (3, 240, 6))
        self.assertTrue(all(not p.requires_grad for p in evidence.model.parameters()))

    def test_reproducible_fit_does_not_change_callers_rng(self):
        x = np.random.default_rng(5).normal(size=(200, 5)).astype(np.float32)
        state = torch.random.get_rng_state().clone()
        models = [AffineResidualPredictor(5, 5, 2, hidden=8, updates=3) for _ in range(2)]
        for model in models:
            model.fit(x, targets(x), 51)
        self.assertTrue(torch.equal(state, torch.random.get_rng_state()))
        np.testing.assert_array_equal(models[0].predict(x), models[1].predict(x))

    def test_action_heads_and_scales_remain_frozen_at_calibration(self):
        rng = np.random.default_rng(4)
        x = rng.normal(size=(240, 5)).astype(np.float32)
        scale_x = rng.normal(size=(120, 5)).astype(np.float32)
        evidence = AffineResidualHeadwiseEvidence(5, 5, 2, hidden=8, updates=3, scale_updates=3, window=10)
        work = evidence.fit(x, targets(x), 120, 14)
        scale_work = evidence.fit_scales(scale_x, targets(scale_x), 17)
        self.assertEqual(evidence.action_heads.tolist(), [False, True])
        self.assertEqual((work["optimizer_steps"], scale_work["optimizer_steps"]), (6, 9))
        cal_x = rng.normal(size=(100, 5)).astype(np.float32)
        prediction = evidence.reward_predictor.predict(cal_x).copy()
        coefficients = evidence.reward_predictor.affine_coefficients.copy()
        scales = evidence.scales(cal_x, "conditional_scale").copy()
        evidence.calibrate(cal_x, targets(cal_x) + 100)
        np.testing.assert_array_equal(evidence.reward_predictor.predict(cal_x), prediction)
        np.testing.assert_array_equal(evidence.reward_predictor.affine_coefficients, coefficients)
        np.testing.assert_array_equal(evidence.scales(cal_x, "conditional_scale"), scales)
        np.testing.assert_allclose(evidence.affine_losses(cal_x, targets(cal_x))[:, 1],
                                  evidence.raw_losses(cal_x, targets(cal_x))[:, 1])

    def test_stored_run_and_compact_summary_preserve_roles_controls_and_budget(self):
        with tempfile.TemporaryDirectory() as tmp, patch.dict(CONFIG, hidden=8, updates=3, scale_updates=3):
            source, root, old = (Path(tmp) / name for name in ("source", "candidate", "control"))
            env = probe.ENVIRONMENTS[0]
            make_source(source, env)
            observed = []
            fit = AffineResidualHeadwiseEvidence.fit
            def record_fit(evidence, inputs, outputs, split_at, seed):
                observed.append((inputs.copy(), outputs.copy()))
                return fit(evidence, inputs, outputs, split_at, seed)
            with patch("builtins.print"), patch.object(original.training, "make_piecewise_mo_env",
                    side_effect=AssertionError("new interaction")):
                baseline = control.run(env, 2101, source, old)
                with patch.object(AffineResidualHeadwiseEvidence, "fit", record_fit):
                    report = probe.run(env, 2101, source, root, old)
            expected = np.concatenate([probe.normal_data(source, env, seed, "expanded_normal")[0]
                                       for seed in report["roles"]["fit_seeds"]])
            np.testing.assert_array_equal(observed[0][0], expected)
            self.assertLess(float(observed[0][1].max()), 500)
            self.assertEqual(report["plain_mlp_control"]["arms"], baseline["arms"])
            self.assertEqual((report["predictor_optimizer_steps"], report["residual_scale_optimizer_steps"]), (6, 9))
            self.assertEqual(report["additional_interactions"], 0)
            self.assertEqual(set(report["normal_losses"]), {
                "fit", "scale", "calibration", "test_stationary", "test_sampler_change", "test_actor_change"})
            for role in ("reward", "dynamics"):
                self.assertEqual(report["predictor_work"][role]["affine_fit_rows"], 10000)
            with patch.object(AffineResidualHeadwiseEvidence, "fit", side_effect=AssertionError("repeated fit")):
                self.assertEqual(probe.run(env, 2101, source, root, old), report)
            with patch.object(probe, "ENVIRONMENTS", (env,)), patch.object(probe, "SEEDS", [2101]):
                summary = probe.summarize(root)
            compact = probe.compact_report(summary)
            self.assertTrue(summary["complete"])
            self.assertFalse(summary["evidence_integrated"])
            self.assertEqual(summary["retained_method"], "v25")
            self.assertEqual(summary["decision"], "hold_conditional_evidence_integration")
            for collection in (compact["environments"][env]["runs"][0]["arms"],
                               compact["environments"][env]["runs"][0]["plain_mlp_control"]["arms"]):
                for arm in collection.values():
                    self.assertNotIn("windows", arm["calibration"])
                    self.assertTrue(all("windows" not in case for case in arm["cases"].values()))
            self.assertIn("windows", summary["environments"][env]["runs"][0]["arms"]["global_scale"]["calibration"])
            changed = copy.deepcopy(baseline)
            changed["roles"]["scale_seed"] = 2101
            with self.assertRaises(AssertionError):
                probe.validate_control(changed, env, 2101)
        print("AFFINE_RESIDUAL_STORED_RUN_TEST interactions=0 predictor_updates=12 scale_updates=18")


if __name__ == "__main__":
    torch.set_num_threads(2)
    unittest.main()
