from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import torch

from conditional_change_evidence import ConditionalChangeEvidence, CONFIG
import diagnose_conditional_residuals as diagnostic
from diagnose_conditional_residuals import describe_phase, episode_age, gaps, neighbors


class ConditionalResidualTests(unittest.TestCase):
    def test_matching_excludes_entire_calibration_window(self):
        inputs = np.asarray([[0.], [.01], [2.], [2.01]], np.float32)
        values = np.asarray([[1.], [3.], [7.], [9.]])
        blocks = np.asarray([0, 0, 1, 1])
        estimate, distance = neighbors(inputs, inputs, values, k=2,
                                       query_blocks=blocks, reference_blocks=blocks)
        np.testing.assert_allclose(estimate[:, 0], [8., 8., 2., 2.])
        self.assertTrue(np.all(distance > 1.9))

    def test_exact_gap_decomposition_and_input_mix_only(self):
        baseline = np.asarray([[1., 2., 3.], [3., 6., 9.]])
        observed = np.repeat(baseline[1:], 4, axis=0)
        result = gaps(observed, observed, baseline, baseline)
        np.testing.assert_allclose(result["remaining_gap"], 0.)
        np.testing.assert_allclose(result["observed_gap"], result["input_mix_gap"])
        changed = gaps(observed + 2., observed, baseline, baseline - .5)
        np.testing.assert_allclose(np.asarray(changed["input_mix_gap"]) + changed["remaining_gap"], changed["observed_gap"])
        np.testing.assert_allclose(changed["remaining_gap"], 1.5)

    def test_unsupported_alarm_retained_but_not_decomposed(self):
        row = dict(end_step=100, original_eligible=True, original_alarm=True,
            matched_eligible=False, matched_steps=0, fit_supported_steps=90,
            early_episode_fraction=1., loss_means=[5., 3., 4.], gaps=None)
        result = describe_phase([row])
        self.assertEqual(result["original_alarms"], 1)
        self.assertEqual(result["matched_alarm_windows"], 0)
        self.assertIsNone(result["observed_gap"])
        self.assertEqual(result["alarm_details"][0]["matched_steps"], 0)

    def test_episode_ages_reset_after_both_terminal_types(self):
        ages, episodes = episode_age([[False, False], [True, False],
                                      [False, True], [False, False]])
        np.testing.assert_array_equal(ages, [0, 1, 0, 0])
        np.testing.assert_array_equal(episodes, [0, 0, 1, 2])

    def test_stored_only_runner_reconstruction_summary_and_reuse(self):
        rng = np.random.default_rng(7)
        inputs = rng.normal(size=(3000, 3)).astype(np.float32)
        targets = np.column_stack([inputs, inputs[:, 0] ** 2]).astype(np.float32)
        evidence = ConditionalChangeEvidence(3, 4, 2)
        evidence.fit(inputs[:2000], targets[:2000], 40002201)
        calibration = evidence.calibrate(inputs[2000:], targets[2000:])
        test_x = rng.normal(size=(1000, 3)).astype(np.float32)
        test_y = np.column_stack([test_x, test_x[:, 0] ** 2]).astype(np.float32)
        evaluation = evidence.evaluate(test_x, test_y)
        data = dict(prefix_inputs=inputs, prefix_targets=targets,
            prefix_weights=np.zeros((3000, 2), np.float32), prefix_terminals=np.zeros((3000, 2), bool))
        original = dict(protocol="conditional_change_evidence_v1", env=diagnostic.ENVS[0],
            seed=2101, probe_seed=2201, complete=True, predictor_config=CONFIG,
            calibration=calibration, cases={})
        for case in diagnostic.CASES:
            original["cases"][case] = dict(windows=evaluation["windows"])
            for key, value in dict(inputs=test_x, targets=test_y,
                terminals=np.zeros((1000, 2), bool), errors=evaluation["errors"],
                distances=evaluation["distances"], supported=evaluation["supported"]).items():
                data[f"{case}_{key}"] = value
        with tempfile.TemporaryDirectory() as tmp:
            source, root = Path(tmp) / "source", Path(tmp) / "diagnostic"
            folder = source / diagnostic.ENVS[0].replace("-", "_") / "seed2101"
            diagnostic.write(folder / "readout.json", original)
            np.savez_compressed(folder / "transitions.npz", **data)
            with patch("builtins.print"):
                report = diagnostic.run(diagnostic.ENVS[0], 2101, source, root)
            self.assertTrue(report["original_reproduced"])
            self.assertEqual(report["maximum_error_delta"], 0.)
            self.assertEqual(report["additional_interactions"], 0)
            self.assertEqual(report["predictor_reconstruction_steps"], 500)
            self.assertEqual(len(report["phases"]), 7)
            self.assertEqual(report["calibration"]["episode_segments"], 1)
            with patch.object(ConditionalChangeEvidence, "fit", side_effect=AssertionError("refit on reuse")):
                self.assertEqual(diagnostic.run(diagnostic.ENVS[0], 2101, source, root), report)
            with patch.object(diagnostic, "ENVS", (diagnostic.ENVS[0],)), patch.object(diagnostic, "SEEDS", (2101,)):
                compact = diagnostic.summarize(root)
            self.assertEqual(compact["decision"], "hold_conditional_evidence_integration")
            self.assertEqual(compact["predictor_reconstruction_steps"], 500)
            for phase in compact["environments"][diagnostic.ENVS[0]]["phases"].values():
                if phase["observed_gap"] is not None:
                    np.testing.assert_allclose(np.asarray(phase["input_mix_gap"]) + phase["remaining_gap"],
                                               phase["observed_gap"], atol=1e-12)
        print("STORED_RESIDUAL_TEST interactions=0 predictor_updates=1000")


if __name__ == "__main__":
    torch.set_num_threads(2)
    unittest.main()
