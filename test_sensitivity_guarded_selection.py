import copy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import torch

from headwise_change_evidence import CONFIG
import run_sensitivity_guarded_selection as probe
import run_window_risk_selection as previous
import run_matched_residual_scale_probe as matched
import run_affine_residual_change_probe as affine
import run_headwise_change_probe as plain
import run_conditional_change_probe as original
from test_affine_residual_change_probe import make_source
from test_matched_residual_scale_probe import make_fresh_fixture
import test_window_risk_selection as risk_fixture


class SensitivityGuardTests(unittest.TestCase):
    def example_folds(self):
        evidence, validation, calibration = risk_fixture.WindowRiskTests().example()
        selection = previous.window_selection(evidence, validation, calibration)
        changes = {case: (np.zeros((1000, 1)), np.full((1000, 2), 4.)) for case in probe.MODE_CASES}
        probe.add_change_sensitivity(evidence, changes, selection)
        return [dict(held_source=s, validation_steps=4, selection=copy.deepcopy(selection)) for s in (21, 22, 23)]

    def test_guard_rejects_low_false_alarm_candidate_that_loses_changes(self):
        result = probe.aggregate_selection(self.example_folds())
        self.assertEqual(result[probe.RISK]["global_weight"], 1.)
        self.assertNotEqual(result[probe.GUARDED]["global_weight"], 1.)
        self.assertEqual(len(result[probe.GUARDED]["protected_probes"]), 9)
        self.assertEqual(result[probe.GUARDED]["lost_protected_probes"], [])
        self.assertFalse(result[probe.GUARDED]["candidates"][0]["sensitivity_feasible"])
        self.assertEqual(result[probe.RISK], previous.aggregate_selection(self.example_folds())[probe.RISK])

    def test_each_probe_is_protected_not_only_mean_detection(self):
        folds = self.example_folds()
        for fold in folds:
            for candidate in fold["selection"]["candidates"]:
                for case in probe.MODE_CASES:
                    candidate["change_probes"][case]["timely_alarm"] = True
        folds[1]["selection"]["candidates"][0]["change_probes"]["mode2"]["timely_alarm"] = False
        result = probe.aggregate_selection(folds)[probe.GUARDED]
        self.assertEqual(result["candidates"][0]["lost_protected_probes"], [dict(source_seed=22, case="mode2")])
        self.assertNotEqual(result["global_weight"], 1.)
        self.assertTrue(result["candidates"][-1]["sensitivity_feasible"])

    def test_uninformative_probe_set_is_reported_not_fabricated(self):
        folds = self.example_folds()
        for fold in folds:
            for candidate in fold["selection"]["candidates"]:
                for case in probe.MODE_CASES:
                    candidate["change_probes"][case]["timely_alarm"] = False
        result = probe.aggregate_selection(folds)
        self.assertEqual(result[probe.GUARDED]["global_weight"], result[probe.RISK]["global_weight"])
        self.assertEqual(result[probe.GUARDED]["protected_probes"], [])

    def test_real_source_labels_and_requested_policy_are_used(self):
        with tempfile.TemporaryDirectory() as tmp:
            source, env = Path(tmp), probe.ENVIRONMENTS[0]
            make_source(source, env)
            rows = probe.training_changes(source, env, 2103)
            self.assertEqual(set(rows), set(probe.MODE_CASES))
            self.assertTrue(all(len(x) == 1000 and float(y[500:, :2].mean()) > 400 for x, y in rows.values()))
            path = probe.source_folder(source, env, 2103) / "readout.json"
            audit = json.loads(path.read_text())
            audit["cases"]["mode2"]["ground_truth_mode_history_audit_only"] = [[0, 0]]
            probe.write(path, audit)
            with self.assertRaises(AssertionError):
                probe.training_changes(source, env, 2103)

    def test_changes_are_not_used_in_scale_training(self):
        evidence, _, calibration = risk_fixture.WindowRiskTests().example()
        calls = []
        def fit(inputs, losses, seed):
            calls.append((inputs.copy(), losses.copy()))
            evidence.global_scales = losses.mean(0)
            return dict(optimizer_steps=0)
        evidence.fit_scale_losses = fit
        rows = [(np.full((4, 1), s), np.full((4, 2), s), dict(seed=s)) for s in (21, 22, 23)]
        changes = {s: {case: (np.full((1000, 1), 777), np.full((1000, 2), 888.))
            for case in probe.MODE_CASES} for s in (21, 22, 23)}
        folds = list(probe.source_validation_folds(evidence, rows, calibration, changes, 7))
        for held, (fold, call) in enumerate(zip(folds, calls)):
            train = [i for i in range(3) if i != held]
            np.testing.assert_array_equal(call[0], np.concatenate([rows[i][0] for i in train]))
            np.testing.assert_array_equal(call[1], np.concatenate([rows[i][1] for i in train]))
            self.assertFalse(np.any(call[0] == 777) or np.any(call[1] == 888))
            self.assertEqual(fold["selection"]["change_probe_steps"], 3000)

    def test_pipeline_isolates_roles_reproduces_controls_and_counts_work(self):
        with tempfile.TemporaryDirectory() as tmp, patch.dict(CONFIG, hidden=8, updates=3, scale_updates=3):
            source, fresh, parent, base, old, prior, root = (Path(tmp) / name for name in
                ("source", "fresh", "matched", "plain", "affine", "previous", "guarded"))
            env, seed = probe.ENVIRONMENTS[0], 2101
            make_source(source, env)
            make_fresh_fixture(fresh, env)
            events, seeds, fit_targets = [], [], []
            changes, normal_data, aggregate, fit = probe.training_changes, probe.normal_data, probe.aggregate_selection, probe.GuardedEvidence.fit
            def read_changes(source, env, seed):
                seeds.append(seed)
                return changes(source, env, seed)
            def read_normal(source, env, seed, arm):
                events.append(("normal", seed))
                return normal_data(source, env, seed, arm)
            def choose(folds):
                events.append(("selection", None))
                return aggregate(folds)
            def fit_normal(evidence, x, y, split, fit_seed):
                fit_targets.append(y.copy())
                return fit(evidence, x, y, split, fit_seed)
            with patch("builtins.print"), patch.object(original.training, "make_piecewise_mo_env", side_effect=AssertionError("No new environment")):
                plain.run(env, seed, source, base)
                affine.run(env, seed, source, old, base)
                matched.run(env, seed, source, parent, old, fresh)
                before = previous.run(env, seed, source, prior, parent, fresh)
                with patch.object(probe, "training_changes", read_changes), patch.object(probe, "normal_data", read_normal), \
                     patch.object(probe, "aggregate_selection", choose), patch.object(probe.GuardedEvidence, "fit", fit_normal):
                    result = probe.run(env, seed, source, root, parent, fresh, prior)
            roles = result["roles"]
            self.assertEqual(seeds, roles["fit_seeds"] + [roles["scale_seed"]])
            self.assertLess(events.index(("selection", None)), events.index(("normal", roles["calibration_seed"])))
            self.assertEqual(len(fit_targets), 1)
            self.assertLess(float(fit_targets[0].max()), 500)
            self.assertEqual(result["training_change_probe_steps"], 9000)
            self.assertTrue(result["prior_selection_controls_reproduced"])
            for arm in (probe.RISK, probe.NLL):
                self.assertEqual(result["selections"][arm], before["selections"][arm])
                self.assertEqual(result["arms"][arm], before["arms"][arm])
            self.assertEqual((result["predictor_optimizer_steps"], result["total_residual_scale_optimizer_steps"]), (6, 36))
            self.assertEqual(result["additional_interactions"], 0)
            self.assertEqual(result["selections"][probe.GUARDED]["lost_protected_probes"], [])
            with patch.object(probe.GuardedEvidence, "fit", side_effect=AssertionError("Repeated fit")):
                self.assertEqual(probe.run(env, seed, source, root, parent, fresh, prior), result)
            with patch.object(probe, "ENVIRONMENTS", (env,)), patch.object(probe, "SEEDS", (seed,)):
                summary = probe.summarize(root)
            self.assertEqual(summary["total_residual_scale_optimizer_steps"], 36)
            compact = probe.compact_report(summary)
            self.assertNotIn("windows", compact["environments"][env]["runs"][0]["arms"][probe.GUARDED]["calibration"])
        print("SENSITIVITY_GUARD_TEST interactions=0 mean_updates=30 total_scale_updates=99")


if __name__ == "__main__":
    torch.set_num_threads(2)
    unittest.main()
