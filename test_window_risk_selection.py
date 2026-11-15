import copy
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import torch

from headwise_change_evidence import CONFIG, HeadwiseChangeEvidence
import run_window_risk_selection as probe
import run_matched_residual_scale_probe as matched
import run_affine_residual_change_probe as affine
import run_headwise_change_probe as plain
import run_conditional_change_probe as original
from test_affine_residual_change_probe import make_source
from test_matched_residual_scale_probe import make_fresh_fixture


class WindowRiskTests(unittest.TestCase):
    def example(self):
        class Evidence:
            config = dict(CONFIG, window=2)
            head_names = ["reward_0", "dynamics"]
            domains = ["state_action", "state_action"]
            global_scales = np.ones(2)
            _windows = HeadwiseChangeEvidence._windows
            def distances(self, inputs):
                return np.zeros((len(inputs), 2))
            def raw_losses(self, inputs, targets):
                return targets
            def scales(self, inputs, arm):
                return np.tile([1., 9.], (len(inputs), 1))
        calibration = (np.full((4, 1), 99), np.tile([1., 9.], (4, 1)), dict(seed=99))
        validation = (np.full((4, 1), 21), np.full((4, 2), 4.), dict(seed=21))
        return Evidence(), validation, calibration

    def test_joint_tail_objective_can_disagree_with_nll(self):
        evidence, validation, calibration = self.example()
        result = probe.window_selection(evidence, validation, calibration)
        self.assertNotEqual(result["global_weight"], 1.)
        self.assertEqual(result["candidates"][0]["window_risk"], 0.)
        self.assertEqual(result["candidates"][-1]["window_risk"], 1.)
        folds = [dict(held_source=s, validation_steps=4, selection=result) for s in (21, 22, 23)]
        selected = probe.aggregate_selection(folds)
        self.assertEqual(selected[probe.RISK]["window_risk"], 0.)
        self.assertNotEqual(selected[probe.RISK]["global_weight"], selected[probe.NLL]["global_weight"])
        self.assertTrue(all(c["validation_eligible_windows"] == 2 for c in result["candidates"]))

    def test_strict_exceedance_and_larger_global_weight_exact_tie(self):
        evidence, validation, calibration = self.example()
        evidence.scales = lambda inputs, arm: np.ones((len(inputs), 2))
        validation = (validation[0], calibration[1], validation[2])
        result = probe.window_selection(evidence, validation, calibration)
        folds = [dict(held_source=s, validation_steps=4, selection=result) for s in (21, 22, 23)]
        for selected in probe.aggregate_selection(folds).values():
            self.assertEqual(selected["global_weight"], 1.)
        self.assertTrue(all(c["joint_window_exceedances"] == 0 for c in result["candidates"]))

    def test_unavailable_validation_cannot_win_by_abstaining(self):
        evidence, validation, calibration = self.example()
        evidence.distances = lambda x: np.zeros((len(x), 2)) if x[0, 0] == 99 else np.ones((len(x), 2))
        with self.assertRaisesRegex(AssertionError, "No supported"):
            probe.window_selection(evidence, validation, calibration)

    def test_source_and_inner_calibration_rows_excluded_from_scale_fits(self):
        evidence, _, calibration = self.example()
        calls = []
        def fit(inputs, losses, seed):
            calls.append((inputs.copy(), losses.copy(), seed))
            evidence.global_scales = losses.mean(0)
            return dict(optimizer_steps=0)
        evidence.fit_scale_losses = fit
        rows = [(np.full((4, 1), seed), np.full((4, 2), seed), dict(seed=seed)) for seed in (21, 22, 23)]
        folds = list(probe.source_validation_folds(evidence, rows, calibration, 7))
        for held, (fold, call) in enumerate(zip(folds, calls)):
            train = [i for i in range(3) if i != held]
            np.testing.assert_array_equal(call[0], np.concatenate([rows[i][0] for i in train]))
            np.testing.assert_array_equal(call[1], np.concatenate([rows[i][1] for i in train]))
            self.assertNotIn(fold["held_source"], fold["train_sources"])
            self.assertFalse(np.any(call[0] == 99))
            self.assertEqual(call[2], 7)
            self.assertEqual(fold["selection"]["inner_calibration_seed"], 99)

    def test_source_risk_is_equally_weighted_not_pooled(self):
        evidence, validation, calibration = self.example()
        result = probe.window_selection(evidence, validation, calibration)
        folds = []
        for seed, size, rate in [(21, 2, 1.), (22, 100, 0.), (23, 100, 0.)]:
            row = copy.deepcopy(result)
            for candidate in row["candidates"]:
                candidate.update(window_risk=rate, validation_eligible_windows=size,
                    validation_total_windows=size, joint_window_exceedances=int(rate * size))
            folds.append(dict(held_source=seed, validation_steps=size * 2, selection=row))
        self.assertAlmostEqual(probe.aggregate_selection(folds)[probe.RISK]["window_risk"], 1 / 3)

    def test_stored_pipeline_same_models_isolated_roles_and_counted_work(self):
        with tempfile.TemporaryDirectory() as tmp, patch.dict(CONFIG, hidden=8, updates=3, scale_updates=3):
            source, fresh, parent, base, old, root = (Path(tmp) / name for name in
                ("source", "fresh", "matched", "plain", "affine", "risk"))
            env = probe.ENVIRONMENTS[0]
            make_source(source, env)
            make_fresh_fixture(fresh, env)
            roles, events, fresh_seeds = probe.fold_roles(2101), [], []
            normal_data, fresh_data, aggregate = probe.normal_data, probe.fresh_data, probe.aggregate_selection
            def read_normal(root, env, seed, pool):
                events.append(("normal", seed))
                return normal_data(root, env, seed, pool)
            def read_fresh(root, env, seed):
                fresh_seeds.append(seed)
                return fresh_data(root, env, seed)
            def select(folds):
                events.append(("selected", None))
                return aggregate(folds)
            with patch("builtins.print"), patch.object(original.training, "make_piecewise_mo_env",
                    side_effect=AssertionError("No new environment")):
                plain.run(env, 2101, source, base)
                affine.run(env, 2101, source, old, base)
                control = matched.run(env, 2101, source, parent, old, fresh)
                with patch.object(probe, "normal_data", read_normal), patch.object(probe, "fresh_data", read_fresh), \
                     patch.object(probe, "aggregate_selection", select):
                    result = probe.run(env, 2101, source, root, parent, fresh)
            self.assertEqual(fresh_seeds, roles["fit_seeds"] + [roles["scale_seed"]])
            self.assertLess(events.index(("selected", None)), events.index(("normal", roles["calibration_seed"])))
            self.assertEqual((result["predictor_optimizer_steps"], result["residual_scale_optimizer_steps"],
                result["validation_scale_optimizer_steps"], result["total_residual_scale_optimizer_steps"]), (6, 9, 27, 36))
            self.assertTrue(result["control_scores_reproduced"])
            self.assertEqual(result["additional_interactions"], 0)
            for arm in probe.ARMS:
                self.assertEqual(result["arms"][arm], control["arms"][arm])
            for arm in (probe.RISK, probe.NLL):
                selected = result["arms"][arm]
                self.assertEqual(selected["calibration"]["support_radii"], control["arms"]["global_scale"]["calibration"]["support_radii"])
                self.assertAlmostEqual(selected["calibration"]["error_threshold"], np.quantile(
                    [w["statistic"] for w in selected["calibration"]["windows"] if w["eligible"]], .99))
            with patch.object(probe.WindowRiskEvidence, "fit", side_effect=AssertionError("Repeated fit")):
                self.assertEqual(probe.run(env, 2101, source, root, parent, fresh), result)
            for seed in probe.SEEDS:
                row = copy.deepcopy(result)
                row["roles"] = probe.fold_roles(seed)
                sources = row["roles"]["fit_seeds"] + [row["roles"]["scale_seed"]]
                row["scale_fit_sources"] = sources
                row["inner_calibration_source"]["seed"] = row["roles"]["scale_seed"]
                for selected in row["selections"].values():
                    selected["validation_sources"] = sources
                for held, fold in enumerate(row["source_validation_folds"]):
                    fold["held_source"] = sources[held]
                    fold["train_sources"] = [s for s in sources if s != sources[held]]
                    fold["selection"]["inner_calibration_seed"] = row["roles"]["scale_seed"]
                probe.write(root / env.replace("-", "_") / f"test_seed{seed}" / "readout.json", row)
            with patch.object(probe, "ENVIRONMENTS", (env,)):
                summary = probe.summarize(root)
            self.assertEqual(summary["total_residual_scale_optimizer_steps"], 180)
            row = probe.compact_report(summary)["environments"][env]["runs"][0]
            self.assertNotIn("windows", row["arms"][probe.RISK]["calibration"])
        print("WINDOW_RISK_TEST interactions=0 mean_updates=24 total_scale_updates=63")


if __name__ == "__main__":
    torch.set_num_threads(2)
    unittest.main()
