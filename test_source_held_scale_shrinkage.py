import copy
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import torch

from headwise_change_evidence import CONFIG
import run_source_held_scale_shrinkage as probe
import run_residual_scale_shrinkage as trajectory
import run_matched_residual_scale_probe as matched
import run_affine_residual_change_probe as affine
import run_headwise_change_probe as plain
import run_conditional_change_probe as original
from test_affine_residual_change_probe import make_source
from test_matched_residual_scale_probe import make_fresh_fixture


class SourceHeldTests(unittest.TestCase):
    def test_aggregation_minimizes_common_nll_not_fold_winners(self):
        folds = []
        for seed, size, values in [(1, 10, [0, 2, 3, 4, 9]), (2, 100, [8, 2, 3, 4, 0]),
                                   (3, 1000, [8, 2, 3, 4, 0])]:
            folds.append(dict(held_source=seed, validation_steps=size, selection=dict(candidates=[
                dict(global_weight=w, normal_nll=v, head_normal_nll=[v, v])
                for w, v in zip(probe.GLOBAL_WEIGHTS, values)])))
        result = probe.aggregate_selection(folds)
        self.assertEqual(result["global_weight"], .75)
        self.assertEqual(result["normal_nll"], 2.)
        self.assertEqual(result["validation_steps"], 1110)
        self.assertEqual(result["source_weighting"], "equal")

    def test_source_folds_exclude_entire_held_block_and_global_statistics(self):
        class Evidence:
            def __init__(self):
                self.calls = []
            def raw_losses(self, inputs, targets):
                return targets
            def fit_scale_losses(self, inputs, losses, seed):
                self.calls.append((inputs.copy(), losses.copy(), seed))
                self.global_scales = losses.mean(0)
                return dict(optimizer_steps=0, global_scales=self.global_scales.tolist())
            def scales(self, inputs, arm):
                return np.broadcast_to(self.global_scales, (len(inputs), 2))
        rows = [(np.full((4, 1), seed), np.full((4, 2), loss), dict(seed=seed))
                for seed, loss in [(21, 1.), (22, 2.), (23, 100.)]]
        evidence = Evidence()
        folds = list(probe.source_validation_folds(evidence, rows, 7))
        for held, (fold, call) in enumerate(zip(folds, evidence.calls)):
            training = [index for index in range(3) if index != held]
            np.testing.assert_array_equal(call[0], np.concatenate([rows[i][0] for i in training]))
            np.testing.assert_array_equal(call[1], np.concatenate([rows[i][1] for i in training]))
            self.assertNotIn(fold["held_source"], fold["train_sources"])
            self.assertEqual(call[2], 7)
            self.assertEqual(fold["selection"]["global_weight"], 1.)

    def test_stored_pipeline_refit_controls_roles_and_full_computation_accounting(self):
        with tempfile.TemporaryDirectory() as tmp, patch.dict(CONFIG, hidden=8, updates=3, scale_updates=3):
            source, fresh, parent, base, old, previous, root = (Path(tmp) / name for name in
                ("source", "fresh", "matched", "plain", "affine", "trajectory", "sourceheld"))
            env = probe.ENVIRONMENTS[0]
            make_source(source, env)
            make_fresh_fixture(fresh, env)
            observed, fresh_seeds, events = [], [], []
            source_folds, fresh_data, normal_data = probe.source_validation_folds, probe.fresh_data, probe.normal_data
            def record_fresh(root, env, seed):
                fresh_seeds.append(seed)
                return fresh_data(root, env, seed)
            def record_normal(root, env, seed, pool):
                events.append(("normal", seed))
                return normal_data(root, env, seed, pool)
            def record_folds(evidence, rows, seed):
                models = [evidence.reward_predictor.model, evidence.dynamics_predictor.model]
                states = [copy.deepcopy(model.state_dict()) for model in models]
                x_mean, y_scale = evidence.x_mean.copy(), evidence.y_scale.copy()
                observed.extend(rows)
                for fold in source_folds(evidence, rows, seed):
                    yield fold
                np.testing.assert_array_equal(evidence.x_mean, x_mean)
                np.testing.assert_array_equal(evidence.y_scale, y_scale)
                self.assertTrue(all(torch.equal(state[key], model.state_dict()[key])
                    for state, model in zip(states, models) for key in state))
                events.append(("selection", None))
            with patch("builtins.print"), patch.object(original.training, "make_piecewise_mo_env",
                    side_effect=AssertionError("No new environment")):
                plain.run(env, 2101, source, base)
                affine.run(env, 2101, source, old, base)
                control = matched.run(env, 2101, source, parent, old, fresh)
                old_result = trajectory.run(env, 2101, source, previous, parent, fresh)
                with patch.object(probe, "source_validation_folds", record_folds), \
                     patch.object(probe, "fresh_data", record_fresh), patch.object(probe, "normal_data", record_normal):
                    result = probe.run(env, 2101, source, root, parent, fresh, previous)
            roles = result["roles"]
            self.assertEqual(fresh_seeds, roles["fit_seeds"])
            self.assertLess(events.index(("selection", None)), events.index(("normal", roles["calibration_seed"])))
            self.assertEqual([row[2]["seed"] for row in observed], roles["fit_seeds"] + [roles["scale_seed"]])
            for index, row in enumerate(observed):
                expected = probe.fresh_data(fresh, env, row[2]["seed"]) if index < 2 else probe.normal_data(source, env, roles["scale_seed"], "expanded_normal")
                for column in (0, 1):
                    np.testing.assert_array_equal(row[column], expected[column])
            self.assertEqual((result["predictor_optimizer_steps"], result["residual_scale_optimizer_steps"],
                              result["validation_scale_optimizer_steps"], result["total_residual_scale_optimizer_steps"]), (6, 9, 27, 36))
            self.assertEqual(result["additional_interactions"], 0)
            self.assertTrue(result["control_scores_reproduced"])
            self.assertEqual(result["trajectory_control"]["arms"][probe.SELECTED], old_result["arms"][probe.SELECTED])
            for arm in probe.ARMS:
                self.assertEqual(result["arms"][arm], control["arms"][arm])
            selected = result["arms"][probe.SELECTED]
            self.assertEqual(selected["calibration"]["support_radii"], control["arms"]["conditional_scale"]["calibration"]["support_radii"])
            self.assertAlmostEqual(selected["calibration"]["error_threshold"], np.quantile(
                [w["statistic"] for w in selected["calibration"]["windows"] if w["eligible"]], .99))
            with patch.object(probe.ShrunkResidualEvidence, "fit", side_effect=AssertionError("Repeated fit")):
                self.assertEqual(probe.run(env, 2101, source, root, parent, fresh, previous), result)
            for seed in probe.SEEDS:
                copied = copy.deepcopy(result)
                copied["roles"] = probe.fold_roles(seed)
                sources = copied["roles"]["fit_seeds"] + [copied["roles"]["scale_seed"]]
                copied["scale_fit_sources"] = copied["selection"]["validation_sources"] = sources
                for held, fold in enumerate(copied["source_validation_folds"]):
                    fold["held_source"] = sources[held]
                    fold["train_sources"] = [s for s in sources if s != sources[held]]
                probe.write(root / env.replace("-", "_") / f"test_seed{seed}" / "readout.json", copied)
            with patch.object(probe, "ENVIRONMENTS", (env,)):
                report = probe.summarize(root)
            self.assertEqual((report["predictor_optimizer_steps"], report["residual_scale_optimizer_steps"],
                              report["validation_scale_optimizer_steps"], report["total_residual_scale_optimizer_steps"]), (30, 45, 135, 180))
            row = probe.compact_report(report)["environments"][env]["runs"][0]
            for collection in (row["arms"], row["matched_scale_control"]["arms"], row["trajectory_control"]["arms"]):
                for arm in collection.values():
                    self.assertNotIn("windows", arm["calibration"])
                    self.assertNotIn("windows", arm["cases"]["stationary"])
        print("SOURCE_HELD_SCALE_TEST interactions=0 mean_updates=30 total_scale_updates=72")


if __name__ == "__main__":
    torch.set_num_threads(2)
    unittest.main()
