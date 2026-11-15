import copy
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import torch

from headwise_change_evidence import CONFIG
import run_residual_scale_shrinkage as probe
import run_matched_residual_scale_probe as matched
import run_affine_residual_change_probe as affine
import run_headwise_change_probe as plain
import run_conditional_change_probe as original
from test_affine_residual_change_probe import make_source
from test_matched_residual_scale_probe import make_fresh_fixture


class ShrinkageTests(unittest.TestCase):
    def test_underestimated_conditional_scale_selects_global(self):
        report = probe.select_weight(np.ones((6, 2)), np.full((6, 2), .01), np.ones(2))
        self.assertEqual(report["global_weight"], 1.)

    def test_well_predicted_conditional_scale_is_retained(self):
        report = probe.select_weight(np.ones((6, 2)), np.ones((6, 2)), np.full(2, 10.))
        self.assertEqual(report["global_weight"], 0.)

    def test_interior_shrinkage_and_single_shared_objective(self):
        report = probe.select_weight(np.ones((6, 2)), np.full((6, 2), .25), np.full(2, 4.))
        self.assertEqual(report["global_weight"], .25)
        for row in report["candidates"]:
            self.assertAlmostEqual(row["normal_nll"], np.mean(row["head_normal_nll"]))
        mixed = probe.select_weight(np.ones((6, 2)), np.tile([.01, 1.], (6, 1)), np.array([1., 10.]))
        self.assertIn(mixed["global_weight"], probe.GLOBAL_WEIGHTS)
        self.assertTrue(mixed["shared_across_heads"])

    def test_equal_predictions_break_ties_toward_global(self):
        report = probe.select_weight(np.ones((6, 2)), np.ones((6, 2)), np.ones(2))
        self.assertEqual(report["global_weight"], 1.)
        self.assertEqual({row["normal_nll"] for row in report["candidates"]}, {1.})

    def test_stored_pipeline_exclusions_controls_frozen_models_and_cost(self):
        with tempfile.TemporaryDirectory() as tmp, patch.dict(CONFIG, hidden=8, updates=3, scale_updates=3):
            source, fresh, parent, base, old, root = (Path(tmp) / name for name in
                ("source", "fresh", "matched", "plain", "affine", "shrunk"))
            env = probe.ENVIRONMENTS[0]
            make_source(source, env)
            make_fresh_fixture(fresh, env)
            with patch("builtins.print"), patch.object(original.training, "make_piecewise_mo_env",
                    side_effect=AssertionError("No new environment")):
                plain.run(env, 2101, source, base)
                affine.run(env, 2101, source, old, base)
                control = matched.run(env, 2101, source, parent, old, fresh)
                read_seeds, observed, events = [], [], []
                fresh_data, normal_data = probe.fresh_data, probe.normal_data
                select = probe.ShrunkResidualEvidence.select_shrinkage
                def record_fresh(root, env, seed):
                    read_seeds.append(seed)
                    return fresh_data(root, env, seed)
                def record_normal(root, env, seed, pool):
                    events.append(("normal", seed))
                    return normal_data(root, env, seed, pool)
                def record_select(evidence, inputs, targets):
                    models = [evidence.reward_predictor.model, evidence.dynamics_predictor.model, *evidence.scale_models]
                    states = [copy.deepcopy(model.state_dict()) for model in models]
                    global_scales = evidence.global_scales.copy()
                    events.append(("select", None))
                    result = select(evidence, inputs, targets)
                    observed.append((inputs.copy(), targets.copy()))
                    np.testing.assert_array_equal(evidence.global_scales, global_scales)
                    self.assertTrue(all(not p.requires_grad for model in models for p in model.parameters()))
                    self.assertTrue(all(torch.equal(state[key], model.state_dict()[key])
                        for state, model in zip(states, models) for key in state))
                    self.assertAlmostEqual(result["normal_nll"], min(r["normal_nll"] for r in result["candidates"]))
                    weight = result["global_weight"]
                    expected = weight * evidence.scales(inputs, "global_scale") + (1 - weight) * evidence.scales(inputs, "conditional_scale")
                    np.testing.assert_allclose(evidence.scales(inputs, probe.SELECTED), expected)
                    return result
                with patch.object(probe, "fresh_data", record_fresh), patch.object(probe, "normal_data", record_normal), \
                     patch.object(probe.ShrunkResidualEvidence, "select_shrinkage", record_select):
                    result = probe.run(env, 2101, source, root, parent, fresh)
            roles = result["roles"]
            self.assertEqual(read_seeds, roles["fit_seeds"] + [roles["scale_seed"]])
            self.assertNotIn(roles["calibration_seed"], read_seeds)
            self.assertNotIn(roles["test_seed"], read_seeds)
            self.assertLess(events.index(("select", None)), events.index(("normal", roles["calibration_seed"])))
            expected = probe.fresh_data(fresh, env, roles["scale_seed"])
            for index in (0, 1):
                np.testing.assert_array_equal(observed[0][index], expected[index])
            self.assertEqual((result["predictor_optimizer_steps"], result["residual_scale_optimizer_steps"]), (6, 9))
            self.assertEqual(result["matched_scale_control"]["arms"], control["arms"])
            self.assertTrue(result["control_scores_reproduced"])
            self.assertEqual(result["additional_interactions"], 0)
            self.assertEqual(result["calibration_steps"], 5000)
            for arm in probe.ARMS:
                self.assertEqual(result["arms"][arm], control["arms"][arm])
            selected = result["arms"][probe.SELECTED]
            self.assertEqual(selected["calibration"]["support_radii"], control["arms"]["conditional_scale"]["calibration"]["support_radii"])
            self.assertAlmostEqual(selected["calibration"]["error_threshold"], np.quantile(
                [w["statistic"] for w in selected["calibration"]["windows"] if w["eligible"]], .99))
            for case in probe.CASES:
                for new, old_window in zip(selected["cases"][case]["windows"], control["arms"]["conditional_scale"]["cases"][case]["windows"]):
                    self.assertEqual({h:v["supported_steps"] for h,v in new["heads"].items()},
                                     {h:v["supported_steps"] for h,v in old_window["heads"].items()})
            with patch.object(probe.ShrunkResidualEvidence, "fit", side_effect=AssertionError("Repeated fit")):
                self.assertEqual(probe.run(env, 2101, source, root, parent, fresh), result)
            for seed in probe.SEEDS:
                copied = copy.deepcopy(result)
                copied["roles"] = probe.fold_roles(seed)
                copied["scale_fit_sources"] = copied["roles"]["fit_seeds"] + [copied["roles"]["scale_seed"]]
                copied["validation_source_seed"] = copied["selection"]["source_seed"] = copied["roles"]["scale_seed"]
                probe.write(root / env.replace("-", "_") / f"test_seed{seed}" / "readout.json", copied)
            with patch.object(probe, "ENVIRONMENTS", (env,)):
                report = probe.summarize(root)
            self.assertEqual((report["predictor_optimizer_steps"], report["residual_scale_optimizer_steps"]), (30, 45))
            compact = probe.compact_report(report)
            row = compact["environments"][env]["runs"][0]
            for collection in (row["arms"], row["matched_scale_control"]["arms"]):
                for arm in collection.values():
                    self.assertNotIn("windows", arm["calibration"])
                    self.assertNotIn("windows", arm["cases"]["stationary"])
        print("RESIDUAL_SCALE_SHRINKAGE_TEST interactions=0 mean_updates=24 scale_updates=36")


if __name__ == "__main__":
    torch.set_num_threads(2)
    unittest.main()
