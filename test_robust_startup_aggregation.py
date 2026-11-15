import copy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import torch

import run_robust_startup_aggregation as probe
from headwise_change_evidence import CONFIG
from test_affine_residual_change_probe import make_source
from test_matched_residual_scale_probe import make_fresh_fixture
from test_headwise_change_probe import targets
from test_startup_confirmation import fixture_collection


def bare_evidence():
    evidence = probe.GuardedEvidence(5, 5, 2)
    evidence.domains = ["state_action", "action", "state_action"]
    return evidence


class RobustStartupAggregationTests(unittest.TestCase):
    def test_one_spike_does_not_dominate_but_sustained_change_remains(self):
        evidence = bare_evidence()
        errors, support = np.ones((100, 3), np.float32), np.ones((100, 3), bool)
        errors[43, 0] = 10001
        original = errors.copy()
        result = probe.aggregate_windows(evidence, errors, support)
        self.assertEqual(result[probe.MEAN][0]["statistic"], 101.)
        self.assertEqual(result[probe.PRIMARY][0]["statistic"], 1.)
        np.testing.assert_array_equal(errors, original)
        errors[:60, 0] = 100
        errors[60:, 0] = 1
        result = probe.aggregate_windows(evidence, errors, support)
        self.assertEqual(result[probe.PRIMARY][0]["statistic"], 100.)
        errors[40:60, 0] = 1
        self.assertEqual(probe.aggregate_windows(evidence, errors, support)[probe.PRIMARY][0]["statistic"], 1.)

    def test_empty_block_and_ineligible_heads_do_not_change_coverage(self):
        errors, support = np.ones((100, 3), np.float32), np.ones((100, 3), bool)
        support[:20, 0] = False
        errors[:20, 0] = 1e8
        support[:21, 1] = False
        support[:, 2] = False
        result = probe.aggregate_windows(bare_evidence(), errors, support)
        probe.verify_coverage(*result.values())
        robust = result[probe.PRIMARY][0]
        self.assertEqual(robust["heads"]["reward_0"]["block_means"], [None, 1., 1., 1., 1.])
        self.assertEqual(robust["heads"]["reward_0"]["block_supported_steps"], [0, 20, 20, 20, 20])
        self.assertEqual(robust["statistic"], 1.)
        self.assertIsNone(robust["heads"]["reward_1"]["statistic"])
        self.assertIsNone(robust["heads"]["dynamics"]["statistic"])
        support[:] = False
        result = probe.aggregate_windows(bare_evidence(), errors, support)
        self.assertFalse(result[probe.PRIMARY][0]["eligible"])
        self.assertIsNone(result[probe.PRIMARY][0]["statistic"])

    def test_joint_score_is_maximum_after_headwise_block_medians(self):
        errors = np.repeat(np.array([[1, 4, 2], [2, 5, 2], [3, 6, 2], [4, 7, 2], [5, 8, 2]], np.float32), 20, axis=0)
        support = np.ones_like(errors, bool)
        robust = probe.aggregate_windows(bare_evidence(), errors, support)[probe.PRIMARY][0]
        self.assertEqual(robust["heads"]["reward_0"]["statistic"], 3.)
        self.assertEqual(robust["heads"]["reward_1"]["statistic"], 6.)
        self.assertEqual(robust["statistic"], 6.)
        probe.mark_alarms([robust], 6.)
        self.assertFalse(robust["alarm"])
        probe.mark_alarms([robust], 5.)
        self.assertTrue(robust["alarm"])
        self.assertFalse(robust["heads"]["reward_0"]["alarm"])

    def test_pipeline_reproduces_control_and_freezes_both_thresholds_before_evaluation(self):
        with tempfile.TemporaryDirectory() as tmp, patch.dict(CONFIG, hidden=8, updates=3, scale_updates=3):
            source, fresh, parent, root = (Path(tmp)/n for n in ("source", "fresh", "parent", "robust"))
            env, seed = probe.ENVIRONMENTS[0], 2101
            make_source(source, env)
            make_fresh_fixture(fresh, env)
            roles = probe.fold_roles(seed)
            fit = [probe.normal_data(source, env, s, "expanded_normal") for s in roles["fit_seeds"]]
            scale = [probe.fresh_data(fresh, env, s) for s in roles["fit_seeds"]]
            scale.append(probe.normal_data(source, env, roles["scale_seed"], "expanded_normal"))
            fit_x, fit_y = (np.concatenate([r[i] for r in fit]) for i in (0, 1))
            scale_x, scale_y = (np.concatenate([r[i] for r in scale]) for i in (0, 1))
            evidence = probe.GuardedEvidence(5, 5, 2)
            evidence.weights = {probe.GUARDED: .5}
            work = evidence.fit(fit_x, fit_y, 5000, 80000000 + 10*seed)
            scale_work = evidence.fit_scale_losses(scale_x, evidence.raw_losses(scale_x, scale_y), 81000000 + 10*seed)
            evidence.radii, evidence.thresholds = np.full(3, 1e10), {}
            rng = np.random.default_rng(7766)
            groups = {}
            for group in ("startup", "evaluation"):
                stored, rows = {}, {}
                for case, count in probe.acquisition.steps(group).items():
                    x = rng.normal(size=(count, 5)).astype(np.float32)
                    y = targets(x)
                    if group == "evaluation" and case.startswith("mode"):
                        y[500:, :2] += 500
                    rows[case] = (x, y)
                    stored.update({f"{case}_inputs": x, f"{case}_targets": y})
                folder = probe.acquisition.folder(parent/"collection", env, seed, group)
                probe.write(folder/"readout.json", fixture_collection(env, seed, group))
                np.savez_compressed(folder/"transitions.npz", **stored)
                groups[group] = rows
            cal_x, cal_y = (np.concatenate([r[i] for r in groups["startup"].values()]) for i in (0, 1))
            calibration = probe.confirmation.calibrate_fixed_support(evidence, cal_x, cal_y)
            old_cases = {case: dict(windows=evidence.evaluate(x, y, probe.GUARDED)["windows"])
                         for case, (x, y) in groups["evaluation"].items()}
            prior = dict(protocol=probe.confirmation.PROTOCOL, env=env, complete=True, config=CONFIG,
                roles=roles, frozen_selection=dict(global_weight=.5), predictor_work=work, scale_work=scale_work,
                scale_fit_sources=roles["fit_seeds"]+[roles["scale_seed"]],
                arms={probe.confirmation.PRIMARY: dict(calibration=calibration, cases=old_cases)})
            parent_path = parent/env.replace("-", "_")/f"test_seed{seed}"/"readout.json"
            probe.write(parent_path, prior)
            events, read, original_fit, original_calibrate = [], probe.acquisition.data, probe.GuardedEvidence.fit, probe.calibrate
            def read_group(root_path, e, s, group):
                events.append(group)
                if group == "evaluation":
                    saved = json.loads((root/env.replace("-", "_")/f"test_seed{seed}"/"readout.json").read_text())
                    self.assertTrue(saved["thresholds_frozen_before_evaluation_read"])
                    self.assertEqual(set(saved["arms"]), set(probe.ARMS))
                return read(root_path, e, s, group)
            def record_fit(model, x, y, split, random_seed):
                np.testing.assert_array_equal(x, fit_x)
                np.testing.assert_array_equal(y, fit_y)
                return original_fit(model, x, y, split, random_seed)
            def freeze(model, x, y):
                events.append("calibrate")
                models = [model.reward_predictor.model, model.dynamics_predictor.model, *model.scale_models]
                states = [copy.deepcopy(m.state_dict()) for m in models]
                radii, weights = model.radii.copy(), model.weights.copy()
                result = original_calibrate(model, x, y)
                self.assertTrue(all(torch.equal(state[k], m.state_dict()[k]) for state, m in zip(states, models) for k in state))
                np.testing.assert_array_equal(model.radii, radii)
                self.assertEqual(model.weights, weights)
                for arm, row in result.items():
                    expected = np.quantile([w["statistic"] for w in row["windows"] if w["eligible"]], .99)
                    self.assertEqual(row["error_threshold"], expected)
                return result
            with patch("builtins.print"), patch.object(probe.acquisition, "data", read_group), \
                 patch.object(probe.acquisition, "run_group", side_effect=AssertionError("Recollection")), \
                 patch.object(probe.GuardedEvidence, "fit", record_fit), patch.object(probe, "calibrate", freeze):
                result = probe.run(env, seed, source, root, parent, fresh)
            self.assertEqual(events, ["startup", "calibrate", "evaluation"])
            self.assertEqual((result["predictor_optimizer_steps"], result["residual_scale_optimizer_steps"]), (6, 9))
            self.assertEqual(result["additional_interactions"], 0)
            self.assertEqual(json.loads(parent_path.read_text()), prior)
            with patch.object(probe.GuardedEvidence, "fit", side_effect=AssertionError("Repeated fit")):
                self.assertEqual(probe.run(env, seed, source, root, parent, fresh), result)
            with patch.object(probe, "ENVIRONMENTS", (env,)), patch.object(probe, "SEEDS", (seed,)):
                summary = probe.summarize(root)
                self.assertFalse(summary["evidence_integrated"])
                self.assertEqual(summary["retained_method"], "v25")
                paired = copy.deepcopy(summary["environments"][env]["paired"])
                paired.update(timely_lost=[dict(seed=seed, case="mode1")], normal_alarms={probe.MEAN: 2, probe.PRIMARY: 0})
                with patch.object(probe, "screen", return_value=dict(development_screen_passed=True)), \
                     patch.object(probe, "paired_readout", return_value=paired):
                    self.assertEqual(probe.summarize(root)["decision"], "hold_conditional_evidence_integration")
                compact = probe.compact_report(summary)
                self.assertNotIn("windows", compact["environments"][env]["runs"][0]["arms"][probe.PRIMARY]["calibration"])
        print("ROBUST_PIPELINE_TEST interactions=0 mean_updates=12 scale_updates=18 affine_fits=4")


if __name__ == "__main__":
    torch.set_num_threads(2)
    unittest.main()
