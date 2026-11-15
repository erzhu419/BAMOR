import copy
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
import torch

import run_startup_mean_offset as probe
from headwise_change_evidence import CONFIG
from test_affine_residual_change_probe import make_source
from test_matched_residual_scale_probe import make_fresh_fixture
from test_headwise_change_probe import targets
from test_robust_startup_confirmation import metadata


class StartupMeanOffsetTests(unittest.TestCase):
    def test_supported_offset_sign_preserves_exact_action_heads(self):
        model = probe.MeanOffsetEvidence(5, 5, 2)
        model.reward_predictor = SimpleNamespace(predict=lambda x: np.tile([4., -7.], (len(x), 1)).astype(np.float32))
        model.dynamics_predictor = SimpleNamespace(predict=lambda x: np.tile([8., -10., 18.], (len(x), 1)).astype(np.float32))
        model.action_heads = np.array([False, True])
        model.domains = ["state_action", "action", "state_action"]
        model.action_coefficients = np.zeros((5, 2))
        model.action_coefficients[3:, 1] = -1
        model.y_scale = np.array([2., 3., 4., 5., 6.], np.float32)
        model.radii = np.ones(3)
        x = np.random.default_rng(19).normal(size=(100, 5)).astype(np.float32)
        y = np.zeros((100, 5), np.float32)
        y[:, 1] = -np.sum(x[:, 3:] ** 2, axis=1)
        y[-1, [0, 2, 3, 4]] = 1e6
        distance = np.zeros((100, 3))
        distance[-1] = 2
        old = model.raw_losses(x, y)
        with patch.object(model, "distances", return_value=distance):
            work = model.fit_offset(x, y)
        np.testing.assert_array_equal(model.normalized_offset, [2., 0., 2., -2., 3.])
        np.testing.assert_array_equal(work["physical_prediction_shift"], [-4., 0., -8., 10., -18.])
        self.assertEqual(work["heads"]["dynamics"]["supported_steps"], 99)
        self.assertFalse(work["heads"]["reward_1"]["corrected"])
        corrected = model.corrected_losses(x, y)
        np.testing.assert_array_equal(corrected[:99, [0, 2]], np.zeros((99, 2)))
        np.testing.assert_array_equal(corrected[:, 1], old[:, 1])
        np.testing.assert_array_equal(model.raw_losses(x, y), old)

    def test_gate_protects_both_controls_and_requires_real_improvement(self):
        baseline = {"env": dict(arms={probe.PRIMARY: dict(development_screen_passed=True)},
            paired={c: dict(timely_lost=[], normal_alarms={c: 2 if c == probe.CONTROL else 1,
                probe.PRIMARY: 1}) for c in (probe.CONTROL, probe.ANCHOR)})}
        self.assertTrue(probe.passes_screen(baseline))
        lost = copy.deepcopy(baseline)
        lost["env"]["paired"][probe.ANCHOR]["timely_lost"] = [dict(seed=2101, case="mode2")]
        self.assertFalse(probe.passes_screen(lost))
        worse = copy.deepcopy(baseline)
        worse["env"]["paired"][probe.ANCHOR]["normal_alarms"][probe.ANCHOR] = 0
        self.assertFalse(probe.passes_screen(worse))
        equal = copy.deepcopy(baseline)
        equal["env"]["paired"][probe.CONTROL]["normal_alarms"][probe.CONTROL] = 1
        self.assertFalse(probe.passes_screen(equal))
        failed = copy.deepcopy(baseline)
        failed["env"]["arms"][probe.PRIMARY]["development_screen_passed"] = False
        self.assertFalse(probe.passes_screen(failed))

    def test_paired_counts_do_not_repeat_shared_prefix(self):
        def case(alarms, timely):
            return dict(timely_alarm=timely, windows=[dict(start_step=i*100+1, end_step=(i+1)*100,
                eligible=True, alarm=i in alarms) for i in range(10)])
        row = dict(roles=dict(test_seed=2101), arms={
            probe.CONTROL: dict(cases=dict(stationary=case([2], False), mode1=case([2, 5], True))),
            probe.PRIMARY: dict(cases=dict(stationary=case([], False), mode1=case([], False)))})
        with patch.object(probe.parent_probe, "CASES", ("stationary", "mode1")):
            result = probe.paired_readout([row], probe.CONTROL)
        self.assertEqual(result["normal_eligible_windows"], 10)
        self.assertEqual(result["mode_eligible_windows"], 5)
        self.assertEqual(result["normal_alarms"], {probe.CONTROL: 1, probe.PRIMARY: 0})
        self.assertEqual(len(result["normal_resolved"]), 1)
        self.assertEqual(len(result["mode_lost"]), 1)
        self.assertEqual(result["timely_lost"], [dict(seed=2101, case="mode1")])

    def test_pipeline_fits_only_stationary_freezes_before_eval_and_keeps_anchor(self):
        parent_probe = probe.parent_probe
        with tempfile.TemporaryDirectory() as tmp, patch.dict(CONFIG, hidden=8, updates=3, scale_updates=3):
            source, fresh, parent, root = (Path(tmp)/n for n in ("source", "fresh", "parent", "offset"))
            env, seed = parent_probe.ENVIRONMENTS[0], 2103
            make_source(source, env)
            make_fresh_fixture(fresh, env)
            roles = parent_probe.fold_roles(seed)
            fit = [parent_probe.normal_data(source, env, s, "expanded_normal") for s in roles["fit_seeds"]]
            scale = [parent_probe.fresh_data(fresh, env, s) for s in roles["fit_seeds"]]
            scale.append(parent_probe.normal_data(source, env, roles["scale_seed"], "expanded_normal"))
            fit_x, fit_y = (np.concatenate([r[i] for r in fit]) for i in (0, 1))
            scale_x, scale_y = (np.concatenate([r[i] for r in scale]) for i in (0, 1))
            model = parent_probe.GuardedEvidence(5, 5, 2)
            model.weights = {parent_probe.GUARDED: .5}
            work = model.fit(fit_x, fit_y, 5000, 80000000 + 10*seed)
            scale_work = model.fit_scale_losses(scale_x, model.raw_losses(scale_x, scale_y), 81000000 + 10*seed)
            model.radii = np.full(3, 1e10)
            groups, rng = {}, np.random.default_rng(12890)
            for group in ("startup", "evaluation"):
                rows, stored = {}, {}
                for case, count in parent_probe.acquisition.steps(group).items():
                    x = rng.normal(size=(count, 5)).astype(np.float32)
                    y = targets(x)
                    y[:, 0] += .2
                    if group == "evaluation" and case.startswith("mode"):
                        y[500:, 0] += 30
                    rows[case] = (x, y)
                    stored.update({f"{case}_inputs": x, f"{case}_targets": y})
                folder = parent_probe.acquisition.folder(parent/"collection", env, seed, group)
                parent_probe.write(folder/"readout.json", metadata(env, seed, group))
                np.savez_compressed(folder/"transitions.npz", **stored)
                groups[group] = rows
            x, y = (np.concatenate([r[i] for r in groups["startup"].values()]) for i in (0, 1))
            calibration = parent_probe.development.calibrate(model, x, y)
            arms = {a: dict(calibration=calibration[a], cases={}) for a in parent_probe.ARMS}
            for case, (x, y) in groups["evaluation"].items():
                paired, support = parent_probe.development.paired_windows(model, x, y)
                for arm in parent_probe.ARMS:
                    parent_probe.development.mark_alarms(paired[arm], calibration[arm]["error_threshold"])
                    arms[arm]["cases"][case] = parent_probe.case_summary(dict(windows=paired[arm], supported=support.any(1)))
            prior = dict(protocol=parent_probe.PROTOCOL, env=env, roles=roles, config=CONFIG, complete=True,
                aggregation=parent_probe.development.AGGREGATION, reward_dim=2, predictor_optimizer_steps=6,
                residual_scale_optimizer_steps=9, predictor_work=work, scale_work=scale_work,
                frozen_selection=dict(global_weight=.5), scale_fit_sources=roles["fit_seeds"]+[roles["scale_seed"]],
                thresholds_frozen_before_evaluation_collection=True, frozen_model_work_reproduced=True,
                acquisition={g: metadata(env, seed, g) for g in groups}, arms=arms,
                additional_interactions=11009, initialization_probes=9, **dict.fromkeys(parent_probe.ZERO_WORK, 0))
            parent_path = parent/env.replace("-", "_")/f"test_seed{seed}"/"readout.json"
            parent_probe.write(parent_path, prior)
            events = []
            original_offset, original_calibrate, original_data = probe.MeanOffsetEvidence.fit_offset, probe.calibrate, parent_probe.acquisition.data
            def fit_offset(evidence, x, y):
                events.append("offset_fit")
                np.testing.assert_array_equal(x, groups["startup"]["stationary"][0])
                np.testing.assert_array_equal(y, groups["startup"]["stationary"][1])
                models = [evidence.reward_predictor.model, evidence.dynamics_predictor.model, *evidence.scale_models]
                states = [copy.deepcopy(m.state_dict()) for m in models]
                scales = evidence.scales(x, parent_probe.GUARDED).copy()
                result = original_offset(evidence, x, y)
                self.assertTrue(all(torch.equal(state[k], m.state_dict()[k]) for state, m in zip(states, models) for k in state))
                np.testing.assert_array_equal(scales, evidence.scales(x, parent_probe.GUARDED))
                return result
            def calibrate(evidence, x, y):
                events.append("calibrate")
                for i, array in enumerate((x, y)):
                    np.testing.assert_array_equal(array, np.concatenate([groups["startup"][c][i] for c in probe.CALIBRATION_CASES]))
                offset = evidence.normalized_offset.copy()
                result = original_calibrate(evidence, x, y)
                np.testing.assert_array_equal(offset, evidence.normalized_offset)
                return result
            def data(collection, e, s, group, seed_fn):
                if group == "evaluation":
                    events.append("evaluation_read")
                    saved = json.loads((root/env.replace("-", "_")/f"test_seed{seed}"/"readout.json").read_text())
                    self.assertTrue(saved["thresholds_and_offset_frozen_before_evaluation_read"])
                    self.assertEqual(set(saved["arms"]), set(probe.ARMS))
                return original_data(collection, e, s, group, seed_fn=seed_fn)
            with patch.object(probe.MeanOffsetEvidence, "fit_offset", fit_offset), patch.object(probe, "calibrate", calibrate), \
                 patch.object(parent_probe.acquisition, "data", data), \
                 patch.object(parent_probe.acquisition, "run_group", side_effect=AssertionError("Recollection")), patch("builtins.print"):
                result = probe.run(env, seed, root, parent, source, fresh)
            self.assertEqual(events, ["offset_fit", "calibrate", "evaluation_read"])
            self.assertEqual((result["predictor_optimizer_steps"], result["residual_scale_optimizer_steps"], result["offset_fit_calls"]), (6, 9, 1))
            self.assertEqual(result["additional_interactions"], 0)
            self.assertTrue(result["original_anchor_reproduced"])
            self.assertEqual(result["arms"][probe.ANCHOR]["calibration"], prior["arms"][parent_probe.MEAN]["calibration"])
            self.assertEqual(json.loads(parent_path.read_text()), prior)
            with patch.object(probe.MeanOffsetEvidence, "fit", side_effect=AssertionError("Repeated fit")):
                self.assertEqual(probe.run(env, seed, root, parent, source, fresh), result)
            with patch.object(parent_probe, "ENVIRONMENTS", (env,)), patch.object(parent_probe, "SEEDS", (seed,)):
                summary = probe.summarize(root)
                self.assertEqual(summary["offset_fit_calls"], 1)
                self.assertEqual(summary["additional_interactions"], 0)
                self.assertEqual(summary["decision"], "hold_conditional_evidence_integration")
        print("MEAN_OFFSET_PIPELINE_TEST interactions=0 mean_updates=12 scale_updates=18 affine_fits=4 offset_fit_calls=1")


if __name__ == "__main__":
    torch.set_num_threads(2)
    unittest.main()
