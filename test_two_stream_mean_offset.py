import copy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import torch

import run_two_stream_mean_offset as probe
from headwise_change_evidence import CONFIG
from test_affine_residual_change_probe import make_source
from test_matched_residual_scale_probe import make_fresh_fixture
from test_headwise_change_probe import targets
from test_robust_startup_confirmation import metadata as new_metadata
from test_startup_confirmation import fixture_collection


def collection_metadata(env, seed, group, old=False):
    row = fixture_collection(env, seed, group) if old else new_metadata(env, seed, group)
    row["source"] = "fixture-policies"
    return row


class TwoStreamMeanOffsetTests(unittest.TestCase):
    def test_streams_match_policy_sources_but_reject_reused_rngs(self):
        env, seed = probe.PARENT.ENVIRONMENTS[0], 2103
        old, new = (collection_metadata(env, seed, "startup", old=b) for b in (True, False))
        probe.verify_streams(old, new, env, seed)
        mismatch = copy.deepcopy(old)
        mismatch["source"] = "different-policies"
        with self.assertRaises(AssertionError):
            probe.verify_streams(mismatch, new, env, seed)
        with self.assertRaises(AssertionError):
            probe.verify_streams(new, new, env, seed)

    def test_shared_gate_uses_new_arm_names_without_relaxing_constraints(self):
        envs = {"env": dict(arms={probe.PRIMARY: dict(development_screen_passed=True)},
            paired={c: dict(timely_lost=[], normal_alarms={c: 2, probe.PRIMARY: 1})
                for c in (probe.CONTROL, probe.ANCHOR)})}
        self.assertTrue(probe.shared.passes_screen(envs, control=probe.CONTROL, primary=probe.PRIMARY))
        envs["env"]["paired"][probe.ANCHOR]["timely_lost"] = [dict(seed=2103, case="mode1")]
        self.assertFalse(probe.shared.passes_screen(envs, control=probe.CONTROL, primary=probe.PRIMARY))

    def test_pipeline_uses_only_old_stationary_and_full_new_calibration(self):
        parent_probe, shared = probe.PARENT, probe.shared
        with tempfile.TemporaryDirectory() as tmp, patch.dict(CONFIG, hidden=8, updates=3, scale_updates=3):
            source, fresh, parent, earlier, root = (Path(tmp)/n for n in ("source", "fresh", "parent", "old", "two_stream"))
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
            groups, rng = {}, np.random.default_rng(95127)
            old_x = rng.normal(size=(3000, 5)).astype(np.float32)
            old_y = targets(old_x)
            old_y[:, 0] -= .45
            old_folder = parent_probe.acquisition.folder(earlier/"collection", env, seed, "startup")
            parent_probe.write(old_folder/"readout.json", collection_metadata(env, seed, "startup", old=True))
            # Only stationary arrays exist: reading old calibration arrays would fail.
            np.savez_compressed(old_folder/"transitions.npz", stationary_inputs=old_x, stationary_targets=old_y)
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
                parent_probe.write(folder/"readout.json", collection_metadata(env, seed, group))
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
                acquisition={g: collection_metadata(env, seed, g) for g in groups}, arms=arms,
                additional_interactions=11009, initialization_probes=9, **dict.fromkeys(parent_probe.ZERO_WORK, 0))
            parent_path = parent/env.replace("-", "_")/f"test_seed{seed}"/"readout.json"
            parent_probe.write(parent_path, prior)
            events = []
            original_offset, original_calibrate, original_data = shared.MeanOffsetEvidence.fit_offset, shared.calibrate, parent_probe.acquisition.data
            def fit_offset(evidence, x, y):
                events.append("old_offset_fit")
                np.testing.assert_array_equal(x, old_x)
                np.testing.assert_array_equal(y, old_y)
                return original_offset(evidence, x, y)
            def calibrate(evidence, x, y, arms):
                events.append("full_new_calibration")
                self.assertEqual(arms, probe.ARMS)
                for i, array in enumerate((x, y)):
                    np.testing.assert_array_equal(array, np.concatenate([r[i] for r in groups["startup"].values()]))
                offset = evidence.normalized_offset.copy()
                result = original_calibrate(evidence, x, y, arms=arms)
                np.testing.assert_array_equal(offset, evidence.normalized_offset)
                return result
            def data(collection, e, s, group, seed_fn):
                if group == "evaluation":
                    events.append("evaluation_read")
                    saved = json.loads((root/env.replace("-", "_")/f"test_seed{seed}"/"readout.json").read_text())
                    self.assertTrue(saved["thresholds_and_offset_frozen_before_evaluation_read"])
                    self.assertEqual((saved["used_startup_steps"], saved["calibration_steps"]), (8000, 5000))
                return original_data(collection, e, s, group, seed_fn=seed_fn)
            with patch.object(shared.MeanOffsetEvidence, "fit_offset", fit_offset), patch.object(shared, "calibrate", calibrate), \
                 patch.object(parent_probe.acquisition, "data", data), \
                 patch.object(parent_probe.acquisition, "run_group", side_effect=AssertionError("Recollection")), patch("builtins.print"):
                result = probe.run(env, seed, root, parent, earlier, source, fresh)
            self.assertEqual(events, ["old_offset_fit", "full_new_calibration", "evaluation_read"])
            self.assertEqual((result["predictor_optimizer_steps"], result["residual_scale_optimizer_steps"], result["offset_fit_calls"]), (6, 9, 1))
            self.assertEqual(result["additional_interactions"], 0)
            self.assertTrue(result["original_anchor_reproduced"])
            self.assertEqual(result["arms"][probe.ANCHOR]["calibration"], prior["arms"][parent_probe.MEAN]["calibration"])
            self.assertEqual(result["arms"][probe.CONTROL]["cases"], result["arms"][probe.ANCHOR]["cases"])
            self.assertEqual(json.loads(parent_path.read_text()), prior)
            with patch.object(shared.MeanOffsetEvidence, "fit", side_effect=AssertionError("Repeated fit")):
                self.assertEqual(probe.run(env, seed, root, parent, earlier, source, fresh), result)
            with patch.object(parent_probe, "ENVIRONMENTS", (env,)), patch.object(parent_probe, "SEEDS", (seed,)):
                summary = probe.summarize(root)
                self.assertEqual((summary["used_startup_steps"], summary["offset_fit_calls"]), (8000, 1))
                self.assertEqual(summary["additional_interactions"], 0)
                self.assertEqual(set(summary["environments"][env]["paired"][probe.CONTROL]["normal_alarms"]), {probe.CONTROL, probe.PRIMARY})
                self.assertEqual(summary["decision"], "hold_conditional_evidence_integration")
        print("TWO_STREAM_PIPELINE_TEST interactions=0 mean_updates=12 scale_updates=18 affine_fits=4 offset_fit_calls=1")


if __name__ == "__main__":
    torch.set_num_threads(2)
    unittest.main()
