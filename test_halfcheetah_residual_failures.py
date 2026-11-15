import copy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import torch

import diagnose_halfcheetah_residual_failures as diagnostic
from headwise_change_evidence import CONFIG
from test_affine_residual_change_probe import make_source
from test_matched_residual_scale_probe import make_fresh_fixture
from test_headwise_change_probe import targets
from test_robust_startup_confirmation import metadata


class HalfCheetahResidualFailureTests(unittest.TestCase):
    def test_bias_variance_identity_does_not_cancel_dynamics_components(self):
        row = diagnostic.moments([[2., -2.], [4., -4.]])
        self.assertEqual(row["normalized_residual_mean"], [3., -3.])
        self.assertEqual(row["mean_squared_bias"], 9.)
        self.assertEqual(row["within_window_variance"], 1.)
        self.assertEqual(row["normalized_mse"], 10.)
        self.assertEqual(row["bias_fraction"], .9)
        self.assertEqual(diagnostic.moments(np.zeros((10, 2)))["bias_fraction"], 0.)

    def test_empirical_lower_tail_is_descriptive_and_preserves_alarms(self):
        keys = ("raw_mean_loss", "affine_only_mean_loss", "guarded_score", "global_score")
        startup = [dict(heads=dict(reward_0=dict(eligible=True, **dict.fromkeys(keys, value))))
                   for value in (10., 12., 14.)]
        normal = copy.deepcopy(startup)
        window = dict(mean_alarm=False, robust_alarm=False,
            heads=dict(reward_0=dict(eligible=True, **dict.fromkeys(keys, 1.))))
        result = diagnostic.compare_reference([window], startup)[0]
        self.assertEqual(result["heads"]["reward_0"]["startup_empirical_cdf"], dict.fromkeys(keys, 0.))
        self.assertEqual(result["heads"]["reward_0"]["raw_loss_to_startup_median"], 1/12)
        self.assertFalse(result["mean_alarm"] or result["robust_alarm"])
        self.assertEqual(startup, normal)

    def test_reconstruction_only_uses_original_fit_sources_and_frozen_scores(self):
        probe = diagnostic.confirmation
        with tempfile.TemporaryDirectory() as tmp, patch.dict(CONFIG, hidden=8, updates=3, scale_updates=3):
            source, fresh, parent, root = (Path(tmp)/n for n in ("source", "fresh", "parent", "diagnostic"))
            env, seed = diagnostic.ENV, 2103
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
            evidence.radii = np.full(3, 1e10)
            rng = np.random.default_rng(9865)
            groups = {}
            for group in ("startup", "evaluation"):
                rows, stored = {}, {}
                for case, count in probe.acquisition.steps(group).items():
                    x = rng.normal(size=(count, 5)).astype(np.float32)
                    y = targets(x)
                    if group == "evaluation" and case.startswith("mode"):
                        y[500:, 0] += 30
                    rows[case] = (x, y)
                    stored.update({f"{case}_inputs": x, f"{case}_targets": y})
                folder = probe.acquisition.folder(parent/"collection", env, seed, group)
                probe.write(folder/"readout.json", metadata(env, seed, group))
                np.savez_compressed(folder/"transitions.npz", **stored)
                groups[group] = rows
            x, y = (np.concatenate([r[i] for r in groups["startup"].values()]) for i in (0, 1))
            calibration = probe.development.calibrate(evidence, x, y)
            arms = {a: dict(calibration=calibration[a], cases={}) for a in probe.ARMS}
            for case, (x, y) in groups["evaluation"].items():
                windows, support = probe.development.paired_windows(evidence, x, y)
                for arm in probe.ARMS:
                    probe.development.mark_alarms(windows[arm], calibration[arm]["error_threshold"])
                    arms[arm]["cases"][case] = probe.case_summary(dict(windows=windows[arm], supported=support.any(1)))
            prior = dict(protocol=probe.PROTOCOL, env=env, roles=roles, config=CONFIG, complete=True,
                aggregation=probe.development.AGGREGATION, reward_dim=2, predictor_optimizer_steps=6,
                residual_scale_optimizer_steps=9, predictor_work=work, scale_work=scale_work,
                frozen_selection=dict(global_weight=.5), scale_fit_sources=roles["fit_seeds"]+[roles["scale_seed"]],
                thresholds_frozen_before_evaluation_collection=True, frozen_model_work_reproduced=True,
                acquisition={g: metadata(env, seed, g) for g in groups}, arms=arms,
                additional_interactions=11009, initialization_probes=9, **dict.fromkeys(probe.ZERO_WORK, 0))
            parent_path = parent/env.replace("-", "_")/f"test_seed{seed}"/"readout.json"
            probe.write(parent_path, prior)
            fits, scale_fits = [], []
            original_fit, original_scale = probe.GuardedEvidence.fit, probe.GuardedEvidence.fit_scale_losses
            def record_fit(model, x, y, split, random_seed):
                fits.append(x.copy())
                np.testing.assert_array_equal(y, fit_y)
                return original_fit(model, x, y, split, random_seed)
            def record_scale(model, x, losses, random_seed):
                scale_fits.append(x.copy())
                return original_scale(model, x, losses, random_seed)
            with patch.object(probe.acquisition, "run_group", side_effect=AssertionError("Recollection")), \
                 patch.object(probe.development, "calibrate", side_effect=AssertionError("Recalibration")), \
                 patch.object(probe.GuardedEvidence, "fit", record_fit), \
                 patch.object(probe.GuardedEvidence, "fit_scale_losses", record_scale), patch("builtins.print"):
                result = diagnostic.run(seed, root, parent, source, fresh)
            self.assertEqual((len(fits), len(scale_fits)), (1, 1))
            np.testing.assert_array_equal(fits[0], fit_x)
            np.testing.assert_array_equal(scale_fits[0], scale_x)
            self.assertTrue(result["frozen_scores_and_alarms_reproduced"])
            self.assertEqual((result["predictor_optimizer_steps"], result["residual_scale_optimizer_steps"], result["affine_fits"]), (6, 9, 2))
            self.assertEqual(result["additional_interactions"], 0)
            self.assertEqual(len(result["startup_windows"]), 50)
            self.assertEqual([len(r["windows"]) for r in result["cases"].values()], [10, 5, 5, 5, 5, 5])
            for window in result["cases"]["mode1"]["windows"]:
                self.assertEqual(window["heads"]["reward_1"]["correction_mse"], 0.)
            self.assertEqual(json.loads(parent_path.read_text()), prior)
            with patch.object(probe.GuardedEvidence, "fit", side_effect=AssertionError("Repeated fit")):
                self.assertEqual(diagnostic.run(seed, root, parent, source, fresh), result)
            with patch.object(diagnostic, "SEEDS", (seed,)):
                compact = diagnostic.summarize(root)
                self.assertEqual(compact["predictor_optimizer_steps"], 6)
                self.assertEqual(compact["residual_scale_optimizer_steps"], 9)
                self.assertEqual(compact["additional_interactions"], 0)
                self.assertEqual(compact["folds"][0]["frozen_thresholds"], result["frozen_thresholds"])
        print("RESIDUAL_DIAGNOSTIC_TEST interactions=0 mean_updates=12 scale_updates=18 affine_fits=4")


if __name__ == "__main__":
    torch.set_num_threads(2)
    unittest.main()
