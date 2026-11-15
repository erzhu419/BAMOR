import copy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import torch

import collect_matched_scale_normals as collection
from headwise_change_evidence import CONFIG
import run_deployment_startup_calibration as probe
import run_cross_policy_calibration as cross
import run_sensitivity_guarded_selection as guarded
import run_window_risk_selection as risk
import run_matched_residual_scale_probe as matched
import run_affine_residual_change_probe as affine
import run_headwise_change_probe as plain
import run_conditional_change_probe as original
from test_affine_residual_change_probe import make_source
from test_matched_residual_scale_probe import make_fresh_fixture


def startup_metadata(seed):
    return dict(seed=seed, source_step=75000, collection_interactions=5000,
        initialization_probes=3, additional_interactions=5003,
        cases={case: dict(seeds=collection.seeds(seed, i)) for i, case in enumerate(collection.COLLECTION_STEPS)})


class DeploymentStartupTests(unittest.TestCase):
    def test_independent_stream_provenance_and_nonzero_startup_cost(self):
        result = probe.startup_provenance(startup_metadata(2103), 2103)
        self.assertEqual((result["calibration_transitions"], result["initialization_probes"],
            result["required_interactions"], result["new_interactions"]), (5000, 3, 5003, 0))
        self.assertTrue(result["already_charged_in_fresh_collection"])
        self.assertEqual(len({r["reset_seed"] for r in result["case_seeds"].values()}), 3)

    def test_wrong_policy_or_evaluation_reset_is_rejected(self):
        meta = startup_metadata(2103)
        with self.assertRaises(AssertionError):
            probe.startup_provenance(meta, 2102)
        meta["cases"]["stationary"]["seeds"]["reset_seed"] = 50000000 + 1000 * 2203
        with self.assertRaises(AssertionError):
            probe.startup_provenance(meta, 2103)

    def test_changed_startup_stream_is_not_accepted_as_normal(self):
        with tempfile.TemporaryDirectory() as tmp:
            root, env = Path(tmp), probe.ENVIRONMENTS[0]
            make_fresh_fixture(root, env)
            p = probe.source_folder(root, env, 2103) / "readout.json"
            meta = json.loads(p.read_text())
            meta["cases"]["actor_change"]["ground_truth_mode_history_audit_only"] = [[0, 0], [500, 1]]
            probe.write(p, meta)
            with self.assertRaises(AssertionError):
                probe.fresh_data(root, env, 2103)

    def test_pipeline_freezes_models_separates_fits_and_evaluation_and_charges_startup(self):
        with tempfile.TemporaryDirectory() as tmp, patch.dict(CONFIG, hidden=8, updates=3, scale_updates=3):
            source, fresh, base, old, parent, risk_root, guarded_root, prior, root = (Path(tmp) / name for name in
                ("source", "fresh", "plain", "affine", "matched", "risk", "guarded", "cross", "startup"))
            env, seed = probe.ENVIRONMENTS[0], 2101
            make_source(source, env)
            make_fresh_fixture(fresh, env)
            for s in probe.SEEDS:
                p = probe.source_folder(fresh, env, s) / "readout.json"
                meta = json.loads(p.read_text())
                for i, case in enumerate(collection.COLLECTION_STEPS):
                    meta["cases"][case]["seeds"] = collection.seeds(s, i)
                probe.write(p, meta)
            events, reads, fits, observed = [], [], [], []
            calibrate, read_fresh, read_source, fit_scale = (
                probe.calibrate_fixed_support, probe.fresh_data, probe.source_readout, probe.GuardedEvidence.fit_scale_losses)
            def record_fresh(root, env, s):
                reads.append(s)
                return read_fresh(root, env, s)
            def record_source(source, env, s):
                events.append("evaluation")
                return read_source(source, env, s)
            def record_fit(evidence, x, losses, fit_seed):
                fits.append(x.copy())
                return fit_scale(evidence, x, losses, fit_seed)
            def record_calibration(evidence, x, y):
                events.append("calibration")
                models = [evidence.reward_predictor.model, evidence.dynamics_predictor.model, *evidence.scale_models]
                states = [copy.deepcopy(m.state_dict()) for m in models]
                radii, weights = evidence.radii.copy(), evidence.weights.copy()
                coefficients = evidence.reward_predictor.affine_coefficients.copy()
                result = calibrate(evidence, x, y)
                np.testing.assert_array_equal(evidence.radii, radii)
                np.testing.assert_array_equal(evidence.reward_predictor.affine_coefficients, coefficients)
                self.assertEqual(evidence.weights, weights)
                self.assertTrue(all(not p.requires_grad for model in models for p in model.parameters()))
                self.assertTrue(all(torch.equal(state[k], model.state_dict()[k])
                    for state, model in zip(states, models) for k in state))
                observed.append((x.copy(), y.copy()))
                return result
            with patch("builtins.print"), patch.object(original.training, "make_piecewise_mo_env",
                    side_effect=AssertionError("No new environment")):
                plain.run(env, seed, source, base)
                affine.run(env, seed, source, old, base)
                matched.run(env, seed, source, parent, old, fresh)
                risk.run(env, seed, source, risk_root, parent, fresh)
                guarded.run(env, seed, source, guarded_root, parent, fresh, risk_root)
                before = cross.run(env, seed, source, prior, guarded_root, fresh)
                with patch.object(probe, "fresh_data", record_fresh), patch.object(probe, "source_readout", record_source), \
                     patch.object(probe, "calibrate_fixed_support", record_calibration), \
                     patch.object(probe.GuardedEvidence, "fit_scale_losses", record_fit):
                    result = probe.run(env, seed, source, root, prior, fresh)
            roles = result["roles"]
            self.assertEqual(reads, roles["fit_seeds"] + [seed])
            self.assertEqual(events, ["calibration", "evaluation"])
            self.assertEqual(len(fits), 1)
            rows = [read_fresh(fresh, env, s) for s in roles["fit_seeds"]]
            rows.append(probe.normal_data(source, env, roles["scale_seed"], "expanded_normal"))
            np.testing.assert_array_equal(fits[0], np.concatenate([r[0] for r in rows]))
            startup = read_fresh(fresh, env, seed)
            self.assertEqual(len(observed), 1)
            for i in (0, 1):
                np.testing.assert_array_equal(observed[0][i], startup[i])
            self.assertEqual(result["frozen_selection"], before["frozen_selection"])
            for arm in probe.CONTROLS:
                self.assertEqual(result["arms"][arm], before["arms"][arm])
            self.assertEqual((result["predictor_optimizer_steps"], result["residual_scale_optimizer_steps"]), (6, 9))
            self.assertTrue(all(result[k] == 0 for k in probe.ZERO_WORK))
            with patch.object(probe.GuardedEvidence, "fit", side_effect=AssertionError("Repeated fit")):
                self.assertEqual(probe.run(env, seed, source, root, prior, fresh), result)
            with patch.object(probe, "ENVIRONMENTS", (env,)), patch.object(probe, "SEEDS", (seed,)):
                summary = probe.summarize(root)
                with patch.object(probe, "screen", side_effect=lambda runs, arm:
                        dict(development_screen_passed=arm != probe.PRIMARY)):
                    self.assertEqual(probe.summarize(root)["decision"], "hold_conditional_evidence_integration")
            self.assertEqual(summary["deployment_startup_required_interactions"], 5003)
            self.assertEqual(summary["additional_interactions"], 0)
            self.assertEqual(summary["residual_scale_optimizer_steps"], 9)
            compact = probe.compact_report(summary)
            self.assertNotIn("windows", compact["environments"][env]["runs"][0]["arms"][probe.PRIMARY]["calibration"])
        print("DEPLOYMENT_STARTUP_TEST added_interactions=0 mean_updates=42 total_scale_updates=117 startup_reuse_required=5003")


if __name__ == "__main__":
    torch.set_num_threads(2)
    unittest.main()
