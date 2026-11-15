import copy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import torch

import run_two_stream_offset_confirmation as probe
import run_conditional_change_probe as original
from headwise_change_evidence import CONFIG
from test_affine_residual_change_probe import make_source
from test_matched_residual_scale_probe import make_fresh_fixture
from test_headwise_change_probe import targets
from test_startup_confirmation import fixture_collection
from test_two_stream_mean_offset import collection_metadata


def metadata(env, seed, group):
    row = fixture_collection(env, seed, group)
    row["source"] = "fixture-policies"
    for index, case in enumerate(probe.acquisition.steps(group)):
        row["cases"][case]["seeds"] = probe.seeds(seed, group, index)
    return row


def development_fixture(source, fresh, parent, env, seed):
    p, shared = probe.PARENT, probe.shared
    make_source(source, env)
    make_fresh_fixture(fresh, env)
    roles = p.fold_roles(seed)
    fit = [p.normal_data(source, env, s, "expanded_normal") for s in roles["fit_seeds"]]
    scale = [p.fresh_data(fresh, env, s) for s in roles["fit_seeds"]]
    scale.append(p.normal_data(source, env, roles["scale_seed"], "expanded_normal"))
    fit_x, fit_y = (np.concatenate([r[i] for r in fit]) for i in (0, 1))
    scale_x, scale_y = (np.concatenate([r[i] for r in scale]) for i in (0, 1))
    model = shared.MeanOffsetEvidence(5, 5, 2)
    model.weights = {p.GUARDED: .5}
    work = model.fit(fit_x, fit_y, 5000, 80000000 + 10 * seed)
    scale_work = model.fit_scale_losses(scale_x, model.raw_losses(scale_x, scale_y), 81000000 + 10 * seed)
    model.radii = np.full(3, 1e10)
    rng = np.random.default_rng(91236)
    old_x = rng.normal(size=(3000, 5)).astype(np.float32)
    offset = model.fit_offset(old_x, targets(old_x))
    cal_x = rng.normal(size=(5000, 5)).astype(np.float32)
    calibration = shared.calibrate(model, cal_x, targets(cal_x), arms=probe.ARMS)
    calibration[probe.ANCHOR] = copy.deepcopy(calibration[probe.CONTROL])
    arms = {a: dict(calibration=calibration[a], cases={}) for a in probe.ARMS}
    with np.load(source / env.replace("-", "_") / f"seed{seed}" / "transitions.npz") as stored:
        evaluation = {c: (stored[f"{c}_inputs"].copy(), stored[f"{c}_targets"].copy()) for c in p.CASES}
    for case, (x, y) in evaluation.items():
        paired, support = shared.windows(model, x, y, arms=probe.ARMS)
        for arm in probe.ARMS:
            p.development.mark_alarms(paired[arm], calibration[arm]["error_threshold"])
            arms[arm]["cases"][case] = p.case_summary(dict(windows=paired[arm], supported=support.any(1)))
    row = dict(protocol=probe.development.PROTOCOL, env=env, roles=roles, config=CONFIG, complete=True,
        reward_dim=2, frozen_selection=dict(global_weight=.5), predictor_work=work, scale_work=scale_work,
        predictor_optimizer_steps=6, residual_scale_optimizer_steps=9, affine_fits=2, offset_fit_calls=1,
        scale_fit_sources=roles["fit_seeds"] + [roles["scale_seed"]], offset_fit=offset,
        offset_fit_case="stationary", calibration_cases=list(p.acquisition.STARTUP_STEPS),
        calibration_steps=5000, used_startup_steps=8000, thresholds_and_offset_frozen_before_evaluation_read=True,
        original_anchor_reproduced=True, arms=arms, startup_sources=dict(
            offset=collection_metadata(env, seed, "startup", old=True),
            calibration=collection_metadata(env, seed, "startup")), **dict.fromkeys(shared.ZERO_WORK, 0))
    probe.development.validate(row, env, seed)
    p.write(parent / env.replace("-", "_") / f"test_seed{seed}" / "readout.json", row)
    return row, evaluation


class TwoStreamOffsetConfirmationTests(unittest.TestCase):
    def test_fresh_streams_and_full_physical_budget(self):
        for seed in probe.PARENT.SEEDS:
            streams = [probe.seeds(seed, group, index) for group in probe.GROUPS
                for index in range(len(probe.acquisition.steps(group))) if group != "evaluation" or index == 0]
            old = [fn(seed, group, index) for fn in (probe.acquisition.seeds, probe.PARENT.seeds)
                for group in ("startup", "evaluation") for index in range(3 if group == "startup" else 1)]
            for key in streams[0]:
                self.assertEqual(len({s[key] for s in streams}), len(streams))
                self.assertFalse({s[key] for s in streams} & {s[key] for s in old})
            self.assertEqual(probe.seeds(seed, "evaluation"), probe.seeds(seed, "evaluation", 5))
            for group in probe.GROUPS:
                probe.acquisition.validate(metadata(probe.PARENT.ENVIRONMENTS[0], seed, group),
                    probe.PARENT.ENVIRONMENTS[0], seed, group, seed_fn=probe.seeds)
        self.assertEqual(probe.acquisition.steps("offset"), dict(stationary=3000))
        self.assertEqual(sum(probe.acquisition.budget(g)["additional_interactions"] for g in probe.GROUPS), 14010)
        self.assertEqual(10 * sum(probe.acquisition.budget(g)["initialization_probes"] for g in probe.GROUPS), 100)

    def test_real_collection_keeps_frozen_policies_pairing_and_cached_groups(self):
        acquisition, training = probe.acquisition, original.training
        with tempfile.TemporaryDirectory() as tmp, patch.object(original, "SOURCE_HIDDEN", 8), \
             patch.object(original, "CHANGE_AT", 10), patch.object(acquisition, "CHANGE_AT", 10), \
             patch.object(acquisition, "CASE_STEPS", 20), \
             patch.dict(acquisition.STARTUP_STEPS, stationary=40, sampler_change=20, actor_change=20):
            root, source = Path(tmp) / "collection", Path(tmp) / "policies"
            for env_name in probe.PARENT.ENVIRONMENTS:
                env = training.make_piecewise_mo_env(env_name, switch_interval=10**9, seed=999)
                try:
                    obs_dim, act_dim, reward_dim = env.observation_space.shape[0], env.action_space.shape[0], env.reward_dim
                    context_dim = training.context_dim_for_method(original.METHODS["fixed"], obs_dim, act_dim, reward_dim, env.num_modes)
                    for index, (name, method) in enumerate(original.METHODS.items()):
                        torch.manual_seed(999 + index)
                        policy = training.PolicyNet(obs_dim, act_dim, context_dim, hidden=8)
                        encoder = training.LatentTransitionEncoder(obs_dim, act_dim, reward_dim, hidden=8, latent_dim=training.LATENT_CONTEXT_DIM)
                        path = source / env_name.replace("-", "_") / name
                        acquisition.write(path / f"{method}_seed2101.json", dict(env=env_name, method=method, seed=2101, total_steps=75000))
                        torch.save(dict(policy=policy.state_dict(), encoder=encoder.state_dict()), path / f"{method}_seed2101_model.pt")
                finally:
                    env.close()
                models, states, loader = [], [], acquisition.load_models
                def record_load(env, seed, source):
                    policies, encoder = loader(env, seed, source)
                    loaded = [*policies.values(), encoder]
                    models.extend(loaded)
                    states.extend(copy.deepcopy(m.state_dict()) for m in loaded)
                    return policies, encoder
                with patch("builtins.print"), patch.object(acquisition, "load_models", record_load):
                    collected = {g: acquisition.run_group(env_name, 2101, g, root, source, seed_fn=probe.seeds) for g in probe.GROUPS}
                self.assertEqual([collected[g]["additional_interactions"] for g in probe.GROUPS], [41, 83, 126])
                self.assertEqual(collected["evaluation"]["paired_prefixes_verified"], 5)
                self.assertTrue(all(not p.requires_grad for m in models for p in m.parameters()))
                self.assertTrue(all(torch.equal(state[k], m.state_dict()[k]) for state, m in zip(states, models) for k in state))
                for group in probe.GROUPS:
                    rows, report = acquisition.data(root, env_name, 2101, group, seed_fn=probe.seeds)
                    self.assertEqual(set(rows), set(acquisition.steps(group)))
                    with patch.object(training, "make_piecewise_mo_env", side_effect=AssertionError("Recollection")):
                        self.assertEqual(acquisition.run_group(env_name, 2101, group, root, source, seed_fn=probe.seeds), report)
        print("REAL_TWO_STREAM_CONFIRMATION_TEST interactions=502 including2 fixture initialization probes")

    def test_pipeline_freezes_before_evaluation_and_records_failure_without_retry(self):
        with tempfile.TemporaryDirectory() as tmp, patch.dict(CONFIG, hidden=8, updates=3, scale_updates=3):
            source, fresh, parent, root = (Path(tmp) / n for n in ("source", "fresh", "parent", "confirmation"))
            env, seed = probe.PARENT.ENVIRONMENTS[0], 2101
            prior, evaluation = development_fixture(source, fresh, parent, env, seed)
            rng = np.random.default_rng(74163)
            offset_x = rng.normal(size=(3000, 5)).astype(np.float32)
            offset_y = targets(offset_x)
            offset_y[:, 0] += .75
            startup = {c: (x, targets(x)) for c, n in probe.acquisition.STARTUP_STEPS.items()
                for x in [rng.normal(size=(n, 5)).astype(np.float32)]}
            groups = dict(offset=dict(stationary=(offset_x, offset_y)), startup=startup, evaluation=evaluation)
            events, original_offset, original_calibrate = [], probe.shared.MeanOffsetEvidence.fit_offset, probe.shared.calibrate
            def collect_group(e, s, group, collection, seed_fn):
                self.assertIs(seed_fn, probe.seeds)
                events.append(group)
                if group == "evaluation":
                    saved = json.loads((root / env.replace("-", "_") / f"test_seed{seed}" / "readout.json").read_text())
                    self.assertTrue(saved["thresholds_and_offset_frozen_before_evaluation_collection"])
                    self.assertEqual(saved["arms"][probe.ANCHOR]["calibration"], prior["arms"][probe.ANCHOR]["calibration"])
                row = metadata(e, s, group)
                probe.PARENT.write(probe.acquisition.folder(collection, e, s, group) / "readout.json", row)
                return row
            def read_group(collection, e, s, group, seed_fn):
                self.assertIs(seed_fn, probe.seeds)
                return groups[group], metadata(e, s, group)
            def fit_offset(model, x, y):
                events.append("fit_offset")
                np.testing.assert_array_equal(x, offset_x)
                np.testing.assert_array_equal(y, offset_y)
                return original_offset(model, x, y)
            def calibrate(model, x, y, arms):
                events.append("freeze")
                self.assertEqual(arms, probe.ARMS)
                for index, array in enumerate((x, y)):
                    np.testing.assert_array_equal(array, np.concatenate([r[index] for r in startup.values()]))
                offset, radii, weights = model.normalized_offset.copy(), model.radii.copy(), model.weights.copy()
                result = original_calibrate(model, x, y, arms=arms)
                np.testing.assert_array_equal(offset, model.normalized_offset)
                np.testing.assert_array_equal(radii, model.radii)
                self.assertEqual(weights, model.weights)
                return result
            with patch("builtins.print"), patch.object(probe.acquisition, "run_group", collect_group), \
                 patch.object(probe.acquisition, "data", read_group), patch.object(probe.shared, "calibrate", calibrate), \
                 patch.object(probe.shared.MeanOffsetEvidence, "fit_offset", fit_offset):
                result = probe.run(env, seed, root, parent, source, fresh)
            self.assertEqual(events, ["offset", "fit_offset", "startup", "freeze", "evaluation"])
            self.assertEqual((result["additional_interactions"], result["initialization_probes"]), (14010, 10))
            self.assertEqual((result["predictor_optimizer_steps"], result["residual_scale_optimizer_steps"]), (6, 9))
            self.assertEqual(result["frozen_selection"], prior["frozen_selection"])
            parent_path = parent / env.replace("-", "_") / f"test_seed{seed}" / "readout.json"
            self.assertEqual(json.loads(parent_path.read_text()), prior)
            with patch.object(probe.shared.MeanOffsetEvidence, "fit", side_effect=AssertionError("Repeated fit")):
                self.assertEqual(probe.run(env, seed, root, parent, source, fresh), result)
            with patch.object(probe.PARENT, "ENVIRONMENTS", (env,)), patch.object(probe.PARENT, "SEEDS", (seed,)):
                summary = probe.summarize(root)
                self.assertEqual(summary["cumulative_acquisition_interactions"], 374290)
                self.assertFalse(summary["evidence_integrated"])
                self.assertEqual(set(summary["environments"][env]["paired"]), {probe.ANCHOR, probe.CONTROL})
            failed_root = Path(tmp) / "failed"
            def fail(e, s, group, collection, seed_fn):
                row = metadata(e, s, group)
                row.update(complete=False, collection_interactions=3, initialization_probes=1, additional_interactions=4)
                probe.PARENT.write(probe.acquisition.folder(collection, e, s, group) / "readout.json", row)
                raise RuntimeError("Interrupted offset collection")
            with patch.object(probe.acquisition, "run_group", fail):
                with self.assertRaisesRegex(RuntimeError, "Interrupted offset"):
                    probe.run(env, seed, failed_root, parent, source, fresh)
            partial = json.loads((failed_root / env.replace("-", "_") / f"test_seed{seed}" / "readout.json").read_text())
            self.assertFalse(partial["complete"])
            self.assertEqual((partial["additional_interactions"], partial["initialization_probes"]), (4, 1))
            with patch.object(probe.shared.MeanOffsetEvidence, "fit", side_effect=AssertionError("Retry")):
                with self.assertRaisesRegex(RuntimeError, "Incomplete offset confirmation"):
                    probe.run(env, seed, failed_root, parent, source, fresh)
        print("TWO_STREAM_CONFIRMATION_PIPELINE_TEST actual_interactions=0 simulated_interactions=14010 simulated_failed=4 mean_updates=18 scale_updates=27 affine_fits=6 offset_fit_calls=2")


if __name__ == "__main__":
    torch.set_num_threads(2)
    unittest.main()
