import copy
import tempfile
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
import torch

from episode_blocked_change_evidence import EpisodeBlockedChangeEvidence, ENSEMBLE_CONFIG
import run_conditional_change_probe as original_probe
import run_episode_blocked_change_probe as probe


class EpisodeBlockedChangeTests(unittest.TestCase):
    def test_source_roles_are_disjoint_in_all_folds(self):
        for seed in probe.SEEDS:
            roles = probe.fold_roles(seed)
            self.assertEqual(len(roles["fit_seeds"]), 3)
            self.assertNotIn(seed, roles["fit_seeds"])
            self.assertNotIn(roles["calibration_seed"], roles["fit_seeds"])
            self.assertNotEqual(seed, roles["calibration_seed"])
            self.assertEqual(set(roles["fit_seeds"] + [roles["calibration_seed"], seed]), set(probe.SEEDS))

    def test_input_only_disagreement_and_no_calibration_updates(self):
        rng = np.random.default_rng(4)
        inputs = rng.normal(size=(100, 3)).astype(np.float32)
        targets = rng.normal(size=(100, 4)).astype(np.float32)
        evidence = EpisodeBlockedChangeEvidence(3, 4, 2, updates=4, hidden=8, window=10)
        work = evidence.fit(inputs, targets, 15)
        self.assertEqual(work["optimizer_steps"], 12)
        states = [copy.deepcopy(member.model.state_dict()) for member in evidence.members]
        predictions, mean = evidence.predict(inputs), evidence.y_mean.copy()
        parts = evidence.components(inputs, targets)
        changed = evidence.components(inputs, targets + 100)
        np.testing.assert_array_equal(parts["disagreement"], changed["disagreement"])
        evidence.calibrate(inputs, targets + 10)
        evidence.evaluate(inputs, targets - 10)
        np.testing.assert_array_equal(evidence.predict(inputs), predictions)
        np.testing.assert_array_equal(evidence.y_mean, mean)
        self.assertTrue(all(torch.equal(state[key], member.model.state_dict()[key])
                            for member, state in zip(evidence.members, states) for key in state))
        unsupported = evidence.evaluate(inputs + 100, targets + 100)
        self.assertTrue(all(not window["eligible"] and not window["alarm"] for window in unsupported["windows"]))

    def test_disagreement_scaling_formula(self):
        evidence = EpisodeBlockedChangeEvidence(3, 4, 2)
        evidence.y_scale = np.ones(4, np.float32)
        evidence.members = [SimpleNamespace(predict=lambda x, value=value: np.full((len(x), 4), value, np.float32))
                            for value in (-2., 0., 2.)]
        parts = evidence.components(np.zeros((10, 3), np.float32), np.full((10, 4), 2., np.float32))
        np.testing.assert_allclose(parts["raw_errors"], 4.)
        np.testing.assert_allclose(parts["disagreement"], 8. / 3.)
        np.testing.assert_allclose(parts["errors"], 12. / 11.)

    def test_development_screen_does_not_hide_missing_coverage_or_controls(self):
        cases = {case: dict(available=True, any_post_alarm=case.startswith("mode"),
            timely_alarm=case.startswith("mode"), pre_available=True, pre_alarm_windows=0,
            intervention_active=True) for case in probe.CASES}
        runs = [dict(arms={arm: dict(cases=copy.deepcopy(cases)) for arm in probe.ARMS}) for _ in probe.SEEDS]
        self.assertTrue(probe.screen(runs, "expanded_normal")["development_screen_passed"])
        for row in runs[:2]:
            row["arms"]["expanded_normal"]["cases"]["actor_change"]["available"] = False
        self.assertFalse(probe.screen(runs, "expanded_normal")["development_screen_passed"])
        for row in runs[:2]:
            row["arms"]["expanded_normal"]["cases"]["actor_change"]["available"] = True
            row["arms"]["expanded_normal"]["cases"]["stationary"]["pre_alarm_windows"] = 1
        self.assertFalse(probe.screen(runs, "expanded_normal")["development_screen_passed"])
        self.assertTrue(probe.screen(runs, "fixed_only")["development_screen_passed"])

    def test_stored_routes_exclude_positive_data_and_shared_prefix_duplicates(self):
        with tempfile.TemporaryDirectory() as tmp, patch.dict(ENSEMBLE_CONFIG, updates=4, hidden=8):
            source, root = Path(tmp) / "source", Path(tmp) / "output"
            env = probe.ENVIRONMENTS[0]
            for seed in probe.SEEDS:
                rng = np.random.default_rng(seed)
                prefix = rng.normal(size=(3000, 3)).astype(np.float32)
                prefix[:, 0] += seed - 2101
                stationary = rng.normal(size=(1000, 3)).astype(np.float32)
                stationary[:, 0] += seed - 2101
                prefix_targets = np.column_stack([prefix, prefix[:, 0] ** 2]).astype(np.float32)
                terminal = np.zeros((3000, 2), bool)
                terminal[999::1000, 1] = True
                data = dict(prefix_inputs=prefix, prefix_targets=prefix_targets, prefix_terminals=terminal,
                            prefix_weights=np.zeros((3000, 2), np.float32))
                cases = {}
                for case in probe.CASES:
                    inputs = stationary.copy()
                    if case in ("actor_change", "sampler_change"):
                        inputs[500:, 2] += .5
                    targets = np.column_stack([inputs, inputs[:, 0] ** 2]).astype(np.float32)
                    if case.startswith("mode"):
                        targets[500:, :2] += 500.
                    end = np.zeros((1000, 2), bool)
                    end[-1, 1] = True
                    data.update({f"{case}_inputs": inputs, f"{case}_targets": targets, f"{case}_terminals": end})
                    cases[case] = dict(ground_truth_mode_history_audit_only=(
                        [[0, 0], [500, int(case[-1])]] if case.startswith("mode") else [[0, 0]]), intervention_active=True)
                folder = probe.source_folder(source, env, seed)
                probe.write(folder / "readout.json", dict(protocol=probe.SOURCE_PROTOCOL, env=env,
                    seed=seed, complete=True, paired_prefixes_verified=5, cases=cases))
                np.savez_compressed(folder / "transitions.npz", **data)
            fixed_x, fixed_y, fixed_meta = probe.normal_data(source, env, 2101, "fixed_only")
            expanded_x, expanded_y, expanded_meta = probe.normal_data(source, env, 2101, "expanded_normal")
            self.assertEqual((len(fixed_x), len(expanded_x)), (4000, 5000))
            np.testing.assert_array_equal(expanded_x[:4000], fixed_x)
            np.testing.assert_array_equal(expanded_y[:4000], fixed_y)
            self.assertLess(float(expanded_y.max()), 500.)
            self.assertEqual(fixed_meta["fixed_rollout_completed_episodes"], 4)
            self.assertEqual(expanded_meta["fixed_rollout_completed_episodes"], 4)
            observed_fits, fit = [], EpisodeBlockedChangeEvidence.fit

            def record_fit(evidence, inputs, targets, seed):
                observed_fits.append((len(inputs), inputs.mean(0), seed))
                return fit(evidence, inputs, targets, seed)

            with patch.object(EpisodeBlockedChangeEvidence, "fit", record_fit), \
                 patch.object(original_probe.training, "make_piecewise_mo_env", side_effect=AssertionError("new interaction")), \
                 patch("builtins.print"):
                report = probe.run(env, 2101, source, root)
            self.assertTrue(report["complete"])
            self.assertTrue(report["development_only"])
            self.assertEqual(report["additional_interactions"], 0)
            self.assertEqual(report["predictor_optimizer_steps"], 24)
            self.assertEqual([row[0] for row in observed_fits], [12000, 15000])
            self.assertEqual(observed_fits[0][2], observed_fits[1][2])
            roles = probe.fold_roles(2101)
            for index, arm in enumerate(probe.ARMS):
                expected = np.concatenate([probe.normal_data(source, env, seed, arm)[0] for seed in roles["fit_seeds"]])
                np.testing.assert_array_equal(observed_fits[index][1], expected.mean(0))
                self.assertEqual(report["arms"][arm]["calibration_source"]["seed"], 2102)
            with patch.object(EpisodeBlockedChangeEvidence, "fit", side_effect=AssertionError("repeated work")):
                self.assertEqual(probe.run(env, 2101, source, root), report)
        print("EPISODE_BLOCKED_TEST interactions=0 predictor_updates=36")


if __name__ == "__main__":
    torch.set_num_threads(2)
    unittest.main()
