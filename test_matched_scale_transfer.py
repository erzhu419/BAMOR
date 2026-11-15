import copy
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import torch

import diagnose_matched_scale_transfer as diagnostic


def window(index, raw, global_scale, conditional_scale, threshold, supported=100):
    heads = {name: dict(eligible=supported >= 80, supported_steps=supported,
        domain="state_action", statistic=value / conditional_scale if supported >= 80 else None)
        for name, value in zip(("reward_0", "dynamics"), raw)}
    values = [head["statistic"] for head in heads.values() if head["eligible"]]
    row = dict(start_step=index * 100 + 1, end_step=(index + 1) * 100,
        eligible=bool(values), supported_steps=supported,
        statistic=max(values) if values else None, heads=heads,
        alarm=bool(values and max(values) > threshold))
    global_row = copy.deepcopy(row)
    for head, value in zip(global_row["heads"].values(), raw):
        head["statistic"] = value / global_scale if head["eligible"] else None
    global_row["statistic"] = max(raw) / global_scale if values else None
    return dict(global_scale=global_row, conditional_scale=row)


def paired(index=0, raw=(5., 1.), supported=100):
    return diagnostic.pair_window(window(index, raw, 2., 2.5, 4., supported),
        window(index, raw, 3., .5, 3., supported), [2., 2.], [3., 3.], (4., 3.))


class MatchedScaleTransferTests(unittest.TestCase):
    def test_exact_factorization_and_effective_scale(self):
        row = paired()
        head = row["heads"]["reward_0"]
        self.assertEqual(head["raw_loss_mean"], 5.)
        self.assertEqual(head["old_effective_scale"], 2.5)
        self.assertEqual(head["new_effective_scale"], .5)
        self.assertEqual(head["scale_factor"], 5.)
        self.assertAlmostEqual(head["new_ratio"], head["old_ratio"] * 5. * 4. / 3.)
        self.assertEqual(diagnostic.group_name(row), "new_only_alarm")
        self.assertEqual(diagnostic.describe([row])["sufficient_changes"], {"scale_sufficient": 1})

    def test_pairing_rejects_changed_error_and_support(self):
        old = window(0, (5., 1.), 2., 2.5, 4.)
        new = window(0, (5., 1.), 3., .5, 3.)
        for key, value in (("supported_steps", 99), ("domain", "action")):
            broken = copy.deepcopy(new)
            broken["conditional_scale"]["heads"]["reward_0"][key] = value
            with self.assertRaises(AssertionError):
                diagnostic.pair_window(old, broken, [2., 2.], [3., 3.], (4., 3.))
        broken = copy.deepcopy(new)
        broken["global_scale"]["heads"]["reward_0"]["statistic"] += .1
        with self.assertRaises(AssertionError):
            diagnostic.pair_window(old, broken, [2., 2.], [3., 3.], (4., 3.))

    def test_zero_error_and_missing_support_are_not_scale_estimates(self):
        zero = paired(raw=(0., 0.))
        self.assertIsNone(zero["heads"]["reward_0"]["new_effective_scale"])
        missing = paired(supported=0)
        self.assertIsNone(missing["dominant_head"])
        self.assertEqual(diagnostic.describe([missing])["eligible_windows"], 0)

    def test_counterfactuals_separate_threshold_and_joint_changes(self):
        threshold = paired()
        threshold.update(old_joint_score=3.5, new_joint_score=3.5, old_threshold=4., new_threshold=3.)
        joint = copy.deepcopy(threshold)
        joint.update(old_joint_score=2.5)
        both = copy.deepcopy(threshold)
        both.update(new_joint_score=5.)
        self.assertEqual(diagnostic.describe([threshold, joint, both])["sufficient_changes"],
            dict(threshold_sufficient=1, joint_change_required=1, both_sufficient=1))

    def test_calibration_tail_interpolation(self):
        rows = [paired(index=i, raw=(value, 1.)) for i, value in enumerate((1., 2., 3.))]
        threshold = float(np.quantile([row["new_joint_score"] for row in rows], .99))
        result = diagnostic.calibration_tail(rows, threshold, .99)
        self.assertAlmostEqual(sum(row["weight"] * row["joint_score"] for row in result["operands"]), threshold)
        self.assertEqual([row["end_step"] for row in result["operands"]], [200, 300])

    def test_geometry_uses_only_inputs_and_predictor_supported_steps(self):
        rng = np.random.default_rng(7)
        reference = rng.normal(size=(240, 2)).astype(np.float32)
        inputs = rng.normal(size=(200, 2)).astype(np.float32)
        distances = diagnostic.nearest(inputs, reference)
        radius = float(np.quantile(distances, .99))
        rows = [paired(index=i, supported=int((distances[i * 100:(i + 1) * 100] <= radius).sum())) for i in range(2)]
        cal, normal = copy.deepcopy(rows), copy.deepcopy(rows)
        refs = dict(predictor=reference, old_scale=reference, new_scale=reference, calibration=inputs)
        domains = dict(reward_0="state_action", dynamics="state_action")
        radii = dict(reward_0=radius, dynamics=radius)
        diagnostic.attach_geometry(cal, normal, refs, inputs, domains, 1, radii, dict(window=100, quantile=.99))
        before = copy.deepcopy(normal[0]["heads"]["reward_0"]["geometry"])
        changed = copy.deepcopy(rows)
        changed[0]["heads"]["reward_0"].update(new_score=1e10, scale_factor=1e12)
        diagnostic.attach_geometry(copy.deepcopy(rows), changed, refs, inputs,
            dict(reward_0="state_action", dynamics="state_action"), 1, radii, dict(window=100, quantile=.99))
        self.assertEqual(before, changed[0]["heads"]["reward_0"]["geometry"])
        self.assertEqual(before["calibration"]["supported_fraction"], 1.)
        broken = copy.deepcopy(rows)
        broken[0]["heads"]["reward_0"]["supported_steps"] -= 1
        with self.assertRaises(AssertionError):
            diagnostic.attach_geometry(copy.deepcopy(rows), broken, refs, inputs,
                dict(reward_0="state_action", dynamics="state_action"), 1, radii, dict(window=100, quantile=.99))

    def test_fold_has_unique_normal_windows_and_no_model_work(self):
        roles = dict(fit_seeds=[2104, 2105], scale_seed=2103, calibration_seed=2102, test_seed=2101)
        old, new = [], []
        for index in range(10):
            old.append(window(index, (5., 1.), 2., 2.5, 4.))
            new.append(window(index, (5., 1.), 3., .5, 3.))
        def saved(rows, scales, threshold):
            return dict(roles=roles, predictor_work=dict(support_domains=["state_action", "state_action"]),
                scale_work=dict(global_scales=scales), config=dict(window=100, quantile=.99),
                arms={arm: dict(calibration=dict(windows=[row[arm] for row in rows],
                    error_threshold=threshold, support_radii=dict(reward_0=1., dynamics=1.)),
                    cases={case: dict(windows=[row[arm] for row in rows])
                           for case in ("stationary", "sampler_change", "actor_change")})
                      for arm in ("global_scale", "conditional_scale")})
        previous, candidate = saved(old, [2., 2.], 4.), saved(new, [3., 3.], 10.)
        # Calibration statistics must reproduce their saved quantile, unlike test labels.
        for arm in candidate["arms"].values():
            arm["calibration"]["windows"] = [copy.deepcopy(row) for row in arm["calibration"]["windows"]]
        references = dict(predictor=np.zeros((2000, 2)))
        tests = {case: np.zeros((1000, 2)) for case in ("stationary", "sampler_change", "actor_change")}
        with patch.object(diagnostic, "build_inputs", return_value=(references, tests, 1)), \
             patch.object(diagnostic, "attach_geometry"), patch.object(torch.optim, "Adam", side_effect=AssertionError("No training")):
            result = diagnostic.analyze_fold(candidate, previous)
        all_rows = [row for group in result["normal_groups"].values() for row in group["windows"]]
        self.assertEqual(len(all_rows), 20)
        self.assertEqual(sum(row["case"] == "shared_prefix" for row in all_rows), 5)
        self.assertEqual(sum(row["case"] == "actor_change" for row in all_rows), 5)
        self.assertEqual(sum(group["summary"]["windows"] for group in result["normal_groups"].values()), 20)

    def test_action_footprint_has_separate_geometry(self):
        rng = np.random.default_rng(9)
        reference = rng.normal(size=(240, 3)).astype(np.float32)
        inputs = rng.normal(size=(200, 3)).astype(np.float32)
        radii, rows = {}, [paired(index=i) for i in range(2)]
        for row in rows:
            row["heads"]["reward_1"] = row["heads"].pop("dynamics")
        for name, domain in (("reward_0", "state_action"), ("reward_1", "action")):
            features, ref = (inputs[:, 2:], reference[:, 2:]) if domain == "action" else (inputs, reference)
            distance = diagnostic.nearest(features, ref)
            radii[name] = float(np.quantile(distance, .99))
            for index, row in enumerate(rows):
                row["heads"][name].update(domain=domain,
                    supported_steps=int((distance[index * 100:(index + 1) * 100] <= radii[name]).sum()))
        domains = dict(reward_0="state_action", reward_1="action")
        diagnostic.attach_geometry(copy.deepcopy(rows), rows,
            dict(predictor=reference, old_scale=reference, new_scale=reference, calibration=inputs),
            inputs, domains, 2, radii, dict(window=100, quantile=.99))
        self.assertEqual(domains["reward_1"]["domain"], "action")
        self.assertNotEqual(domains["reward_0"]["radii"]["calibration"], domains["reward_1"]["radii"]["calibration"])

    def test_input_roles_exclude_calibration_and_test_from_scale_references(self):
        roles = dict(fit_seeds=[2104, 2105], scale_seed=2103, calibration_seed=2102, test_seed=2101)
        row = dict(env="mo-halfcheetah-v5", roles=roles, fit_steps=200, scale_steps=100,
            calibration_steps=100, scale_fit_steps=300, config=dict(scale_floor=1e-3))
        old_seen, fresh_seen = [], []
        def original(source, env, seed, arm):
            old_seen.append(seed)
            return np.full((100, 3), seed, np.float32), np.zeros((100, 3)), dict(reward_dim=1)
        def fresh(source, env, seed):
            fresh_seen.append(seed)
            return np.full((100, 3), seed + 100, np.float32), None, {}
        with tempfile.TemporaryDirectory() as tmp:
            source = Path(tmp)
            folder = diagnostic.source_folder(source, row["env"], 2101)
            folder.mkdir(parents=True)
            np.savez(folder / "transitions.npz", **{f"{case}_inputs": np.zeros((1000, 3), np.float32)
                for case in ("stationary", "sampler_change", "actor_change")})
            with patch.object(diagnostic, "normal_data", original), patch.object(diagnostic, "fresh_data", fresh):
                references, tests, obs_dim = diagnostic.build_inputs(row, source, source)
        self.assertEqual(old_seen, [2104, 2105, 2103, 2102])
        self.assertEqual(fresh_seen, [2104, 2105])
        self.assertEqual(obs_dim, 2)
        np.testing.assert_array_equal(references["new_scale"][:100], np.full((100, 3), 199.))
        np.testing.assert_array_equal(references["new_scale"][100:200], np.full((100, 3), 201.))
        np.testing.assert_array_equal(references["new_scale"][200:], references["old_scale"])


if __name__ == "__main__":
    torch.set_num_threads(2)
    unittest.main()
