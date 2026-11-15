import copy
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import torch

import diagnose_scale_support as diagnostic


def saved_windows(distances, radius):
    rows = []
    for start in range(0, len(distances), 100):
        supported = distances[start:start + 100] <= radius
        eligible = bool(supported.mean() >= .8)
        rows.append(dict(start_step=start + 1, end_step=start + 100,
            heads={diagnostic.HEAD: dict(supported_steps=int(supported.sum()), eligible=eligible)}))
    return rows


class ScaleSupportTests(unittest.TestCase):
    def test_chunked_distances_match_direct_geometry(self):
        rng = np.random.default_rng(4)
        inputs = rng.normal(size=(280, 3)).astype(np.float32)
        reference = rng.normal(size=(190, 3)).astype(np.float32)
        direct = np.linalg.norm(inputs[:, None] - reference[None, :], axis=2).min(1)
        np.testing.assert_allclose(diagnostic.nearest(inputs, reference), direct, atol=1e-5, rtol=1e-5)

    def test_leave_window_excludes_whole_block_not_only_self(self):
        inputs = np.array([[0.], [.1], [4.], [4.1], [8.], [8.1]], dtype=np.float32)
        direct = np.abs(inputs - inputs.T)
        blocks = np.arange(len(inputs)) // 2
        direct[blocks[:, None] == blocks[None, :]] = np.inf
        np.testing.assert_allclose(diagnostic.nearest(inputs, inputs, exclude_window=2), direct.min(1), atol=1e-5)
        self.assertGreater(float(diagnostic.nearest(inputs, inputs, exclude_window=2).min()), 3.)

    def test_scale_support_is_reported_on_predictor_supported_steps(self):
        original = [dict(start_step=1, end_step=4, heads={diagnostic.HEAD: dict(supported_steps=2)})]
        decomposed = {4: dict(start_step=1, end_step=4, eligible=False)}
        radii = dict(predictor=1., scale_calibration=1., scale_leave_window=.5)
        row = diagnostic.geometry_windows(np.array([.2, .2, 2., 2.]),
            np.array([2., 2., 0., 0.]), radii, original, decomposed)[0]
        self.assertEqual(row["scale_calibration_supported_fraction"], .5)
        self.assertEqual(row["scale_calibration_supported_on_predictor_steps"], 0.)
        broken = copy.deepcopy(original)
        broken[0]["heads"][diagnostic.HEAD]["supported_steps"] = 3
        with self.assertRaises(AssertionError):
            diagnostic.geometry_windows(np.array([.2, .2, 2., 2.]), np.zeros(4), radii, broken, decomposed)

    def test_no_predictor_support_is_missing_not_zero_scale_coverage(self):
        original = [dict(start_step=1, end_step=4, heads={diagnostic.HEAD: dict(supported_steps=0)})]
        row = diagnostic.geometry_windows(np.ones(4) * 2, np.zeros(4),
            dict(predictor=1., scale_calibration=1., scale_leave_window=1.), original,
            {4: dict(start_step=1, end_step=4, eligible=False)})[0]
        self.assertIsNone(row["scale_calibration_supported_on_predictor_steps"])
        self.assertIsNone(diagnostic.summary([row])["scale_calibration"]["mean_supported_fraction"])

    def test_roles_input_only_geometry_and_unique_normal_windows(self):
        rng = np.random.default_rng(6)
        arrays = {seed: rng.normal(size=(1000, 2)).astype(np.float32) for seed in diagnostic.SEEDS}
        arrays[2105] += 10.
        roles = dict(fit_seeds=[2101, 2102], scale_seed=2105, calibration_seed=2104, test_seed=2103)
        fit = np.concatenate([arrays[seed] for seed in roles["fit_seeds"]])
        mean, scale = fit.mean(0), fit.std(0)
        reference = (fit - mean) / scale
        cal_distance = diagnostic.nearest((arrays[2104] - mean) / scale, reference)
        radius = float(np.quantile(cal_distance, .99))
        calibration = saved_windows(cal_distance, radius)
        test_distance = diagnostic.nearest((arrays[2103] - mean) / scale, reference)
        windows = saved_windows(test_distance, radius)
        paired = [dict(start_step=row["start_step"], end_step=row["end_step"],
            eligible=row["heads"][diagnostic.HEAD]["eligible"], raw_loss_mean=.1,
            effective_conditional_scale=.01, global_joint_alarm=False, conditional_joint_alarm=False)
            for row in windows]
        normal = [dict(row, case="shared_prefix") for row in paired[:5]]
        normal += [dict(row, case=case) for case in diagnostic.CASES[:3] for row in paired[5:]]
        scores = dict(roles=roles, calibration_windows=[dict(start_step=row["start_step"],
            end_step=row["end_step"], eligible=row["heads"][diagnostic.HEAD]["eligible"]) for row in calibration],
            normal_groups={"neither_alarm": dict(windows=normal)},
            cases={case: dict(windows=copy.deepcopy(paired[5:]), outcomes={}) for case in diagnostic.CASES})
        raw = dict(roles=roles, env=diagnostic.ENV, complete=True,
            predictor_work=dict(support_domains=["state_action"]), config=dict(scale_floor=1e-3, window=100, quantile=.99),
            arms=dict(conditional_scale=dict(calibration=dict(support_radii={diagnostic.HEAD: radius}, windows=calibration),
                      cases={case: dict(windows=windows) for case in diagnostic.CASES})))
        seen = []

        def normal_inputs(source, env, seed, arm):
            seen.append(seed)
            return arrays[seed], None, {}

        with tempfile.TemporaryDirectory() as tmp, patch.object(diagnostic, "normal_data", normal_inputs):
            source = Path(tmp)
            folder = diagnostic.source_folder(source, diagnostic.ENV, 2103)
            folder.mkdir(parents=True)
            np.savez(folder / "transitions.npz", **{f"{case}_inputs": arrays[2103] for case in diagnostic.CASES})
            result = diagnostic.analyze_fold(raw, scores, source)
            changed = copy.deepcopy(scores)
            for row in changed["normal_groups"]["neither_alarm"]["windows"]:
                row.update(raw_loss_mean=1e6, effective_conditional_scale=1e-12, conditional_joint_alarm=True)
            repeated = diagnostic.analyze_fold(raw, changed, source)
        self.assertEqual(seen, [2101, 2102, 2105, 2104] * 2)
        self.assertEqual(result["radii"], repeated["radii"])
        self.assertEqual(result["normal_groups"]["neither_alarm"]["summary"], repeated["normal_groups"]["neither_alarm"]["summary"])
        self.assertEqual(result["normal_groups"]["neither_alarm"]["summary"]["windows"], 20)
        normal_windows = result["normal_groups"]["neither_alarm"]["windows"]
        self.assertEqual(sum(row["case"] == "shared_prefix" for row in normal_windows), 5)
        self.assertEqual(sum(row["case"] == "actor_change" and row["phase"] == "post" for row in normal_windows), 5)
        self.assertEqual(result["cases"]["mode2"]["summary"]["windows"], 5)


if __name__ == "__main__":
    torch.set_num_threads(2)
    unittest.main()
