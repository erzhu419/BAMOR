import copy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import torch
from torch import nn

import diagnose_action_radius as diagnostic
import test_mean_action_gradient as fixtures
from test_return_critic import toy_data


class ActionRadiusTests(unittest.TestCase):
    def test_checkpoint_load_restores_normalization_values_and_action_gradients(self):
        p = diagnostic.pilot
        critics = nn.ModuleList([p.RawQ(toy_data()), p.RawQ(toy_data())]).eval().requires_grad_(False)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "critic.pt"
            torch.save(critics.state_dict(), path)
            rng = torch.get_rng_state().clone()
            restored = diagnostic.load_critics(path)
        torch.testing.assert_close(torch.get_rng_state(), rng, rtol=0, atol=0)
        for key, value in restored.state_dict().items():
            torch.testing.assert_close(value, critics.state_dict()[key], rtol=0, atol=0)
        action = torch.tensor([[.1, -.2]], requires_grad=True)
        for old, new in zip(critics, restored):
            a, b = [q(torch.zeros(1, 3), action, torch.zeros(1, 2)) for q in (old, new)]
            torch.testing.assert_close(a, b, rtol=0, atol=0)
            torch.testing.assert_close(torch.autograd.grad(a.sum(), action)[0],
                                       torch.autograd.grad(b.sum(), action)[0], rtol=0, atol=0)
        self.assertTrue(all(not p.requires_grad for p in restored.parameters()))

    def test_curvature_and_wrong_direction_are_different_and_no_radius_is_selected(self):
        cases = []
        for repeat in range(3):
            case = {"by_radius": {}}
            for radius in diagnostic.RADII:
                values = {"base": 0., "repeat": 0.,
                          "soft_plus": -radius, "soft_minus": radius,
                          "raw_plus": radius - 30 * radius ** 2,
                          "raw_minus": -radius - 30 * radius ** 2}
                scores = {"base": 0., "soft_plus": radius, "raw_plus": radius,
                          "soft_minus": -radius, "raw_minus": -radius}
                case["by_radius"][str(radius)] = {"reused": radius == .05, "case": {
                    "mode": 0, "weight_index": 1, "reset_seed": repeat, "weight": [.5, .5],
                    "branches": {k: {"discounted_returns": {"512": [v, v]}} for k, v in values.items()},
                    "predictions": {name: scores for name in ("soft", "raw")}}}
            cases.append(case)
        result = diagnostic.summarize_cases(cases)
        self.assertEqual(set(result["by_radius"]), {"0.05", "0.01", "0.002"})
        self.assertEqual(result["diagnosis_counts"]["raw"]["consistent_with_oversized_step"], 1)
        self.assertEqual(result["diagnosis_counts"]["raw"]["direction_mismatch_at_both_small_radii"], 0)
        self.assertEqual(result["diagnosis_counts"]["soft"]["direction_mismatch_at_both_small_radii"], 1)
        for radius in diagnostic.RADII:
            row = result["by_radius"][str(radius)]["cells"][0]
            self.assertAlmostEqual(row["raw"]["centered_return_slope"], 1.)
            self.assertAlmostEqual(row["raw"]["centered_return_curvature"], -60.)
            self.assertAlmostEqual(row["soft"]["centered_return_slope"], -1.)
        self.assertTrue(result["by_radius"]["0.05"]["cells"][0]["raw"]["both_sides_below_base"])
        # One favorable radius, or a zero contrast, must not qualify as recovery.
        for case in cases:
            case["by_radius"]["0.002"]["case"]["branches"]["raw_plus"]["discounted_returns"]["512"] = [0., 0.]
        self.assertEqual(diagnostic.summarize_cases(cases)["diagnosis_counts"]["raw"]["consistent_with_oversized_step"], 0)

    def test_real_replay_runner_leaves_all_networks_fixed_and_charges_only_new_radii(self):
        p, g, t = diagnostic.pilot, diagnostic.g, diagnostic.training
        nets = fixtures.networks()
        data = {k: torch.randn(8, n) for k, n in (("obs", 17), ("actions", 6),
                                                 ("contexts", 13), ("rewards", 2))}
        critics = nn.ModuleList([p.RawQ(data), p.RawQ(data)]).eval().requires_grad_(False)
        old = {k: copy.deepcopy(n.state_dict()) for k, n in nets.items()}
        old_raw = copy.deepcopy(critics.state_dict())
        with (tempfile.TemporaryDirectory() as directory, patch.object(g, "WARMUP", 2),
              patch.object(g, "HORIZONS", (2, 4)), patch.object(g, "REPEATS", 1),
              patch.object(p, "fit_critics", side_effect=AssertionError("No fitting")),
              patch.object(t, "train_method", side_effect=AssertionError("No training"))):
            reference = {"cases": []}
            env = t.make_piecewise_mo_env("mo-halfcheetah-v5", switch_interval=25000, seed=2031)
            env._lock_mode = True
            try:
                for mode in range(4):
                    env._apply_mode(mode)
                    for wi, weight in enumerate(g.WEIGHTS):
                        case = p.evaluate_case(env, nets, critics, g.METHODS["v27"], weight,
                                               p.reset_seed("test", 31, mode, 0))
                        case.update(mode=mode, weight_index=wi, repeat=0)
                        reference["cases"].append(case)
            finally:
                env.close()
            unchanged = copy.deepcopy(reference)
            with patch.object(diagnostic, "load_inputs", return_value=(nets, critics, reference, {})):
                result = diagnostic.run("v27", 31, directory)
            self.assertTrue(result["complete"])
            self.assertEqual(result["evaluation_interactions"], 12 * 2 * (2 + 6 * 4))
            self.assertEqual(result["environment_initialization_probe_steps"], 1)
            self.assertEqual(result["training_interactions"], 0)
            self.assertEqual(result["critic_updates"] + result["policy_updates"], 0)
            self.assertTrue(all(c["historical_replay_matched"] for c in result["cases"]))
            self.assertEqual(reference, unchanged)
            self.assertEqual(result, json.loads((Path(directory) / "v27/seed31.json").read_text()))
            current = result["cases"][0]["by_radius"]["0.01"]["case"]
            original = reference["cases"][0]
            for key in ("return", "gradient"):
                changed = copy.deepcopy(current)
                if key == "return":
                    changed["branches"]["base"]["discounted_returns"]["4"][0] += 1.
                else:
                    changed["diagnostics"]["raw"]["gradient"][0] += 1.
                with self.assertRaises(AssertionError):
                    diagnostic.check_replay(changed, original)
        for name, network in nets.items():
            for key, value in network.state_dict().items():
                torch.testing.assert_close(value, old[name][key], rtol=0, atol=0)
        for key, value in critics.state_dict().items():
            torch.testing.assert_close(value, old_raw[key], rtol=0, atol=0)
        self.assertTrue(all(p.grad is None for n in [*nets.values(), critics] for p in n.parameters()))


if __name__ == "__main__":
    unittest.main()
