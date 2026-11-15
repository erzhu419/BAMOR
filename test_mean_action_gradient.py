import copy
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import gymnasium as gym
import numpy as np
import torch
from torch import nn

import diagnose_mean_action_gradient as diagnostic


class ConstantPolicy(nn.Module):
    def __init__(self, mean):
        super().__init__()
        self.mean = nn.Parameter(torch.tensor(mean))

    def forward(self, s, c):
        return self.mean[None].expand(len(s), -1), torch.zeros(len(s), len(self.mean))


class QuadraticQ(nn.Module):
    def __init__(self, target, offset=0.):
        super().__init__()
        self.target, self.offset = torch.tensor(target), offset

    def forward(self, s, a, c):
        return -(a - self.target).square() + self.offset


def networks(hidden=8):
    t = diagnostic.training
    torch.manual_seed(7)
    result = {"policy": t.PolicyNet(17, 6, 13, hidden=hidden),
              "encoder": t.LatentTransitionEncoder(17, 6, 2, latent_dim=8, hidden=hidden),
              "q1": t.SoftQNet(17, 6, 13, 2, hidden=hidden),
              "q2": t.SoftQNet(17, 6, 13, 2, hidden=hidden)}
    for network in result.values():
        network.eval().requires_grad_(False)
    return result


class MeanActionGradientTests(unittest.TestCase):
    def test_mean_output_chain_rule_perturbation_and_zero_gradient(self):
        nets = {"policy": ConstantPolicy([.1, -.2]),
                "q1": QuadraticQ([.4, -.3]), "q2": QuadraticQ([.4, -.3], -.1)}
        weight = np.array([.3, .7], dtype=np.float32)
        actions, stats = diagnostic.mean_directions(nets, np.zeros(3), np.zeros(4), weight, .001)
        a = torch.tanh(nets["policy"].mean).detach().numpy()
        expected = weight * -2 * (a - [.4, -.3]) * (1 - a*a)
        np.testing.assert_allclose(stats["gradient"], expected, rtol=1e-6)
        for sign, name in ((1, "plus"), (-1, "minus")):
            np.testing.assert_allclose(actions[name], np.tanh(
                [.1, -.2] + sign * .001 * expected / np.linalg.norm(expected)), atol=1e-7)
        finite = (stats["q_scalar"]["plus"] - stats["q_scalar"]["minus"]) / .002
        self.assertAlmostEqual(finite, np.linalg.norm(expected), places=4)
        self.assertIsNone(nets["policy"].mean.grad)
        nets = {"policy": ConstantPolicy([0., 0.]),
                "q1": QuadraticQ([0., 0.]), "q2": QuadraticQ([0., 0.])}
        actions, stats = diagnostic.mean_directions(nets, np.zeros(3), np.zeros(4), weight)
        self.assertEqual(stats["gradient_norm"], 0.)
        np.testing.assert_array_equal(actions["base"], actions["plus"])
        np.testing.assert_array_equal(actions["base"], actions["minus"])

    def test_real_forks_match_native_execution_and_leave_networks_unchanged(self):
        t, nets = diagnostic.training, networks()
        states = {k: copy.deepcopy(n.state_dict()) for k, n in nets.items()}
        env = t.make_piecewise_mo_env("mo-halfcheetah-v5", seed=1033)
        env._lock_mode = True
        env._apply_mode(3)
        weight = diagnostic.WEIGHTS[1]
        seed = diagnostic.reset_seed(33, 3, 0)
        try:
            case = diagnostic.run_case(env, nets, diagnostic.METHODS["v28"], weight, seed,
                                       warmup=3, horizons=(2, 5))
            self.assertTrue(case["identity_matched"])
            self.assertEqual(case["evaluation_interactions"], 23)
            self.assertEqual(env._total_steps, 8)
            physical_state = env.env.unwrapped.data.qpos.copy()
            # Direct continuation without any simulator copy must agree too.
            obs, _ = env.reset(seed=seed)
            trackers = (t.TransitionContextTracker(2), t.LearnedTransitionContextTracker(nets["encoder"]))
            obs, warm = diagnostic.rollout(env, nets, diagnostic.METHODS["v28"], weight,
                                          obs, trackers, 3, (3,))
            np.testing.assert_array_equal(obs, case["anchor_observation"])
            with patch.object(nets["q1"], "forward", side_effect=AssertionError("Continuation must not query Q")):
                _, direct = diagnostic.rollout(env, nets, diagnostic.METHODS["v28"], weight,
                                              obs, trackers, 5, (2, 5))
            self.assertEqual(direct, case["branches"]["base"])
            np.testing.assert_array_equal(env.env.unwrapped.data.qpos, physical_state)
            env.env = gym.wrappers.TimeLimit(env.env, max_episode_steps=8)
            expected = t.evaluate_pareto_v2(
                env, SimpleNamespace(reward_dim=2, select_action=nets["policy"].get_action),
                diagnostic.METHODS["v28"], nets["encoder"], [weight], num_episodes=1,
                episode_seeds=[seed])[0]["return"]
            np.testing.assert_allclose(np.asarray(warm["raw_returns"]["3"]) + direct["raw_returns"]["5"],
                                       expected, rtol=0, atol=1e-12)
            identity = diagnostic.run_case(env, nets, diagnostic.METHODS["v28"], weight, seed,
                                           warmup=2, horizons=(2, 4), epsilon=0)
            self.assertTrue(all(b == identity["branches"]["base"] for b in identity["branches"].values()))
        finally:
            env.close()
        for name, network in nets.items():
            for key, value in network.state_dict().items():
                torch.testing.assert_close(value, states[name][key], rtol=0, atol=0)
            self.assertTrue(all(p.grad is None for p in network.parameters()))

    def test_summary_uses_raw_outcomes_and_keeps_q_disagreement(self):
        cases = []
        for repeat in range(3):
            cases.append({"mode": 0, "weight_index": 1, "weight": [.5, .5], "reset_seed": 10 + repeat,
                "gradient_diagnostic": {"q_scalar": {"plus": 10., "minus": 8.}, "gradient_norm": 2.},
                "branches": {name: {"discounted_returns": {str(h): [value, value] for h in diagnostic.HORIZONS}}
                             for name, value in (("base", 0.), ("plus", -2.), ("minus", 1.))}})
        report = diagnostic.summarize_cases(cases)
        self.assertEqual(report["q_ascent_cells"], 1)
        self.assertEqual(report["return_positive_among_q_ascent"], 0)
        self.assertEqual(report["mean_plus_minus_return"], -3.)
        self.assertEqual(report["mean_plus_base_return"], -2.)
        with self.assertRaises(AssertionError):
            diagnostic.summarize_cases(cases[:-1])

    def test_short_runner_loads_both_models_and_accounts_every_branch(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            roots = {name: root / name for name in diagnostic.METHODS}
            nets = networks(hidden=64)
            states = {k: n.state_dict() for k, n in nets.items()}
            states["policy_archive"] = [{"step": 50000, "policy": states["policy"]}]
            for condition, method in diagnostic.METHODS.items():
                roots[condition].mkdir()
                source = roots[condition] / f"{method}_seed31"
                reference = {"method": method, "seed": 31, "total_steps": 50000,
                    "mean_value_pilot": {"complete": True, "protocol": diagnostic.pilot.PROTOCOL},
                    "replay_context_confirmation": {"complete": True, "condition": "stored"}}
                source.with_suffix(".json").write_text(json.dumps(reference))
                torch.save(states, str(source) + "_model.pt")
                with (patch.object(diagnostic, "SOURCE_ROOTS", roots),
                      patch.object(diagnostic, "WARMUP", 2), patch.object(diagnostic, "HORIZONS", (2, 4)),
                      patch.object(diagnostic, "REPEATS", 1),
                      patch.object(diagnostic.training, "train_method", side_effect=AssertionError("No training"))):
                    report = diagnostic.run(condition, 31, root / "output")
                self.assertTrue(report["complete"])
                self.assertEqual(report["evaluation_interactions"], 12 * (2 + 4 * 4))
                self.assertEqual(report["training_interactions"], 0)
                self.assertEqual(report["summary"]["total_cells"], 12)
                self.assertEqual(report, json.loads((root / "output" / condition / "seed31.json").read_text()))
                self.assertEqual(len({(c["mode"], c["weight_index"], c["repeat"]) for c in report["cases"]}), 12)


if __name__ == "__main__":
    unittest.main()
