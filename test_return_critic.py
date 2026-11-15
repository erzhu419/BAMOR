import copy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import gymnasium as gym
import numpy as np
import torch
from torch import nn

import diagnose_return_critic as diagnostic
import test_mean_action_gradient as fixtures


def toy_data():
    rng = torch.Generator().manual_seed(7)
    actions = torch.randn(64, 2, generator=rng)
    return {"obs": torch.randn(64, 3, generator=rng), "contexts": torch.randn(64, 2, generator=rng),
            "actions": actions, "rewards": actions * torch.tensor([2., 4.]),
            "next_obs": torch.randn(64, 3, generator=rng), "next_contexts": torch.randn(64, 2, generator=rng),
            "next_actions": torch.zeros(64, 2), "terminated": torch.ones(64, 1), "truncated": torch.zeros(64, 1)}


class ActionQ(nn.Module):
    def forward(self, obs, action, context):
        return 2 * action


class ReturnCriticTests(unittest.TestCase):
    def test_raw_units_gradient_and_terminal_not_time_limit_target(self):
        data = toy_data()
        q = diagnostic.RawQ(data)
        q.net = ActionQ()
        action = torch.tensor([[.2, -.1]], requires_grad=True)
        result = q(torch.zeros(1, 3), action, torch.zeros(1, 2))
        expected = 2 * (action - q.actions_mean) / q.actions_scale * q.reward_scale
        torch.testing.assert_close(result, expected)
        grad, = torch.autograd.grad(result.sum(), action)
        torch.testing.assert_close(grad[0], 2 * q.reward_scale / q.actions_scale)
        batch = {k: v[:3].clone() for k, v in data.items()}
        batch["terminated"] = torch.tensor([[1.], [0.], [0.]])
        batch["truncated"] = torch.tensor([[0.], [1.], [0.]])
        batch["next_actions"] = torch.ones(3, 2)
        target = diagnostic.bellman_target(batch, [ActionQ(), ActionQ()])
        torch.testing.assert_close(target, batch["rewards"] + diagnostic.gradient.GAMMA *
                                   torch.tensor([[0., 0.], [2., 2.], [2., 2.]]))
        self.assertFalse(target.requires_grad)
        self.assertEqual(q.obs_mean.shape, (3,))

    def test_fit_is_deterministic_train_only_and_learns_raw_terminal_returns(self):
        data = toy_data()
        original = {k: v.clone() for k, v in data.items()}
        rng = torch.get_rng_state().clone()
        initial, _ = diagnostic.fit_critics(data, 31, updates=0)
        fitted, metadata = diagnostic.fit_critics(data, 31, updates=300)
        repeated, _ = diagnostic.fit_critics(data, 31, updates=300)
        torch.testing.assert_close(torch.get_rng_state(), rng, rtol=0, atol=0)
        for key, value in data.items():
            torch.testing.assert_close(value, original[key], rtol=0, atol=0)
        for key, value in fitted.state_dict().items():
            torch.testing.assert_close(value, repeated.state_dict()[key], rtol=0, atol=0)
        loss = lambda q: ((q(data["obs"], data["actions"], data["contexts"]) - data["rewards"]) ** 2).mean()
        self.assertLess(float(loss(fitted[0])), float(loss(initial[0])) * .1)
        self.assertEqual(metadata["updates"], 300)
        self.assertTrue(all(not p.requires_grad and p.grad is None for p in fitted.parameters()))

    def test_real_runner_separates_collection_fit_evaluation_and_counts_steps(self):
        g, t = diagnostic.gradient, diagnostic.training
        nets = fixtures.networks()
        old = {name: copy.deepcopy(n.state_dict()) for name, n in nets.items()}
        factory, fitter, collector = t.make_piecewise_mo_env, diagnostic.fit_critics, diagnostic.collect_data
        created, raw = [], {}

        def env_factory(*args, **kwargs):
            env = factory(*args, **kwargs)
            env.env = gym.wrappers.TimeLimit(env.env, max_episode_steps=8)
            created.append(env)
            return env

        def collect(*args):
            with (patch.object(nets["q1"], "forward", side_effect=AssertionError("Collection cannot query Q")),
                  patch.object(nets["q2"], "forward", side_effect=AssertionError("Collection cannot query Q"))):
                data, episodes = collector(*args)
            self.assertEqual(len(data["obs"]), 96)
            self.assertEqual(int(data["truncated"].sum()), 12)
            torch.testing.assert_close(data["contexts"][:, :2], data["next_contexts"][:, :2])
            with torch.no_grad():
                torch.testing.assert_close(data["next_actions"], nets["policy"].get_action(
                    data["next_obs"], data["next_contexts"], deterministic=True))
            return data, episodes

        def fit(data, seed, updates):
            self.assertEqual(len(created), 1, "Held-out environment exists before fitting ends")
            critics, metadata = fitter(data, seed, updates)
            raw["critics"] = critics
            raw["before"] = copy.deepcopy(critics.state_dict())
            return critics, metadata

        with (tempfile.TemporaryDirectory() as directory,
              patch.object(g, "load_models", return_value=(nets, Path("fixed.pt"))),
              patch.object(t, "make_piecewise_mo_env", side_effect=env_factory),
              patch.object(diagnostic, "collect_data", side_effect=collect),
              patch.object(diagnostic, "fit_critics", side_effect=fit),
              patch.object(diagnostic, "TRAIN_STEPS", 8), patch.object(diagnostic, "TRAIN_REPEATS", 1),
              patch.object(diagnostic, "FIT_UPDATES", 5), patch.object(g, "WARMUP", 2),
              patch.object(g, "HORIZONS", (2, 4)), patch.object(g, "REPEATS", 1),
              patch.object(t, "train_method", side_effect=AssertionError("No actor training"))):
            result = diagnostic.run("v28", 31, directory)
            self.assertTrue(result["complete"])
            self.assertEqual(result["critic_data_interactions"], 96)
            self.assertEqual(result["evaluation_interactions"], 12 * (2 + 6 * 4))
            self.assertEqual(result["environment_initialization_probe_steps"], 2)
            self.assertEqual(result["policy_updates"], 0)
            self.assertTrue(all(c["identity_matched"] for c in result["cases"]))
            train_seeds = {e["reset_seed"] for e in result["collection_episodes"]}
            test_seeds = {c["reset_seed"] for c in result["cases"]}
            self.assertTrue(train_seeds.isdisjoint(test_seeds))
            self.assertEqual(result, json.loads((Path(directory) / "v28/seed31.json").read_text()))
        for name, network in nets.items():
            for key, value in network.state_dict().items():
                torch.testing.assert_close(value, old[name][key], rtol=0, atol=0)
        for key, value in raw["critics"].state_dict().items():
            torch.testing.assert_close(value, raw["before"][key], rtol=0, atol=0)

    def test_summary_compares_returns_not_prediction_offsets(self):
        values = {"base": 0., "repeat": 0., "soft_plus": -2., "soft_minus": 1.,
                  "raw_plus": 2., "raw_minus": -1.}
        cases = [{"mode": 0, "weight_index": 1, "reset_seed": i, "weight": [.5, .5],
                  "branches": {k: {"discounted_returns": {"512": [v, v]}} for k, v in values.items()},
                  "predictions": {"soft": {k: 100. - v for k, v in values.items()},
                                  "raw": {k: 1000. + v for k, v in values.items()}}} for i in range(3)]
        result = diagnostic.summarize_cases(cases)
        self.assertEqual(result["aggregate"]["soft"]["mean_plus_base"], -2.)
        self.assertEqual(result["aggregate"]["raw"]["mean_plus_base"], 2.)
        self.assertEqual(result["same_action_contrast_calibration"]["raw"]["rmse"], 0.)
        self.assertEqual(result["same_action_contrast_calibration"]["soft"]["sign_matches"], 0)
        with self.assertRaises(AssertionError):
            diagnostic.summarize_cases(cases[:-1])


if __name__ == "__main__":
    unittest.main()
