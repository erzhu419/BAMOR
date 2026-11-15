import copy
import inspect
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import gymnasium as gym
import numpy as np
import torch
from torch import nn

import run_mean_value_pilot as pilot
import test_aligned_target as fixtures


training = pilot.training


class LegacyObjective(training.PreferenceAlignedTargetMOSACTrainer):
    def actor_objective(self, s, w, c, q_scalar, log_prob):
        return (self.alpha * log_prob - q_scalar).mean()


class MeanPolicy(nn.Module):
    def __init__(self):
        super().__init__()
        self.mean = nn.Parameter(torch.tensor([.25, -.1]))

    def get_action(self, states, context, deterministic=False):
        assert deterministic
        return torch.tanh(self.mean + context[:, :2])


class LinearContextQ(nn.Module):
    def forward(self, states, action, context):
        return action * torch.tensor([2., -3.]) + context[:, :2]


class MeanValueTests(unittest.TestCase):
    def make_trainer(self, cls, dim=2):
        return fixtures.AlignedTargetTests().make_trainer(cls, dim)

    def assert_network_equal(self, a, b):
        for name in a.state_dict():
            torch.testing.assert_close(a.state_dict()[name], b.state_dict()[name], rtol=0, atol=0)

    def test_value_mix_preserves_entropy_and_differentiates_own_mean(self):
        trainer = self.make_trainer(training.MeanValueMOSACTrainer)
        trainer.policy = MeanPolicy()
        trainer.q1, trainer.q2 = LinearContextQ(), LinearContextQ()
        s = torch.zeros(2, 3)
        w = torch.tensor([[.2, .8], [.9, .1]])
        c = torch.cat([w, torch.ones(2, 2)], -1)
        selected = torch.tensor([[3.], [5.]], requires_grad=True)
        log_prob = torch.tensor([[-2.], [-4.]], requires_grad=True)
        with patch.object(trainer.q1, "forward", wraps=trainer.q1.forward) as q:
            loss = trainer.actor_objective(s, w, c, selected, log_prob)
        torch.testing.assert_close(q.call_args.args[2], c, rtol=0, atol=0)
        value = (w * (torch.tanh(trainer.policy.mean + w) * torch.tensor([2., -3.]) + w)).sum(-1, keepdim=True)
        expected = (trainer.alpha * log_prob - .5 * selected - .5 * value).mean()
        torch.testing.assert_close(loss, expected)
        variables = (selected, log_prob, trainer.policy.mean)
        actual_grad = torch.autograd.grad(loss, variables)
        expected_grad = torch.autograd.grad(expected, variables)
        for actual, desired in zip(actual_grad, expected_grad):
            torch.testing.assert_close(actual, desired)
        torch.testing.assert_close(actual_grad[0], torch.full_like(selected, -.25))
        torch.testing.assert_close(actual_grad[1], torch.full_like(log_prob, trainer.alpha / 2))
        self.assertGreater(actual_grad[2].abs().sum().item(), 0)

    def test_all_operator_updates_preserve_legacy_rng_and_critic(self):
        any_actor_change = False
        for dim in (2, 3):
            selector = training.FixedExtremeHierarchicalOperatorSelector(dim)
            for arm in selector.operators:
                with self.subTest(dim=dim, arm=arm):
                    config = selector.trainer_config(arm)
                    runs = []
                    for cls in (LegacyObjective, training.PreferenceAlignedTargetMOSACTrainer,
                                training.MeanValueMOSACTrainer):
                        trainer = self.make_trainer(cls, dim)
                        for key, value in config.items():
                            setattr(trainer, key, value)
                        np.random.seed(17)
                        torch.manual_seed(17)
                        value = trainer.update()
                        runs.append((trainer, value, torch.get_rng_state(), np.random.get_state()))
                    legacy, baseline, mixed = runs
                    for other in (baseline, mixed):
                        self.assertEqual(legacy[1][0], other[1][0])
                        np.testing.assert_array_equal(legacy[1][1], other[1][1])
                        torch.testing.assert_close(legacy[2], other[2], rtol=0, atol=0)
                        np.testing.assert_array_equal(legacy[3][1], other[3][1])
                        self.assertEqual(legacy[3][2:], other[3][2:])
                        for name in ("q1", "q2", "q1_target", "q2_target"):
                            self.assert_network_equal(getattr(legacy[0], name), getattr(other[0], name))
                    self.assert_network_equal(legacy[0].policy, baseline[0].policy)
                    any_actor_change |= any(not torch.equal(a, b) for a, b in zip(
                        baseline[0].policy.parameters(), mixed[0].policy.parameters()))
                    stats = mixed[0].actor_update_diagnostics()
                    self.assertEqual(stats["updates"], 1)
                    self.assertEqual(stats["single_candidate_updates"], int(config["envelope_policy_candidates"] == 1))
                    self.assertEqual(stats["envelope_updates"], int(config["envelope_policy_candidates"] > 1))
                    self.assertTrue(all(torch.isfinite(p).all() for p in mixed[0].policy.parameters()))
        self.assertTrue(any_actor_change)

    def test_feature_parity_and_real_training_pairing(self):
        self.assertIn(pilot.METHOD, training.METHODS)
        for name, function in vars(training).items():
            if name.startswith("uses_") and inspect.isfunction(function):
                self.assertEqual(function(pilot.METHOD), function(pilot.BASE), name)
        env_factory = training.make_piecewise_mo_env
        selector_factory = training.FixedExtremeHierarchicalOperatorSelector
        weights = np.array([[0., 1.], [.5, .5], [1., 0.]], dtype=np.float32)

        def short_env(*args, **kwargs):
            env = env_factory(*args, **kwargs)
            env.env = gym.wrappers.TimeLimit(env.env, max_episode_steps=8)
            return env

        with (tempfile.TemporaryDirectory() as directory,
              patch.object(training, "make_piecewise_mo_env", side_effect=short_env),
              patch.object(training, "FixedExtremeHierarchicalOperatorSelector",
                           side_effect=lambda d: selector_factory(d, block_steps=4)),
              patch.object(training, "full_simplex_grid", return_value=weights),
              patch.object(pilot, "LEARNING_STARTS", 8)):
            runs = []
            for method in (pilot.BASE, pilot.METHOD):
                runs.append(training.train_method(
                    "mo-halfcheetah-v5", method, 64, 20, 29, directory,
                    hidden=8, batch_size=4, learning_starts=8, eval_freq=100000,
                    device="cpu", archive_steps=(16, 32, 48, 64)))
            baseline, mixed = runs
            self.assertTrue(all(pilot.paired_protocol_check(mixed, json.loads(json.dumps(baseline))).values()))
            self.assertEqual(mixed["actor_update"]["updates"], 57)
            self.assertGreater(mixed["actor_update"]["single_candidate_updates"], 0)
            self.assertGreater(mixed["actor_update"]["envelope_updates"], 0)
            self.assertEqual(mixed["training_interactions"], 64)
            self.assertEqual(mixed["evaluation_interactions"], 4 * 4 * 3 * 3 * 8)
            self.assertEqual(mixed["environment_initialization_probe_steps"], 2)
            for field in ("method", "warmup", "schedule", "refresh"):
                bad = copy.deepcopy(mixed)
                if field == "method":
                    bad["method"] = pilot.BASE
                elif field == "warmup":
                    bad["episode_rewards"][0]["return"][0] += 1
                elif field == "schedule":
                    bad["adaptive_operator_final"]["block_history"][0]["operator"] = "wrong"
                else:
                    bad["replay_context_refresh"] = {"complete": True}
                with self.subTest(field=field), self.assertRaises(AssertionError):
                    pilot.paired_protocol_check(bad, baseline)

            # Exercise production postprocessing with the real short-run output.
            source = Path(directory) / "reference"
            source.mkdir()
            baseline["replay_context_confirmation"] = {"complete": True, "condition": "stored"}
            (source / f"{pilot.BASE}_seed29.json").write_text(json.dumps(baseline))
            full = copy.deepcopy(mixed)
            full["training_interactions"] = 50000
            full["evaluation_interactions"] = 19 * 4 * 21 * 3 * 1000
            full["actor_update"]["updates"] = 45001
            full["policy_archive_snapshots"] = [{"step": s} for s in pilot.controls.ARCHIVE_STEPS]
            with (patch.object(pilot, "REFERENCE_ROOT", source),
                  patch.object(pilot, "LEARNING_STARTS", 5000),
                  patch.object(pilot, "paired_protocol_check", return_value={"tested_above": True}),
                  patch.object(training, "train_method", return_value=full) as train):
                saved = pilot.run(29, directory)
            self.assertEqual(train.call_args.args[:5], ("mo-halfcheetah-v5", pilot.METHOD, 50000, 25000, 29))
            self.assertEqual(train.call_args.kwargs,
                             dict(hidden=64, learning_starts=5000, eval_freq=100000, device="cpu"))
            self.assertTrue(saved["mean_value_pilot"]["complete"])
            self.assertEqual(json.loads((Path(directory) / f"{pilot.METHOD}_seed29.json").read_text()),
                             json.loads(json.dumps(saved)))


if __name__ == "__main__":
    unittest.main()
