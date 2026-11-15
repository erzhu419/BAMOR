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

import diagnose_execution_policy as diagnostic


class ExecutionPolicyTests(unittest.TestCase):
    def test_actions_match_native_policy_and_statistics(self):
        torch.manual_seed(7)
        policy = diagnostic.training.PolicyNet(17, 6, 13, hidden=8)
        states, contexts = torch.randn(4, 17), torch.randn(4, 13)
        for execution in diagnostic.EXECUTIONS:
            expected = []
            with diagnostic.snapshots.pilot.drift.forks.isolated_rng(13), torch.no_grad():
                for state, context in zip(states, contexts):
                    action = policy.get_action(state[None], context[None], deterministic=execution == "deterministic")
                    expected.append(action[0] if isinstance(action, tuple) else action)
            probe = diagnostic.ExecutionProbe(policy, execution)
            with diagnostic.snapshots.pilot.drift.forks.isolated_rng(13), torch.no_grad():
                actual = [probe.select_action(s[None], c[None]) for s, c in zip(states, contexts)]
            torch.testing.assert_close(torch.cat(actual), torch.cat(expected), rtol=0, atol=0)
            statistics = probe.statistics()
            actions = torch.cat(actual).numpy().astype(np.float64)
            np.testing.assert_allclose(statistics["empirical_action_mean_per_dim"], actions.mean(axis=0))
            np.testing.assert_allclose(statistics["empirical_action_std_per_dim"], actions.std(axis=0))
            self.assertAlmostEqual(statistics["executed_action_rms"], np.sqrt(np.mean(actions ** 2)))
            self.assertEqual(statistics["steps"], 4)
            if execution == "deterministic":
                self.assertEqual(statistics["noise_displacement_rms"], 0)
            else:
                self.assertGreater(statistics["noise_displacement_rms"], 0)

    def test_short_run_preserves_protocol_networks_and_noise_reproducibility(self):
        training = diagnostic.training
        factory = training.make_piecewise_mo_env
        weights = diagnostic.snapshots.pilot.drift.forks.WEIGHTS

        def short_env(*args, **kwargs):
            env = factory(*args, **kwargs)
            env.env = gym.wrappers.TimeLimit(env.env, max_episode_steps=8)
            return env

        with (tempfile.TemporaryDirectory() as directory,
              patch.object(training, "make_piecewise_mo_env", side_effect=short_env),
              patch.object(training, "full_simplex_grid", return_value=weights),
              patch.object(diagnostic, "SOURCE_ROOT", Path(directory) / "source")):
            torch.manual_seed(31)
            policy = training.PolicyNet(17, 6, 13, hidden=64)
            encoder = training.LatentTransitionEncoder(17, 6, 2, latent_dim=8, hidden=64)
            before = {"policy": copy.deepcopy(policy.state_dict()), "encoder": copy.deepcopy(encoder.state_dict())}
            env = short_env("mo-halfcheetah-v5", switch_interval=25000, seed=1031)
            trainer = SimpleNamespace(reward_dim=2, select_action=policy.get_action)
            try:
                reference = training.evaluate_locked_modes(
                    env, lambda w, seeds: training.evaluate_pareto_v2(
                        env, trainer, diagnostic.METHOD, encoder, w, num_episodes=3, episode_seeds=seeds),
                    31, training.reference_point_for_env("mo-halfcheetah-v5", 2))
                env._lock_mode = True
                actor = diagnostic.ExecutionProbe(policy, "stochastic")
                rng = torch.get_rng_state().clone()
                first = diagnostic.evaluate_front(env, actor, encoder, weights, [131000, 131001, 131002])
                torch.testing.assert_close(torch.get_rng_state(), rng, rtol=0, atol=0)
                torch.manual_seed(987)
                second = diagnostic.evaluate_front(env, actor, encoder, weights, [131000, 131001, 131002])
                self.assertEqual(first, second)
            finally:
                env.close()
            for name, network in (("policy", policy), ("encoder", encoder)):
                for key, value in network.state_dict().items():
                    torch.testing.assert_close(value, before[name][key], rtol=0, atol=0)
            reference.update(actor_step=45000, encoder_step=45000)
            source = diagnostic.SOURCE_ROOT / "stored"
            source.mkdir(parents=True)
            torch.save({45000: before}, source / "seed31_snapshots.pt")
            data = {"complete": True, "protocol": diagnostic.snapshots.PROTOCOL,
                    "training_reproduction": {"matched": True}, "cells": [reference]}
            (source / "seed31.json").write_text(json.dumps(data))
            with patch.object(training, "train_method", side_effect=AssertionError("No training allowed")):
                report = diagnostic.run("stored", 45000, Path(directory) / "result")
            self.assertTrue(report["complete"])
            self.assertEqual(report["deterministic_reproduction"]["max_return_error"], 0)
            self.assertEqual(report["training_interactions"], 0)
            self.assertEqual(report["evaluation_interactions"], 2 * 4 * 3 * 3 * 8)
            saved = json.loads((Path(directory) / "result/stored/step45000.json").read_text())
            self.assertEqual(report, saved)
            for cell in report["cells"].values():
                for mode, front in enumerate(cell["pareto_per_mode"]):
                    for index, point in enumerate(front):
                        self.assertEqual(point["episode_lengths"], [8] * 3)
                        self.assertEqual(point["action_statistics"]["steps"], 24)
                        self.assertEqual(point["action_noise_seeds"],
                                         [diagnostic.action_noise_seed(mode, index, r) for r in range(3)])
                        np.testing.assert_array_equal(point["return"], np.mean(point["episode_returns"], axis=0))
            bad = copy.deepcopy(reference)
            bad["pareto_per_mode"][0][0]["return"][0] += .01
            self.assertFalse(diagnostic.deterministic_reference_check(report["cells"]["deterministic"], bad)["matched"])
            bad = copy.deepcopy(reference)
            bad["episode_seeds_per_mode"][0][0] += 1
            self.assertFalse(diagnostic.deterministic_reference_check(report["cells"]["deterministic"], bad)["matched"])

    def test_noise_streams_are_distinct_and_frozen(self):
        self.assertEqual(diagnostic.action_noise_seed(3, 20, 2), 31003202)
        seeds = [diagnostic.action_noise_seed(m, w, r) for m in range(4) for w in range(21) for r in range(3)]
        self.assertEqual(len(seeds), len(set(seeds)))
        self.assertEqual(diagnostic.snapshots.STEPS, (45000, 47500, 50000))


if __name__ == "__main__":
    unittest.main()
