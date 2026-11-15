import copy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import gymnasium as gym
import numpy as np
import torch

import diagnose_context_drift as drift


training = drift.training


def fixture():
    torch.manual_seed(9)
    encoder = training.LatentTransitionEncoder(3, 2, 2, latent_dim=8, hidden=8)
    buffer = training.ReplayBuffer(64, 3, 2, 2, 13, 2, torch.device("cpu"))
    tracker = training.LearnedTransitionContextTracker(encoder)
    for i in range(32):
        state = np.full(3, (i % 16) / 16, dtype=np.float32)
        action = np.array([.2, -.3], dtype=np.float32)
        reward = np.array([.5, -.1], dtype=np.float32)
        nxt = state + .1
        prefix = np.array([.4, .6, .7, .8, .9], dtype=np.float32)
        context = np.concatenate([prefix, tracker.features()])
        tracker.update(state, action, reward, nxt, "cpu")
        next_context = np.concatenate([prefix, tracker.features()])
        buffer.push(state, action, prefix[:2], context, reward, nxt, next_context,
                    i == 15, t=i + 1, regime=123456)
    trainer = training.PreferenceAlignedTargetMOSACTrainer(
        3, 2, 2, hidden=8, context_dim=13, buffer_capacity=64, batch_size=4, device="cpu")
    trainer.buffer = buffer
    return trainer, encoder


class ContextDriftTests(unittest.TestCase):
    def test_frozen_encoder_recovers_ema_before_and_after_across_episode(self):
        trainer, encoder = fixture()
        buffer = trainer.buffer
        history = drift.reencode_history(buffer, encoder, chunk_size=8)
        np.testing.assert_allclose(history[:-1], buffer.contexts[:32, -8:], atol=2e-7, rtol=2e-6)
        np.testing.assert_allclose(history[1:], buffer.next_contexts[:32, -8:], atol=2e-7, rtol=2e-6)
        np.testing.assert_array_equal(history[0], np.zeros(8))
        self.assertGreater(np.linalg.norm(history[16]), .01)
        self.assertTrue(encoder.training)
        encoder.eval()
        drift.reencode_history(buffer, encoder)
        self.assertFalse(encoder.training)

    def test_future_transition_cannot_change_its_pre_transition_context(self):
        trainer, encoder = fixture()
        buffer = trainer.buffer
        before = drift.reencode_history(buffer, encoder, chunk_size=8)
        buffer.rewards[16:] = 100
        buffer.states[16:] += 3
        after = drift.reencode_history(buffer, encoder, chunk_size=8)
        np.testing.assert_array_equal(before[:17], after[:17])
        self.assertGreater(np.linalg.norm(before[17:] - after[17:]), .01)
        full = after.copy()
        buffer.size = buffer.ptr = 16
        np.testing.assert_array_equal(full[:17], drift.reencode_history(buffer, encoder, chunk_size=8))

    def test_only_latent_changes_and_drift_is_not_confused_with_network_updates(self):
        trainer, encoder = fixture()
        buffer, idx = trainer.buffer, np.array([4, 8, 16, 24])
        history = drift.reencode_history(buffer, encoder)
        panel = drift.capture_panel(trainer, history, idx, 32)
        before_buffer = buffer.contexts.copy()
        measured = drift.measure_panel(trainer, history, panel)
        self.assertLess(measured["latent_current_minus_stored"]["rms"], 2e-7)
        for p in measured["preferences"]:
            self.assertLess(p["actor_context_effect"]["rms"], 1e-7)
            self.assertEqual(p["actor_weight_drift_fixed_anchor_context"]["rms"], 0.)
            self.assertEqual(p["critic_weight_drift_fixed_anchor_inputs"]["rms"], 0.)
        with torch.no_grad():
            encoder.encoder[-1].bias.add_(.5)
        current = drift.reencode_history(buffer, encoder)
        changed = drift.replace_latent(buffer.contexts[idx], current[idx])
        np.testing.assert_array_equal(changed[:, :-8], buffer.contexts[idx, :-8])
        measured = drift.measure_panel(trainer, current, panel)
        self.assertGreater(measured["latent_current_minus_anchor"]["rms"], .01)
        for p in measured["preferences"]:
            self.assertEqual(p["actor_weight_drift_fixed_anchor_context"]["rms"], 0.)
            self.assertEqual(p["critic_weight_drift_fixed_anchor_inputs"]["rms"], 0.)
        with torch.no_grad():
            trainer.policy.mean_head.bias.add_(.5)
            trainer.q1.net[-1].bias.add_(1.)
            trainer.q2.net[-1].bias.add_(1.)
        measured = drift.measure_panel(trainer, current, panel)
        self.assertTrue(all(p["actor_weight_drift_fixed_anchor_context"]["rms"] > .01
                            for p in measured["preferences"]))
        self.assertTrue(all(p["critic_weight_drift_fixed_anchor_inputs"]["rms"] > .9
                            for p in measured["preferences"]))
        np.testing.assert_array_equal(before_buffer, buffer.contexts)

    def test_posthoc_labels_respect_transition_clock_and_do_not_choose_samples(self):
        report = {"cohorts": [{"sample_timesteps": [1, 16, 17, 32]}]}
        drift.annotate_modes(report, {"num_modes": 2, "mode_history": [[0, 0], [16, 1]]})
        self.assertEqual(report["cohorts"][0]["posthoc_sample_counts_per_mode"], [2, 2])
        self.assertEqual(report["cohorts"][0]["sample_timesteps"], [1, 16, 17, 32])

    def test_real_probe_preserves_trajectory_optimizer_rng_and_uses_no_env_access(self):
        class ForbiddenEnv:
            def __getattr__(self, name):
                raise AssertionError(f"Probe accessed environment: {name}")

        snapshots, logs = [], []
        factory = training.FixedExtremeHierarchicalOperatorSelector
        env_factory = training.make_piecewise_mo_env

        def short_env(*args, **kwargs):
            env = env_factory(*args, **kwargs)
            env.env = gym.wrappers.TimeLimit(env.env, max_episode_steps=8)
            return env

        with tempfile.TemporaryDirectory() as directory:
            probe = drift.ContextDriftProbe(26, Path(directory) / "drift.json",
                                            steps=(16, 32, 48, 64), cohort_width=16, samples=4)
            for enabled in (False, True):
                def callback(step, trainer, encoder, env, operator):
                    if enabled:
                        probe(step, trainer, encoder, ForbiddenEnv(), operator)
                    if step != 64:
                        return False
                    snapshots.append({name: copy.deepcopy(getattr(trainer, name).state_dict())
                                      for name in ("policy", "q1", "q2", "q1_target", "q2_target",
                                                   "policy_opt", "q_opt")})
                    snapshots[-1].update(encoder=copy.deepcopy(encoder.state_dict()),
                                         torch_rng=torch.get_rng_state(), numpy_next=np.random.random(4),
                                         contexts=trainer.buffer.contexts[:64].copy())
                    return True

                with (patch.object(training, "FixedExtremeHierarchicalOperatorSelector",
                                   side_effect=lambda d: factory(d, block_steps=4)),
                      patch.object(training, "make_piecewise_mo_env", side_effect=short_env)):
                    logs.append(training.train_method(
                        "mo-halfcheetah-v5", drift.forks.METHOD, 64, 20, 26, directory,
                        hidden=8, batch_size=4, learning_starts=8, device="cpu",
                        eval_freq=100000, training_probe=callback))
            self.assertTrue(drift.forks.reproduction_check(logs[0], logs[1])["matched"])
            self.assertTrue(probe.report["measurements_complete"])
            self.assertFalse(probe.report["complete"])
            self.assertEqual([len(c["panels"]) for c in probe.report["cases"]], [1, 2, 3, 4])
            self.assertEqual(probe.report["extra_environment_steps"], 0)
            for name in ("policy", "q1", "q2", "q1_target", "q2_target", "encoder"):
                for key, value in snapshots[0][name].items():
                    torch.testing.assert_close(value, snapshots[1][name][key], rtol=0, atol=0)
            for name in ("policy_opt", "q_opt"):
                self.assertEqual(snapshots[0][name]["param_groups"], snapshots[1][name]["param_groups"])
                for param, values in snapshots[0][name]["state"].items():
                    for key, value in values.items():
                        torch.testing.assert_close(value, snapshots[1][name]["state"][param][key],
                                                   rtol=0, atol=0)
            torch.testing.assert_close(snapshots[0]["torch_rng"], snapshots[1]["torch_rng"], rtol=0, atol=0)
            np.testing.assert_array_equal(snapshots[0]["numpy_next"], snapshots[1]["numpy_next"])
            np.testing.assert_array_equal(snapshots[0]["contexts"], snapshots[1]["contexts"])
            json.dumps(probe.report, allow_nan=False)


if __name__ == "__main__":
    unittest.main()
