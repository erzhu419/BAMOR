import copy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import gymnasium as gym
import numpy as np
import torch

import run_replay_context_refresh as refresh
from test_context_drift import fixture


class ReplayRefreshTests(unittest.TestCase):
    def assert_nested_equal(self, a, b):
        if isinstance(a, dict):
            self.assertEqual(a.keys(), b.keys())
            for k in a:
                self.assert_nested_equal(a[k], b[k])
        elif isinstance(a, (list, tuple)):
            self.assertEqual(len(a), len(b))
            for x, y in zip(a, b):
                self.assert_nested_equal(x, y)
        elif torch.is_tensor(a):
            torch.testing.assert_close(a, b, rtol=0, atol=0)
        elif isinstance(a, np.ndarray):
            np.testing.assert_array_equal(a, b)
        else:
            self.assertEqual(a, b)

    def test_refresh_changes_only_both_latent_suffixes_and_preserves_rng(self):
        trainer, encoder = fixture()
        trainer.update()
        with torch.no_grad():
            encoder.encoder[-1].bias.add_(.5)
        buffer = trainer.buffer
        arrays = {k: v.copy() for k, v in vars(buffer).items() if isinstance(v, np.ndarray)}
        nets = (trainer.policy, trainer.q1, trainer.q2, trainer.q1_target, trainer.q2_target,
                trainer.policy_opt, trainer.q_opt, encoder)
        states = [copy.deepcopy(n.state_dict()) for n in nets]
        expected = refresh.drift.reencode_history(buffer, encoder)
        rng = (np.random.get_state(), torch.get_rng_state())

        class ForbiddenEnv:
            def __getattr__(self, name):
                raise AssertionError(f"Refresh accessed environment: {name}")

        with tempfile.TemporaryDirectory() as directory:
            hook = refresh.ReplayContextRefresh(Path(directory) / "trace.json", steps=(32,))
            self.assertFalse(hook(31, trainer, encoder, ForbiddenEnv(), "test"))
            self.assertEqual(hook.cases, [])
            self.assertFalse(hook(32, trainer, encoder, ForbiddenEnv(), "test"))
            for key, before in arrays.items():
                after = getattr(buffer, key)
                if key in ("contexts", "next_contexts"):
                    np.testing.assert_array_equal(before[:, :-8], after[:, :-8])
                    np.testing.assert_array_equal(before[32:], after[32:])
                    np.testing.assert_array_equal(after[:32, -8:],
                                                  expected[:-1] if key == "contexts" else expected[1:])
                    self.assertGreater(np.linalg.norm(before[:32, -8:] - after[:32, -8:]), .01)
                else:
                    np.testing.assert_array_equal(before, after)
            for before, net in zip(states, nets):
                self.assert_nested_equal(before, net.state_dict())
            self.assert_nested_equal(rng, (np.random.get_state(), torch.get_rng_state()))
            self.assertTrue(encoder.training)
            self.assertEqual((buffer.ptr, buffer.size), (32, 32))
            state = json.loads(hook.output.read_text())
            self.assertEqual((state["extra_environment_steps"], state["extra_updates"]), (0, 0))
            self.assertFalse(state["complete"])

    def test_real_training_prefix_update_timing_online_tracker_and_evaluation(self):
        training = refresh.training
        env_factory = training.make_piecewise_mo_env
        selector_factory = training.FixedExtremeHierarchicalOperatorSelector
        tracker_factory = training.LearnedTransitionContextTracker
        runs, prefixes, snapshots, final_policies = [], [], [], []

        def short_env(*args, **kwargs):
            env = env_factory(*args, **kwargs)
            env.env = gym.wrappers.TimeLimit(env.env, max_episode_steps=8)
            return env

        with tempfile.TemporaryDirectory() as directory:
            for enabled in (False, True):
                trackers, captured = [], {}
                hook = refresh.ReplayContextRefresh(Path(directory) / f"{enabled}_trace.json",
                                                    steps=(16, 32, 48))

                def make_tracker(*args, **kwargs):
                    tracker = tracker_factory(*args, **kwargs)
                    trackers.append(tracker)
                    return tracker

                def callback(step, trainer, encoder, env, operator):
                    if step == 16:
                        prefixes.append(copy.deepcopy(trainer.policy.state_dict()))
                    tracker_state = (trackers[0].z.copy(), trackers[0].count)
                    policy_state = copy.deepcopy(trainer.policy.state_dict())
                    if enabled:
                        hook(step, trainer, encoder, env, operator)
                    self.assert_nested_equal(tracker_state, (trackers[0].z, trackers[0].count))
                    self.assert_nested_equal(policy_state, trainer.policy.state_dict())
                    if step in (16, 32, 48, 64):
                        captured[step] = policy_state
                    if step == 17:
                        snapshots.append(policy_state)
                    if step == 64:
                        final_policies.append(policy_state)
                    return False

                output = Path(directory) / str(enabled)
                with (patch.object(training, "make_piecewise_mo_env", side_effect=short_env),
                      patch.object(training, "LearnedTransitionContextTracker", side_effect=make_tracker),
                      patch.object(training, "FixedExtremeHierarchicalOperatorSelector",
                                   side_effect=lambda d: selector_factory(d, block_steps=4)),
                      patch.object(training, "full_simplex_grid", return_value=refresh.drift.forks.WEIGHTS)):
                    result = training.train_method(
                        "mo-halfcheetah-v5", refresh.METHOD, 64, 20, 26, str(output),
                        hidden=8, batch_size=4, learning_starts=8, eval_freq=100000,
                        device="cpu", training_probe=callback, archive_steps=(16, 32, 48, 64))
                runs.append(result)
                model = torch.load(output / f"{refresh.METHOD}_seed26_model.pt", weights_only=False)
                for snap in model["policy_archive"]:
                    self.assert_nested_equal(snap["policy"], captured[snap["step"]])
                if enabled:
                    self.assertEqual([c["step"] for c in hook.cases], [16, 32, 48])
                    self.assertEqual(hook.state_dict()["transitions_reencoded"], 96)
            self.assert_nested_equal(*prefixes)
            self.assertTrue(any(not torch.equal(snapshots[0][k], snapshots[1][k]) for k in snapshots[0]))
            self.assertTrue(any(not torch.equal(final_policies[0][k], final_policies[1][k]) for k in final_policies[0]))
            with patch.object(refresh, "LEARNING_STARTS", 16):
                reference = json.loads(json.dumps(runs[0]))
                checks = refresh.paired_protocol_check(runs[1], reference)
                self.assertTrue(all(checks.values()))
                bad_schedule = copy.deepcopy(runs[1])
                bad_schedule["ground_truth"]["mode_history"] = [(0, 0), (17, 3)]
                with self.assertRaisesRegex(AssertionError, "mode_schedule.*False"):
                    refresh.paired_protocol_check(bad_schedule, reference)
                bad = copy.deepcopy(runs[1])
                bad["episode_seeds_per_mode"][0][0] += 1
                with self.assertRaises(AssertionError):
                    refresh.paired_protocol_check(bad, runs[0])

    def test_production_schedule_has_no_useless_terminal_refresh(self):
        self.assertEqual(refresh.REFRESH_STEPS, tuple(range(5000, 50000, 1000)))
        self.assertEqual(len(refresh.REFRESH_STEPS), 45)
        self.assertEqual(sum(refresh.REFRESH_STEPS), 1215000)


if __name__ == "__main__":
    unittest.main()
