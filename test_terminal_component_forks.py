import copy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import torch

import run_terminal_component_forks as experiment
from piecewise_mo_env import PiecewiseStationaryMOEnv


class TerminalForkTests(unittest.TestCase):
    def test_completed_result_reuse_and_incomplete_attempt_preservation(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / f"seed{experiment.SEED}.json"
            report = {"protocol": experiment.PROTOCOL, "complete": True, "controls_passed": True}
            path.write_text(json.dumps(report))
            with patch.object(experiment.training, "train_method", side_effect=AssertionError("No retraining")):
                self.assertEqual(experiment.run(directory), report)
                report["complete"] = False
                path.write_text(json.dumps(report))
                with self.assertRaisesRegex(RuntimeError, "Incomplete attempt preserved"):
                    experiment.run(directory)
                self.assertEqual(json.loads(path.read_text()), report)

    def test_one_factor_overrides_do_not_change_copies_or_sampler(self):
        selector = experiment.training.ProgressExtremeHierarchicalOperatorSelector(2)
        base = {**selector.trainer_config(experiment.EXTREME),
                "preference_relabeling_rate": selector.relabel_rate(experiment.EXTREME)}
        for name, key in (("actor_one", "envelope_policy_candidates"), ("relabel_one", "preference_relabeling_rate")):
            modified = {**base, **experiment.OVERRIDES[name]}
            self.assertEqual([k for k in base if base[k] != modified[k]], [key])
            self.assertEqual(modified["critic_relabel_copies"], 1)
            self.assertEqual(modified["preference_relabeling_strategy"], "edge")

    def test_control_reproduction_rejects_changed_returns_or_blocks(self):
        data = {"evaluation_weights": [[0., 1.]], "episode_seeds_per_mode": [[1]],
                "ref_point": [-50., -3000.], "ground_truth": {"mode_history": [(0, 0)]},
                "episode_rewards": [{"step": 50000, "return": [1., 2.]}],
                "adaptive_operator_final": {"block_history": [{"operator": "x"}]},
                "pareto_per_mode": [[{"archive_step": 50000, "return": [1., 2.]} for _ in range(21)] for _ in range(4)]}
        reference = json.loads(json.dumps(data))
        self.assertTrue(experiment.reproduction(data, reference)["passed"])
        for kind in ("return", "block"):
            changed = copy.deepcopy(data)
            if kind == "return":
                changed["pareto_per_mode"][0][0]["return"][0] += .01
            else:
                changed["adaptive_operator_final"]["block_history"][0]["operator"] = "y"
            self.assertFalse(experiment.reproduction(changed, reference)["passed"])

    def test_twice_forked_environment_preserves_unseeded_resets(self):
        env = experiment.training.make_piecewise_mo_env("mo-halfcheetah-v5", seed=1004)
        copies = [env]
        try:
            env.reset(seed=1004)
            copies.append(experiment.fork_env(env))
            copies.append(experiment.fork_env(copies[-1]))
            for branch in copies[1:]:
                self.assertIsNot(branch.unwrapped._np_random, env.unwrapped._np_random)
                self.assertIsNotNone(branch.unwrapped._np_random)
                self.assertEqual(branch.unwrapped._np_random_seed, env.unwrapped._np_random_seed)
            for _ in range(2):
                expected, _ = env.reset()
                for branch in copies[1:]:
                    actual, _ = branch.reset()
                    np.testing.assert_array_equal(actual, expected)
        finally:
            for branch in copies:
                branch.close()
            print("TERMINAL_FORK_RESET_TEST_INTERACTIONS", json.dumps(
                dict(training=0, evaluation=0, initialization=1)))

    def test_full_state_continuation_matches_uninterrupted_training_for_both_controls(self):
        training = experiment.training
        make_env, trainer_cls = training.make_piecewise_mo_env, training.ConsistentTargetMOSACTrainer
        init = training.ProgressExtremeHierarchicalOperatorSelector.__init__
        step_env = PiecewiseStationaryMOEnv.step
        cost = dict(training=0, evaluation=0, initialization=0)

        def short_env(*args, **kwargs):
            env = make_env(*args, **kwargs)
            # Resume at45, reset at48, then update once in the next episode.
            env.env._max_episode_steps = 6
            env.smoke_kind = "training" if kwargs["seed"] == 7 else "evaluation"
            cost["initialization"] += 1
            return env

        def counted_step(env, action):
            result = step_env(env, action)
            cost[env.smoke_kind] += 1
            return result

        def short_blocks(self, reward_dim, block_steps=1500, learning_starts=5000):
            init(self, reward_dim, block_steps=4, learning_starts=learning_starts)

        def small_trainer(*args, **kwargs):
            return trainer_cls(*args, buffer_capacity=128, **kwargs)

        try:
            with (tempfile.TemporaryDirectory() as directory,
                  patch.object(training, "make_piecewise_mo_env", side_effect=short_env),
                  patch.object(PiecewiseStationaryMOEnv, "step", new=counted_step),
                  patch.object(training, "ConsistentTargetMOSACTrainer", side_effect=small_trainer),
                  patch.object(training.ProgressExtremeHierarchicalOperatorSelector, "__init__", new=short_blocks),
                  patch("builtins.print")):
                root = Path(directory)
                methods = {"factual": experiment.previous.METHOD,
                           "recovery": experiment.previous.previous.METHODS["legacy"]}

                def run(name, method, **kwargs):
                    return training.train_method("mo-halfcheetah-v5", method, 49, 25, 7, str(root / name),
                        hidden=8, batch_size=4, learning_starts=5, eval_freq=1000, device="cpu", **kwargs)

                originals = {name: run(name, method) for name, method in methods.items()}
                capture = experiment.CaptureState(45)
                run("prefix", methods["factual"], state_probe=capture)
                state = capture.state
                self.assertEqual(state["episode_steps"], 3)
                self.assertEqual(state["trainer"].buffer.size, 45)
                self.assertIs(state["learned_tracker"].encoder, state["encoder"])
                self.assertIs(state["encoder_opt"].param_groups[0]["params"][0], next(state["encoder"].parameters()))
                initial_buffer = state["trainer"].buffer.states.copy()
                try:
                    for name, method in methods.items():
                        branch = experiment.clone_state(state)
                        if name == "recovery":
                            experiment.adaptive_state(branch)
                        router = experiment.TerminalIntervention(name, start=46, end=49)
                        try:
                            result = run("resumed_" + name, method, resume_state=branch, update_router=router)
                        finally:
                            router.close()
                        self.assertEqual(result["training_interactions_this_call"], 4)
                        self.assertEqual(result["evaluation_interactions"], 1512)
                        for key in ("episode_rewards", "adaptive_operator_final", "ground_truth"):
                            self.assertEqual(result[key], originals[name][key], key)
                        fronts = [[{k: v for k, v in p.items() if k != "archive_interaction_step"}
                                   for p in front] for front in result["pareto_per_mode"]]
                        self.assertEqual(fronts, originals[name]["pareto_per_mode"])
                        left = torch.load(root / name / f"{method}_seed7_model.pt", weights_only=False)
                        right = torch.load(root / ("resumed_" + name) / f"{method}_seed7_model.pt", weights_only=False)
                        for net in ("policy", "q1", "q2", "encoder"):
                            for key in left[net]:
                                torch.testing.assert_close(left[net][key], right[net][key], rtol=0, atol=0)
                        self.assertEqual(len(router.records), 2)
                        self.assertTrue(all(np.isfinite(r["critic"]["gradient_norm"]) for r in router.records))
                        np.testing.assert_array_equal(state["trainer"].buffer.states, initial_buffer)
                finally:
                    state["env"].close()
                    state["eval_env"].close()
                self.assertEqual(cost, dict(training=151, evaluation=6048, initialization=10))
        finally:
            print("TERMINAL_FORK_SMOKE_INTERACTIONS", json.dumps(cost))


if __name__ == "__main__":
    unittest.main()
