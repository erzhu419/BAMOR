import copy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import torch
from torch.nn import functional as F

import run_terminal_rng_contrasts as experiment
from piecewise_mo_env import PiecewiseStationaryMOEnv


class NativeBCContrastTests(unittest.TestCase):
    def make_trainer(self):
        trainer = experiment.training.ConsistentTargetMOSACTrainer(
            3, 2, 2, hidden=8, buffer_capacity=16, batch_size=4, device="cpu",
            preference_relabeling=True)
        for key, value in experiment.CONFIG.items():
            setattr(trainer, key, value)
        for i in range(8):
            s, w = np.full(3, i / 10, dtype=np.float32), np.array([.3, .7], dtype=np.float32)
            trainer.buffer.push(s, np.full(2, .2), w, w, np.ones(2), s + .1, w, False)
        return trainer

    def test_native_config_and_reject_accidental_bc_removal(self):
        for name in experiment.BRANCHES:
            trainer = self.make_trainer()
            router = experiment.ContrastIntervention(name, [], start=1, end=4)
            try:
                router.configure(1, trainer)
                self.assertEqual(router.config, experiment.CONFIG)
                self.assertEqual(router.state_dict()["protocol"], experiment.PROTOCOL)
                self.assertEqual(router.state_dict()["overrides"], {})
                trainer.behavior_clone_coef = 0.
                with self.assertRaises(AssertionError):
                    router.configure(2, trainer)
            finally:
                router.close()

    def test_expanded_relabel_batch_bc_gradient_and_rng_parity(self):
        torch.manual_seed(7)
        base, trace, first_critics = self.make_trainer(), [], []
        for name in experiment.BRANCHES:
            trainer = copy.deepcopy(base)
            router = experiment.ContrastIntervention(name, trace, start=1, end=4)
            np.random.seed(17)
            torch.manual_seed(17)
            recorded, gradient_checks = {}, []
            original_sample = trainer.buffer.sample

            def sample(n):
                batch = original_sample(n)
                recorded["actions"] = batch[1]
                return batch

            def check_gradients(optimizer, args, kwargs):
                for parameter, expected in zip(trainer.policy.parameters(), recorded["grads"]):
                    torch.testing.assert_close(parameter.grad, expected, rtol=0, atol=0)
                gradient_checks.append(True)

            hook = trainer.policy_opt.register_step_pre_hook(check_gradients)
            try:
                for step in range(1, 5):
                    router.configure(step, trainer)
                    original_loss = trainer.envelope_actor_loss

                    def loss(s, w, c):
                        self.assertEqual(len(s), 12)
                        self.assertTrue(torch.all((w == torch.tensor([.3, .7])).all(-1).logical_not()))
                        result = original_loss(s, w, c)
                        expected = result[2] + .004 * F.mse_loss(
                            result[0], recorded["actions"].repeat(3, 1).detach())
                        recorded["grads"] = torch.autograd.grad(
                            expected, tuple(trainer.policy.parameters()), retain_graph=True)
                        return result

                    with (patch.object(trainer.buffer, "sample", side_effect=sample),
                          patch.object(trainer, "envelope_actor_loss", side_effect=loss)):
                        trainer.update()
                    router.after_step(step, trainer, None, None, experiment.OPERATOR)
                    if step == 1:
                        first_critics.append(copy.deepcopy(trainer.q1.state_dict()))
                self.assertEqual(len(gradient_checks), 4)
                if name == "own_matched":
                    self.assertTrue(router.state_dict()["rng_alignment"]["passed"])
            finally:
                hook.remove()
                router.close()
        for key in first_critics[0]:
            torch.testing.assert_close(first_critics[0][key], first_critics[1][key], rtol=0, atol=0)
        print("NATIVE_BC_UNIT_LEARNER_UPDATES", 8)

    def test_real_continuation_control_and_rng_after_reset(self):
        training = experiment.training
        make_env, trainer_cls = training.make_piecewise_mo_env, training.ConsistentTargetMOSACTrainer
        init = training.ProgressExtremeHierarchicalOperatorSelector.__init__
        original_step = PiecewiseStationaryMOEnv.step
        cost = dict(training=0, evaluation=0, initialization=0)

        def short_env(*args, **kwargs):
            env = make_env(*args, **kwargs)
            env.env._max_episode_steps = 6
            env.smoke_kind = "training" if kwargs["seed"] == 7 else "evaluation"
            cost["initialization"] += 1
            return env

        def counted_step(env, action):
            result = original_step(env, action)
            cost[env.smoke_kind] += 1
            return result

        def short_blocks(self, reward_dim, block_steps=1500, learning_starts=5000):
            init(self, reward_dim, block_steps=4, learning_starts=learning_starts)

        try:
            with (tempfile.TemporaryDirectory() as directory,
                  patch.object(training, "make_piecewise_mo_env", side_effect=short_env),
                  patch.object(PiecewiseStationaryMOEnv, "step", new=counted_step),
                  patch.object(training, "ConsistentTargetMOSACTrainer",
                               side_effect=lambda *a, **kw: trainer_cls(*a, buffer_capacity=128, **kw)),
                  patch.object(training.ProgressExtremeHierarchicalOperatorSelector, "__init__", new=short_blocks),
                  patch("builtins.print")):
                root = Path(directory)

                def run(name, **kwargs):
                    return training.train_method("mo-halfcheetah-v5", experiment.METHOD, 37, 25, 7,
                        str(root / name), hidden=8, batch_size=4, learning_starts=5,
                        eval_freq=1000, device="cpu", **kwargs)

                original = run("uninterrupted")
                capture = experiment.forks.CaptureState(33)
                run("prefix", state_probe=capture)
                state, trace = capture.state, []
                self.assertEqual(state["episode_steps"], 3)
                try:
                    for name in experiment.BRANCHES:
                        router = experiment.ContrastIntervention(name, trace, start=34, end=37)
                        branch = experiment.forks.clone_state(state)
                        try:
                            result = run(name, resume_state=branch, update_router=router)
                        finally:
                            router.close()
                        self.assertEqual(result["training_interactions_this_call"], 4)
                        self.assertEqual(result["episode_rewards"][-1]["step"], 36)
                        self.assertEqual(result["evaluation_interactions"], 1512)
                        self.assertEqual(result["update_routing"]["config"], experiment.CONFIG)
                        if name == "factual":
                            for key in ("episode_rewards", "adaptive_operator_final", "ground_truth"):
                                self.assertEqual(result[key], original[key])
                            fronts = [[{k: v for k, v in p.items() if k != "archive_interaction_step"}
                                       for p in front] for front in result["pareto_per_mode"]]
                            self.assertEqual(fronts, original["pareto_per_mode"])
                            left = torch.load(root / "uninterrupted" / f"{experiment.METHOD}_seed7_model.pt", weights_only=False)
                            right = torch.load(root / name / f"{experiment.METHOD}_seed7_model.pt", weights_only=False)
                            for net in ("policy", "q1", "q2", "encoder"):
                                for key in left[net]:
                                    torch.testing.assert_close(left[net][key], right[net][key], rtol=0, atol=0)
                        else:
                            self.assertTrue(result["update_routing"]["rng_alignment"]["passed"])
                    self.assertEqual(state["trainer"].buffer.size, 33)
                    self.assertEqual(cost, dict(training=78, evaluation=4536, initialization=8))
                finally:
                    state["env"].close()
                    state["eval_env"].close()
        finally:
            print("NATIVE_BC_SMOKE_INTERACTIONS", json.dumps(cost))

    def test_reference_seed_and_terminal_arm_must_match(self):
        for seed in experiment.SEEDS:
            data = dict(method=experiment.METHOD, env="mo-halfcheetah-v5", seed=seed,
                total_steps=50000, switch_interval=25000,
                progress_factorial_completion=dict(protocol=experiment.forks.previous.PROTOCOL),
                adaptive_operator_final=dict(block_history=[dict(operator=experiment.OPERATOR,
                    end_step=50000, steps=1500, complete=True)]),
                evaluation_protocol="locked_full_simplex_v1",
                evaluation_weights=experiment.training.full_simplex_grid(2).tolist(),
                ref_point=[-50., -3000.],
                episode_seeds_per_mode=[[100000 + 1000 * seed + 10 * m + i for i in range(3)] for m in range(4)])
            experiment.validate_reference(data, seed)
            with self.assertRaises(AssertionError):
                experiment.validate_reference(data, seed + 1)
            data["adaptive_operator_final"]["block_history"][-1]["operator"] = experiment.forks.EXTREME
            with self.assertRaises(AssertionError):
                experiment.validate_reference(data, seed)

    def test_completed_reuse_and_seed_isolated_incomplete_preservation(self):
        with tempfile.TemporaryDirectory() as directory:
            for seed in experiment.SEEDS:
                path = Path(directory) / f"seed{seed}.json"
                report = dict(protocol=experiment.PROTOCOL, seed=seed, complete=True,
                              controls_passed=True, rng_alignment_passed=True)
                path.write_text(json.dumps(report))
                with patch.object(experiment.training, "train_method", side_effect=AssertionError("No training")):
                    self.assertEqual(experiment.run(seed, directory), report)
                    report["complete"] = False
                    path.write_text(json.dumps(report))
                    with self.assertRaisesRegex(RuntimeError, "Incomplete attempt preserved"):
                        experiment.run(seed, directory)
                    self.assertEqual(json.loads(path.read_text()), report)


if __name__ == "__main__":
    unittest.main()
