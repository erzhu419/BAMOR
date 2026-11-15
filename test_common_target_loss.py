import copy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import torch

import run_common_target_loss as experiment


training = experiment.training


class CommonTargetLossTests(unittest.TestCase):
    def trainer(self, dim, projected=True):
        cls = training.ProjectedCommonTargetMOSACTrainer if projected else training.CommonTargetMOSACTrainer
        return cls(3, 2, dim, hidden=8, buffer_capacity=16, batch_size=4, device="cpu")

    def test_loss_metric_trace_and_invisible_errors(self):
        for dim in (2, 3):
            trainer = self.trainer(dim)
            for weights in (torch.eye(dim)[0:1], torch.full((1, dim), 1 / dim)):
                target = torch.zeros_like(weights)
                basis_losses = [trainer.critic_objective(e[None], e[None], weights, torch.zeros(1, 1), target)
                                for e in torch.eye(dim)]
                self.assertAlmostEqual(float(sum(basis_losses)), 1., places=6)
                error = torch.zeros_like(weights, requires_grad=True)
                with torch.no_grad():
                    error[0, 0], error[0, 1] = 1., -1.
                loss = trainer.critic_objective(error, error, weights, torch.zeros(1, 1), target)
                self.assertGreater(float(loss.detach()), 0.)
                loss.backward()
                self.assertTrue(torch.isfinite(error.grad).all())
            weights = torch.eye(dim)[0:1]
            parallel, orthogonal = torch.eye(dim)[0:1], torch.eye(dim)[1:2]
            self.assertGreater(float(trainer.critic_objective(parallel, parallel, weights, torch.zeros(1, 1), target)),
                               float(trainer.critic_objective(orthogonal, orthogonal, weights, torch.zeros(1, 1), target)))

    def test_twin_batch_formula_and_gradients(self):
        torch.manual_seed(913)
        trainer = self.trainer(3)
        q1, q2 = torch.randn(4, 3, requires_grad=True), torch.randn(4, 3, requires_grad=True)
        target = torch.randn(4, 3)
        weights = torch.softmax(torch.randn(4, 3), dim=-1)
        expected = torch.stack([((q - target).square().mean(-1) + ((q - target) * weights).sum(-1).square())
                                / (1 + weights.square().sum(-1)) for q in (q1, q2)]).mean()
        actual = trainer.critic_objective(q1, q2, weights, (target * weights).sum(-1, keepdim=True), target)
        self.assertTrue(torch.equal(actual, expected))
        self.assertTrue(all(torch.equal(a, b) for a, b in zip(
            torch.autograd.grad(actual, (q1, q2), retain_graph=True), torch.autograd.grad(expected, (q1, q2)))))

    def test_own_target_and_same_operator_update_work(self):
        for dim in (2, 3):
            trainer, base = self.trainer(dim), self.trainer(dim, projected=False)
            weights = torch.full((4, dim), 1 / dim)
            reward, next_vector = torch.randn(4, dim), torch.randn(4, dim)
            next_scalar, done = torch.randn(4, 1), torch.tensor([[0.], [1.], [0.], [1.]])
            self.assertTrue(all(torch.equal(a, b) for a, b in zip(
                trainer.critic_targets(reward, weights, done, next_scalar, next_vector),
                base.critic_targets(reward, weights, done, next_scalar, next_vector))))
            for i in range(8):
                state = np.full(3, i / 10, dtype=np.float32)
                w = np.full(dim, 1 / dim, dtype=np.float32)
                trainer.buffer.push(state, np.zeros(2), w, w, np.ones(dim), state + .1, w, i % 3 == 0)
            states = []
            selector = training.FixedCommonTargetSamplingSelector(dim)
            for op in selector.operators:
                branch = copy.deepcopy(trainer)
                for key, value in selector.trainer_config(op).items():
                    setattr(branch, key, value)
                np.random.seed(833)
                torch.manual_seed(833)
                branch.update()
                self.assertEqual(branch.update_work, dict(critic_optimizer_steps=1, actor_optimizer_steps=1,
                    critic_rows=4, target_candidate_rows=4, actor_candidate_rows=4))
                states.append({name: copy.deepcopy(getattr(branch, name).state_dict())
                               for name in ("policy", "q1", "q2", "q1_target", "q2_target")})
            for state in states[1:]:
                self.assertTrue(all(torch.equal(value, state[name][key])
                    for name, values in states[0].items() for key, value in values.items()))

    def test_short_routes_same_sampling_readout_and_no_retraining(self):
        original_env = training.make_piecewise_mo_env
        original_selector_init = training.CommonTargetSamplingSelector.__init__
        kwargs = dict(hidden=8, lr=3e-4, gamma=.99, batch_size=4, learning_starts=5,
                      eval_freq=100000, log_freq=10000, device="cpu")

        def short_env(*args, **kw):
            kw.update(switch_interval=12, mode_order="sequential")
            env = original_env(*args, **kw)
            env.env._max_episode_steps = 4
            return env

        def short_blocks(selector, reward_dim, **kw):
            original_selector_init(selector, reward_dim, block_steps=6, **kw)

        with tempfile.TemporaryDirectory() as folder, patch.object(experiment, "SEEDS", (912347,)), \
             patch.object(experiment, "DIAGNOSTIC_START", 40), \
             patch.object(experiment.pilot, "COMMON_KWARGS", kwargs), \
             patch.object(experiment.previous, "TOTAL_STEPS", 80), \
             patch.object(experiment.previous.pilot, "TRAIN_KWARGS", kwargs), \
             patch.object(training, "make_piecewise_mo_env", side_effect=short_env), \
             patch.object(training.CommonTargetSamplingSelector, "__init__", new=short_blocks), patch("builtins.print"):
            root = Path(folder)
            for env in experiment.ENVIRONMENTS:
                for method in experiment.METHODS:
                    report = experiment.run(env, method, 912347, root)
                    self.assertEqual(report["learning_work"]["critic_optimizer_steps"], 76)
                    self.assertEqual(report["completed_budget"], dict(training_interactions=80,
                        evaluation_interactions=1008, environment_initialization_probe_steps=2))
                    self.assertTrue(report["completed_training_episode_profile"])
                    with patch.object(training, "train_method", side_effect=AssertionError("retrained")):
                        self.assertEqual(experiment.run(env, method, 912347, root), report)
                path = root / env.replace("-", "_") / "projected" / f"{experiment.METHODS['projected']}_seed912347.json"
                raw = json.loads(path.read_text())
                raw["common_target_learning"]["critic_loss_rule"] = experiment.LOSS_RULES["vector"]
                with self.assertRaises(AssertionError):
                    experiment.validate(raw, env, "projected", 912347, path.with_name(
                        f"{experiment.METHODS['projected']}_seed912347_model.pt"))
            readout = experiment.summarize(root)
            self.assertEqual(readout["completed_budget"], dict(training_interactions=480,
                evaluation_interactions=6048, environment_initialization_probe_steps=12))
            path = root / "mo_hopper_v5" / "projected" / "seed912347.json"
            row = json.loads(path.read_text())
            row["learning_work"]["critic_rows"] += 1
            experiment.write_report(path, row)
            with self.assertRaises(AssertionError):
                experiment.summarize(root)
        print("COMMON_TARGET_LOSS_SMOKE training=480 evaluation=6048 probes=12")


if __name__ == "__main__":
    unittest.main()
