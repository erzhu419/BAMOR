import copy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import torch

import run_context_descent_probe as probe


class ContextDescentTests(unittest.TestCase):
    def test_multi_constraint_projection_is_nearest_and_handles_opposition(self):
        d = torch.tensor([2., 3., 4.], dtype=torch.float64)
        g = torch.tensor([[1., 0., 0.], [0., 1., 0.]], dtype=d.dtype)
        projected, directions, changed = probe.project_displacement(d, g)
        torch.testing.assert_close(projected, torch.tensor([0., 0., 4.], dtype=d.dtype), atol=1e-12, rtol=0)
        self.assertTrue(changed)
        self.assertLessEqual(float((directions @ projected).max()), 1e-12)
        opposed = torch.tensor([[1., 0., 0.], [-1., 0., 0.], [0., 0., 0.]], dtype=d.dtype)
        projected, _, _ = probe.project_displacement(d, opposed)
        torch.testing.assert_close(projected, torch.tensor([0., 3., 4.], dtype=d.dtype), atol=1e-12, rtol=0)
        free = -torch.ones(3, dtype=d.dtype)
        projected, _, changed = probe.project_displacement(free, g)
        self.assertFalse(changed)
        self.assertTrue(torch.equal(projected, free))

    def test_warm_actual_adam_not_raw_gradient_and_moments_unchanged(self):
        p = torch.nn.Parameter(torch.ones(2, dtype=torch.float64))
        opt = probe.ContextAdam([p], lr=.1)
        zero = torch.zeros((1, 2), dtype=p.dtype)
        for _ in range(3):
            p.grad = torch.tensor([-10., 1.], dtype=p.dtype)
            opt.constraints = zero
            opt.step()
        native = torch.nn.Parameter(p.detach().clone())
        ref = torch.optim.Adam([native], lr=.1)
        ref.load_state_dict(copy.deepcopy(opt.state_dict()))
        before = p.detach().clone()
        p.grad = native.grad = torch.tensor([1., -1.], dtype=p.dtype)
        opt.constraints = torch.tensor([[1., 0.], [0., 1.]], dtype=p.dtype)
        opt.step()
        ref.step()
        self.assertGreater(float(native.detach()[0] - before[0]), 0.)
        self.assertLessEqual(float(p.detach()[0] - before[0]), 1e-12)
        self.assertIsNone(opt.constraints)
        for k, v in opt.state_dict()["state"][0].items():
            torch.testing.assert_close(v, ref.state_dict()["state"][0][k], atol=0, rtol=0)

    def test_grouping_excludes_preferences_latents_and_labels(self):
        class Buffer:
            size = 8
            contexts = np.arange(8 * 13, dtype=float).reshape(8, 13)
        class Trainer:
            reward_dim, buffer = 2, Buffer()
        grouping = probe.ObservableGroups(Trainer())
        contexts = torch.as_tensor(Buffer.contexts.copy())
        expected = grouping.assign(contexts)
        contexts[:, :2] += 1e6
        contexts[:, 5:] -= 1e6
        np.testing.assert_array_equal(grouping.assign(contexts), expected)
        self.assertEqual(grouping.state_dict()["feature_slice"], [2, 5])

    def test_real_paired_pipeline_native_identity_and_prices(self):
        factory, train = probe.training.make_piecewise_mo_env, probe.train
        kwargs = dict(hidden=8, lr=3e-4, gamma=.99, batch_size=4, learning_starts=5,
                      eval_freq=1000000, log_freq=1000000, device="cpu")
        reference_work = dict(training_interactions=0, initialization_probes=0, actor_steps=0, critic_steps=0, encoder_steps=0)
        costs, checks = {}, []

        def short_env(*args, **options):
            options.update(switch_interval=12, mode_order="sequential")
            env = factory(*args, **options)
            env.env._max_episode_steps = 7
            return env

        def checked_train(env, seed, output, steps, resume=None, intervention=None):
            state = train(env, seed, output, steps, resume=resume, intervention=intervention)
            if resume is not None and intervention is None:
                expected = train(env, seed, output, steps)
                reference_work["training_interactions"] += steps
                reference_work["initialization_probes"] += 2
                for k in ("actor_steps", "critic_steps", "encoder_steps"):
                    reference_work[k] += steps - kwargs["learning_starts"] + 1
                try:
                    probe.base.same_learner(state["trainer"], expected["trainer"])
                    for k, v in state["encoder"].state_dict().items():
                        torch.testing.assert_close(v, expected["encoder"].state_dict()[k], atol=0, rtol=0)
                    for k in (*probe.shared.SHARED_FIELDS, "contexts", "next_contexts"):
                        np.testing.assert_array_equal(getattr(state["trainer"].buffer, k)[:steps],
                                                      getattr(expected["trainer"].buffer, k)[:steps])
                    checks.append(env)
                finally:
                    expected["env"].close()
                    expected["eval_env"].close()
            return state

        with tempfile.TemporaryDirectory() as directory, \
             patch.multiple(probe, CUT=24, END=40, FIT_ROWS=16, KWARGS=kwargs, SWITCH_INTERVAL=12, SEEDS=(9135,)), \
             patch.multiple(probe.router, PREFIX=2, HORIZON=4, REPEATS=1), \
             patch.object(probe.training, "make_piecewise_mo_env", side_effect=short_env), \
             patch.object(probe.training, "policy_archive_interval_for_method", return_value=10), \
             patch.object(probe, "train", side_effect=checked_train), patch("builtins.print"):
            root = Path(directory)
            probe.write(root / "protocol.json", dict(protocol=probe.PROTOCOL, experiment=probe.specification()))
            for env in probe.ENVIRONMENTS:
                record = probe.run(env, 9135, root)
                self.assertEqual(record["path_prices"]["native"], record["path_prices"]["grouped"])
                self.assertEqual(record["intervention"]["updates"], 16)
                self.assertEqual(record["intervention"]["group_gradient_rows"], 64)
                self.assertLess(record["intervention"]["max_positive_actual_normalized_group_inner_product"], 1e-7)
                for k, v in record["actual_work"].items():
                    costs[k] = costs.get(k, 0) + v
                with patch.object(probe.training, "train_method", side_effect=AssertionError("retrained")):
                    self.assertEqual(probe.run(env, 9135, root), record)
                bad = copy.deepcopy(record)
                bad["actual_work"]["initialization_probes"] -= 2
                with self.assertRaises(AssertionError):
                    probe.check(bad)
            self.assertEqual(probe.summarize(root)["actual_work"], costs)
        self.assertEqual(checks, list(probe.ENVIRONMENTS))
        print("CONTEXT_DESCENT_SMOKE_WORK", json.dumps(costs))
        print("CONTEXT_DESCENT_REFERENCE_WORK", json.dumps(reference_work))


if __name__ == "__main__":
    unittest.main()
