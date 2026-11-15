import copy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import torch

import run_return_labelled_guard as probe


class ReturnLabelledGuardTests(unittest.TestCase):
    def test_projects_actual_warm_adam_displacement_and_retains_moments(self):
        p = torch.nn.Parameter(torch.ones(2, dtype=torch.float64))
        opt = probe.TangentAdam([p], lr=.1)
        for _ in range(3):
            p.grad = torch.tensor([-10., 1.], dtype=p.dtype)
            opt.step()
        native = torch.nn.Parameter(p.detach().clone())
        ref = torch.optim.Adam([native], lr=.1)
        ref.load_state_dict(copy.deepcopy(opt.state_dict()))
        before = p.detach().clone()
        p.grad = native.grad = torch.tensor([1., -1.], dtype=p.dtype)
        opt.constraint = [torch.tensor([1., 0.], dtype=p.dtype)]
        opt.step()
        ref.step()
        self.assertGreater(float(native[0] - before[0]), 0)
        self.assertEqual(float(p[0]), float(before[0]))
        self.assertEqual(float(p[1]), float(native[1]))
        self.assertEqual(opt.projections, 1)
        self.assertIsNone(opt.constraint)
        for k, value in opt.state_dict()["state"][0].items():
            torch.testing.assert_close(value, ref.state_dict()["state"][0][k], rtol=0, atol=0)
        opt.constraint = [torch.zeros_like(p)]
        p.grad = native.grad = torch.ones_like(p)
        old_difference = p.detach() - native.detach()
        opt.step()
        ref.step()
        torch.testing.assert_close(p.detach() - native.detach(), old_difference, atol=1e-15, rtol=0)
        self.assertEqual(opt.projections, 1)

    def test_labels_use_only_actual_scalar_returns_and_prefer_current_on_ties(self):
        self.assertEqual(probe.choose_teacher([1., -2.], [2., -5.], [.9, .1]), "new")
        self.assertEqual(probe.choose_teacher([1., -2.], [2., -5.], [.1, .9]), "old")
        self.assertEqual(probe.choose_teacher([1., 2., 3.], [1., 2., 3.], [.2, .3, .5]), "new")

    def test_real_full_pipeline_matched_prices_clocks_and_native_identity(self):
        factory, train, collect = probe.training.make_piecewise_mo_env, probe.train, probe.collect
        kwargs = dict(hidden=8, lr=3e-4, gamma=.99, batch_size=4, learning_starts=5,
                      eval_freq=1000000, log_freq=10000, device="cpu")
        native_costs = dict(training_interactions=0, initialization_probes=0,
                            actor_steps=0, critic_steps=0, encoder_steps=0, evaluation_interactions=0)
        costs = {}
        native_checks = []

        def short_env(*args, **options):
            options.update(switch_interval=12, mode_order="sequential")
            env = factory(*args, **options)
            env.env._max_episode_steps = 4
            return env

        def same(state, expected):
            probe.base.same_learner(state["trainer"], expected["trainer"])
            for k, v in state["encoder"].state_dict().items():
                self.assertTrue(torch.equal(v, expected["encoder"].state_dict()[k]))
            for k in probe.shared.SHARED_FIELDS:
                np.testing.assert_array_equal(getattr(state["trainer"].buffer, k)[:state["step"]],
                    getattr(expected["trainer"].buffer, k)[:expected["step"]])
            self.assertEqual(state["env"].mode_history, expected["env"].mode_history)

        def checked_train(env, seed, output, steps, resume=None, guard=None):
            state = train(env, seed, output, steps, resume=resume, guard=guard)
            if resume is not None and guard is None and steps > probe.END:
                expected = train(env, seed, output, steps)
                native_costs["training_interactions"] += steps
                native_costs["initialization_probes"] += 2
                for key in ("actor_steps", "critic_steps", "encoder_steps"):
                    native_costs[key] += steps - kwargs["learning_starts"] + 1
                try:
                    same(state, expected)
                    native_checks.append(env)
                finally:
                    expected["env"].close()
                    expected["eval_env"].close()
            return state

        def checked_collect(state, models, seed):
            rng, trng = np.random.get_state(), torch.get_rng_state().clone()
            frozen = {name: {k: copy.deepcopy(model[k].state_dict()) for k in ("policy", "encoder")}
                      for name, model in models.items()}
            pairs, report = collect(state, models, seed)
            now = np.random.get_state()
            self.assertEqual(rng[0], now[0])
            np.testing.assert_array_equal(rng[1], now[1])
            self.assertEqual(rng[2:], now[2:])
            self.assertTrue(torch.equal(trng, torch.get_rng_state()))
            self.assertEqual(state["env"]._total_steps, probe.PROPOSAL)
            for name, model in models.items():
                for key in frozen[name]:
                    for k, v in model[key].state_dict().items():
                        self.assertTrue(torch.equal(v, frozen[name][key][k]))
            weight = probe.training.full_simplex_grid(models["new"]["reward_dim"])[0]
            left, right = probe.base.fork_env(state["env"]), probe.base.fork_env(state["env"])
            try:
                data, vector = probe.trajectory(left, models["new"], weight, 114124)
                reference = probe.router.episode(right, {"soft": models["new"]}, weight, 114124, "budget_soft")
                native_costs["evaluation_interactions"] += len(data["states"]) + reference["interactions"]
                np.testing.assert_array_equal(vector, reference["return_vector"])
            finally:
                left.close()
                right.close()
            for pair, row in zip(pairs, report["rows"]):
                self.assertEqual(pair["scored"], row["scored"])
                self.assertEqual(pair["blind"], row["blind"])
                for name, data in pair["trajectories"].items():
                    self.assertEqual(len(data["states"]), row["steps"][name])
                    np.testing.assert_allclose(data["contexts"][:, :len(row["weight"])],
                        np.broadcast_to(row["weight"], (len(data["states"]), len(row["weight"]))), atol=1e-7)
            return pairs, report

        with tempfile.TemporaryDirectory() as directory, \
             patch.multiple(probe, CUT=24, PROPOSAL=32, END=48, HORIZON=4, SWITCH_INTERVAL=12,
                            KWARGS=kwargs, SEEDS=(9133,)), \
             patch.multiple(probe.router, PREFIX=2, HORIZON=4, REPEATS=1), \
             patch.object(probe.training, "make_piecewise_mo_env", side_effect=short_env), \
             patch.object(probe.training, "policy_archive_interval_for_method", return_value=10), \
             patch.object(probe, "train", side_effect=checked_train), \
             patch.object(probe, "collect", side_effect=checked_collect), patch("builtins.print"):
            root = Path(directory)
            probe.write(root / "protocol.json", dict(protocol=probe.PROTOCOL, experiment=probe.specification()))
            for env in probe.ENVIRONMENTS:
                record = probe.run(env, 9133, root)
                n = record["calibration"]["interactions"]
                self.assertEqual(n, 168)
                self.assertTrue(all(r["paired_initial_observation"] for r in record["calibration"]["rows"]))
                totals = [sum(v[k] for k in ("training_interactions", "calibration_interactions", "initialization_probes"))
                          for v in record["path_prices"].values()]
                self.assertEqual(totals, [probe.END + n + 6] * 3)
                for c in ("scored", "blind"):
                    self.assertEqual(record["guards"][c]["constraint_gradient_calls"], 184)
                    self.assertEqual(record["guards"][c]["auxiliary_rows"], 736)
                for k, value in record["actual_work"].items():
                    costs[k] = costs.get(k, 0) + value
                with patch.object(probe.training, "train_method", side_effect=AssertionError("retrained")):
                    self.assertEqual(probe.run(env, 9133, root), record)
                bad = copy.deepcopy(record)
                bad["clocks"]["scored"]["actor"] -= 1
                with self.assertRaises(AssertionError):
                    probe.check(bad)
            summary = probe.summarize(root)
            self.assertEqual(summary["actual_work"], costs)
            self.assertEqual(native_checks, list(probe.ENVIRONMENTS))
        print("RETURN_GUARD_SMOKE_WORK", json.dumps(costs))
        print("RETURN_GUARD_NATIVE_WORK", json.dumps(native_costs))


if __name__ == "__main__":
    unittest.main()
