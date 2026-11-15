import copy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import torch

import run_preference_retention as e
from piecewise_mo_env import PiecewiseStationaryMOEnv


class RetentionTests(unittest.TestCase):
    def trainer(self):
        torch.manual_seed(9)
        trainer = e.training.ConsistentTargetMOSACTrainer(3, 2, 3, hidden=8,
            context_dim=5, batch_size=4, buffer_capacity=16, device="cpu")
        trainer.buffer.size = 8
        trainer.buffer.states[:8] = np.arange(24).reshape(8, 3) / 30.
        trainer.buffer.next_states[:8] = trainer.buffer.states[:8] + .1
        trainer.buffer.contexts[:8] = [1., 0., 0., .4, .5]
        trainer.buffer.next_contexts[:8] = trainer.buffer.contexts[:8]
        trainer.buffer.weights[:8] = [1., 0., 0.]
        trainer.buffer.rewards[:8] = [.1, .3, -.2]
        return trainer

    def configure(self, router, trainer, step):
        selector = router.selector
        selector.current_operator = e.credit.B
        selector.block_start_step = 11 if step <= 12 else 13
        for k, v in {**selector.trainer_config(e.credit.B),
                     "preference_relabeling_rate": selector.relabel_rate(e.credit.B)}.items():
            setattr(trainer, k, v)
        router.configure(step, trainer)

    def router(self, condition, coefficient=1.):
        return e.RetentionRouter("legacy_adaptive", (10, 12), condition,
            e.training.ProgressExtremeHierarchicalOperatorSelector(3), coefficient, warmup=1, end=13)

    def test_templates_equal_compute_frozen_teacher_and_nonzero_gradient(self):
        for condition in e.CONDITIONS:
            trainer, router = self.trainer(), self.router(condition)
            try:
                self.configure(router, trainer, 11)
                s, c = torch.ones(4, 3), torch.tensor([[1., 0., 0., .4, .5]]).repeat(4, 1)
                states, contexts = router.anchor_batch(s, c, condition == "coverage")
                self.assertEqual(states.shape, (28, 3))
                torch.testing.assert_close(contexts[:, 3:], c.repeat_interleave(7, 0)[:, 3:])
                if condition == "coverage":
                    torch.testing.assert_close(contexts[:7, :3], router.templates)
                else:
                    torch.testing.assert_close(contexts, c.repeat_interleave(7, 0))
                teacher_before = copy.deepcopy(router.teacher.state_dict())
                with torch.no_grad():
                    trainer.policy.mean_head.bias.add_(.2)
                n, t = np.random.get_state(), torch.get_rng_state()
                loss = router.objective(s, c[:, :3], c, torch.zeros(4, 1), torch.zeros(4, 1))
                loss.backward()
                self.assertGreater(float(loss.detach()), 0.)
                self.assertGreater(float(trainer.policy.mean_head.bias.grad.norm()), 0.)
                self.assertTrue(all(p.grad is None and not p.requires_grad for p in router.teacher.parameters()))
                self.assertTrue(all(p.grad is None for p in trainer.q1.parameters()))
                for key, value in teacher_before.items():
                    torch.testing.assert_close(router.teacher.state_dict()[key], value, rtol=0, atol=0)
                np.testing.assert_array_equal(np.random.get_state()[1], n[1])
                torch.testing.assert_close(torch.get_rng_state(), t, rtol=0, atol=0)
                self.assertEqual(router.action_pairs, 28)
                self.assertTrue(router.state_dict()["selection_uses_scores"])
                self.assertEqual(router.state_dict()["retention"]["condition"], condition)
                self.configure(router, trainer, 13)
                self.assertNotIn("actor_objective", trainer.__dict__)
            finally:
                router.close()

    def test_zero_coefficient_preserves_native_updates_and_rng(self):
        reference = self.trainer()
        candidate = copy.deepcopy(reference)
        native = e.credit.PulseRouter("legacy_adaptive", True, (10, 12),
            e.training.ProgressExtremeHierarchicalOperatorSelector(3), warmup=1, end=13)
        zero = self.router("coverage", 0.)
        try:
            for step in (11, 12, 13):
                outputs = []
                for router, trainer in ((native, reference), (zero, candidate)):
                    self.configure(router, trainer, step)
                    np.random.seed(step)
                    torch.manual_seed(step)
                    trainer.update()
                    outputs.append((np.random.get_state(), torch.get_rng_state()))
                for name in ("policy", "q1", "q2", "q1_target", "q2_target"):
                    for key, value in getattr(reference, name).state_dict().items():
                        torch.testing.assert_close(getattr(candidate, name).state_dict()[key], value, rtol=0, atol=0)
                np.testing.assert_array_equal(outputs[0][0][1], outputs[1][0][1])
                torch.testing.assert_close(outputs[0][1], outputs[1][1], rtol=0, atol=0)
            self.assertEqual(zero.loss_updates, 2)
        finally:
            native.close()
            zero.close()

    def test_budget_and_completed_reuse_without_training(self):
        budgets = [e.planned_budget(cut) for _, cut in e.credit.CASES.values()]
        self.assertEqual(sum(b["training_interactions"] for b in budgets), 143500)
        self.assertEqual(sum(b["environment_initialization_probe_steps"] for b in budgets), 12)
        self.assertEqual(sum(b["retention_action_pairs"] for b in budgets), 10752000)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "seed1013.json"
            report = dict(protocol=e.PROTOCOL, seed=1013, prefix_complete=True,
                          branches={name: dict(complete=True) for name in e.CONDITIONS})
            path.write_text(json.dumps(report))
            with (patch.object(e, "load_source", return_value=({}, Path("source"))),
                  patch.object(e, "finalize", return_value={}),
                  patch.object(e.training, "train_method", side_effect=AssertionError("No retraining"))):
                self.assertTrue(e.run(1013, directory)["complete"])
                report["branches"]["current"]["complete"] = False
                path.write_text(json.dumps(report))
                with self.assertRaisesRegex(RuntimeError, "Incomplete attempt preserved"):
                    e.run(1013, directory)

    def test_finalize_completed_measurements_and_budget(self):
        from test_scoped_bc_credit import CreditTests
        source = CreditTests().example_report()
        for branch in source["branches"].values():
            for row in branch["measurements"]:
                row.update(encoder="contemporaneous", policy="current_only", returns=[[[1., 2., 3.]]])
        branches = {}
        for condition in e.CONDITIONS:
            branches[condition] = copy.deepcopy(source["branches"]["pulse"])
            branches[condition]["routing"]["retention"] = dict(condition=condition, coefficient=1.,
                loss_updates=1500, action_pairs=1500 * 256 * 7)
        budget = e.planned_budget(32000)
        report = dict(prefix_complete=True, window=[32000, 33500], branches=branches,
            prefix_measurement=source["branches"]["native"]["measurements"][0],
            planned_budget_upper_bound=budget, completed_budget={**budget, "evaluation_interactions": 63})
        result = e.finalize(report, source)
        self.assertEqual([r["hv_delta"] for r in result["coverage_vs_current"]], [0.] * 4)
        report["completed_budget"]["evaluation_interactions"] += 1
        with self.assertRaises(AssertionError):
            e.finalize(report, source)

    def test_hopper_prefix_reproduction_continuation_and_patch_expiration(self):
        t = e.training
        make_env, trainer_cls = t.make_piecewise_mo_env, t.ConsistentTargetMOSACTrainer
        init = t.ProgressExtremeHierarchicalOperatorSelector.__init__
        original_step = PiecewiseStationaryMOEnv.step
        cost = dict(training=0, evaluation=0, initialization=0)

        def make(*args, **kwargs):
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

        def networks(state):
            return {name: copy.deepcopy(net.state_dict()) for name, net in (
                ("policy", state["trainer"].policy), ("q1", state["trainer"].q1),
                ("q2", state["trainer"].q2), ("encoder", state["encoder"]))}

        try:
            with (tempfile.TemporaryDirectory() as directory,
                  patch.object(t, "make_piecewise_mo_env", side_effect=make),
                  patch.object(PiecewiseStationaryMOEnv, "step", new=counted_step),
                  patch.object(t, "ConsistentTargetMOSACTrainer",
                               side_effect=lambda *a, **kw: trainer_cls(*a, buffer_capacity=128, **kw)),
                  patch.object(t.ProgressExtremeHierarchicalOperatorSelector, "__init__", new=short_blocks),
                  patch("builtins.print")):
                path = Path(directory) / "report.json"
                report = dict(seed=7, prefix_checks={}, completed_budget=dict(training_interactions=0,
                    learner_updates=0, evaluation_interactions=0),
                    branches={name: dict(complete=False, measurements=[]) for name in ("zero", *e.CONDITIONS)})
                final_nets = {}

                def run(router, probe, **kwargs):
                    try:
                        return t.train_method(e.credit.ENV, e.credit.METHOD, 41, 25, 7, directory,
                            hidden=8, batch_size=4, learning_starts=5, eval_freq=1000, device="cpu",
                            archive_steps=[41], state_probe=probe, update_router=router, **kwargs)
                    finally:
                        router.close()

                baseline = e.credit.PulseRouter("envelope_only", True, (33, 37), warmup=5, end=41)

                class BaselineProbe:
                    step = 5
                    rows = []

                    def __call__(self, state):
                        baseline.selector = state["adaptive_selector"]
                        if state["step"] > 5:
                            self.rows.append(e.credit.measure(state, 7))
                        if state["step"] == 41:
                            final_nets["baseline"] = networks(state)
                            self.reference = {**copy.deepcopy(state["log_data"]),
                                "ground_truth": state["env"].get_ground_truth(),
                                "adaptive_operator_final": dict(block_history=copy.deepcopy(baseline.selector.block_history))}
                            return None
                        self.step = next(x for x in (33, 37, 41) if x > state["step"])
                        return False

                reference = BaselineProbe()
                run(baseline, reference)
                native = e.credit.PulseRouter("envelope_only", False, (33, 37), warmup=5, end=41)
                prefix = e.Probe(native, report, path, "prefix", (5, 33),
                    reference=reference.reference, expected_prefix=reference.rows[0], learning_starts=5)
                run(native, prefix)
                self.assertTrue(report["prefix_complete"])
                self.assertEqual(prefix.state["episode_steps"], 3)

                class Probe(e.Probe):
                    def __call__(self, state):
                        if state["step"] == 41:
                            final_nets[self.branch] = networks(state)
                            self.assert_unpatched = "actor_objective" not in state["trainer"].__dict__
                        return super().__call__(state)

                for i, name in enumerate(("zero", *e.CONDITIONS)):
                    state = e.forks.clone_state(prefix.state)
                    router = e.RetentionRouter("envelope_only", (33, 37),
                        "coverage" if name == "zero" else name, state["adaptive_selector"],
                        coefficient=0. if name == "zero" else 1., warmup=5, end=41)
                    probe = Probe(router, report, path, name, (37, 41), start=33, offset=33 + i * 8,
                                  learning_starts=5)
                    run(router, probe, resume_state=state)
                    self.assertTrue(probe.assert_unpatched)
                    self.assertEqual(router.loss_updates, 4)
                    self.assertEqual(router.action_pairs, 112)
                for net, values in final_nets["baseline"].items():
                    for key, value in values.items():
                        torch.testing.assert_close(final_nets["zero"][net][key], value, rtol=0, atol=0)
                self.assertEqual(report["branches"]["zero"]["measurements"][-1]["returns"], reference.rows[-1]["returns"])
                self.assertTrue(any(not torch.equal(final_nets["coverage"]["policy"][k], v)
                                    for k, v in final_nets["current"]["policy"].items()))
                prefix.state["env"].close()
                prefix.state["eval_env"].close()
                self.assertEqual(cost, dict(training=98, evaluation=15120, initialization=10))
        finally:
            print("RETENTION_SMOKE_INTERACTIONS", json.dumps(cost))


if __name__ == "__main__":
    unittest.main()
