import copy
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
import torch

import run_scoped_bc_credit as experiment
from piecewise_mo_env import PiecewiseStationaryMOEnv


class CreditTests(unittest.TestCase):
    def test_pulse_expires_even_on_another_bc_block(self):
        for background in ("legacy_adaptive", "envelope_only"):
            selector = experiment.training.ProgressExtremeHierarchicalOperatorSelector(3)
            trainer = SimpleNamespace()
            router = experiment.PulseRouter(background, True, (10, 12), selector, warmup=1, end=15)
            try:
                for step, op in ((9, experiment.E), (11, experiment.B), (12, experiment.B),
                                 (13, experiment.B), (15, experiment.E)):
                    selector.current_operator = op
                    selector.block_start_step = 11 if step == 12 else step
                    for k, v in {**selector.trainer_config(op),
                                 "preference_relabeling_rate": selector.relabel_rate(op)}.items():
                        setattr(trainer, k, v)
                    router.configure(step, trainer)
                    expected = step in (11, 12) or (op == experiment.E and background == "envelope_only")
                    self.assertEqual("envelope_actor_loss" in trainer.__dict__, expected)
                self.assertEqual(router.state_dict()["pulse_updates"], 2)
                selector.current_operator = experiment.E
                with self.assertRaisesRegex(AssertionError, "not a BC block"):
                    router.configure(11, trainer)
            finally:
                router.close()
            self.assertNotIn("envelope_actor_loss", trainer.__dict__)

    def example_report(self):
        cut, end = 32000, 33500
        times = experiment.measurement_steps(cut)
        rows = []
        for step in times:
            rows.append(dict(step=step, evaluation_weights=[[0., 0., 1.]],
                episode_seeds_per_mode=[[1, 2, 3]], ref_point=[0., 0., 0.],
                hv_mean_mode=2., hv_per_mode=[2.], matched_utility_per_mode=[2.],
                objective_means_per_mode=[[2., 2., 2.]], evaluation_interactions=7,
                scores={experiment.B: .1}, feedback=[dict(end_step=end, operator=experiment.B,
                                                        complete=True, progress_score=.1)]))
        pulse = copy.deepcopy(rows[1:])
        for i, row in enumerate(pulse):
            row["hv_mean_mode"] = 3. if i == 0 else 1.
            row["feedback"][0]["progress_score"] = .2
        budget = experiment.planned_budget(cut)
        return dict(protocol=experiment.PROTOCOL, seed=1013, window=[cut, end], complete=False,
            planned_budget_upper_bound=budget,
            completed_budget={**budget, "evaluation_interactions": 63},
            branches={"native": dict(complete=True, measurements=rows, mode_history=[[0, 0]],
                                     routing=dict(pulse_updates=0)),
                      "pulse": dict(complete=True, measurements=pulse, mode_history=[[0, 0]],
                                    routing=dict(pulse_updates=1500))})

    def test_summary_keeps_immediate_delayed_sign_reversal_and_checks_budget(self):
        report = self.example_report()
        summary = experiment.summarize(report)
        self.assertGreater(summary["block_progress_delta"], 0)
        self.assertEqual([r["hv_delta"] for r in summary["contrasts"]], [1., -1., -1., -1.])
        report["completed_budget"]["evaluation_interactions"] += 1
        with self.assertRaises(AssertionError):
            experiment.summarize(report)

    def test_postprocessing_recovery_does_not_retrain(self):
        report = self.example_report()
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "seed1013.json"
            path.write_text(json.dumps(report))
            with patch.object(experiment.training, "train_method", side_effect=AssertionError("No retraining")):
                self.assertTrue(experiment.run(1013, directory)["complete"])
                report["branches"]["pulse"]["complete"] = False
                path.write_text(json.dumps(report))
                with self.assertRaisesRegex(RuntimeError, "Incomplete attempt preserved"):
                    experiment.run(1013, directory)

    def test_real_hopper_mid_episode_clone_and_contemporaneous_evaluation(self):
        t = experiment.training
        make_env, trainer_cls = t.make_piecewise_mo_env, t.ConsistentTargetMOSACTrainer
        init = t.ProgressExtremeHierarchicalOperatorSelector.__init__
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

        def networks(state):
            trainer = state["trainer"]
            return {name: copy.deepcopy(net.state_dict()) for name, net in (
                ("policy", trainer.policy), ("q1", trainer.q1), ("q2", trainer.q2), ("encoder", state["encoder"]))}

        try:
            with (tempfile.TemporaryDirectory() as directory,
                  patch.object(t, "make_piecewise_mo_env", side_effect=short_env),
                  patch.object(PiecewiseStationaryMOEnv, "step", new=counted_step),
                  patch.object(t, "ConsistentTargetMOSACTrainer",
                               side_effect=lambda *a, **kw: trainer_cls(*a, buffer_capacity=128, **kw)),
                  patch.object(t.ProgressExtremeHierarchicalOperatorSelector, "__init__", new=short_blocks),
                  patch("builtins.print")):
                path = Path(directory) / "report.json"
                report = dict(seed=7, completed_budget=dict(training_interactions=0, learner_updates=0,
                    evaluation_interactions=0), control_reproduction={},
                    branches={name: dict(complete=False, measurements=[])
                        for name in ("native", "native_fork", "pulse")})
                endpoint_nets = {}

                class Probe(experiment.TrajectoryProbe):
                    def __call__(self, state):
                        if state["step"] == 41:
                            endpoint_nets[self.branch] = networks(state)
                        return super().__call__(state)

                def router(pulse=False, selector=None):
                    return experiment.PulseRouter("envelope_only", pulse, (33, 37), selector,
                                                  warmup=5, end=41)

                def run(route, probe, **kwargs):
                    try:
                        return t.train_method(experiment.ENV, experiment.METHOD, 41, 25, 7, directory,
                            hidden=8, batch_size=4, learning_starts=5, eval_freq=1000, device="cpu",
                            archive_steps=[41], state_probe=probe, update_router=route, **kwargs)
                    finally:
                        route.close()

                plain = router()

                class PlainProbe:
                    step = 5

                    def __call__(self, state):
                        if state["step"] == 5:
                            plain(state)
                            self.step = 41
                            return False
                        endpoint_nets["plain"] = networks(state)
                        self.row = experiment.measure(state, 7)
                        self.feedback = copy.deepcopy(state["adaptive_selector"].block_history)
                        self.reference = {**copy.deepcopy(state["log_data"]),
                            "ground_truth": state["env"].get_ground_truth(),
                            "adaptive_operator_final": dict(block_history=self.feedback)}
                        self.reference["pareto_per_mode"] = [[{"return": r, "archive_step": experiment.forks.END}
                            for r in mode] for mode in self.row["returns"]]

                plain_probe = PlainProbe()
                run(plain, plain_probe)
                native = router()
                probe = Probe(native, report, path, "native", 33, (33, 37, 41),
                              reference=plain_probe.reference, warmup=5, learning_starts=5, end=41)
                run(native, probe)
                self.assertTrue(report["control_reproduction"]["41"]["final_returns_matched"])
                self.assertEqual(probe.state["episode_steps"], 3)
                self.assertNotIn("envelope_actor_loss", probe.state["trainer"].__dict__)
                self.assertEqual(report["branches"]["native"]["measurements"][-1]["returns"], plain_probe.row["returns"])
                for name in ("native_fork", "pulse"):
                    state = experiment.forks.clone_state(probe.state)
                    route = router(name == "pulse", state["adaptive_selector"])
                    resumed = Probe(route, report, path, name, 33, (33, 37, 41),
                                    warmup=5, learning_starts=5, end=41)
                    run(route, resumed, resume_state=state)
                for name in ("native", "native_fork"):
                    for net in endpoint_nets["plain"]:
                        for key, value in endpoint_nets["plain"][net].items():
                            torch.testing.assert_close(endpoint_nets[name][net][key], value, rtol=0, atol=0)
                    row = report["branches"][name]["measurements"][-1]
                    self.assertEqual(row["feedback"], plain_probe.feedback)
                    self.assertEqual(row["returns"], plain_probe.row["returns"])
                self.assertTrue(any(not torch.equal(endpoint_nets["pulse"]["policy"][key], value)
                                    for key, value in endpoint_nets["plain"]["policy"].items()))
                self.assertEqual(report["branches"]["pulse"]["routing"]["pulse_updates"], 4)
                probe.state["env"].close()
                probe.state["eval_env"].close()
                self.assertEqual(cost, dict(training=98, evaluation=12096, initialization=8))
        finally:
            print("BC_CREDIT_SMOKE_INTERACTIONS", json.dumps(cost))


if __name__ == "__main__":
    unittest.main()
