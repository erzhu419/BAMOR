import copy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import torch

import run_terminal_rng_matched as experiment
from piecewise_mo_env import PiecewiseStationaryMOEnv


class MatchedRngTests(unittest.TestCase):
    def make_trainer(self):
        trainer = experiment.training.ConsistentTargetMOSACTrainer(
            3, 2, 2, hidden=8, buffer_capacity=16, batch_size=4, device="cpu")
        selector = experiment.training.FixedProgressExtremeHierarchicalOperatorSelector(2)
        for key, value in selector.trainer_config(experiment.previous.EXTREME).items():
            setattr(trainer, key, value)
        trainer.preference_relabeling_rate = 0.
        for i in range(8):
            s, w = np.full(3, i / 10, dtype=np.float32), np.array([.3, .7], dtype=np.float32)
            trainer.buffer.push(s, np.zeros(2), w, w, np.ones(2), s + .1, w, False)
        return trainer

    def test_same_candidates_draw_order_rng_and_exact_own_gradient(self):
        torch.manual_seed(7)
        trainer = self.make_trainer()
        s, w = torch.zeros(4, 3), torch.tensor([[.3, .7]]).repeat(4, 1)
        rng = torch.get_rng_state()
        original_action = trainer.policy.get_action
        runs = []
        for loss_fn in (trainer.envelope_actor_loss,
                        lambda *args: experiment.own_candidate_loss(trainer, *args)):
            calls = []

            def record(s, c):
                action, log_prob = original_action(s, c)
                calls.append((c.clone(), action, log_prob))
                return action, log_prob

            torch.set_rng_state(rng)
            with patch.object(trainer.policy, "get_action", side_effect=record):
                output = loss_fn(s, w, w)
            runs.append((calls, output, torch.get_rng_state()))
        self.assertEqual(len(runs[0][0]), 5)
        self.assertEqual(len(runs[1][0]), 5)
        self.assertTrue(torch.equal(runs[0][2], runs[1][2]))
        for left, right in zip(runs[0][0], runs[1][0]):
            for a, b in zip(left, right):
                torch.testing.assert_close(a, b, rtol=0, atol=0)
        own_c, own_a, own_logp = runs[1][0][0]
        a, q, loss = runs[1][1]
        expected_q = (w * torch.min(trainer.q1(s, own_a, w), trainer.q2(s, own_a, w))).sum(-1, keepdim=True)
        expected_loss = trainer.actor_objective(s, w, own_c, expected_q, own_logp)
        for actual, expected in ((a, own_a), (q, expected_q), (loss, expected_loss)):
            torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        expected_grad = torch.autograd.grad(expected_loss, tuple(trainer.policy.parameters()), retain_graph=True)
        actual_grad = torch.autograd.grad(loss, tuple(trainer.policy.parameters()))
        for actual, expected in zip(actual_grad, expected_grad):
            torch.testing.assert_close(actual, expected, rtol=0, atol=0)

    def test_rng_alignment_across_updates_and_mismatch_detection(self):
        torch.manual_seed(7)
        base, trace, first_critics = self.make_trainer(), [], []
        for name in ("factual", "own_matched"):
            trainer = copy.deepcopy(base)
            router = experiment.MatchedIntervention(name, trace, start=1, end=4)
            np.random.seed(17)
            torch.manual_seed(17)
            try:
                for step in range(1, 5):
                    router.configure(step, trainer)
                    trainer.update()
                    router.after_step(step, trainer, None, None, experiment.previous.EXTREME)
                    if step == 1:
                        first_critics.append(copy.deepcopy(trainer.q1.state_dict()))
                if name == "own_matched":
                    self.assertTrue(router.state_dict()["rng_alignment"]["passed"])
                    torch.rand(1)
                    router.after_step(4, trainer, None, None, experiment.previous.EXTREME)
                    self.assertEqual(router.first_mismatch, 4)
                    self.assertFalse(router.state_dict()["rng_alignment"]["passed"])
            finally:
                router.close()
            self.assertNotIn("envelope_actor_loss", trainer.__dict__)
        for key in first_critics[0]:
            torch.testing.assert_close(first_critics[0][key], first_critics[1][key], rtol=0, atol=0)
        print("MATCHED_RNG_UNIT_LEARNER_UPDATES", 8)

    def test_real_training_rng_alignment_across_episode_reset(self):
        training = experiment.training
        make_env, trainer_cls = training.make_piecewise_mo_env, training.ConsistentTargetMOSACTrainer
        selector_init = training.ProgressExtremeHierarchicalOperatorSelector.__init__
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
            selector_init(self, reward_dim, block_steps=4, learning_starts=learning_starts)

        try:
            with (tempfile.TemporaryDirectory() as directory,
                  patch.object(training, "make_piecewise_mo_env", side_effect=short_env),
                  patch.object(PiecewiseStationaryMOEnv, "step", new=counted_step),
                  patch.object(training, "ConsistentTargetMOSACTrainer",
                               side_effect=lambda *a, **kw: trainer_cls(*a, buffer_capacity=128, **kw)),
                  patch.object(training.ProgressExtremeHierarchicalOperatorSelector, "__init__", new=short_blocks),
                  patch("builtins.print")):
                def run(name, **kwargs):
                    return training.train_method("mo-halfcheetah-v5", experiment.METHOD, 33, 25, 7,
                        str(Path(directory) / name), hidden=8, batch_size=4, learning_starts=5,
                        eval_freq=1000, device="cpu", **kwargs)

                capture = experiment.previous.CaptureState(29)
                run("prefix", state_probe=capture)
                state, trace = capture.state, []
                self.assertEqual(state["episode_steps"], 5)
                try:
                    configs = {}
                    for name in experiment.BRANCHES:
                        branch = experiment.previous.clone_state(state)
                        router = experiment.MatchedIntervention(name, trace, start=30, end=33)
                        try:
                            data = run(name, resume_state=branch, update_router=router)
                        finally:
                            router.close()
                        self.assertEqual(data["training_interactions_this_call"], 4)
                        self.assertEqual(data["episode_rewards"][-1]["step"], 30)
                        self.assertEqual(data["evaluation_interactions"], 1512)
                        configs[name] = data["update_routing"]["config"]
                        if name == "own_matched":
                            self.assertTrue(data["update_routing"]["rng_alignment"]["passed"])
                    self.assertEqual(configs["factual"], configs["own_matched"])
                    self.assertEqual(configs["actor_one"], {**configs["factual"], "envelope_policy_candidates": 1})
                    self.assertEqual(state["trainer"].buffer.size, 29)
                    self.assertEqual(cost, dict(training=41, evaluation=4536, initialization=8))
                finally:
                    state["env"].close()
                    state["eval_env"].close()
        finally:
            print("MATCHED_RNG_SMOKE_INTERACTIONS", json.dumps(cost))

    def test_paired_readout_keeps_negative_preference_effects(self):
        rows = {}
        for name, reward, hv in (("factual", [1., 2.], 10.), ("actor_one", [2., 0.], 12.),
                                 ("own_matched", [3., 1.], 13.)):
            rows[name] = dict(final_hv=hv, final_mode_hv=[hv] * 4,
                              terminal_returns=[[reward] * 2] * 4)
        out = experiment.paired_readout(rows, [[0., 1.], [1., 0.]])
        self.assertEqual(out["actor_one_minus_factual"]["matched_weight_utility_differences"], [[-2., 1.]] * 4)
        self.assertEqual(out["actor_one_minus_factual"]["utility_wins_per_mode"], [1] * 4)
        self.assertEqual(out["own_matched_minus_actor_one"]["final_hv"], 1.)

    def test_completed_reuse_and_incomplete_preservation(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / f"seed{experiment.SEED}.json"
            report = dict(protocol=experiment.PROTOCOL, complete=True, controls_passed=True,
                          rng_alignment_passed=True)
            path.write_text(json.dumps(report))
            with patch.object(experiment.training, "train_method", side_effect=AssertionError("No training")):
                self.assertEqual(experiment.run(directory), report)
                report["complete"] = False
                path.write_text(json.dumps(report))
                with self.assertRaisesRegex(RuntimeError, "Incomplete attempt preserved"):
                    experiment.run(directory)
                self.assertEqual(json.loads(path.read_text()), report)


if __name__ == "__main__":
    unittest.main()
