import copy
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import torch

import run_common_target_pilot as experiment


training = experiment.training


class CommonTargetTests(unittest.TestCase):
    def trainer(self, dim):
        return training.CommonTargetMOSACTrainer(3, 2, dim, hidden=8,
            buffer_capacity=16, batch_size=4, device="cpu")

    def test_full_vector_loss_sees_errors_invisible_to_scalar_loss(self):
        trainer = self.trainer(2)
        weights = torch.tensor([[.5, .5]])
        error = torch.tensor([[1., -1.]], requires_grad=True)
        target = torch.zeros_like(error)
        scalar = torch.zeros(1, 1)
        legacy = training.MOSACTrainer.critic_objective(trainer, error, error, weights, scalar, target)
        common = trainer.critic_objective(error, error, weights, scalar, target)
        self.assertEqual(float(legacy.detach()), 0.)
        self.assertEqual(float(common.detach()), 1.)
        common.backward()
        self.assertTrue(torch.all(error.grad != 0))

    def test_operator_invariance_at_the_same_replay_and_rng(self):
        for dim in (2, 3):
            trainer = self.trainer(dim)
            for i in range(8):
                s = np.full(3, i / 10, dtype=np.float32)
                w = np.full(dim, 1 / dim, dtype=np.float32)
                trainer.buffer.push(s, np.zeros(2), w, w, np.ones(dim), s + .1, w, i % 3 == 0)
            selector = training.CommonTargetSamplingSelector(dim)
            states = []
            for op in selector.operators:
                branch = copy.deepcopy(trainer)
                for key, value in selector.trainer_config(op).items():
                    setattr(branch, key, value)
                branch.preference_relabeling_rate = selector.relabel_rate(op)
                np.random.seed(832)
                torch.manual_seed(832)
                mean_q, target = branch.update()
                self.assertTrue(np.isfinite(mean_q) and np.isfinite(target).all())
                self.assertEqual(branch.update_work, dict(critic_optimizer_steps=1, actor_optimizer_steps=1,
                    critic_rows=4, target_candidate_rows=4, actor_candidate_rows=4))
                self.assertEqual(training.optimizer_step_count(branch.q_opt), 1)
                states.append({name: {k:v.clone() for k,v in getattr(branch, name).state_dict().items()}
                               for name in ("policy", "q1", "q2", "q1_target", "q2_target")})
            for state in states[1:]:
                for name in states[0]:
                    for key in states[0][name]:
                        self.assertTrue(torch.equal(state[name][key], states[0][name][key]), (dim,name,key))

    def test_scores_only_affect_adaptive_and_learning_config_is_constant(self):
        a = training.CommonTargetSamplingSelector(3)
        b = training.FixedCommonTargetSamplingSelector(3)
        self.assertEqual(a.state_dict()["operator_trainer_configs"], b.state_dict()["operator_trainer_configs"])
        a.counts = b.counts = dict.fromkeys(a.operators, 1)
        a.scores = b.scores = dict(uniform=-1., stratified=1., mixed=0.)
        self.assertGreater(a.probabilities()["stratified"], .8)
        self.assertEqual(b.probabilities(), dict.fromkeys(b.operators, 1 / 3))
        a.scores = b.scores = dict(uniform=1., stratified=-1., mixed=0.)
        self.assertGreater(a.probabilities()["uniform"], .8)
        self.assertEqual(b.probabilities(), dict.fromkeys(b.operators, 1 / 3))
        for method in training.COMMON_TARGET_METHODS:
            self.assertTrue(training.uses_learned_context(method))
            self.assertTrue(training.uses_transition_context(method))
            self.assertFalse(training.uses_envelope_adaptive_operator_selection(method))
            self.assertFalse(training.uses_policy_envelope_adaptive_operator_selection(method))
            self.assertEqual(training.context_dim_for_method(method, 17, 6, 2, 4), 13)

    @unittest.skipUnless(importlib.util.find_spec("morl_baselines"), "CAPQL checkout is on the server")
    def test_short_five_condition_routes_and_paired_readout(self):
        training.ensure_wandb_stub()
        from morl_baselines.multi_policy.capql.capql import CAPQL

        original_env, original_capql_init = training.make_piecewise_mo_env, CAPQL.__init__
        original_selector_init = training.CommonTargetSamplingSelector.__init__
        kwargs = dict(hidden=8, lr=3e-4, gamma=.99, batch_size=4, learning_starts=5,
                      eval_freq=100000, log_freq=10000, device="cpu")

        def short_env(*args, **kw):
            kw.update(switch_interval=12, mode_order="sequential")
            env = original_env(*args, **kw)
            env.env._max_episode_steps = 4
            return env

        def small_capql(agent, *args, **kw):
            kw.update(buffer_size=128, net_arch=[8, 8], batch_size=4, learning_starts=5)
            original_capql_init(agent, *args, **kw)

        def short_blocks(selector, reward_dim, **kw):
            original_selector_init(selector, reward_dim, block_steps=6, **kw)

        with tempfile.TemporaryDirectory() as folder, \
             patch.multiple(experiment, SEEDS=(912346,), COMMON_KWARGS=kwargs, CAPQL_LEARNING_STARTS=5, CAPQL_BATCH_SIZE=4), \
             patch.object(experiment.previous, "TOTAL_STEPS", 80), \
             patch.object(experiment.previous.pilot, "TRAIN_KWARGS", kwargs), \
             patch.object(training, "make_piecewise_mo_env", side_effect=short_env), \
             patch.object(CAPQL, "__init__", new=small_capql), \
             patch.object(training.CommonTargetSamplingSelector, "__init__", new=short_blocks), patch("builtins.print"):
            root = Path(folder)
            for env in experiment.ENVIRONMENTS:
                common = []
                for method in experiment.METHODS:
                    report = experiment.run(env, method, 912346, root)
                    self.assertTrue(report["complete"])
                    self.assertEqual(report["completed_budget"], dict(training_interactions=80,
                        evaluation_interactions=1008, environment_initialization_probe_steps=2))
                    work = report["learning_work"]
                    self.assertEqual(work["critic_optimizer_steps"], 76)
                    self.assertEqual(work["actor_optimizer_steps"], 76)
                    self.assertEqual(work["encoder_optimizer_steps"], 0 if method.startswith("capql") else 76)
                    if method in ("adaptive", "fixed"):
                        raw_path = root / env.replace("-", "_") / method / f"{experiment.METHODS[method]}_seed912346.json"
                        common.append(json.loads(raw_path.read_text()))
                    with patch.object(training, "train_method", side_effect=AssertionError("retrained")), \
                         patch.object(training, "train_capql", side_effect=AssertionError("retrained")):
                        self.assertEqual(experiment.run(env, method, 912346, root), report)
                self.assertEqual(common[0]["learning_work"], common[1]["learning_work"])
                self.assertEqual([r for r in common[0]["episode_rewards"] if r["step"] <= 23],
                                 [r for r in common[1]["episode_rewards"] if r["step"] <= 23])
            readout = experiment.summarize(root)
            self.assertEqual(readout["completed_budget"], dict(training_interactions=800,
                evaluation_interactions=10080, environment_initialization_probe_steps=20))
            path = root / "mo_hopper_v5" / "fixed" / "seed912346.json"
            row = json.loads(path.read_text())
            row["learning_work"]["critic_rows"] += 1
            experiment.write_report(path, row)
            with self.assertRaises(AssertionError):
                experiment.summarize(root)
        print("COMMON_TARGET_SMOKE training=800 evaluation=10080 probes=20")


if __name__ == "__main__":
    unittest.main()
