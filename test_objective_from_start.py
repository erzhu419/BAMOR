import copy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import torch

import run_objective_from_start as probe
from test_learning_credit_probe import toy_trainer


KWARGS = dict(hidden=8, lr=3e-4, gamma=.99, batch_size=4, learning_starts=5,
              eval_freq=100000, log_freq=10000, device="cpu")


class ObjectiveFromStartTests(unittest.TestCase):
    def test_cold_initialization_raw_identity_and_gaussian_behavior(self):
        soft = toy_trainer()
        torch.manual_seed(7)
        raw = probe.MeanRawTrainer(3, 2, 2, hidden=8, buffer_capacity=16,
                                   batch_size=4, device="cpu")
        probe.base.same_learner(soft, raw)
        self.assertFalse(raw.policy_opt.state or raw.q_opt.state)
        raw.buffer = copy.deepcopy(soft.buffer)
        old_std = {name: value.detach().clone() for name, value in raw.policy.named_parameters()
                   if name.startswith("log_std_head")}
        with probe.base.isolated_rng(510):
            self.assertEqual(raw.update(), (None, None))
        with probe.base.isolated_rng(510):
            probe.local.raw_update(soft)
        probe.base.same_learner(soft, raw)
        self.assertEqual(raw.objective_updates, 1)
        self.assertEqual(probe.training.optimizer_step_count(raw.policy_opt), 1)
        for name, value in raw.policy.named_parameters():
            if name in old_std:
                self.assertTrue(torch.equal(value, old_std[name]))
                self.assertIsNone(value.grad)
                self.assertNotIn(value, raw.policy_opt.state)
        state, context = torch.zeros(1, 3), torch.full((1, 2), .5)
        with torch.no_grad(), probe.base.isolated_rng(512):
            action, log_prob = raw.select_action(state, context)
            second, _ = raw.select_action(state, context)
            mean = raw.policy.get_action(state, context, deterministic=True)
        self.assertTrue(torch.isfinite(action).all() and torch.isfinite(log_prob).all())
        self.assertFalse(torch.equal(action, second) or torch.equal(action, mean))
        print("FROM_START_STANDALONE actor_optimizer_steps=2 critic_optimizer_steps=2")

    def test_default_training_entry_matches_explicit_soft_factory(self):
        factory = probe.training.make_piecewise_mo_env

        def short_env(*args, **kwargs):
            kwargs.update(switch_interval=12, mode_order="sequential")
            env = factory(*args, **kwargs)
            env.env._max_episode_steps = 4
            return env

        states = []
        with tempfile.TemporaryDirectory() as directory, \
             patch.object(probe.training, "make_piecewise_mo_env", side_effect=short_env), \
             patch("builtins.print"):
            try:
                for injected in (False, True):
                    capture = probe.base.Capture()
                    capture.step = 22
                    options = dict(trainer_factory=probe.TRAINERS["soft"]) if injected else {}
                    probe.training.train_method("mo-hopper-v5", probe.METHOD, 22, 12, 91300,
                        directory, state_probe=capture, **options, **KWARGS)
                    states.append(capture.state)
                probe.base.same_learner(states[0]["trainer"], states[1]["trainer"])
                self.assertEqual(states[0]["env"].mode_history, states[1]["env"].mode_history)
                self.assertEqual(states[0]["trainer"].update_work["actor_optimizer_steps"], 18)
                for key, value in states[0]["encoder"].state_dict().items():
                    self.assertTrue(torch.equal(value, states[1]["encoder"].state_dict()[key]))
                self.assertEqual(probe.training.optimizer_step_count(states[0]["encoder_opt"]), 18)
            finally:
                for state in states:
                    state["env"].close()
                    state["eval_env"].close()
        print("FROM_START_DEFAULT_IDENTITY training_interactions=44 initialization_probes=4 "
              "actor_optimizer_steps=36 critic_optimizer_steps=36 encoder_optimizer_steps=36")

    def test_real_both_environments_first_update_budget_summary_and_reuse(self):
        factory, evaluate = probe.training.make_piecewise_mo_env, probe.base.evaluate
        costs = dict.fromkeys(probe.budget(probe.ENVIRONMENTS[0]), 0)
        evaluated = []
        root = None

        def short_env(*args, **kwargs):
            kwargs.update(switch_interval=12, mode_order="sequential")
            env = factory(*args, **kwargs)
            env.env._max_episode_steps = 4
            return env

        def checked_evaluate(state, trainer, seed):
            condition = "mean_raw" if isinstance(trainer, probe.MeanRawTrainer) else "soft"
            env = state["run_config"]["env_name"]
            record = json.loads((root / env.replace("-", "_") / condition / f"seed{seed}.json").read_text())
            self.assertTrue(record["cold_initialization_passed"])
            self.assertEqual(record["first_update_step"], 5)
            self.assertEqual(state["env"]._total_steps, 22)
            self.assertEqual(record["optimizer_clocks"], dict(actor=18, critic=18, encoder=18))
            self.assertEqual(record["raw_objective_updates"], 18 if condition == "mean_raw" else 0)
            evaluated.append((env, condition))
            return evaluate(state, trainer, seed)

        with tempfile.TemporaryDirectory() as directory, \
             patch.multiple(probe, TOTAL_STEPS=22, SWITCH_INTERVAL=12, KWARGS=KWARGS, SEEDS=(91301,)), \
             patch.multiple(probe.base, REPEATS=1, HORIZON=4), \
             patch.object(probe.training, "make_piecewise_mo_env", side_effect=short_env), \
             patch.object(probe.base, "evaluate", side_effect=checked_evaluate), patch("builtins.print"):
            root = Path(directory)
            probe.write_report(root / "protocol.json", dict(protocol=probe.PROTOCOL, experiment=probe.specification()))
            records = {}
            for env in probe.ENVIRONMENTS:
                for condition in probe.CONDITIONS:
                    record = probe.run(env, condition, 91301, root)
                    records[env, condition] = record
                    for key, value in record["completed_budget"].items():
                        costs[key] += value
                    with patch.object(probe.training, "train_method", side_effect=AssertionError("retrained")):
                        self.assertEqual(probe.run(env, condition, 91301, root), record)
            self.assertEqual(len(evaluated), 4)
            self.assertEqual(costs["training_interactions"], 88)
            self.assertEqual(costs["environment_initialization_probe_steps"], 8)
            self.assertEqual(costs["actor_optimizer_steps"], 72)
            self.assertLessEqual(costs["evaluation_interactions"], 1344)
            summary = probe.summarize(root)
            self.assertEqual(summary["completed_budget"], costs)
            for env, result in summary["environments"].items():
                for metric in ("utility", "hv"):
                    delta = records[env, "mean_raw"]["endpoint"][metric] - records[env, "soft"]["endpoint"][metric]
                    self.assertEqual(result["mean_raw_minus_soft"][metric]["mean"], delta)
                delta = result["mean_raw_minus_soft"]
                expected = delta["utility"]["mean"] > 0 and delta["hv"]["mean"] >= 0
                self.assertEqual(result["robust_lead_screen_passed"], expected)
            bad = copy.deepcopy(records[probe.ENVIRONMENTS[0], "mean_raw"])
            bad["raw_objective_updates"] -= 1
            with self.assertRaises(AssertionError):
                probe.check_record(bad)
        print("FROM_START_PIPELINE", json.dumps(costs))


if __name__ == "__main__":
    unittest.main()
