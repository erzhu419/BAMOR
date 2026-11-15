import copy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import torch

import run_projected_allocation_pilot as experiment


training = experiment.training


class ProjectedAllocationTests(unittest.TestCase):
    def test_registration_and_scores_only_affect_collection(self):
        adaptive = training.CommonTargetSamplingSelector(3)
        fixed = training.FixedCommonTargetSamplingSelector(3)
        for selector in (adaptive, fixed):
            selector.counts = dict.fromkeys(selector.operators, 1)
            selector.scores = dict(uniform=-1., stratified=1., mixed=0.)
        self.assertGreater(adaptive.probabilities()["stratified"], .8)
        self.assertEqual(fixed.probabilities(), dict.fromkeys(fixed.operators, 1 / 3))
        self.assertEqual(adaptive.state_dict()["operator_trainer_configs"], fixed.state_dict()["operator_trainer_configs"])
        for method in (experiment.METHODS["adaptive"], experiment.METHODS["fixed"]):
            self.assertIn(method, training.COMMON_TARGET_METHODS)
            self.assertTrue(training.uses_learned_context(method))
            self.assertTrue(training.uses_transition_context(method))
            self.assertFalse(training.uses_envelope_adaptive_operator_selection(method))
            self.assertFalse(training.uses_policy_envelope_adaptive_operator_selection(method))
            self.assertEqual(training.context_dim_for_method(method, 17, 6, 2, 4), 13)

    def test_identical_same_replay_updates_across_allocation_and_operators(self):
        for dim in (2, 3):
            trainer = training.ProjectedCommonTargetMOSACTrainer(3, 2, dim, hidden=8,
                buffer_capacity=16, batch_size=4, device="cpu")
            for i in range(8):
                state = np.full(3, i / 10, dtype=np.float32)
                weights = np.full(dim, 1 / dim, dtype=np.float32)
                trainer.buffer.push(state, np.zeros(2), weights, weights, np.ones(dim), state + .1, weights, i % 3 == 0)
            states = []
            for cls in (training.CommonTargetSamplingSelector, training.FixedCommonTargetSamplingSelector):
                selector = cls(dim)
                for op in selector.operators:
                    branch = copy.deepcopy(trainer)
                    for key, value in selector.trainer_config(op).items():
                        setattr(branch, key, value)
                    branch.preference_relabeling_rate = selector.relabel_rate(op)
                    np.random.seed(834)
                    torch.manual_seed(834)
                    mean_q, target = branch.update()
                    self.assertTrue(np.isfinite(mean_q) and np.isfinite(target).all())
                    self.assertEqual(branch.update_work, dict(critic_optimizer_steps=1, actor_optimizer_steps=1,
                        critic_rows=4, target_candidate_rows=4, actor_candidate_rows=4))
                    states.append({name: copy.deepcopy(getattr(branch, name).state_dict())
                                   for name in ("policy", "q1", "q2", "q1_target", "q2_target")})
            for state in states[1:]:
                self.assertTrue(all(torch.equal(value, state[name][key])
                    for name, values in states[0].items() for key, value in values.items()))

    def test_short_three_condition_routes_and_matched_readout(self):
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

        with tempfile.TemporaryDirectory() as folder, \
             patch.multiple(experiment, SEEDS=(912348,), COMMON_KWARGS=kwargs, DIAGNOSTIC_START=40), \
             patch.object(experiment.previous, "TOTAL_STEPS", 80), \
             patch.object(experiment.previous.pilot, "TRAIN_KWARGS", kwargs), \
             patch.object(training, "make_piecewise_mo_env", side_effect=short_env), \
             patch.object(training.CommonTargetSamplingSelector, "__init__", new=short_blocks), patch("builtins.print"):
            root = Path(folder)
            for env in experiment.ENVIRONMENTS:
                pair = []
                for method in experiment.METHODS:
                    report = experiment.run(env, method, 912348, root)
                    self.assertEqual(report["completed_budget"], dict(training_interactions=80,
                        evaluation_interactions=1008, environment_initialization_probe_steps=2))
                    self.assertEqual(report["learning_work"]["critic_optimizer_steps"], 76)
                    if method != "v25":
                        self.assertEqual(report["common_target_learning"], experiment.LEARNING_CONTRACT)
                        raw_path = root / env.replace("-", "_") / method / f"{experiment.METHODS[method]}_seed912348.json"
                        raw = json.loads(raw_path.read_text())
                        pair.append(raw)
                        raw["adaptive_operator_final"]["selection_uses_scores"] = method != "adaptive"
                        with self.assertRaises(AssertionError):
                            experiment.validate(raw, env, method, 912348, raw_path.with_name(
                                f"{experiment.METHODS[method]}_seed912348_model.pt"))
                    with patch.object(training, "train_method", side_effect=AssertionError("retrained")):
                        self.assertEqual(experiment.run(env, method, 912348, root), report)
                self.assertEqual(pair[0]["learning_work"], pair[1]["learning_work"])
                self.assertEqual([r for r in pair[0]["episode_rewards"] if r["step"] <= 23],
                                 [r for r in pair[1]["episode_rewards"] if r["step"] <= 23])
            readout = experiment.summarize(root)
            self.assertEqual(readout["completed_budget"], dict(training_interactions=480,
                evaluation_interactions=6048, environment_initialization_probe_steps=12))
            path = root / "mo_hopper_v5" / "fixed" / "seed912348.json"
            saved = json.loads(path.read_text())
            changed = copy.deepcopy(saved)
            changed["learning_work"]["critic_rows"] += 1
            experiment.write_report(path, changed)
            with self.assertRaises(AssertionError):
                experiment.summarize(root)
            changed = copy.deepcopy(saved)
            changed["common_target_learning"]["critic_loss_rule"] = "mean_twin_full_vector_mse"
            experiment.write_report(path, changed)
            with self.assertRaises(AssertionError):
                experiment.summarize(root)
        print("PROJECTED_ALLOCATION_SMOKE training=480 evaluation=6048 probes=12")


if __name__ == "__main__":
    unittest.main()
