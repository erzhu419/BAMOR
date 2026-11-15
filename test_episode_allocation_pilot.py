import copy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import torch

import run_episode_allocation_pilot as experiment


training = experiment.training


class EpisodeAllocationTests(unittest.TestCase):
    def test_full_episode_return_and_causal_baseline(self):
        selector = training.EpisodeReturnSamplingSelector(2, learning_starts=1)
        selector.baseline_ready = True
        selector.return_mean, selector.return_var = 100., 400.
        for step in range(2, 12):
            operator = selector.select_for_step(step)
            self.assertEqual(operator, "uniform")
            selector.observe_transition(operator, [20., 20.], step, step == 11)
            if step < 11:
                self.assertEqual(sum(selector.counts.values()), 0)
                self.assertEqual(selector.return_mean, 100.)
        record = selector.episode_history[-1]
        self.assertEqual(record["utility"], 200.)
        self.assertEqual(record["baseline_before"], 100.)
        self.assertAlmostEqual(record["score"], np.tanh(5.))
        self.assertEqual(selector.select_for_step(12), "stratified")

        shorter = training.EpisodeReturnSamplingSelector(2, learning_starts=1)
        shorter.baseline_ready, shorter.return_mean, shorter.return_var = True, 100., 400.
        operator = shorter.select_for_step(2)
        shorter.observe_transition(operator, [20., 20.], 2, True)
        self.assertLess(shorter.episode_history[-1]["score"], 0)

    def test_change_reset_drops_mixed_episode_and_partial_return(self):
        selector = training.EpisodeReturnSamplingSelector(2, learning_starts=1)
        for step in range(1, 2501):
            operator = selector.select_for_step(step)
            selector.observe_transition(operator, [1., 1.], step, step % 100 == 0)
        selector.scores = dict.fromkeys(selector.operators, .7)
        for step in range(2501, 2507):
            operator = selector.select_for_step(step)
            selector.observe_transition(operator, [1000., 1000.], step, step == 2506)
        self.assertEqual(selector.epoch, 1)
        self.assertTrue(selector.episode_history[-1]["mixed_epoch"])
        self.assertFalse(selector.episode_history[-1]["credited"])
        self.assertFalse(selector.baseline_ready)
        self.assertEqual(selector.scores, dict.fromkeys(selector.operators, 0.))
        operator = selector.select_for_step(2507)
        selector.observe_transition(operator, [1000., 1000.], 2507, False)
        self.assertEqual(selector.state_dict()["partial_episode_length"], 1)
        self.assertEqual(selector.segment_counts, dict.fromkeys(selector.operators, 0))

    def test_registration_and_fixed_ignores_scores(self):
        adaptive = training.EpisodeReturnSamplingSelector(3)
        fixed = training.FixedEpisodeReturnSamplingSelector(3)
        for selector in (adaptive, fixed):
            selector.scores = dict(uniform=-1., stratified=1., mixed=0.)
        self.assertGreater(adaptive.probabilities()["stratified"], .8)
        self.assertEqual(fixed.probabilities(), dict.fromkeys(fixed.operators, 1 / 3))
        for method in (experiment.METHODS["adaptive"], experiment.METHODS["fixed"]):
            self.assertIn(method, training.METHODS)
            self.assertTrue(training.uses_learned_context(method))
            self.assertTrue(training.uses_transition_context(method))
            self.assertFalse(training.uses_block_adaptive_operator_selection(method))
            self.assertFalse(training.uses_policy_envelope_adaptive_operator_selection(method))
            self.assertEqual(training.context_dim_for_method(method, 17, 6, 2, 4), 13)

    def test_same_replay_updates_do_not_depend_on_selector_or_operator(self):
        for dim in (2, 3):
            trainer = training.ProjectedCommonTargetMOSACTrainer(3, 2, dim, hidden=8,
                buffer_capacity=16, batch_size=4, device="cpu")
            for i in range(8):
                state = np.full(3, i / 10, dtype=np.float32)
                weight = np.full(dim, 1 / dim, dtype=np.float32)
                trainer.buffer.push(state, np.zeros(2), weight, weight, np.ones(dim), state + .1, weight, i % 3 == 0)
            states = []
            for cls in (training.EpisodeReturnSamplingSelector, training.FixedEpisodeReturnSamplingSelector):
                selector = cls(dim)
                for op in selector.operators:
                    branch = copy.deepcopy(trainer)
                    for key, value in selector.trainer_config(op).items():
                        setattr(branch, key, value)
                    branch.preference_relabeling_rate = selector.relabel_rate(op)
                    np.random.seed(834)
                    torch.manual_seed(834)
                    branch.update()
                    self.assertEqual(branch.update_work["critic_optimizer_steps"], 1)
                    states.append({name: copy.deepcopy(getattr(branch, name).state_dict())
                        for name in ("policy", "q1", "q2", "q1_target", "q2_target")})
            for state in states[1:]:
                self.assertTrue(all(torch.equal(value, state[name][key])
                    for name, values in states[0].items() for key, value in values.items()))

    def test_real_routes_feedback_budget_and_paired_readout(self):
        original_env = training.make_piecewise_mo_env
        kwargs = dict(hidden=8, lr=3e-4, gamma=.99, batch_size=4, learning_starts=5,
                      eval_freq=100000, log_freq=10000, device="cpu")

        def short_env(*args, **kw):
            kw.update(switch_interval=12, mode_order="sequential")
            env = original_env(*args, **kw)
            env.env._max_episode_steps = 4
            return env

        contract = dict(experiment.SELECTOR_CONTRACT, learning_starts=5)
        with tempfile.TemporaryDirectory() as folder, \
             patch.multiple(experiment, SEEDS=(912349,), COMMON_KWARGS=kwargs, SELECTOR_CONTRACT=contract), \
             patch.object(experiment.previous, "TOTAL_STEPS", 80), \
             patch.object(experiment.previous.pilot, "TRAIN_KWARGS", kwargs), \
             patch.object(training, "make_piecewise_mo_env", side_effect=short_env), patch("builtins.print"):
            root = Path(folder)
            for env in experiment.ENVIRONMENTS:
                pair = []
                for method in experiment.METHODS:
                    report = experiment.run(env, method, 912349, root)
                    self.assertEqual(report["completed_budget"], dict(training_interactions=80,
                        evaluation_interactions=1008, environment_initialization_probe_steps=2))
                    self.assertEqual(report["learning_work"]["critic_optimizer_steps"], 76)
                    if method != "v25":
                        self.assertGreater(sum(report["allocation"]["counts"].values()), 3)
                        raw_path = root / env.replace("-", "_") / method / f"{experiment.METHODS[method]}_seed912349.json"
                        raw = json.loads(raw_path.read_text())
                        pair.append(raw)
                        raw["adaptive_operator_final"]["utility_rule"] = "return_per_step"
                        with self.assertRaises(AssertionError):
                            experiment.validate(raw, env, method, 912349, raw_path.with_name(
                                f"{experiment.METHODS[method]}_seed912349_model.pt"))
                    with patch.object(training, "train_method", side_effect=AssertionError("retrained")):
                        self.assertEqual(experiment.run(env, method, 912349, root), report)
                self.assertEqual(pair[0]["learning_work"], pair[1]["learning_work"])
                forced = [r for r in pair[0]["adaptive_operator_final"]["episode_history"] if r["credited"]][:3]
                self.assertEqual([r["operator"] for r in forced], ["uniform", "stratified", "mixed"])
                prefix_end = forced[-1]["end_step"]
                self.assertEqual([r for r in pair[0]["episode_rewards"] if r["step"] <= prefix_end],
                                 [r for r in pair[1]["episode_rewards"] if r["step"] <= prefix_end])
            readout = experiment.summarize(root)
            self.assertEqual(readout["completed_budget"], dict(training_interactions=480,
                evaluation_interactions=6048, environment_initialization_probe_steps=12))
            path = root / "mo_hopper_v5" / "fixed" / "seed912349.json"
            changed = json.loads(path.read_text())
            changed["learning_work"]["critic_rows"] += 1
            experiment.write_report(path, changed)
            with self.assertRaises(AssertionError):
                experiment.summarize(root)
        print("EPISODE_ALLOCATION_SMOKE training=480 evaluation=6048 probes=12")


if __name__ == "__main__":
    unittest.main()
