import copy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import torch

import run_episode_forecast_pilot as experiment
from episode_credit_blend import EpisodeCreditBlend


training = experiment.training


def selector_for(method, dim=2, starts=0):
    return training.ForecastEpisodeReturnSamplingSelector(dim, starts,
        credit_rule=experiment.CREDIT_RULES[method], adaptive=method != "fixed")


class EpisodeForecastPilotTests(unittest.TestCase):
    def test_full_return_frozen_forecast_and_past_weight_round_trip(self):
        selector = selector_for("past_mix")
        reference = EpisodeCreditBlend(.13)
        step = 1
        for utility, length in zip((0., 10., 20., 30., 0., -5., 20.), (3, 1, 5, 2, 6, 1, 4)):
            before = reference.predict(step)
            start = step
            operator = selector.select_for_step(step)
            frozen = copy.deepcopy(selector.forecast_before)
            for index in range(length):
                self.assertEqual(selector.select_for_step(step), operator)
                selector.observe_transition(operator, [utility / length] * 2, step, index == length - 1)
                self.assertEqual(selector.forecast_before, frozen)
                step += 1
            record = selector.episode_history[-1]
            self.assertAlmostEqual(record["utility"], utility)
            self.assertEqual(record["forecast_before"], before)
            reference.observe(start, record["utility"], before, False)
        contract = dict(experiment.SELECTOR_CONTRACT, learning_starts=0)
        with patch.object(experiment, "SELECTOR_CONTRACT", contract):
            experiment.validate_forecasts(selector.state_dict(), "past_mix")

    def test_shared_three_episode_readiness_and_zero_dispersion(self):
        for method in experiment.METHODS:
            selector = selector_for(method)
            for step, utility in enumerate((0., 0., 0., 1., 2.), 1):
                operator = selector.select_for_step(step)
                selector.observe_transition(operator, [utility] * 2, step, True)
            self.assertEqual([r["credited"] for r in selector.episode_history], [False] * 4 + [True])
            self.assertGreater(selector.episode_history[-1]["scale_before"], 0.)
            selector = selector_for(method)
            for step, utility in enumerate((0., 2., 4., 6.), 1):
                operator = selector.select_for_step(step)
                selector.observe_transition(operator, [utility] * 2, step, True)
            self.assertEqual([r["credited"] for r in selector.episode_history], [False] * 3 + [True])

    def test_reset_in_final_partial_episode_is_replayed(self):
        selector = selector_for("past_mix")
        for step in range(1, 6):
            operator = selector.select_for_step(step)
            selector.observe_transition(operator, [float(step)] * 2, step, True)
        operator = selector.select_for_step(6)
        with patch.object(selector.belief, "update", return_value={"mode_switched": True}):
            selector.observe_transition(operator, [1000.] * 2, 6, False)
        self.assertEqual(selector.epoch, 1)
        self.assertEqual(selector.forecaster.comparisons, 0)
        self.assertEqual(selector.state_dict()["partial_episode_length"], 1)
        contract = dict(experiment.SELECTOR_CONTRACT, learning_starts=0)
        with patch.object(experiment, "SELECTOR_CONTRACT", contract):
            experiment.validate_forecasts(selector.state_dict(), "past_mix")

    def test_detected_crossing_excluded_without_changing_frozen_prediction(self):
        selector = selector_for("past_mix")
        for step in range(1, 7):
            operator = selector.select_for_step(step)
            selector.observe_transition(operator, [float(step)] * 2, step, True)
        operator = selector.select_for_step(7)
        frozen = copy.deepcopy(selector.forecast_before)
        with patch.object(selector.belief, "update", return_value={"mode_switched": True}):
            selector.observe_transition(operator, [1000.] * 2, 7, True)
        record = selector.episode_history[-1]
        self.assertTrue(record["mixed_epoch"])
        self.assertFalse(record["credited"])
        self.assertEqual(record["forecast_before"], frozen)
        self.assertEqual(selector.forecaster.baseline.scale.count, 0)
        self.assertEqual(selector.scores, dict.fromkeys(selector.operators, 0.))

    def test_registration_fixed_blindness_and_legacy_methods_retained(self):
        for method in experiment.METHODS:
            mapped = experiment.METHODS[method]
            self.assertIn(mapped, training.METHODS)
            self.assertTrue(training.uses_learned_context(mapped))
            self.assertTrue(training.uses_transition_context(mapped))
            self.assertFalse(training.uses_block_adaptive_operator_selection(mapped))
            self.assertEqual(training.context_dim_for_method(mapped, 17, 6, 2, 4), 13)
        fixed = selector_for("fixed")
        fixed.scores = dict(uniform=-1., stratified=1., mixed=0.)
        self.assertEqual(fixed.probabilities(), dict.fromkeys(fixed.operators, 1 / 3))
        adaptive = selector_for("past_mix")
        adaptive.scores = dict(fixed.scores)
        self.assertGreater(adaptive.probabilities()["stratified"], .8)
        self.assertNotIn("bamor_v25_progress_archive_mix", training.FORECAST_EPISODIC_METHODS)

    def test_same_replay_updates_for_all_credit_rules_and_operators(self):
        for dim in (2, 3):
            trainer = training.ProjectedCommonTargetMOSACTrainer(3, 2, dim, hidden=8,
                buffer_capacity=16, batch_size=4, device="cpu")
            for i in range(8):
                state = np.full(3, i / 10, dtype=np.float32)
                weight = np.full(dim, 1 / dim, dtype=np.float32)
                trainer.buffer.push(state, np.zeros(2), weight, weight, np.ones(dim), state + .1, weight, i % 3 == 0)
            expected = None
            for method in experiment.METHODS:
                selector = selector_for(method, dim)
                for op in selector.operators:
                    branch = copy.deepcopy(trainer)
                    for key, value in selector.trainer_config(op).items():
                        setattr(branch, key, value)
                    branch.preference_relabeling_rate = selector.relabel_rate(op)
                    np.random.seed(834)
                    torch.manual_seed(834)
                    branch.update()
                    state = {name: copy.deepcopy(getattr(branch, name).state_dict())
                        for name in ("policy", "q1", "q2", "q1_target", "q2_target")}
                    if expected is None:
                        expected = state
                    self.assertTrue(all(torch.equal(value, state[name][key])
                        for name, values in expected.items() for key, value in values.items()))

    def test_real_routes_paired_prefix_budget_validation_reuse_and_compact_summary(self):
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
             patch.multiple(experiment, SEEDS=(912350,), COMMON_KWARGS=kwargs, SELECTOR_CONTRACT=contract), \
             patch.object(experiment.previous.previous, "TOTAL_STEPS", 80), \
             patch.object(training, "make_piecewise_mo_env", side_effect=short_env), patch("builtins.print"):
            root = Path(folder)
            for env in experiment.ENVIRONMENTS:
                reports = []
                for method in experiment.METHODS:
                    report = experiment.run(env, method, 912350, root)
                    self.assertEqual(report["completed_budget"], dict(training_interactions=80,
                        evaluation_interactions=1008, environment_initialization_probe_steps=2))
                    self.assertEqual(report["learning_work"]["critic_optimizer_steps"], 76)
                    self.assertGreater(report["allocation"]["counts"]["uniform"], 0)
                    self.assertIsNotNone(report["first_free_start_step"])
                    reports.append(report)
                    raw_path = root / env.replace("-", "_") / method / f"{experiment.METHODS[method]}_seed912350.json"
                    raw = json.loads(raw_path.read_text())
                    changed = copy.deepcopy(raw)
                    changed["adaptive_operator_final"]["episode_history"][-1]["credit_trend_weight"] += .1
                    with self.assertRaises(AssertionError):
                        experiment.validate(changed, env, method, 912350, raw_path.with_name(
                            f"{experiment.METHODS[method]}_seed912350_model.pt"))
                    with patch.object(training, "train_method", side_effect=AssertionError("retrained")):
                        self.assertEqual(experiment.run(env, method, 912350, root), report)
                for report in reports[1:]:
                    self.assertEqual(report["paired_prefix"], reports[0]["paired_prefix"])
                    self.assertEqual(report["first_free_start_step"], reports[0]["first_free_start_step"])
                    self.assertEqual(report["learning_work"], reports[0]["learning_work"])
            readout = experiment.summarize(root)
            self.assertEqual(readout["completed_budget"], dict(training_interactions=640,
                evaluation_interactions=8064, environment_initialization_probe_steps=16))
            self.assertNotIn("episode_history", json.dumps(readout))
            self.assertEqual(json.loads(json.dumps(readout, allow_nan=False)), readout)
            for env in experiment.ENVIRONMENTS:
                for method, utility in dict(fixed=10., ema=15., equal_mix=25., past_mix=20.).items():
                    path = root / env.replace("-", "_") / method / "seed912350.json"
                    record = json.loads(path.read_text())
                    record["metrics"]["direct_utility"] = dict(mean=utility, per_mode=[utility] * 4)
                    record["metrics"]["final_hv"] = 10.
                    experiment.write_report(path, record)
            readout = experiment.summarize(root)
            self.assertTrue(readout["allocation_gate_passed"])
            self.assertFalse(readout["weight_adaptation_gate_passed"])
            self.assertFalse(readout["adaptive_weight_claim_supported"])
            path = root / "mo_hopper_v5" / "fixed" / "seed912350.json"
            changed = json.loads(path.read_text())
            changed["learning_work"]["critic_rows"] += 1
            experiment.write_report(path, changed)
            with self.assertRaises(AssertionError):
                experiment.summarize(root)
        print("EPISODE_FORECAST_SMOKE training=640 evaluation=8064 probes=16 standalone_updates=24")


if __name__ == "__main__":
    unittest.main()
