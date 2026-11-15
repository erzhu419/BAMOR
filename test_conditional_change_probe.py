import copy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import torch

import conditional_change_evidence as evidence_module
from conditional_change_evidence import ConditionalChangeEvidence
import run_conditional_change_probe as probe


class ConditionalChangeTests(unittest.TestCase):
    def test_prediction_uses_only_inputs_and_fitting_normalization(self):
        rng = np.random.default_rng(1)
        inputs = rng.normal(size=(100, 3)).astype(np.float32)
        targets = rng.normal(size=(100, 4)).astype(np.float32)
        evidence = ConditionalChangeEvidence(3, 4, 2, updates=4, window=10)
        work = evidence.fit(inputs, targets, 3)
        self.assertEqual(work["optimizer_steps"], 4)
        state = copy.deepcopy(evidence.model.state_dict())
        means = evidence.x_mean.copy(), evidence.y_mean.copy()
        predictions = evidence.predict(inputs)
        evidence.calibrate(inputs, targets * 100)
        evidence.evaluate(inputs, targets * -100)
        np.testing.assert_array_equal(evidence.predict(inputs), predictions)
        np.testing.assert_array_equal(evidence.x_mean, means[0])
        np.testing.assert_array_equal(evidence.y_mean, means[1])
        self.assertTrue(all(torch.equal(value, evidence.model.state_dict()[key]) for key, value in state.items()))
        self.assertTrue(all(not parameter.requires_grad for parameter in evidence.model.parameters()))

    def test_supported_error_and_unsupported_abstention(self):
        rng = np.random.default_rng(2)
        inputs = rng.normal(size=(100, 3)).astype(np.float32)
        targets = rng.normal(size=(100, 4)).astype(np.float32)
        evidence = ConditionalChangeEvidence(3, 4, 2, updates=4, window=10)
        evidence.fit(inputs, targets, 4)
        evidence.calibrate(inputs, targets)
        changed = evidence.evaluate(inputs, targets + 100)
        self.assertTrue(all(window["alarm"] for window in changed["windows"]))
        unsupported = evidence.evaluate(inputs + 100, targets + 100)
        self.assertTrue(all(not window["eligible"] and not window["alarm"] for window in unsupported["windows"]))
        self.assertTrue(all(window["statistic"] is None for window in unsupported["windows"]))

    def test_delay_and_availability_gate(self):
        windows = [dict(start_step=i + 1, end_step=i + 100, eligible=True, alarm=i == 500)
                   for i in range(0, 1000, 100)]
        result = probe.case_summary(dict(windows=windows, supported=np.ones(1000, bool)))
        self.assertEqual(result["first_post_alarm_delay"], 100)
        self.assertTrue(result["timely_alarm"])
        for window in windows[6:]:
            window["eligible"] = False
        result = probe.case_summary(dict(windows=windows, supported=np.ones(1000, bool)))
        self.assertFalse(result["available"])
        self.assertFalse(result["timely_alarm"])

    def test_screen_holds_for_missing_positive_coverage_or_inactive_control(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            for env in probe.ENVIRONMENTS:
                for seed in probe.SEEDS:
                    path = root / env.replace("-", "_") / f"seed{seed}" / "readout.json"
                    path.parent.mkdir(parents=True)
                    cases = {case: dict(available=True, pre_available=True, pre_alarm_windows=0,
                        any_post_alarm=case.startswith("mode"),
                        timely_alarm=case.startswith("mode"), intervention_active=True) for case in probe.CASES}
                    probe.write(path, dict(protocol=probe.PROTOCOL, env=env, seed=seed, complete=True,
                        predictor_config=probe.CONFIG, completed_budget=probe.budget(),
                        paired_prefixes_verified=5, policy_optimizer_steps=0, critic_optimizer_steps=0,
                        encoder_optimizer_steps=0, predictor_work=dict(optimizer_steps=500), cases=cases))
            self.assertTrue(probe.summarize(root)["screen_passed"])
            for seed in probe.SEEDS[:2]:
                path = root / "mo_hopper_v5" / f"seed{seed}" / "readout.json"
                row = json.loads(path.read_text())
                row["cases"]["mode1"].update(available=False, timely_alarm=False)
                probe.write(path, row)
            self.assertEqual(probe.summarize(root)["decision"], "hold_conditional_evidence_integration")
            for seed in probe.SEEDS[:2]:
                path = root / "mo_hopper_v5" / f"seed{seed}" / "readout.json"
                row = json.loads(path.read_text())
                row["cases"]["mode1"].update(available=True, timely_alarm=True)
                row["cases"]["actor_change"]["intervention_active"] = False
                probe.write(path, row)
            self.assertFalse(probe.summarize(root)["screen_passed"])
            for seed in probe.SEEDS[:2]:
                path = root / "mo_hopper_v5" / f"seed{seed}" / "readout.json"
                row = json.loads(path.read_text())
                row["cases"]["actor_change"]["intervention_active"] = True
                row["cases"]["stationary"]["pre_alarm_windows"] = 1
                probe.write(path, row)
            self.assertFalse(probe.summarize(root)["screen_passed"])

    def test_real_routes_pairing_data_budget_reuse_and_readout(self):
        training = probe.training
        config = {**evidence_module.CONFIG, "updates": 8, "window": 20}
        with tempfile.TemporaryDirectory() as tmp, \
             patch.multiple(probe, SEEDS=(2101,), SOURCE_HIDDEN=8, FIT_STEPS=300,
                            CALIBRATION_STEPS=200, CASE_STEPS=200, CHANGE_AT=100,
                            MINIMUM_POST_WINDOWS=3, MAXIMUM_DELAY=60, CONFIG=config), \
             patch.dict(evidence_module.CONFIG, config):
            root, source = Path(tmp) / "output", Path(tmp) / "source"
            for env_name in probe.ENVIRONMENTS:
                env = training.make_piecewise_mo_env(env_name, switch_interval=10**9, seed=991)
                try:
                    obs_dim, act_dim, reward_dim = env.observation_space.shape[0], env.action_space.shape[0], env.reward_dim
                    context_dim = training.context_dim_for_method(probe.METHODS["fixed"], obs_dim, act_dim, reward_dim, env.num_modes)
                    for index, (name, method) in enumerate(probe.METHODS.items()):
                        torch.manual_seed(991 + index)
                        policy = training.PolicyNet(obs_dim, act_dim, context_dim, hidden=8)
                        encoder = training.LatentTransitionEncoder(obs_dim, act_dim, reward_dim, hidden=8,
                                                                  latent_dim=training.LATENT_CONTEXT_DIM)
                        folder = source / env_name.replace("-", "_") / name
                        folder.mkdir(parents=True)
                        (folder / f"{method}_seed2101.json").write_text(json.dumps(
                            dict(env=env_name, method=method, seed=2101, total_steps=75000)))
                        torch.save(dict(policy=policy.state_dict(), encoder=encoder.state_dict()),
                                   folder / f"{method}_seed2101_model.pt")
                finally:
                    env.close()
                with patch("builtins.print"):
                    report = probe.run(env_name, 2101, root, source)
                self.assertTrue(report["complete"])
                self.assertEqual(report["paired_prefixes_verified"], 5)
                self.assertEqual(report["predictor_work"]["optimizer_steps"], 8)
                self.assertEqual(report["completed_budget"], dict(fit_interactions=300,
                    calibration_interactions=200, controlled_probe_interactions=1200,
                    environment_initialization_probe_steps=7))
                self.assertTrue(report["cases"]["sampler_change"]["intervention_active"])
                self.assertTrue(report["cases"]["actor_change"]["intervention_active"])
                for case in probe.CASES[3:]:
                    self.assertEqual(report["cases"][case]["ground_truth_mode_history_audit_only"], [[0, 0], [100, int(case[-1])]])
                dataset = root / env_name.replace("-", "_") / "seed2101" / "transitions.npz"
                with np.load(dataset) as data:
                    self.assertEqual(data["prefix_inputs"].shape[0], 500)
                    self.assertEqual(data["actor_change_inputs"].shape[0], 200)
                with patch.object(training, "make_piecewise_mo_env", side_effect=AssertionError("re-collected")):
                    self.assertEqual(probe.run(env_name, 2101, root, source), report)
            readout = probe.summarize(root)
            self.assertEqual(readout["completed_budget"]["controlled_probe_interactions"], 2400)
            self.assertEqual(readout["predictor_optimizer_steps"], 16)
            self.assertFalse(readout["screen_passed"])
            path = root / "mo_hopper_v5" / "seed2101" / "readout.json"
            report = json.loads(path.read_text())
            report["complete"] = False
            probe.write(path, report)
            with self.assertRaises(RuntimeError):
                probe.run("mo-hopper-v5", 2101, root, source)
        print("CONDITIONAL_PROBE_SMOKE interactions=3400 initialization_probes=16 predictor_updates=24 policy_updates=0")


if __name__ == "__main__":
    torch.set_num_threads(2)
    unittest.main()
