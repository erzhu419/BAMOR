import copy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import torch

import run_scoped_extreme_pilot as experiment
import summarize_scoped_extreme_pilot as summary
from piecewise_mo_env import PiecewiseStationaryMOEnv


ARMS = (experiment.TARGETS[0], "coverage:edge_relabel_envelope_policy_bc",
        experiment.TARGETS[1], "recovery:uniform_relabel_envelope_policy")


class ScopedExtremeTests(unittest.TestCase):
    def test_live_arm_scope_native_settings_gradients_and_random_draws(self):
        training = experiment.training
        torch.manual_seed(3)
        base = training.ConsistentTargetMOSACTrainer(3, 2, 2, hidden=8, batch_size=4,
                                                    buffer_capacity=8, device="cpu")
        selector = training.FixedProgressExtremeHierarchicalOperatorSelector(2)
        trainer, trace = copy.deepcopy(base), []
        router = experiment.ScopedRouter("own_matched", trace, cut=1, end=5, selector=selector)
        states, prefs = torch.ones(4, 3), torch.tensor([[.4, .6]]).repeat(4, 1)
        router.configure(1, trainer)
        self.assertNotIn("envelope_actor_loss", trainer.__dict__)
        try:
            for step, op in enumerate(ARMS, start=2):
                selector.current_operator, selector.block_start_step = op, step
                config = {**selector.trainer_config(op), "preference_relabeling_rate": selector.relabel_rate(op)}
                reference = copy.deepcopy(base)
                for k, v in config.items():
                    setattr(trainer, k, v)
                    setattr(reference, k, v)
                np.random.seed(19)
                torch.manual_seed(19)
                target = (experiment.matched.own_candidate_loss(reference, states, prefs, prefs)
                          if op in experiment.TARGETS else reference.envelope_actor_loss(states, prefs, prefs))
                gradients = torch.autograd.grad(target[2], tuple(reference.policy.parameters()))
                n, t = np.random.get_state(), torch.get_rng_state()
                trace.append((n, t))
                np.random.seed(19)
                torch.manual_seed(19)
                router.configure(step, trainer)
                actual = trainer.envelope_actor_loss(states, prefs, prefs)
                for left, right in zip(actual, target):
                    torch.testing.assert_close(left, right, rtol=0, atol=0)
                for left, right in zip(torch.autograd.grad(actual[2], tuple(trainer.policy.parameters())), gradients):
                    torch.testing.assert_close(left, right, rtol=0, atol=0)
                router.after_step(step, trainer, None, None, op)
                self.assertEqual("envelope_actor_loss" in trainer.__dict__, op in experiment.TARGETS)
                self.assertEqual({k: getattr(trainer, k) for k in config}, config)
            record = router.state_dict()
            self.assertEqual(record["own_selected_updates"], 2)
            self.assertTrue(record["rng_alignment"]["passed"])
            self.assertEqual([r["own_selected"] for r in record["blocks"]], [True, False, True, False])
            trainer.behavior_clone_coef += 1
            with self.assertRaises(AssertionError):
                router.configure(5, trainer)
        finally:
            router.close()
        self.assertNotIn("envelope_actor_loss", trainer.__dict__)

    def test_real_full_warmup_checkpoint_control_and_scoped_continuation(self):
        training = experiment.training
        make_env, trainer_cls = training.make_piecewise_mo_env, training.ConsistentTargetMOSACTrainer
        original_init = training.ProgressExtremeHierarchicalOperatorSelector.__init__
        original_select = training.FixedProgressExtremeHierarchicalOperatorSelector.select
        original_step = PiecewiseStationaryMOEnv.step
        cost = dict(training=0, evaluation=0, initialization=0)

        def short_env(*args, **kwargs):
            env = make_env(*args, **kwargs)
            env.env._max_episode_steps = 6
            env.smoke_kind = "training" if kwargs["seed"] == 7 else "evaluation"
            cost["initialization"] += 1
            return env

        def counted_step(env, action):
            value = original_step(env, action)
            cost[env.smoke_kind] += 1
            return value

        def short_blocks(self, reward_dim, block_steps=1500, learning_starts=5000):
            original_init(self, reward_dim, block_steps=4, learning_starts=learning_starts)

        def fixed_test_sequence(self):
            index = len(self.block_history) - 10
            return original_select(self) if index < 0 else ARMS[index]

        try:
            with (tempfile.TemporaryDirectory() as directory,
                  patch.object(training, "make_piecewise_mo_env", side_effect=short_env),
                  patch.object(PiecewiseStationaryMOEnv, "step", new=counted_step),
                  patch.object(training, "ConsistentTargetMOSACTrainer",
                               side_effect=lambda *a, **kw: trainer_cls(*a, buffer_capacity=128, **kw)),
                  patch.object(training.ProgressExtremeHierarchicalOperatorSelector, "__init__", new=short_blocks),
                  patch.object(training.FixedProgressExtremeHierarchicalOperatorSelector, "select", new=fixed_test_sequence),
                  patch("builtins.print")):
                root, cut, end = Path(directory), 45, 61
                archive_steps = [5, 15, 25, 35, 45, 49, 53, 57, 61]

                def run(name, **kwargs):
                    return training.train_method("mo-halfcheetah-v5", experiment.METHOD, end, 25, 7,
                        str(root / name), hidden=8, batch_size=4, learning_starts=5,
                        eval_freq=1000, device="cpu", archive_steps=archive_steps, **kwargs)

                reference = run("reference")
                model = torch.load(root / "reference" / f"{experiment.METHOD}_seed7_model.pt", weights_only=False)
                report = dict(controls_passed=False, completed_budget=dict.fromkeys(experiment.NEW_BUDGET, 0))
                trace = []
                control = experiment.ScopedRouter("factual", trace, cut=cut, end=end)
                capture = experiment.ReplayCapture(control, reference, model, root / "report.json", report,
                                                   cut=cut, end=end, learning_starts=5)
                try:
                    result = run("control", state_probe=capture, update_router=control)
                    self.assertTrue(report["controls_passed"])
                    self.assertEqual(result["episode_rewards"], reference["episode_rewards"])
                    self.assertEqual(report["control_reproduction"]["archive_snapshots_matched"], 9)
                    self.assertEqual(set(report["control_reproduction"]["endpoint_network_tensors_matched"]),
                                     {"policy", "q1", "q2", "encoder"})
                    self.assertEqual(report["completed_budget"], dict(training_interactions=61,
                        learner_updates=57, evaluation_interactions=0, environment_initialization_probe_steps=2))
                    self.assertFalse((root / "control").exists(), "Control must stop before evaluation/save")
                    self.assertEqual(capture.state["step"], cut)
                    self.assertEqual(capture.state["episode_steps"], 3)
                    branch = experiment.forks.clone_state(capture.state)
                    router = experiment.ScopedRouter("own_matched", trace, cut=cut, end=end,
                                                      selector=branch["adaptive_selector"])
                    try:
                        own = run("own", resume_state=branch, update_router=router)
                    finally:
                        router.close()
                    route = own["update_routing"]
                    self.assertTrue(route["rng_alignment"]["passed"])
                    self.assertEqual(route["rng_alignment"]["checked_steps"], 16)
                    self.assertEqual(route["own_selected_updates"], 8)
                    self.assertEqual([r["operator"] for r in route["blocks"]], list(ARMS))
                    self.assertEqual([r["own_selected"] for r in route["blocks"]], [True, False, True, False])
                    self.assertEqual([r["step"] for r in own["policy_archive_snapshots"]], archive_steps)
                    self.assertEqual(own["evaluation_interactions"], 13608)
                    self.assertEqual(own["training_interactions_this_call"], 16)
                    self.assertEqual(own["adaptive_operator_final"]["block_history"][:10],
                                     reference["adaptive_operator_final"]["block_history"][:10])
                    corrupted = copy.deepcopy(model)
                    first = corrupted["policy_archive"][0]["policy"]
                    first[next(iter(first))].add_(1.)
                    with self.assertRaises(AssertionError):
                        experiment.state_check(capture.state, reference, corrupted)
                    self.assertEqual(cost, dict(training=138, evaluation=27216, initialization=6))
                finally:
                    control.close()
                    if capture.state is not None:
                        capture.state["env"].close()
                        capture.state["eval_env"].close()
        finally:
            print("SCOPED_PILOT_SMOKE_INTERACTIONS", json.dumps(cost))

    def test_exact_tensor_mismatch_rejected(self):
        a = dict(weight=torch.ones(2))
        self.assertEqual(experiment.weights_match(a, copy.deepcopy(a)), 1)
        b = dict(weight=torch.tensor([1., 1.00001]))
        with self.assertRaises(AssertionError):
            experiment.weights_match(a, b)

    def candidate_fixture(self):
        selector = experiment.training.FixedProgressExtremeHierarchicalOperatorSelector(2)
        operators = list(selector.warmup_sequence) + list(ARMS) * 5
        blocks = [dict(operator=op, end_step=6500 + 1500 * i, steps=1500)
                  for i, op in enumerate(operators)]
        routing = [dict(start_step=b["end_step"] - 1499, end_step=b["end_step"], steps=1500,
                        operator=b["operator"], own_selected=b["operator"] in experiment.TARGETS,
                        config={**selector.trainer_config(b["operator"]),
                                "preference_relabeling_rate": selector.relabel_rate(b["operator"])})
                   for b in blocks[10:]]
        count = sum(b["steps"] for b in routing if b["own_selected"])
        data = dict(training_interactions=50000, continuation_from_step=20000,
                    training_interactions_this_call=30000, evaluation_interactions=4788000,
                    environment_initialization_probe_steps=2,
                    critic_target_rule="scalar_projection_of_soft_vector",
                    bootstrap_terminal_rule="terminated_only", critic_candidate_context_rule="candidate_preference",
                    episode_rewards=[dict(step=20000, **{"return": [1., 2.]})],
                    evaluation_weights=[[0., 1.], [1., 0.]],
                    episode_seeds_per_mode=[[1, 2, 3]], ref_point=[-50., -3000.],
                    ground_truth=dict(mode_history=[(0, 0), (22770, 2)]),
                    adaptive_operator_final=dict(block_history=blocks),
                    update_routing=dict(protocol=experiment.PROTOCOL, branch="own_matched", blocks=routing,
                        rng_alignment=dict(passed=True, checked_steps=30000),
                        own_selected_updates=count, targeted_updates=count,
                        extra_updates=0, extra_environment_steps=0))
        return data, json.loads(json.dumps(data))

    def test_candidate_validation_accepts_live_history_but_rejects_schedule_change(self):
        data, reference = self.candidate_fixture()
        # The pre-fix direct comparison rejects the supported live-vs-JSON pair.
        self.assertNotEqual(data["ground_truth"]["mode_history"], reference["ground_truth"]["mode_history"])
        with (patch.object(experiment.baseline_summary.previous.common, "validate_selector"),
              patch.object(experiment, "compact_metrics", return_value={"checked": True})):
            self.assertEqual(experiment.validate_candidate(data, reference, 1003), {"checked": True})
            self.assertEqual(experiment.validate_candidate(json.loads(json.dumps(data)), reference, 1003),
                             {"checked": True})
            data["ground_truth"]["mode_history"][1] = (22771, 2)
            with self.assertRaises(AssertionError):
                experiment.validate_candidate(data, reference, 1003)

    def test_saved_recovery_preserves_sources_and_never_trains(self):
        data, reference = self.candidate_fixture()
        metrics = dict(final_hv=1., final_mode_hv=[1.], archive_hv=2., late_mean_hv=1.,
                       terminal_returns=[[[0., 1.], [1., 0.]]])
        checks = dict(passed=True, episodes_matched=True, blocks_matched=True, schedule_matched=True)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "source"
            raw = root / "seed1003" / "own_matched" / f"{experiment.METHOD}_seed1003.json"
            raw.parent.mkdir(parents=True)
            ref_path = root / "reference.json"
            raw.write_text(json.dumps(data))
            ref_path.write_text(json.dumps(reference))
            report = dict(protocol=experiment.PROTOCOL, seed=1003, warmup_steps=20000, end_step=50000,
                          complete=False, controls_passed=True, rng_alignment_passed=True,
                          completed_budget=experiment.NEW_BUDGET, target_operators=list(experiment.TARGETS),
                          reference_path=str(ref_path), branches={"factual": metrics},
                          evaluation={k: data[k] for k in ("evaluation_weights", "episode_seeds_per_mode", "ref_point")},
                          candidate_routing=data["update_routing"],
                          prefix_reproduction=dict(passed=True, history=checks, archive_snapshots_matched=7),
                          control_reproduction=dict(passed=True, history=checks, archive_snapshots_matched=19,
                              endpoint_network_tensors_matched=dict(policy=8, q1=6, q2=6, encoder=18)))
            source = root / "seed1003.json"
            source.write_text(json.dumps(report))
            original = {p: p.read_bytes() for p in (source, raw, ref_path)}
            with (patch.object(experiment.training, "train_method", side_effect=AssertionError("Must not train")),
                  patch.object(experiment.baseline_summary, "validate_new_run"),
                  patch.object(experiment.baseline_summary.previous.common, "validate_selector"),
                  patch.object(experiment, "compact_metrics", return_value=metrics)):
                recovered = experiment.recover_saved(1003, root, Path(directory) / "recovered")
                self.assertTrue(recovered["complete"])
                self.assertEqual(recovered["completed_budget"], experiment.NEW_BUDGET)
                self.assertEqual(recovered["postprocessing_recovery"]["new_evaluation_interactions"], 0)
                self.assertEqual(experiment.recover_saved(1003, root, Path(directory) / "recovered"), recovered)
                with self.assertRaises(AssertionError):
                    experiment.recover_saved(1003, root, root)
                for path, before in original.items():
                    self.assertEqual(path.read_bytes(), before)
                report["controls_passed"] = False
                source.write_text(json.dumps(report))
                with self.assertRaises(AssertionError):
                    experiment.recover_saved(1003, root, Path(directory) / "rejected")
                self.assertFalse((Path(directory) / "rejected").exists())

    def test_completed_reuse_and_incomplete_preservation(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "seed1003.json"
            report = dict(protocol=experiment.PROTOCOL, seed=1003, complete=True,
                          controls_passed=True, rng_alignment_passed=True)
            path.write_text(json.dumps(report))
            with patch.object(experiment.training, "train_method", side_effect=AssertionError("No training")):
                self.assertEqual(experiment.run(1003, directory), report)
                report["complete"] = False
                path.write_text(json.dumps(report))
                with self.assertRaisesRegex(RuntimeError, "Incomplete attempt preserved"):
                    experiment.run(1003, directory)
                self.assertEqual(json.loads(path.read_text()), report)

    def test_frozen_gates_require_complete_set_and_retain_mode_loss(self):
        reports = []
        for seed in experiment.SEEDS:
            reports.append(dict(protocol=experiment.PROTOCOL, seed=seed, complete=True,
                controls_passed=True, rng_alignment_passed=True, completed_budget=experiment.NEW_BUDGET,
                branches={"factual": dict(final_hv=1., late_mean_hv=1., archive_hv=2.),
                          "own_matched": dict(final_hv=2., late_mean_hv=1., archive_hv=2.)},
                paired_comparisons={"own_matched_minus_factual": dict(utility_mean_per_mode=[1., 0., 0., 0.])}))
        self.assertTrue(summary.summarize(reports)["eligible_to_propose_adaptive_pilot"])
        with self.assertRaises(AssertionError):
            summary.summarize(reports[:2])
        reports[0]["paired_comparisons"]["own_matched_minus_factual"]["utility_mean_per_mode"][1] = -1.
        result = summary.summarize(reports)
        self.assertFalse(result["eligible_to_propose_adaptive_pilot"])
        self.assertFalse(result["development_gates"]["every_mode_mean_matched_utility_not_worse"])
        reports[0]["branches"]["own_matched"]["late_mean_hv"] = .5
        reports[0]["branches"]["own_matched"]["archive_hv"] = 1.
        result = summary.summarize(reports)
        self.assertFalse(result["development_gates"]["late_mean_not_worse"])
        self.assertFalse(result["development_gates"]["archive_mean_not_worse"])


if __name__ == "__main__":
    unittest.main()
