import copy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import torch

import run_scoped_adaptive_confirmation as experiment
import summarize_scoped_adaptive_confirmation as summary
from piecewise_mo_env import PiecewiseStationaryMOEnv
from test_scoped_extreme_pilot import ARMS, ScopedExtremeTests


class ScopedAdaptiveTests(unittest.TestCase):
    def test_live_feedback_scope_and_no_global_rng_claim(self):
        training = experiment.training
        for condition in experiment.CONDITIONS:
            update, selection = experiment.factors(condition)
            cls = (training.ProgressExtremeHierarchicalOperatorSelector if selection == "adaptive"
                   else training.FixedProgressExtremeHierarchicalOperatorSelector)
            selector = cls(2)
            trainer = training.ConsistentTargetMOSACTrainer(3, 2, 2, hidden=8, batch_size=4,
                                                            buffer_capacity=8, device="cpu")
            router = experiment.IntegratedRouter(condition, cut=1, end=5)
            self.assertFalse(router(dict(step=1, adaptive_selector=selector)))
            router.configure(1, trainer)
            self.assertNotIn("envelope_actor_loss", trainer.__dict__)
            try:
                for step, op in enumerate(ARMS, 2):
                    selector.current_operator, selector.block_start_step = op, step
                    selector.scores[op] = step / 10
                    scores = selector.scores.copy()
                    config = {**selector.trainer_config(op),
                              "preference_relabeling_rate": selector.relabel_rate(op)}
                    for k, value in config.items():
                        setattr(trainer, k, value)
                    router.configure(step, trainer)
                    router.after_step(step, trainer, None, None, op)
                    self.assertEqual("envelope_actor_loss" in trainer.__dict__,
                                     update == "scoped" and op in experiment.scoped.TARGETS)
                    self.assertEqual(selector.scores, scores)
                result = router.state_dict()
                self.assertEqual(result["own_selected_updates"], 2 if update == "scoped" else 0)
                self.assertEqual(result["selection_uses_scores"], selection == "adaptive")
                self.assertNotIn("rng_alignment", result)
            finally:
                router.close()

    def test_formal_routing_validation_and_saved_result_recovery(self):
        for condition in experiment.CONDITIONS:
            update, selection = experiment.factors(condition)
            data, _ = ScopedExtremeTests().candidate_fixture()
            data.pop("continuation_from_step")
            data.pop("training_interactions_this_call")
            data["update_routing"].pop("rng_alignment")
            route = data["update_routing"]
            route.update(protocol=experiment.PROTOCOL, condition=condition, cut=20000, end=50000,
                         target_operators=list(experiment.scoped.TARGETS), selection_uses_scores=selection == "adaptive")
            if update == "legacy":
                route["own_selected_updates"] = 0
                for block in route["blocks"]:
                    block["own_selected"] = False
            metrics = dict(posthoc_late_mean_hv=1, posthoc_late_snapshot_hv=[1, 1, 1],
                           posthoc_late_mode_mean_hv=[1] * 4, final_hv=1)
            data["policy_archive_snapshots"] = []
            with (patch.object(experiment.common, "validate_selector", return_value={}),
                  patch.object(experiment.common, "metrics", side_effect=lambda *a, **kw: metrics.copy()),
                  patch.object(experiment.scoped.forks, "terminal_returns", return_value=np.zeros((4, 21, 2)))):
                report = experiment.validate(data, condition, 1006)
                changed = copy.deepcopy(data)
                changed["update_routing"]["blocks"][0]["own_selected"] ^= True
                with self.assertRaises(AssertionError):
                    experiment.validate(changed, condition, 1006)
                with tempfile.TemporaryDirectory() as directory:
                    folder = Path(directory) / condition
                    folder.mkdir()
                    output = folder / "seed1006.json"
                    incomplete = dict(protocol=experiment.PROTOCOL, condition=condition, seed=1006, complete=False)
                    output.write_text(json.dumps(incomplete))
                    with patch.object(experiment.training, "train_method") as train:
                        with self.assertRaisesRegex(RuntimeError, "Incomplete training"):
                            experiment.run(condition, 1006, directory)
                        raw = folder / f"{experiment.METHODS[selection]}_seed1006.json"
                        raw.write_text(json.dumps(data))
                        original = raw.read_bytes()
                        self.assertEqual(experiment.run(condition, 1006, directory)["metrics"], report["metrics"])
                        experiment.run(condition, 1006, directory)
                        train.assert_not_called()
                        self.assertEqual(raw.read_bytes(), original)

    def test_four_arm_real_training_same_warmup_and_budget(self):
        training = experiment.training
        make_env, trainer_cls = training.make_piecewise_mo_env, training.ConsistentTargetMOSACTrainer
        original_init = training.ProgressExtremeHierarchicalOperatorSelector.__init__
        original_step = PiecewiseStationaryMOEnv.step
        cost = dict(training=0, evaluation=0, initialization=0)

        def short_env(*args, **kwargs):
            env = make_env(*args, **kwargs)
            env.env._max_episode_steps = 3
            env.smoke_kind = "training" if kwargs["seed"] == 7 else "evaluation"
            cost["initialization"] += 1
            return env

        def counted_step(env, action):
            result = original_step(env, action)
            cost[env.smoke_kind] += 1
            return result

        def short_blocks(self, reward_dim, block_steps=1500, learning_starts=5000):
            original_init(self, reward_dim, block_steps=4, learning_starts=learning_starts)

        runs = {}
        try:
            with (tempfile.TemporaryDirectory() as directory,
                  patch.object(training, "make_piecewise_mo_env", side_effect=short_env),
                  patch.object(PiecewiseStationaryMOEnv, "step", new=counted_step),
                  patch.object(training.ProgressExtremeHierarchicalOperatorSelector, "__init__", new=short_blocks),
                  patch.object(training, "ConsistentTargetMOSACTrainer",
                               side_effect=lambda *a, **kw: trainer_cls(*a, buffer_capacity=128, **kw)),
                  patch("builtins.print")):
                for condition in experiment.CONDITIONS:
                    router = experiment.IntegratedRouter(condition, cut=45, end=61)
                    try:
                        result = training.train_method("mo-halfcheetah-v5",
                            experiment.METHODS[experiment.factors(condition)[1]], 61, 25, 7,
                            str(Path(directory) / condition), hidden=8, batch_size=4, learning_starts=5,
                            eval_freq=1000, device="cpu", archive_steps=[5, 45, 61],
                            state_probe=router, update_router=router)
                        runs[condition] = result
                        self.assertEqual(result["training_interactions"], 61)
                        self.assertEqual(result["evaluation_interactions"], 2268)
                        self.assertEqual([r["steps"] for r in result["update_routing"]["blocks"]], [4] * 4)
                        self.assertEqual(len(result["adaptive_operator_final"]["block_history"]), 14)
                    finally:
                        router.close()
            base = runs["legacy_blind"]
            for result in runs.values():
                self.assertEqual(result["ground_truth"]["mode_history"], base["ground_truth"]["mode_history"])
                self.assertEqual([r for r in result["episode_rewards"] if r["step"] <= 45],
                                 [r for r in base["episode_rewards"] if r["step"] <= 45])
                self.assertEqual(result["adaptive_operator_final"]["block_history"][:10],
                                 base["adaptive_operator_final"]["block_history"][:10])
            self.assertEqual(cost, dict(training=244, evaluation=9072, initialization=8))
        finally:
            print("SCOPED_ADAPTIVE_SMOKE_INTERACTIONS", json.dumps(cost))

    def test_complete_factorial_contrasts_and_secondary_tradeoffs(self):
        reports = []
        for c in experiment.CONDITIONS:
            update, selection = experiment.factors(c)
            u, a = update == "scoped", selection == "adaptive"
            for i, seed in enumerate(experiment.SEEDS):
                hv = 100 + i + 10 * u + 3 * a + 2 * u * a
                metrics = dict(final_hv=hv, late_mean_hv=hv, archive_hv=150 + i - u,
                    final_mode_hv=[hv] * 4, mode_hv=[150 + i - u] * 4, mode_history=[[0, 0]],
                    terminal_returns=[[[1, 1]] * 2, [[1 - 5 * u, 1 - 5 * u]] * 2,
                                      [[1, 1]] * 2, [[1, 1]] * 2])
                reports.append(dict(protocol=experiment.PROTOCOL, condition=c, seed=seed, complete=True,
                    completed_budget=experiment.BUDGET.copy(), metrics=metrics, warmup={"same": True},
                    evaluation=dict(evaluation_weights=[[0, 1], [1, 0]]),
                    selector=dict(operators=["a"] * 30), routing={}))
        result = summary.summarize(reports)
        self.assertEqual(result["contrasts"]["update_under_adaptive"]["mean"]["final_hv"], 12)
        self.assertEqual(result["contrasts"]["selector_under_scoped"]["mean"]["final_hv"], 5)
        self.assertEqual(result["selector_by_update_interaction"][1006]["final_hv"], 2)
        self.assertEqual(result["contrasts"]["update_under_adaptive"]["mode_mean_matched_utility"][1], -5)
        self.assertEqual(result["contrasts"]["update_under_adaptive"]["mean"]["archive_hv"], -1)
        self.assertEqual(result["completed_run_budget"]["training_interactions"], 1000000)
        for invalid in (reports[:-1], reports + reports[:1]):
            with self.assertRaises(AssertionError):
                summary.summarize(invalid)
        reports[-1]["warmup"] = {"same": False}
        with self.assertRaisesRegex(AssertionError, "before the scoped"):
            summary.summarize(reports)


if __name__ == "__main__":
    unittest.main()
