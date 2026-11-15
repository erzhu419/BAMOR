import copy
import inspect
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

import run_aligned_progress_pilot as pilot
import summarize_aligned_progress_pilot as summary
from test_progress_selector import observe
from test_replay_refresh_summary import result as front_fixture


def fixture(condition="adaptive", seed=26, aligned=True):
    data = front_fixture()
    methods = pilot.METHODS if aligned else pilot.LEGACY_METHODS
    cls = (pilot.training.ProgressExtremeHierarchicalOperatorSelector if condition == "adaptive"
           else pilot.training.FixedProgressExtremeHierarchicalOperatorSelector)
    selector = cls(2)
    np.random.seed(seed)
    observe(selector, np.zeros((50000, 2)))
    data.update(method=methods[condition], seed=seed,
        episode_seeds_per_mode=[[100000 + seed * 1000 + m * 10 + r for r in range(3)] for m in range(4)],
        critic_candidate_context_rule="update_preference" if aligned else "candidate_preference",
        critic_target_rule="scalar_projection_of_soft_vector", bootstrap_terminal_rule="terminated_only",
        episode_rewards=[{"step": 1000, "reward": [0., 0.]}, {"step": 50000, "reward": [0., 0.]}],
        adaptive_operator_final=selector.state_dict(), training_interactions=50000,
        evaluation_interactions=4788000, environment_initialization_probe_steps=2)
    if aligned:
        data["aligned_progress_pilot"] = pilot.metadata(condition)
    else:
        data.pop("critic_candidate_context_rule")
    return data


class AlignedProgressTests(unittest.TestCase):
    def test_new_routes_keep_all_existing_features_and_arm_configs(self):
        for method in pilot.METHODS.values():
            self.assertIn(method, pilot.training.METHODS)
            self.assertIn(method, pilot.training.V27_MUJOCO_METHODS)
            for name, function in vars(pilot.training).items():
                if name.startswith("uses_") and inspect.isfunction(function):
                    self.assertEqual(function(method), function("bamor_v27_aligned_target_fixed_archive_mix"), name)
            self.assertEqual(pilot.training.context_dim_for_method(method, 17, 6, 2, 4), 13)
            self.assertEqual(pilot.training.policy_archive_interval_for_method(method), 2500)
        a = pilot.training.ProgressExtremeHierarchicalOperatorSelector(2)
        b = pilot.training.FixedProgressExtremeHierarchicalOperatorSelector(2)
        legacy = pilot.training.FixedExtremeHierarchicalOperatorSelector(2)
        for op in a.operators:
            self.assertEqual(a.trainer_config(op), b.trainer_config(op))
            self.assertEqual(a.trainer_config(op), legacy.trainer_config(op))

    def test_real_training_uses_aligned_targets_and_paired_progress_warmup(self):
        training = pilot.training
        make_env, trainer_cls = training.make_piecewise_mo_env, training.PreferenceAlignedTargetMOSACTrainer
        original_init = training.ProgressExtremeHierarchicalOperatorSelector.__init__
        constructed = []

        def short_env(*args, **kwargs):
            env = make_env(*args, **kwargs)
            env.env._max_episode_steps = 3
            return env

        def short_blocks(self, reward_dim, block_steps=1500, learning_starts=5000):
            constructed.append(type(self))
            original_init(self, reward_dim, block_steps=4, learning_starts=learning_starts)

        pair = {}
        with tempfile.TemporaryDirectory() as directory:
            for condition, method in pilot.METHODS.items():
                with (patch.object(training, "make_piecewise_mo_env", side_effect=short_env),
                      patch.object(training, "PreferenceAlignedTargetMOSACTrainer", side_effect=lambda *a, **kw: trainer_cls(*a, buffer_capacity=128, **kw)),
                      patch.object(training.ProgressExtremeHierarchicalOperatorSelector, "__init__", new=short_blocks),
                      patch("builtins.print")):
                    pair[condition] = training.train_method("mo-halfcheetah-v5", method, 64, 25, 7, directory,
                        hidden=8, batch_size=4, learning_starts=5, eval_freq=1000, log_freq=1000, device="cpu")
                data = pair[condition]
                self.assertEqual(data["critic_candidate_context_rule"], "update_preference")
                self.assertEqual(data["critic_target_rule"], "scalar_projection_of_soft_vector")
                self.assertEqual(data["bootstrap_terminal_rule"], "terminated_only")
                self.assertEqual(data["adaptive_operator_final"]["score_signal"], "within_block_normalized_vector_progress")
                self.assertEqual(sum(data["adaptive_operator_final"]["counts"].values()), 14)
                self.assertEqual(data["evaluation_protocol"], "locked_full_simplex_v1")
                self.assertNotIn("actor_update", data)
                self.assertEqual(data["training_interactions"], 64)
                self.assertEqual(data["evaluation_interactions"], 4 * 21 * 3 * 3)
        a, b = pair["adaptive"], pair["blind"]
        self.assertEqual(constructed, [training.ProgressExtremeHierarchicalOperatorSelector,
                                      training.FixedProgressExtremeHierarchicalOperatorSelector])
        self.assertEqual(a["adaptive_operator_final"]["block_history"][:10], b["adaptive_operator_final"]["block_history"][:10])
        self.assertEqual([r for r in a["episode_rewards"] if r["step"] <= 45], [r for r in b["episode_rewards"] if r["step"] <= 45])

    def test_summary_rejects_changed_protocol_controls_and_score_formula(self):
        data = fixture()
        result = summary.validate_run(data, "adaptive", 26, aligned=True)
        self.assertEqual(result["late_mean_hv"], result["final_hv"])
        self.assertEqual(summary.validate_pair(data, fixture("blind"))["postwarmup_different_arms"], 0)
        for kind in ("score", "warmup", "grid", "target", "budget", "config", "metadata"):
            bad = copy.deepcopy(data)
            if kind == "score":
                bad["adaptive_operator_final"]["block_history"][10]["progress_score"] += .1
            elif kind == "warmup":
                bad["adaptive_operator_final"]["block_history"][0]["end_step"] -= 1
            elif kind == "grid":
                bad["pareto_per_mode"][0].pop()
            elif kind == "target":
                bad["critic_candidate_context_rule"] = "candidate_preference"
            elif kind == "budget":
                bad["training_interactions"] += 1
            elif kind == "config":
                op = bad["adaptive_operator_final"]["operators"][0]
                bad["adaptive_operator_final"]["operator_trainer_configs"][op]["envelope_policy_candidates"] += 1
            else:
                bad["aligned_progress_pilot"]["selection_uses_scores"] = False
            with self.subTest(kind=kind), self.assertRaises(AssertionError):
                summary.validate_run(bad, "adaptive", 26, aligned=True)
        bad = fixture("blind")
        bad["episode_rewards"][0]["reward"][0] += .1
        with self.assertRaises(AssertionError):
            summary.validate_pair(data, bad)

    def test_gate_does_not_promote_archive_only_or_unstable_final_gains(self):
        def conditions():
            return {c: {"per_seed": {seed: {"final_hv": float(i + 1), "archive_hv": 5.,
                         "late_mean_hv": 2., "late_min_hv": 1.} for i, seed in enumerate(pilot.SEEDS)}} for c in pilot.METHODS}
        c = conditions()
        c["adaptive"]["per_seed"][28]["archive_hv"] = 100.
        self.assertFalse(summary.aggregate(c)["advance_to_fresh_seed_test"])
        c["adaptive"]["per_seed"][28]["final_hv"] += 2.
        self.assertTrue(summary.aggregate(c)["advance_to_fresh_seed_test"])
        c["adaptive"]["per_seed"][26]["final_hv"] -= .1
        self.assertFalse(summary.aggregate(c)["advance_to_fresh_seed_test"])
        c = conditions()
        c["adaptive"]["per_seed"][28]["final_hv"] += 2.
        c["adaptive"]["per_seed"][28]["late_mean_hv"] -= .1
        self.assertFalse(summary.aggregate(c)["advance_to_fresh_seed_test"])

    def test_runner_reuses_terminal_result_without_retraining(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)
            data = fixture()
            with patch.object(pilot.training, "train_method", return_value=data) as train:
                pilot.run("adaptive", 26, output)
                args, kwargs = train.call_args
                self.assertEqual(args[:5], ("mo-halfcheetah-v5", pilot.METHODS["adaptive"], 50000, 25000, 26))
                self.assertEqual(kwargs, pilot.TRAIN_KWARGS)
            with patch.object(pilot.training, "train_method", side_effect=AssertionError("No rerun")):
                repeated = pilot.run("adaptive", 26, output)
            self.assertEqual(repeated["aligned_progress_pilot"], pilot.metadata("adaptive"))

    def test_full_summary_counts_pairs_and_target_interaction(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for aligned, folder in ((True, root / "new"), (False, root / "old")):
                folder.mkdir()
                for condition in pilot.METHODS:
                    for seed in pilot.SEEDS:
                        data = fixture(condition, seed, aligned)
                        (folder / f"{data['method']}_seed{seed}.json").write_text(json.dumps(data))
            report = summary.summarize(root / "new", root / "old")
            self.assertEqual((report["new_runs"], report["reused_runs"]), (6, 6))
            self.assertEqual(report["completed_run_budget"], dict(training_interactions=300000,
                evaluation_interactions=28728000, environment_initialization_probe_steps=12))
            self.assertTrue(all(all(v == 0 for v in row.values()) for row in report["selector_by_target_interaction"].values()))
            self.assertFalse(report["results"]["aligned"]["advance_to_fresh_seed_test"])


if __name__ == "__main__":
    unittest.main()
