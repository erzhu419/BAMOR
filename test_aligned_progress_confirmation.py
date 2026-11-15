import copy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import run_aligned_progress_confirmation as run
import summarize_aligned_progress_confirmation as summary
from test_aligned_progress_pilot import fixture as pilot_fixture


class ConfirmationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.base = pilot_fixture()

    def fixture(self, condition="adaptive", seed=1001):
        data = copy.deepcopy(self.base)
        data.pop("aligned_progress_pilot")
        data.update(method=run.METHODS[condition], seed=seed,
            episode_seeds_per_mode=[[100000 + seed * 1000 + m * 10 + r for r in range(3)] for m in range(4)],
            critic_candidate_context_rule=run.metadata(condition)["critic_candidate_context_rule"],
            aligned_progress_confirmation=run.metadata(condition))
        return data

    def test_runner_routes_three_unchanged_methods_and_reuses_saved_results(self):
        with tempfile.TemporaryDirectory() as directory:
            for condition in run.METHODS:
                data = self.fixture(condition)
                with patch.object(run.pilot.training, "train_method", return_value=data) as train:
                    result = run.run(condition, 1001, Path(directory))
                    args, kwargs = train.call_args
                    self.assertEqual(args[:5], ("mo-halfcheetah-v5", run.METHODS[condition], 50000, 25000, 1001))
                    self.assertEqual(kwargs, run.pilot.TRAIN_KWARGS)
                with patch.object(run.pilot.training, "train_method", side_effect=AssertionError("No retraining")):
                    self.assertEqual(result, run.run(condition, 1001, Path(directory)))
            with patch.object(run.pilot.training, "train_method", side_effect=AssertionError("No development seeds")):
                with self.assertRaises(AssertionError):
                    run.run("adaptive", 26, Path(directory))

    def test_all_three_results_validate_and_wrong_protocol_or_target_is_rejected(self):
        for condition in run.METHODS:
            result = summary.validate_run(self.fixture(condition), condition, 1001)
            self.assertEqual(result["final_hv"], result["late_mean_hv"])
        for kind in ("seed", "condition", "metadata", "target", "budget", "grid", "score", "probe"):
            bad = self.fixture()
            if kind == "seed":
                bad["seed"] = 26
            elif kind == "condition":
                bad["method"] = run.METHODS["legacy"]
            elif kind == "metadata":
                bad["aligned_progress_confirmation"]["training_kwargs"]["hidden"] = 256
            elif kind == "target":
                bad["critic_candidate_context_rule"] = "candidate_preference"
            elif kind == "budget":
                bad["evaluation_interactions"] -= 1
            elif kind == "grid":
                bad["pareto_per_mode"][0].pop()
            elif kind == "score":
                bad["adaptive_operator_final"]["block_history"][10]["progress_score"] += .1
            else:
                bad["update_routing"] = {}
            with self.subTest(kind=kind), self.assertRaises(AssertionError):
                summary.validate_run(bad, "adaptive", 1001)

    def test_triplet_matches_legacy_only_before_learning_and_aligned_pair_to_20k(self):
        data = {c: self.fixture(c) for c in run.METHODS}
        for condition, d in data.items():
            d["episode_rewards"].insert(1, {"step": 10000, "reward": [7. if condition == "legacy" else 0., 0.]})
        result = summary.validate_triplet(data)
        self.assertEqual(result["aligned_pair"]["warmup_episodes_matched"], 2)
        self.assertEqual(result["legacy_prelearning_episodes_matched"], 1)
        for kind in ("prelearning", "aligned_warmup", "schedule"):
            bad = copy.deepcopy(data)
            if kind == "prelearning":
                bad["legacy"]["episode_rewards"][0]["reward"][0] = 1.
            elif kind == "aligned_warmup":
                bad["blind"]["episode_rewards"][1]["reward"][0] = 1.
            else:
                bad["legacy"]["ground_truth"]["mode_history"][1][0] += 1
            with self.subTest(kind=kind), self.assertRaises(AssertionError):
                summary.validate_triplet(bad)

    def conditions(self):
        return {c: {"per_seed": {s: {"final_hv": float(i + 1), "archive_hv": 10.,
                     "late_mean_hv": 2., "late_min_hv": 1., "final_mode_hv": [float(i + 1)] * 4}
                     for i, s in enumerate(run.SEEDS)}} for c in run.METHODS}

    def test_selector_benefit_and_legacy_regression_are_distinct(self):
        c = self.conditions()
        c["adaptive"]["per_seed"][1005]["final_hv"] += 1.
        c["legacy"]["per_seed"][1005]["final_hv"] += 2.
        report = summary.aggregate(c)
        self.assertTrue(report["selector_signal_replicated"])
        self.assertFalse(report["legacy_regression_not_observed"])
        self.assertFalse(report["advance_to_larger_confirmation"])
        c["legacy"]["per_seed"][1005]["final_hv"] -= 2.
        self.assertTrue(summary.aggregate(c)["advance_to_larger_confirmation"])
        c["adaptive"]["per_seed"][1001]["final_hv"] -= .01
        self.assertFalse(summary.aggregate(c)["advance_to_larger_confirmation"])

    def test_archive_or_terminal_peak_cannot_replace_frozen_metrics(self):
        c = self.conditions()
        c["adaptive"]["per_seed"][1001]["archive_hv"] += 100.
        self.assertFalse(summary.aggregate(c)["selector_signal_replicated"])
        c["adaptive"]["per_seed"][1005]["final_hv"] += 2.
        c["adaptive"]["per_seed"][1001]["late_mean_hv"] -= .01
        self.assertFalse(summary.aggregate(c)["selector_signal_replicated"])
        c["adaptive"]["per_seed"].pop(1002)
        with self.assertRaises(AssertionError):
            summary.aggregate(c)

    def test_complete_summary_requires_all_fifteen_runs_and_counts_budget(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for seed in run.SEEDS:
                for condition, method in run.METHODS.items():
                    (root / f"{method}_seed{seed}.json").write_text(json.dumps(self.fixture(condition, seed)))
            report = summary.summarize(root)
            self.assertEqual(report["validated_runs"], 15)
            self.assertEqual(report["completed_run_budget"], dict(training_interactions=750000,
                evaluation_interactions=71820000, environment_initialization_probe_steps=30))
            self.assertFalse(report["selector_signal_replicated"])
            self.assertTrue(report["legacy_regression_not_observed"])
            self.assertEqual(set(report["validation"]), set(run.SEEDS))
            for comparator in ("blind", "legacy"):
                self.assertEqual(report["comparisons"][comparator]["final_wins"], 0)
                self.assertEqual(set(report["comparisons"][comparator]["paired_adaptive_minus_control"]), set(run.SEEDS))
            (root / f"{run.METHODS['legacy']}_seed1005.json").unlink()
            with self.assertRaises(FileNotFoundError):
                summary.summarize(root)


if __name__ == "__main__":
    unittest.main()
