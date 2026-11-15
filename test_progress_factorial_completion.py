import copy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import run_progress_factorial_completion as run
import summarize_progress_factorial_completion as summary
from test_aligned_progress_pilot import fixture


class FactorialCompletionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.base = fixture("blind", 1001, aligned=False)
        cls.base.update(critic_candidate_context_rule="candidate_preference",
                        progress_factorial_completion=run.metadata())

    def data(self, seed=1001):
        data = copy.deepcopy(self.base)
        data.update(seed=seed, episode_seeds_per_mode=[
            [100000 + seed * 1000 + m * 10 + r for r in range(3)] for m in range(4)])
        return data

    def conditions(self):
        return {c: {"per_seed": {s: {"final_hv": float(i + 1), "archive_hv": 10.,
                   "late_mean_hv": 2., "late_min_hv": 1., "final_mode_hv": [float(i + 1)] * 4}
                   for i, s in enumerate(run.SEEDS)}} for c in summary.ARMS}

    def test_runner_only_adds_legacy_blind_and_recovers_terminal_results(self):
        with tempfile.TemporaryDirectory() as directory:
            with patch.object(run.previous.pilot.training, "train_method", return_value=self.data()) as train:
                first = run.run(1001, directory)
                args, kwargs = train.call_args
                self.assertEqual(args[:5], ("mo-halfcheetah-v5", "bamor_v25_progress_fixed_archive_mix", 50000, 25000, 1001))
                self.assertEqual(kwargs, run.previous.pilot.TRAIN_KWARGS)
                self.assertFalse(first["progress_factorial_completion"]["selection_uses_scores"])
            with patch.object(run.previous.pilot.training, "train_method", side_effect=AssertionError("No rerun")):
                self.assertEqual(first, run.run(1001, directory))
                with self.assertRaises(AssertionError):
                    run.run(26, directory)

    def test_new_run_rejects_wrong_arm_target_metadata_and_cost(self):
        result = summary.validate_new_run(self.data(), 1001)
        self.assertEqual(result["final_hv"], result["late_mean_hv"])
        for key, value in (("method", run.previous.METHODS["legacy"]),
                           ("critic_candidate_context_rule", "update_preference"),
                           ("progress_factorial_completion", {}),
                           ("evaluation_interactions", 0), ("seed", 1002)):
            bad = self.data()
            bad[key] = value
            with self.subTest(key=key), self.assertRaises(AssertionError):
                summary.validate_new_run(bad, 1001)

    def test_factorial_contrast_signs_are_paired_not_ratios(self):
        conditions = self.conditions()
        for arm, value in zip(summary.ARMS, (8., 5., 6., 4.)):
            for row in conditions[arm]["per_seed"].values():
                row["final_hv"] = value
                row["final_mode_hv"] = [value] * 4
        result = summary.aggregate(conditions)
        for name, value in zip(("aligned_selector", "legacy_selector", "target_with_adaptive",
                                "target_with_blind", "selector_by_target_interaction"), (3., 2., 2., 1., 1.)):
            self.assertEqual(result["contrasts"][name]["mean"]["final_hv"], value)
            self.assertEqual(result["contrasts"][name]["per_seed"][1001]["final_mode_hv"], [value] * 4)
        self.assertTrue(result["legacy_selector_checks_passed"])

    def test_legacy_gate_keeps_minimum_and_late_metric_and_requires_all_seeds(self):
        c = self.conditions()
        c["legacy_adaptive"]["per_seed"][1005]["archive_hv"] += 100.
        self.assertFalse(summary.aggregate(c)["legacy_selector_checks_passed"])
        c["legacy_adaptive"]["per_seed"][1005]["final_hv"] += 2.
        self.assertTrue(summary.aggregate(c)["legacy_selector_checks_passed"])
        for metric in ("final_hv", "late_mean_hv"):
            bad = copy.deepcopy(c)
            bad["legacy_adaptive"]["per_seed"][1001][metric] -= .1
            self.assertFalse(summary.aggregate(bad)["legacy_selector_checks_passed"])
        c["legacy_blind"]["per_seed"].pop(1005)
        with self.assertRaises(AssertionError):
            summary.aggregate(c)

    def test_completion_reuses_fifteen_runs_preserves_decision_and_counts_only_five_new(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            conditions = self.conditions()
            old = summary.previous.aggregate({old: conditions[arm] for arm, old in
                zip(summary.ARMS[:3], ("adaptive", "blind", "legacy"))})
            old.update(protocol=run.previous.PROTOCOL, seeds=list(run.SEEDS), validated_runs=15,
                       expected=15, completed_run_budget=dict(zip(summary.BUDGET_KEYS, (750000, 71820000, 30))))
            ref = root / "reference.json"
            ref.write_text(json.dumps(old))
            for seed in run.SEEDS:
                data = self.data(seed)
                (root / f"{run.METHOD}_seed{seed}.json").write_text(json.dumps(data))
                data["method"] = run.previous.METHODS["legacy"]
                (root / f"{data['method']}_seed{seed}.json").write_text(json.dumps(data))
            report = summary.summarize(root, root, ref)
            self.assertEqual((report["validated_new_runs"], report["reused_validated_runs"]), (5, 15))
            self.assertEqual(report["new_completed_run_budget"], dict(zip(summary.BUDGET_KEYS, (250000, 23940000, 10))))
            self.assertEqual(report["all_completed_run_budget"], dict(zip(summary.BUDGET_KEYS, (1000000, 95760000, 40))))
            self.assertFalse(report["previous_decision"]["advance_to_larger_confirmation"])
            for k, value in report["previous_decision"].items():
                self.assertEqual(value, old[k])
            bad = self.data(1005)
            bad["episode_rewards"][0]["reward"][0] = 1.
            path = root / f"{run.METHOD}_seed1005.json"
            path.write_text(json.dumps(bad))
            with self.assertRaises(AssertionError):
                summary.summarize(root, root, ref)
            path.unlink()
            with self.assertRaises(FileNotFoundError):
                summary.summarize(root, root, ref)


if __name__ == "__main__":
    unittest.main()
