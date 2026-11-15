import copy
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
import torch

import diagnose_ridge_expansion as diagnostic
import summarize_ridge_expansion as summary
from test_paired_advantage_summary import terminal_fixture
from test_ridge_learning_curve import cached_fixture


def fixture():
    d = diagnostic
    source = terminal_fixture()
    source["sources"] = {"policy": "frozen_policy.pt", "raw_critics": "frozen_raw.pt", "diagnostic": "frozen.json"}
    new = copy.deepcopy(source["train_cases"])
    for c in new:
        c["repeat"] += 16
        c["reset_seed"] = d.p.reset_seed("train", 31, c["mode"], c["repeat"])
    labels = {"protocol": d.LABEL_PROTOCOL, "condition": "v27", "seed": 31,
        "method": d.p.g.METHODS["v27"], "complete": True, "sources": source["sources"],
        "repeats": list(d.REPEATS), "epsilon": .05, "warmup": 128, "horizons": [32, 128, 512], "gamma": .99,
        "weights": d.p.g.WEIGHTS.tolist(), "candidates": list(d.p.CANDIDATES), "train_cases": new,
        "label_collection_interactions": 614400, "environment_initialization_probe_steps": 1,
        "new_environment_interactions": 614401, "policy_updates": 0, "critic_updates": 0,
        "evaluation_interactions": 0, "policy_training_interactions": 0}
    return source, labels


class RidgeExpansionTests(unittest.TestCase):
    def test_new_ids_and_fold_grouping_exclude_old_held_and_new_same_fold(self):
        d = diagnostic
        source, labels = fixture()
        new = d.validate_labels(labels, "v27", 31, source["sources"])
        for fold in range(4):
            fit, held = d.split_expanded(source["train_cases"], new, fold)
            self.assertEqual((len(fit), len(held)), (288, 48))
            self.assertTrue(all(c["repeat"] % 4 != fold for c in fit))
            self.assertEqual(held, d.cv.split_fold(source["train_cases"], fold)[1])
            for reset in {c["reset_seed"] for c in fit}:
                self.assertEqual({c["weight_index"] for c in fit if c["reset_seed"] == reset}, {0, 1, 2})
        for kind in ("reset", "branch", "budget", "paths"):
            bad = copy.deepcopy(labels)
            if kind == "reset":
                bad["train_cases"][0]["reset_seed"] = source["train_cases"][0]["reset_seed"]
            elif kind == "branch":
                bad["train_cases"][0]["branches"]["raw_plus"]["steps"] -= 1
            elif kind == "budget":
                bad["new_environment_interactions"] -= 1
            else:
                bad["sources"]["policy"] = "different.pt"
            with self.subTest(kind=kind), self.assertRaises(AssertionError):
                d.validate_labels(bad, "v27", 31, source["sources"])

    def test_collector_counts_all_branches_reuses_complete_and_refuses_partial(self):
        d = diagnostic
        source, expected = fixture()
        by_reset = {(c["reset_seed"], c["weight_index"]): c for c in expected["train_cases"]}
        env = SimpleNamespace(num_modes=4, _total_steps=0, _lock_mode=False, _apply_mode=lambda _: None, close=lambda: None)

        def evaluate(environment, networks, critics, method, weight, seed, epsilon):
            self.assertIs(environment, env)
            self.assertTrue(env._lock_mode)
            self.assertEqual(epsilon, .05)
            wi = next(i for i, w in enumerate(d.p.g.WEIGHTS) if np.array_equal(w, weight))
            env._total_steps += 640
            return copy.deepcopy(by_reset[seed, wi])

        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "labels.json"
            with (patch.object(d.p.sources, "load_inputs", return_value=({}, [], {}, source["sources"])) as loader,
                  patch.object(d.p.training, "make_piecewise_mo_env", return_value=env) as factory,
                  patch.object(d.p.p, "evaluate_case", side_effect=evaluate) as evaluator,
                  patch.object(d.p, "fit_evaluator", side_effect=AssertionError("No MLP training"))):
                report = d.collect_labels("v27", 31, source["sources"], path)
                self.assertEqual(report, expected)
                self.assertEqual(evaluator.call_count, 192)
                self.assertEqual(factory.call_count, 1)
                repeated = d.collect_labels("v27", 31, source["sources"], path)
                self.assertEqual(repeated, report)
                self.assertEqual(loader.call_count, 1)
                self.assertEqual(evaluator.call_count, 192)
                report["complete"] = False
                path.write_text(json.dumps(report))
                with self.assertRaisesRegex(AssertionError, "Incomplete collection"):
                    d.collect_labels("v27", 31, source["sources"], path)

    def test_runner_fits_only_expanded_training_and_preserves_old_controls(self):
        d = diagnostic
        source, labels = fixture()
        source["validation_cases"] = source["summary"] = None
        previous = cached_fixture(source["train_cases"])
        fit_predict = d.curve.fit_predict

        def fit(cases, held_features):
            self.assertEqual(len(cases), 288)
            self.assertEqual(sum(c["repeat"] >= 16 for c in cases), 144)
            self.assertEqual(held_features.shape, (48, 5, 42))
            return fit_predict(cases, held_features)

        with (tempfile.TemporaryDirectory() as directory,
              patch.object(d.p.training, "make_piecewise_mo_env", side_effect=AssertionError("Cached labels only")),
              patch.object(d.p.sources, "load_inputs", side_effect=AssertionError("No checkpoint reload")),
              patch.object(d.curve, "fit_predict", side_effect=fit) as fitter,
              patch.object(d.cv, "fit_fold", side_effect=AssertionError("No control refit"))):
            root = Path(directory)
            for name, data in (("source/v27/seed31.json", source), ("previous/v27/seed31.json", previous),
                               ("out/v27/seed31_labels.json", labels)):
                path = root / name
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text(json.dumps(data))
            result = d.run("v27", 31, root / "source", root / "previous", root / "out")
            self.assertEqual(fitter.call_count, 4)
            self.assertEqual(summary.validate_run(result, labels, previous, source, "v27", 31), result["summary"])
            for kind in ("fold", "control", "budget", "gate", "grid"):
                bad = copy.deepcopy(result)
                if kind == "fold":
                    bad["folds"][0]["fit_reset_ids"] = bad["folds"][0]["held_reset_ids"]
                elif kind == "control":
                    bad["rows"][0]["scores"]["ridge"]["soft_plus"] += 1.
                elif kind == "budget":
                    bad["label_collection_interactions"] -= 1
                elif kind == "gate":
                    bad["summary"]["group_passes"] = not result["summary"]["group_passes"]
                else:
                    bad["rows"][0] = copy.deepcopy(bad["rows"][1])
                with self.subTest(kind=kind), self.assertRaises(AssertionError):
                    summary.validate_run(bad, labels, previous, source, "v27", 31)

    def test_more_labels_do_not_relax_downside_gate_or_mix_score_units(self):
        d = diagnostic
        cases = fixture()[0]["train_cases"]
        for c in cases:
            value = -4. if c["repeat"] == 0 else 1.
            c["branches"]["soft_plus"]["discounted_returns"]["512"] = [value, value]
        old = cached_fixture(cases)
        rows = [d.reference.join_row(c, r, {"ridge96": np.array([0., 2., 0., 1., 0.])}, selectors=d.SELECTORS)
                for c, r in zip(cases, old["rows"])]
        result = d.summarize_rows(rows)
        self.assertTrue(result["checks"]["positive_mean_gain"])
        self.assertFalse(result["checks"]["downside_not_worse_than_controls"])
        self.assertFalse(result["group_passes"])
        self.assertFalse({"ridge128", "margin", "cost_sensitive"}.intersection(result["calibration"]))


if __name__ == "__main__":
    unittest.main()
