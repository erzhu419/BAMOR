"""Validate the expanded-label ridge experiment and export a compact summary."""

import argparse
import json
from itertools import product
from pathlib import Path

import numpy as np

import diagnose_ridge_expansion as pilot


def validate_run(data, labels, previous, source, condition, seed):
    new_cases = pilot.validate_labels(labels, condition, seed, source["sources"])
    cases = pilot.cv.training_cases(source, condition, seed)
    assert data["complete"] and data["protocol"] == pilot.PROTOCOL
    assert previous["complete"] and previous["protocol"] == pilot.previous.PROTOCOL
    for report in (data, previous):
        assert (report["condition"], report["seed"], report["source_split"]) == (condition, seed, "train_cases")
    assert (data["train_horizon"], data["primary_horizon"], data["penalty"]) == (512, 512, 1.)
    assert (data["fold_rule"], data["new_repeats"]) == ("repeat % 4", list(pilot.REPEATS))
    assert (data["ridge_solves"], data["old_anchors"], data["new_anchors"]) == (4, 192, 192)
    for key in ("label_collection_interactions", "environment_initialization_probe_steps", "new_environment_interactions"):
        assert data[key] == labels[key]
    for key in ("evaluation_interactions", "policy_training_interactions", "policy_updates", "critic_updates",
                "selector_updates", "mlp_updates"):
        assert data[key] == 0
    case_map = {pilot.reference.case_key(c): c for c in cases}
    cached = {pilot.reference.case_key(r): r for r in previous["rows"]}
    assert len(previous["rows"]) == len(cached) == 192
    for row in data["rows"]:
        key = pilot.reference.case_key(row)
        old, case = cached[key], case_map[key]
        assert row["fold"] == row["repeat"] % 4
        assert row["reset_seed"] == old["reset_seed"] == case["reset_seed"]
        assert row["candidate_gains"] == old["candidate_gains"] == pilot.p.realized_gains(case)
        for name in pilot.previous.SELECTORS:
            assert all(row[k][name] == old[k][name] for k in ("scores", "choices", "selected_gains"))
        assert row["scores"]["ridge96"]["base"] == 0.
        for name in pilot.SELECTORS:
            assert np.isfinite(list(row["scores"][name].values())).all()
            assert row["choices"][name] == pilot.p.choose(row["scores"][name])
            assert row["selected_gains"][name] == row["candidate_gains"][row["choices"][name]]
    assert len(data["folds"]) == 4 and {f["fold"] for f in data["folds"]} == set(range(4))
    for fold in data["folds"]:
        fit, held = pilot.split_expanded(cases, new_cases, fold["fold"])
        assert fold["fit_reset_ids"] == sorted({c["reset_seed"] for c in fit})
        assert fold["held_reset_ids"] == sorted({c["reset_seed"] for c in held})
        assert (fold["parameters"], fold["penalty"], fold["train_horizon"], fold["fit_anchors"], fold["fit_nonbase_rows"]) == (
            243, 1., 512, 288, 1152)
    result = pilot.summarize_rows(data["rows"])
    assert data["summary"] == result
    assert not {"ridge128", "margin", "cost_sensitive"}.intersection(result["calibration"])
    return result


def summarize(root, reference_root, source_root):
    runs = []
    budget_keys = ("label_collection_interactions", "environment_initialization_probe_steps",
                   "new_environment_interactions", "evaluation_interactions", "policy_training_interactions",
                   "policy_updates", "critic_updates", "selector_updates", "mlp_updates", "ridge_solves")
    budget = dict.fromkeys(budget_keys, 0)
    for condition, seed in product(("v27", "v28"), (31, 33)):
        relative = Path(condition) / f"seed{seed}.json"
        data, previous, source = [json.loads((folder / relative).read_text())
                                  for folder in (root, reference_root, source_root)]
        labels = json.loads((root / condition / f"seed{seed}_labels.json").read_text())
        result = validate_run(data, labels, previous, source, condition, seed)
        runs.append({"condition": condition, "seed": seed, "summary": result,
            "training": [{k: f[k] for k in ("fold", "target_scale", "train_rmse512", "train_normalized_mse")}
                         for f in data["folds"]]})
        for key in budget:
            budget[key] += data[key]
    return {"protocol": pilot.PROTOCOL, "validated_runs": 4, "new_anchors": 768,
        "old_oof_anchors": 768, "budget": budget, "runs": runs,
        "passing_groups": sum(r["summary"]["group_passes"] for r in runs),
        "scaling_improves_groups": sum(r["summary"]["scaling_improves"] for r in runs),
        "eligible_for_fresh_validation_proposal": all(r["summary"]["group_passes"] for r in runs),
        "scope": "Extra-label development comparison on reused holdouts, not fresh confirmation "
                 "or equal-budget policy improvement."}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=pilot.OUTPUT_ROOT)
    parser.add_argument("--reference-root", type=Path, default=pilot.previous.OUTPUT_ROOT)
    parser.add_argument("--source-root", type=Path, default=pilot.p.OUTPUT_ROOT)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = summarize(args.root, args.reference_root, args.source_root)
    args.output.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    print(json.dumps({k: v for k, v in report.items() if k != "runs"}))
