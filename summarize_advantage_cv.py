"""Compact server-side aggregation of the frozen training-only CV diagnosis."""

import argparse
import json
from itertools import product
from pathlib import Path

import diagnose_advantage_cv as pilot


def validate_run(data, condition, seed):
    assert data["complete"] and data["protocol"] == pilot.PROTOCOL
    assert (data["condition"], data["seed"], data["source_split"]) == (condition, seed, "train_cases")
    assert data["new_environment_interactions"] == data["policy_updates"] == data["critic_updates"] == 0
    assert (data["reused_anchors"], data["reused_nonbase_labels"]) == (192, 768)
    assert data["fold_rule"] == "repeat % 4"
    assert (data["mlp_updates"], data["ridge_solves"]) == (4 * pilot.MLP_UPDATES, 4)
    assert len(data["folds"]) == 4 and {f["fold"] for f in data["folds"]} == set(range(4))
    for row in data["rows"]:
        assert row["fold"] == row["repeat"] % 4
        assert row["reset_seed"] == pilot.pilot.reset_seed("train", seed, row["mode"], row["repeat"])
        for name in pilot.SELECTORS:
            assert row["choices"][name] == pilot.pilot.choose(row["scores"][name])
            assert row["selected_gains"][name] == row["candidate_gains"][row["choices"][name]]
    for fold in data["folds"]:
        held = sorted({r["reset_seed"] for r in data["rows"] if r["fold"] == fold["fold"]})
        fit = sorted({r["reset_seed"] for r in data["rows"] if r["fold"] != fold["fold"]})
        assert fold["fit_reset_ids"] == fit and fold["held_reset_ids"] == held
        assert set(fit).isdisjoint(held) and (len(fit), len(held)) == (48, 16)
        assert (fold["fit_anchors"], fold["held_anchors"], fold["fit_seed"]) == (144, 48, seed + 1000 * (fold["fold"] + 1))
        assert (fold["mlp"]["updates"], fold["mlp"]["data_rows"]) == (pilot.MLP_UPDATES, 576)
        assert (fold["ridge_coefficients"], fold["ridge_penalty"], fold["ridge_solves"]) == (243, 1., 1)
    result = pilot.summarize_rows(data["rows"])
    assert result == data["summary"], "Saved CV metrics/gates differ from recomputation"
    return result


def summarize(root):
    runs = []
    budget = dict.fromkeys(("new_environment_interactions", "policy_updates", "critic_updates", "mlp_updates", "ridge_solves"), 0)
    for condition, seed in product(("v27", "v28"), (31, 33)):
        path = root / condition / f"seed{seed}.json"
        data = json.loads(path.read_text())
        result = validate_run(data, condition, seed)
        runs.append({"condition": condition, "seed": seed, "source": str(path), "summary": result,
            "fold_fit_rmse": [{"fold": f["fold"], **f["train_rmse"]} for f in data["folds"]]})
        for key in budget:
            budget[key] += data[key]
    return {"protocol": pilot.PROTOCOL, "complete": True, "validated_runs": len(runs),
            "source_split": "train_cases", "oof_anchors": 768, "reused_nonbase_labels": 3072,
            "budget": budget, "runs": runs,
            "passing_groups": sum(r["summary"]["group_passes"] for r in runs),
            "eligible_for_fresh_validation_proposal": all(r["summary"]["group_passes"] for r in runs),
            "scope": "Development-only OOF comparison. No old validation split, pooled override, tuning or deployment approval."}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=pilot.OUTPUT_ROOT)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = summarize(args.root)
    args.output.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    print(json.dumps({k: v for k, v in report.items() if k != "runs"}))
