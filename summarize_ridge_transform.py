"""Validate donor-transform controls and retain compact per-cell diagnostics."""

import argparse
import json
from itertools import product
from pathlib import Path

import numpy as np

import diagnose_ridge_transform as pilot


def validate_run(data, cached, source, labels, condition, seed):
    old = pilot.cv.training_cases(source, condition, seed)
    new = pilot.previous.previous.validate_labels(labels, condition, seed, source["sources"])
    assert data["complete"] and data["protocol"] == pilot.PROTOCOL
    assert cached["complete"] and cached["protocol"] == pilot.previous.PROTOCOL
    for report in (data, cached):
        assert (report["condition"], report["seed"], report["source_split"]) == (condition, seed, "train_cases")
    assert (data["train_horizon"], data["penalty"], data["fold_rule"]) == (512, 1., "repeat % 4")
    assert data["transform_donors"] == pilot.DONORS
    assert (data["ridge_solves"], data["reused_old_anchors"], data["reused_new_anchors"]) == (8, 192, 192)
    assert data["previous_all_control_pass"] == cached["previous_all_control_pass"]
    for key in ("new_environment_interactions", "selector_updates", "policy_updates", "critic_updates", "mlp_updates"):
        assert data[key] == 0
    cases = {pilot.reference.case_key(c): c for c in old + new}
    lookup = {pilot.reference.case_key(r): r for r in cached["rows"]}
    assert len(cached["rows"]) == len(lookup) == 384
    for row in data["rows"]:
        key = pilot.reference.case_key(row)
        case, original = cases[key], lookup[key]
        assert row["batch"] == ("old" if case["repeat"] < 16 else "new")
        assert (row["fold"], row["reset_seed"]) == (case["repeat"] % 4, case["reset_seed"])
        assert row["candidate_gains"] == pilot.p.realized_gains(case)
        for name in pilot.previous.SELECTORS:
            assert all(row[k][name] == original[k][name] for k in ("scores", "choices", "selected_gains"))
        expected = pilot.join_row(original, {name: [row["scores"][name][label] for label in pilot.p.CANDIDATES]
                                             for name in pilot.DONORS})
        assert row == expected
        for name in pilot.SELECTORS:
            assert row["scores"][name]["base"] == 0. and np.isfinite(list(row["scores"][name].values())).all()
    assert len(data["folds"]) == 4 and {f["fold"] for f in data["folds"]} == set(range(4))
    for fold in data["folds"]:
        fits, held = pilot.previous.split_batches(old, new, fold["fold"])
        assert fold["fit_reset_ids"] == sorted({c["reset_seed"] for c in fits["combined96"]})
        assert fold["held_reset_ids"] == {batch: sorted({c["reset_seed"] for c in cases}) for batch, cases in held.items()}
        assert set(fold["fits"]) == set(pilot.DONORS)
        target_scale = float(pilot.p.training_data(fits["combined96"])["targets"].double().std(unbiased=False).clamp_min(1e-3))
        for name, meta in fold["fits"].items():
            assert meta["transform_reset_ids"] == sorted({c["reset_seed"] for c in fits[pilot.DONORS[name]]})
            assert (meta["transform_donor"], meta["parameters"], meta["penalty"], meta["train_horizon"]) == (
                pilot.DONORS[name], 243, 1., 512)
            assert (meta["fit_anchors"], meta["fit_nonbase_rows"], meta["transform_anchors"], meta["transform_nonbase_rows"]) == (
                288, 1152, 144, 576)
            assert meta["target_scale"] == target_scale
    assert pilot.previous.summarize_rows(data["rows"]) == cached["summary"]
    result = pilot.summarize_rows(data["rows"])
    assert result == data["summary"]
    return result


def compact_result(result):
    return {**result, "batches": {batch: {**block, "cells": [
        {**cell, "metrics": {name: {k: v for k, v in metrics.items() if k != "selection_counts"}
                            for name, metrics in cell["metrics"].items()}}
        for cell in block["cells"]]} for batch, block in result["batches"].items()}}


def summarize(root, reference_root, expansion_root, source_root):
    runs = []
    for condition, seed in product(("v27", "v28"), (31, 33)):
        relative = Path(condition) / f"seed{seed}.json"
        data, cached, source = [json.loads((folder / relative).read_text()) for folder in (root, reference_root, source_root)]
        labels = json.loads((expansion_root / condition / f"seed{seed}_labels.json").read_text())
        result = validate_run(data, cached, source, labels, condition, seed)
        runs.append({"condition": condition, "seed": seed, "summary": compact_result(result),
            "previous_all_control_pass": data["previous_all_control_pass"],
            "training": {name: {key: float(np.mean([f["fits"][name][key] for f in data["folds"]]))
                                for key in ("train_rmse512", "train_normalized_mse")}
                         for name in pilot.DONORS}})
    return {"protocol": pilot.PROTOCOL, "validated_runs": 4, "oof_anchors": 1536, "ridge_solves": 32,
        "new_environment_interactions": 0, "selector_updates": 0,
        "previous_passing_groups": sum(r["previous_all_control_pass"] for r in runs), "runs": runs,
        "scope": "Fixed-label input-transform diagnostic, with cached controls unchanged. "
                 "No transform selection, new acceptance gate or independent confirmation."}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=pilot.OUTPUT_ROOT)
    parser.add_argument("--reference-root", type=Path, default=pilot.previous.OUTPUT_ROOT)
    parser.add_argument("--expansion-root", type=Path, default=pilot.previous.previous.OUTPUT_ROOT)
    parser.add_argument("--source-root", type=Path, default=pilot.p.OUTPUT_ROOT)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = summarize(args.root, args.reference_root, args.expansion_root, args.source_root)
    args.output.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    print(json.dumps({k: v for k, v in report.items() if k != "runs"}))
