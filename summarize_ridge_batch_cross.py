"""Validate both held batches and export the frozen ridge cross-evaluation."""

import argparse
import json
from itertools import product
from pathlib import Path

import numpy as np

import diagnose_ridge_batch_cross as pilot


def validate_run(data, expanded, source, labels, condition, seed):
    old = pilot.cv.training_cases(source, condition, seed)
    new = pilot.previous.validate_labels(labels, condition, seed, source["sources"])
    assert data["complete"] and data["protocol"] == pilot.PROTOCOL
    assert expanded["complete"] and expanded["protocol"] == pilot.previous.PROTOCOL
    for report in (data, expanded):
        assert (report["condition"], report["seed"], report["source_split"]) == (condition, seed, "train_cases")
    assert (data["train_horizon"], data["penalty"], data["fold_rule"]) == (512, 1., "repeat % 4")
    assert (data["ridge_solves"], data["reused_old_anchors"], data["reused_new_anchors"]) == (12, 192, 192)
    assert data["previous_all_control_pass"] == expanded["summary"]["group_passes"]
    for key in ("new_environment_interactions", "selector_updates", "policy_updates", "critic_updates", "mlp_updates"):
        assert data[key] == 0
    cases = {pilot.reference.case_key(c): c for c in old + new}
    cached = {pilot.reference.case_key(r): r for r in expanded["rows"]}
    assert len(cases) == 384 and len(expanded["rows"]) == len(cached) == 192
    for row in data["rows"]:
        case = cases[pilot.reference.case_key(row)]
        assert row["batch"] == ("old" if case["repeat"] < 16 else "new")
        assert (row["fold"], row["reset_seed"]) == (case["repeat"] % 4, case["reset_seed"])
        assert row["candidate_gains"] == pilot.p.realized_gains(case)
        expected = pilot.decision_row(case, row["batch"], {
            name: [row["scores"][name][label] for label in pilot.p.CANDIDATES] for name in pilot.MODELS})
        assert row == expected
        for name in pilot.SELECTORS:
            assert row["scores"][name]["base"] == 0. and np.isfinite(list(row["scores"][name].values())).all()
    assert len(data["folds"]) == 4 and {f["fold"] for f in data["folds"]} == set(range(4))
    for fold in data["folds"]:
        fits, held = pilot.split_batches(old, new, fold["fold"])
        assert fold["held_reset_ids"] == {batch: sorted({c["reset_seed"] for c in cases}) for batch, cases in held.items()}
        assert set(fold["fits"]) == set(pilot.MODELS)
        for name, cases in fits.items():
            meta = fold["fits"][name]
            assert meta["fit_reset_ids"] == sorted({c["reset_seed"] for c in cases})
            assert (meta["parameters"], meta["penalty"], meta["train_horizon"], meta["fit_anchors"], meta["fit_nonbase_rows"]) == (
                243, 1., 512, len(cases), len(cases) * 4)
        rows = [r for r in data["rows"] if r["fold"] == fold["fold"]]
        assert fold["reproduction_errors"] == pilot.check_reproduction(rows, cached)
    result = pilot.summarize_rows(data["rows"])
    assert result == data["summary"]
    return result


def summarize(root, expansion_root, source_root):
    runs = []
    for condition, seed in product(("v27", "v28"), (31, 33)):
        relative = Path(condition) / f"seed{seed}.json"
        data, expanded, source = [json.loads((folder / relative).read_text()) for folder in (root, expansion_root, source_root)]
        labels = json.loads((expansion_root / condition / f"seed{seed}_labels.json").read_text())
        result = validate_run(data, expanded, source, labels, condition, seed)
        runs.append({"condition": condition, "seed": seed, "summary": result,
            "previous_all_control_pass": data["previous_all_control_pass"],
            "reproduction_max_abs_error": {name: max(f["reproduction_errors"][name] for f in data["folds"])
                                           for name in ("old48", "combined96")},
            "training": {name: {key: float(np.mean([f["fits"][name][key] for f in data["folds"]]))
                                for key in ("train_rmse512", "train_normalized_mse")}
                         for name in pilot.MODELS}})
    return {"protocol": pilot.PROTOCOL, "validated_runs": 4, "oof_anchors": 1536,
        "ridge_solves": 48, "new_environment_interactions": 0, "selector_updates": 0,
        "previous_passing_groups": sum(r["previous_all_control_pass"] for r in runs), "runs": runs,
        "scope": "Descriptive batch cross-evaluation; no new acceptance gate or model selection. "
                 "Reused development labels, not independent confirmation or evidence of physical distribution shift."}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=pilot.OUTPUT_ROOT)
    parser.add_argument("--expansion-root", type=Path, default=pilot.previous.OUTPUT_ROOT)
    parser.add_argument("--source-root", type=Path, default=pilot.p.OUTPUT_ROOT)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = summarize(args.root, args.expansion_root, args.source_root)
    args.output.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    print(json.dumps({k: v for k, v in report.items() if k != "runs"}))
