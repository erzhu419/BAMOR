"""Validate learning-curve subsets and export only compact server-side results."""

import argparse
import json
from itertools import product
from pathlib import Path

import numpy as np

import diagnose_ridge_learning_curve as pilot


def validate_run(data, previous, source, condition, seed):
    assert data["complete"] and data["protocol"] == pilot.PROTOCOL
    assert previous["complete"] and previous["protocol"] == pilot.previous.PROTOCOL
    for report in (data, previous):
        assert (report["condition"], report["seed"], report["source_split"]) == (condition, seed, "train_cases")
    assert (data["train_horizon"], data["primary_horizon"], data["penalty"]) == (512, 512, 1.)
    assert (data["subset_rule"], data["sizes"], data["chains"], data["fold_rule"]) == (
        pilot.SUBSET_RULE, list(pilot.SIZES), 3, "repeat % 4")
    assert (data["ridge_solves"], data["reused_anchors"], data["reused_nonbase_labels"]) == (28, 192, 768)
    for key in ("new_environment_interactions", "selector_updates", "policy_updates", "critic_updates", "mlp_updates"):
        assert data[key] == 0
    cases = pilot.cv.training_cases(source, condition, seed)
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
        for name in pilot.SELECTORS:
            assert np.isfinite(list(row["scores"][name].values())).all()
            assert row["choices"][name] == pilot.p.choose(row["scores"][name])
            assert row["selected_gains"][name] == row["candidate_gains"][row["choices"][name]]
        assert all(row["scores"][name]["base"] == 0. for name in pilot.CURVE_SELECTORS)
    assert len(data["folds"]) == 4 and {f["fold"] for f in data["folds"]} == set(range(4))
    for fold in data["folds"]:
        fit, held = pilot.cv.split_fold(cases, fold["fold"])
        assert fold["held_reset_ids"] == sorted({c["reset_seed"] for c in held})
        assert len(fold["fits"]) == 7
        assert {(f["size"], f["chain"]) for f in fold["fits"]} == set(pilot.FIT_SPECS)
        for f in fold["fits"]:
            subset = pilot.fitting_subset(fit, fold["fold"], f["size"], f["chain"])
            assert f["fit_reset_ids"] == sorted({c["reset_seed"] for c in subset})
            assert set(f["fit_reset_ids"]).isdisjoint(fold["held_reset_ids"])
            assert (f["parameters"], f["penalty"], f["train_horizon"], f["fit_anchors"], f["fit_nonbase_rows"]) == (
                243, 1., 512, f["size"] * 3, f["size"] * 12)
        rows = [r for r in data["rows"] if r["fold"] == fold["fold"]]
        scores = np.asarray([[r["scores"]["ridge48"][label] for label in pilot.p.CANDIDATES] for r in rows])
        error = pilot.check_endpoint(scores, [cached[pilot.reference.case_key(r)] for r in rows])
        assert error == fold["endpoint_max_abs_error"]
    result = pilot.summarize_rows(data["rows"])
    assert result == data["summary"]
    return result


def summarize(root, reference_root, source_root):
    runs = []
    for condition, seed in product(("v27", "v28"), (31, 33)):
        relative = Path(condition) / f"seed{seed}.json"
        data, previous, source = [json.loads((folder / relative).read_text())
                                  for folder in (root, reference_root, source_root)]
        result = validate_run(data, previous, source, condition, seed)
        runs.append({"condition": condition, "seed": seed, "summary": result,
            "endpoint_max_abs_error": max(f["endpoint_max_abs_error"] for f in data["folds"]),
            "training": [{"size": size, "chain": chain,
                **{key: float(np.mean([f[key] for fold in data["folds"] for f in fold["fits"]
                                       if (f["size"], f["chain"]) == (size, chain)]))
                   for key in ("train_rmse512", "train_normalized_mse")}}
                for size, chain in pilot.FIT_SPECS]})
    return {"protocol": pilot.PROTOCOL, "validated_runs": 4, "unique_oof_anchors": 768,
        "ridge_solves": 112, "new_environment_interactions": 0, "selector_updates": 0,
        "runs": runs, "recent_improvement_groups": sum(r["summary"]["recent_improvement"] for r in runs),
        "broad_data_scaling_support": all(r["summary"]["recent_improvement"] for r in runs),
        "scope": "Reused development labels and overlapping folds/chains; no independent confirmation, "
                 "selector selection, ensemble, deployment or new collection authorization."}


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
