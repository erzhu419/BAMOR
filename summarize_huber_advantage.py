"""Validate frozen Huber fits and retain both-batch decision and error results."""

import argparse
import json
from itertools import product
from pathlib import Path

import numpy as np

import diagnose_huber_advantage as pilot
from summarize_ridge_transform import compact_result


def validate_run(data, cached, legacy, source, labels, condition, seed):
    old = pilot.cv.training_cases(source, condition, seed)
    new = pilot.expansion.validate_labels(labels, condition, seed, source["sources"])
    assert data["complete"] and data["protocol"] == pilot.PROTOCOL
    assert cached["complete"] and cached["protocol"] == pilot.previous.PROTOCOL
    for report in (data, cached):
        assert (report["condition"], report["seed"], report["source_split"]) == (condition, seed, "train_cases")
    settings = {"delta": pilot.DELTA, "penalty": pilot.PENALTY, "max_iterations": pilot.MAX_ITERATIONS,
                "gradient_tolerance": pilot.GRAD_TOL}
    assert all(data[k] == v for k, v in settings.items())
    assert (data["train_horizon"], data["fold_rule"], data["huber_fits"]) == (512, "repeat % 4", 4)
    assert all(data[k] == 0 for k in ("new_environment_interactions", "policy_updates", "critic_updates", "mlp_updates"))
    assert data["legacy_controls"] == list(pilot.expansion.CONTROLS) and data["batch_controls"] == list(pilot.CONTROLS)
    assert data["legacy_aggregate"] == pilot.validate_legacy(legacy, cached, condition, seed)
    lookup = {pilot.reference.case_key(r): r for r in cached["rows"]}
    cases = {pilot.reference.case_key(c): c for c in old + new}
    for row in data["rows"]:
        key = pilot.reference.case_key(row)
        assert row["candidate_gains"] == pilot.p.realized_gains(cases[key])
        expected = pilot.join_row(lookup[key], [row["scores"][pilot.TARGET][name] for name in pilot.p.CANDIDATES])
        assert row == expected
        for name in pilot.SELECTORS:
            assert row["scores"][name]["base"] == 0. and np.isfinite(list(row["scores"][name].values())).all()
    assert len(data["folds"]) == 4 and {f["fold"] for f in data["folds"]} == set(range(4))
    for fold in data["folds"]:
        fits, held = pilot.previous.split_batches(old, new, fold["fold"])
        assert fold["fit_reset_ids"] == sorted({c["reset_seed"] for c in fits["combined96"]})
        assert fold["held_reset_ids"] == {batch: sorted({c["reset_seed"] for c in cases}) for batch, cases in held.items()}
        assert all(fold[k] == v for k, v in settings.items())
        assert (fold["parameters"], fold["fit_anchors"], fold["fit_nonbase_rows"]) == (243, 288, 1152)
        scale = float(pilot.p.training_data(fits["combined96"])["targets"].double().std(unbiased=False).clamp_min(1e-3))
        assert fold["target_scale"] == scale and fold["converged"]
        assert 0. <= fold["stationarity_inf"] <= pilot.GRAD_TOL
        assert 1 <= fold["linear_solves"] <= pilot.MAX_ITERATIONS
        history = fold["objective_history"]
        assert len(history) == fold["linear_solves"] + 1 and np.isfinite(history).all()
        assert all(0. <= b <= a + 1e-12 * (1. + abs(a)) for a, b in zip(history, history[1:]))
        groups = fold["weight_groups"]
        assert len(groups) == 6 and {(g["batch"], g["weight_index"]) for g in groups} == set(product(pilot.previous.BATCHES, range(3)))
        assert all(g["rows"] == 192 and 0 <= g["downweighted_rows"] <= 192 and 0. < g["min_weight"] <= g["mean_weight"] <= 1. for g in groups)
    assert data["linear_solves"] == sum(f["linear_solves"] for f in data["folds"])
    assert pilot.previous.summarize_rows(data["rows"]) == cached["summary"]
    result = pilot.summarize_rows(data["rows"], data["legacy_aggregate"])
    assert result == data["summary"]
    return {"condition": condition, "seed": seed, "summary": compact_result(result),
        "legacy_aggregate": data["legacy_aggregate"], "linear_solves": data["linear_solves"],
        "fitting": [{k: v for k, v in fold.items() if k not in ("fit_reset_ids", "held_reset_ids", "objective_history")}
                    for fold in data["folds"]]}


def summarize(root, reference_root, expansion_root, source_root):
    runs = []
    for condition, seed in product(("v27", "v28"), (31, 33)):
        relative = Path(condition) / f"seed{seed}.json"
        data, cached, legacy, source = [json.loads((folder / relative).read_text()) for folder in (root, reference_root, expansion_root, source_root)]
        labels = json.loads((expansion_root / condition / f"seed{seed}_labels.json").read_text())
        runs.append(validate_run(data, cached, legacy, source, labels, condition, seed))
    return {"protocol": pilot.PROTOCOL, "validated_runs": 4, "huber_fits": 16,
        "linear_solves": sum(r["linear_solves"] for r in runs), "new_environment_interactions": 0,
        "passing_groups": sum(r["summary"]["group_passes"] for r in runs),
        "all_groups_pass": all(r["summary"]["group_passes"] for r in runs), "runs": runs,
        "scope": "Reused development folds; a pass only warrants fresh confirmation, not main-algorithm integration."}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=pilot.OUTPUT_ROOT)
    parser.add_argument("--reference-root", type=Path, default=pilot.previous.OUTPUT_ROOT)
    parser.add_argument("--expansion-root", type=Path, default=pilot.expansion.OUTPUT_ROOT)
    parser.add_argument("--source-root", type=Path, default=pilot.p.OUTPUT_ROOT)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = summarize(args.root, args.reference_root, args.expansion_root, args.source_root)
    args.output.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    print(json.dumps({k: v for k, v in report.items() if k != "runs"}))
