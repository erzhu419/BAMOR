"""Validate and compact the fixed-coordinate label influence diagnostic."""

import argparse
import json
from itertools import product
from pathlib import Path

import numpy as np

import diagnose_ridge_influence as pilot
from summarize_ridge_transform import compact_result


def validate_run(data, cached, source, labels, condition, seed):
    old = pilot.cv.training_cases(source, condition, seed)
    new = pilot.previous.previous.validate_labels(labels, condition, seed, source["sources"])
    assert data["complete"] and data["protocol"] == pilot.PROTOCOL
    assert cached["complete"] and cached["protocol"] == pilot.previous.PROTOCOL
    for report in (data, cached):
        assert (report["condition"], report["seed"], report["source_split"]) == (condition, seed, "train_cases")
    assert (data["train_horizon"], data["penalty"], data["fold_rule"]) == (512, 1., "repeat % 4")
    assert (data["transform"], data["target_scale"]) == ("combined96 fitting inputs", "combined96 fitting labels")
    assert (data["model_solves"], data["attribution_solves"], data["new_environment_interactions"], data["optimizer_updates"]) == (8, 4, 0, 0)
    assert data["previous_all_control_pass"] == cached["previous_all_control_pass"]
    lookup = {pilot.reference.case_key(r): r for r in cached["rows"]}
    cases = {pilot.reference.case_key(c): c for c in old + new}
    for row in data["rows"]:
        key = pilot.reference.case_key(row)
        assert row["candidate_gains"] == pilot.p.realized_gains(cases[key])
        for field in ("batch", "mode", "weight_index", "repeat", "reset_seed", "fold"):
            assert row[field] == lookup[key][field]
        for name in pilot.SELECTORS:
            assert row["scores"][name]["base"] == 0. and np.isfinite(list(row["scores"][name].values())).all()
            assert row["choices"][name] == pilot.p.choose(row["scores"][name])
            assert row["selected_gains"][name] == row["candidate_gains"][row["choices"][name]]
        for name in pilot.previous.SELECTORS:
            assert all(row[field][name] == lookup[key][field][name] for field in ("scores", "choices", "selected_gains"))
    assert len(data["folds"]) == 4 and {f["fold"] for f in data["folds"]} == set(range(4))
    for fold in data["folds"]:
        fits, held = pilot.previous.split_batches(old, new, fold["fold"])
        assert fold["fit_reset_ids"] == {name: sorted({c["reset_seed"] for c in fit}) for name, fit in fits.items()}
        assert fold["held_reset_ids"] == {name: sorted({c["reset_seed"] for c in fit}) for name, fit in held.items()}
        assert len(fold["resets"]) == 48
        assert sorted(r["reset_seed"] for r in fold["resets"]) == fold["fit_reset_ids"]["new48"]
        assert [r["weight_index"] for r in fold["preferences"]] == [0, 1, 2]
        assert fold["parameters"] == 243 and fold["target_scale"] > 0.
        for key in ("coefficient_error", "score_error", "cached_score_error"):
            assert np.isfinite(fold[key]) and 0. <= fold[key] < 1e-7
        for batch in pilot.previous.BATCHES:
            rows = [r for r in data["rows"] if r["batch"] == batch and r["fold"] == fold["fold"]]
            assert fold["changed_choices"][batch] == sum(r["choices"][pilot.OLD] != r["choices"]["combined96"] for r in rows)
            components = fold["components"]
            margin = lambda r: r["batches"][batch]["mean_switch_margin"]
            for group in (fold["resets"], fold["preferences"], [components["labels"], components["feedback"]]):
                np.testing.assert_allclose(sum(map(margin, group)) + margin(components["ridge"]), margin(components["total"]), rtol=1e-9, atol=1e-10)
    assert pilot.previous.summarize_rows(data["rows"]) == cached["summary"]
    result = pilot.summarize_rows(data["rows"])
    assert result == data["summary"]
    return {"condition": condition, "seed": seed, "summary": compact_result(result),
            "folds": [pilot.compact_fold(f) for f in data["folds"]]}


def summarize(root, reference_root, expansion_root, source_root):
    runs = []
    for condition, seed in product(("v27", "v28"), (31, 33)):
        relative = Path(condition) / f"seed{seed}.json"
        data, cached, source = [json.loads((folder / relative).read_text()) for folder in (root, reference_root, source_root)]
        labels = json.loads((expansion_root / condition / f"seed{seed}_labels.json").read_text())
        runs.append(validate_run(data, cached, source, labels, condition, seed))
    return {"protocol": pilot.PROTOCOL, "validated_runs": 4, "model_solves": 32, "attribution_solves": 16,
        "new_environment_interactions": 0, "optimizer_updates": 0, "runs": runs,
        "scope": "Exact fixed-coordinate attribution, not deletion effects or new selector acceptance. "
                 "Top shares use summed reset RMS magnitudes, not an additive share of total score change; folds overlap."}


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
