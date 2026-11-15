"""Validate vector-head predictions, scalarization, and unchanged cached controls."""

import argparse
import json
from itertools import product
from pathlib import Path

import numpy as np

import diagnose_vector_advantage as pilot
from summarize_ridge_transform import compact_result


def validate_run(data, cached, legacy, source, labels, condition, seed):
    old = pilot.cv.training_cases(source, condition, seed)
    new = pilot.expansion.validate_labels(labels, condition, seed, source["sources"])
    assert data["complete"] and data["protocol"] == pilot.PROTOCOL
    assert cached["complete"] and cached["protocol"] == pilot.previous.PROTOCOL
    for report in (data, cached):
        assert (report["condition"], report["seed"], report["source_split"]) == (condition, seed, "train_cases")
    assert (data["train_horizon"], data["fold_rule"], data["penalty"], data["vector_fits"], data["linear_solves"]) == (512, "repeat % 4", 1., 4, 4)
    assert all(data[k] == 0 for k in ("new_environment_interactions", "policy_updates", "critic_updates", "mlp_updates"))
    assert data["legacy_controls"] == list(pilot.expansion.CONTROLS) and data["batch_controls"] == list(pilot.CONTROLS)
    assert data["legacy_aggregate"] == cached["legacy_aggregate"] == pilot.previous.validate_legacy(legacy, cached, condition, seed)
    lookup = {pilot.reference.case_key(r): r for r in cached["rows"]}
    cases = {pilot.reference.case_key(c): c for c in old + new}
    for row in data["rows"]:
        key = pilot.reference.case_key(row)
        case = cases[key]
        expected = pilot.join_row(lookup[key], row["vector_scores"], case["weight"])
        expected["vector_gains"] = pilot.vector_gains(case).tolist()
        assert row == expected
        assert row["candidate_gains"] == pilot.p.realized_gains(case)
        assert np.isfinite(row["vector_scores"]).all() and row["vector_scores"][0] == [0., 0.]
        np.testing.assert_allclose(np.asarray(row["vector_gains"]) @ np.asarray(row["weight"]),
            [row["candidate_gains"][name] for name in pilot.p.CANDIDATES], rtol=1e-12, atol=1e-12)
        for name in pilot.SELECTORS:
            assert row["scores"][name]["base"] == 0. and np.isfinite(list(row["scores"][name].values())).all()
    assert len(data["folds"]) == 4 and {f["fold"] for f in data["folds"]} == set(range(4))
    for fold in data["folds"]:
        fits, held = pilot.cross.split_batches(old, new, fold["fold"])
        assert fold["fit_reset_ids"] == sorted({c["reset_seed"] for c in fits["combined96"]})
        assert fold["held_reset_ids"] == {batch: sorted({c["reset_seed"] for c in cases}) for batch, cases in held.items()}
        assert (fold["parameters"], fold["heads"], fold["coefficients_per_head"]) == (486, 2, 243)
        assert (fold["penalty"], fold["linear_solves"], fold["fit_anchors"], fold["fit_nonbase_rows"]) == (1., 1, 288, 1152)
        scale = pilot.training_data(fits["combined96"])["targets"].double().std(0, unbiased=False).clamp_min(1e-3).tolist()
        assert fold["target_scales"] == scale
        assert 0. <= fold["normal_equation_residual_inf"] < 1e-10
        assert len(fold["train_vector_rmse512"]) == 2 and np.isfinite(fold["train_vector_rmse512"]).all()
    assert pilot.previous.summarize_rows(data["rows"], data["legacy_aggregate"]) == cached["summary"]
    result = pilot.summarize_rows(data["rows"], data["legacy_aggregate"])
    assert result == data["summary"]
    return {"condition": condition, "seed": seed, "summary": compact_result(result),
        "legacy_aggregate": data["legacy_aggregate"],
        "fitting": [{k: v for k, v in fold.items() if k not in ("fit_reset_ids", "held_reset_ids")} for fold in data["folds"]]}


def summarize(root, reference_root, expansion_root, source_root):
    runs = []
    for condition, seed in product(("v27", "v28"), (31, 33)):
        relative = Path(condition) / f"seed{seed}.json"
        data, cached, legacy, source = [json.loads((folder / relative).read_text()) for folder in (root, reference_root, expansion_root, source_root)]
        labels = json.loads((expansion_root / condition / f"seed{seed}_labels.json").read_text())
        runs.append(validate_run(data, cached, legacy, source, labels, condition, seed))
    return {"protocol": pilot.PROTOCOL, "validated_runs": 4, "vector_fits": 16, "objective_heads_fitted": 32,
        "linear_solves": 16, "new_environment_interactions": 0,
        "passing_groups": sum(r["summary"]["group_passes"] for r in runs),
        "all_groups_pass": all(r["summary"]["group_passes"] for r in runs), "runs": runs,
        "scope": "Structured vector predictor with doubled coefficients, reused development folds; not fresh confirmation."}


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
