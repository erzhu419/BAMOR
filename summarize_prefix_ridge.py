"""Server-side validation and compact export of the fixed prefix ridge pilot."""

import argparse
import json
from itertools import product
from pathlib import Path

import diagnose_prefix_ridge as pilot


def validate_run(data, previous, source, condition, seed):
    assert data["complete"] and data["protocol"] == pilot.PROTOCOL
    assert previous["complete"] and previous["protocol"] == pilot.previous.PROTOCOL
    for report in (data, previous):
        assert (report["condition"], report["seed"], report["source_split"]) == (condition, seed, "train_cases")
    assert (data["train_horizon"], data["primary_horizon"]) == (128, 512)
    for key in ("new_environment_interactions", "mlp_updates", "policy_updates", "critic_updates", "selector_updates"):
        assert data[key] == 0
    assert (data["ridge_solves"], data["reused_anchors"], data["reused_nonbase_labels"]) == (4, 192, 768)
    assert len(data["folds"]) == 4 and {f["fold"] for f in data["folds"]} == set(range(4))
    cases = {pilot.reference.case_key(c): c for c in pilot.cv.training_cases(source, condition, seed)}
    cached = {pilot.reference.case_key(r): r for r in previous["rows"]}
    assert len(previous["rows"]) == len(cached) == 192
    for row in data["rows"]:
        key = pilot.reference.case_key(row)
        old, case = cached[key], cases[key]
        assert row["fold"] == row["repeat"] % 4
        assert row["reset_seed"] == old["reset_seed"] == case["reset_seed"]
        assert row["candidate_gains"] == old["candidate_gains"] == pilot.p.realized_gains(case)
        assert row["prefix_candidate_gains"] == dict(zip(pilot.p.CANDIDATES, pilot.horizon.paired_gains(case, 128)))
        for name in pilot.previous.SELECTORS:
            assert all(row[k][name] == old[k][name] for k in ("scores", "choices", "selected_gains"))
        for name in pilot.SELECTORS:
            assert row["choices"][name] == pilot.p.choose(row["scores"][name])
            assert row["selected_gains"][name] == row["candidate_gains"][row["choices"][name]]
    for fold in data["folds"]:
        for name, matches in (("fit", False), ("held", True)):
            ids = {r["reset_seed"] for r in data["rows"] if (r["fold"] == fold["fold"]) == matches}
            assert fold[name + "_reset_ids"] == sorted(ids)
        assert set(fold["fit_reset_ids"]).isdisjoint(fold["held_reset_ids"])
        expected = {"train_horizon": 128, "ridge_solves": 1, "parameters": 243, "penalty": 1.,
            "fit_anchors": 144, "fit_nonbase_rows": 576, "target_dtype": "float32", "solve_dtype": "float64"}
        assert all(fold[k] == v for k, v in expected.items())
    result = pilot.summarize_rows(data["rows"])
    assert result == data["summary"]
    assert not {"ridge128", "margin", "cost_sensitive"}.intersection(result["calibration"])
    return result


def summarize(root, reference_root, source_root):
    runs = []
    for condition, seed in product(("v27", "v28"), (31, 33)):
        relative = Path(condition) / f"seed{seed}.json"
        data = json.loads((root / relative).read_text())
        previous = json.loads((reference_root / relative).read_text())
        source = json.loads((source_root / relative).read_text())
        result = validate_run(data, previous, source, condition, seed)
        runs.append({"condition": condition, "seed": seed, "source": str(root / relative), "summary": result,
            "fits": [{k: f[k] for k in ("fold", "target_scale", "train_rmse128")} for f in data["folds"]]})
    return {"protocol": pilot.PROTOCOL, "validated_runs": 4, "oof_anchors": 768,
        "new_environment_interactions": 0, "selector_updates": 0, "ridge_solves": 16,
        "runs": runs, "passing_groups": sum(r["summary"]["group_passes"] for r in runs),
        "eligible_for_fresh_validation_proposal": all(r["summary"]["group_passes"] for r in runs),
        "scope": "Repeated development CV, not fresh validation or deployment approval."}


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
