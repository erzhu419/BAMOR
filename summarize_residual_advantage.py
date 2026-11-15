"""Small result export for the fixed critic-residual CV experiment."""

import argparse
import json
from itertools import product
from pathlib import Path

import diagnose_residual_advantage as pilot


def validate_run(data, previous, condition, seed):
    assert data["complete"] and data["protocol"] == pilot.PROTOCOL
    assert (data["condition"], data["seed"], data["source_split"]) == (condition, seed, "train_cases")
    for key in ("new_environment_interactions", "mlp_updates", "policy_updates", "critic_updates"):
        assert data[key] == 0
    assert (data["nnls_fits"], data["ridge_solves"], data["reused_anchors"], data["reused_nonbase_labels"]) == (4, 4, 192, 768)
    assert len(data["folds"]) == 4 and {f["fold"] for f in data["folds"]} == set(range(4))
    cached = {pilot.case_key(r): r for r in previous["rows"]}
    for row in data["rows"]:
        assert row["fold"] == row["repeat"] % 4
        old = cached[pilot.case_key(row)]
        assert row["reset_seed"] == old["reset_seed"] == pilot.p.reset_seed("train", seed, row["mode"], row["repeat"])
        assert row["candidate_gains"] == old["candidate_gains"]
        for name in pilot.cv.SELECTORS:
            assert all(row[key][name] == old[key][name] for key in ("scores", "choices", "selected_gains"))
        for name in pilot.SELECTORS:
            assert row["choices"][name] == pilot.p.choose(row["scores"][name])
            assert row["selected_gains"][name] == row["candidate_gains"][row["choices"][name]]
    for fold in data["folds"]:
        for name, matches in (("fit", False), ("held", True)):
            ids = {r["reset_seed"] for r in data["rows"] if (r["fold"] == fold["fold"]) == matches}
            assert fold[name + "_reset_ids"] == sorted(ids)
        assert set(fold["fit_reset_ids"]).isdisjoint(fold["held_reset_ids"])
        assert (fold["fit_rows"], fold["residual_coefficients"], fold["penalty"], fold["nnls_fits"], fold["ridge_solves"]) == (576, 77, 1., 1, 1)
    result = pilot.summarize_rows(data["rows"])
    assert result == data["summary"]
    return result


def summarize(root, reference_root):
    runs = []
    for condition, seed in product(("v27", "v28"), (31, 33)):
        path = root / condition / f"seed{seed}.json"
        data = json.loads(path.read_text())
        previous = json.loads((reference_root / condition / f"seed{seed}.json").read_text())
        result = validate_run(data, previous, condition, seed)
        runs.append({"condition": condition, "seed": seed, "source": str(path), "summary": result,
            "fits": [{k: f[k] for k in ("fold", "prior_coefficients", "train_rmse")} for f in data["folds"]]})
    return {"protocol": pilot.PROTOCOL, "validated_runs": 4, "oof_anchors": 768,
        "new_environment_interactions": 0, "mlp_updates": 0, "nnls_fits": 16, "ridge_solves": 16,
        "runs": runs, "passing_groups": sum(r["summary"]["group_passes"] for r in runs),
        "eligible_for_fresh_validation_proposal": all(r["summary"]["group_passes"] for r in runs),
        "scope": "Repeated development CV, not independent validation or deployment approval."}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=pilot.OUTPUT_ROOT)
    parser.add_argument("--reference-root", type=Path, default=pilot.cv.OUTPUT_ROOT)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = summarize(args.root, args.reference_root)
    args.output.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    print(json.dumps({k: v for k, v in report.items() if k != "runs"}))
