"""Validate and export the deterministic margin pilot without raw case pulls."""

import argparse
import json
from itertools import product
from pathlib import Path

import diagnose_margin_selection as pilot


def validate_run(data, previous, condition, seed):
    assert data["complete"] and data["protocol"] == pilot.PROTOCOL
    assert previous["complete"] and previous["protocol"] == pilot.previous.PROTOCOL
    for report in (data, previous):
        assert (report["condition"], report["seed"], report["source_split"]) == (condition, seed, "train_cases")
    for key in ("new_environment_interactions", "mlp_updates", "policy_updates", "critic_updates"):
        assert data[key] == 0
    assert (data["selector_updates"], data["reused_anchors"], data["reused_nonbase_labels"]) == (4 * pilot.UPDATES, 192, 768)
    assert len(data["folds"]) == 4 and {f["fold"] for f in data["folds"]} == set(range(4))
    cached = {pilot.reference.case_key(r): r for r in previous["rows"]}
    assert len(previous["rows"]) == len(cached) == 192
    for row in data["rows"]:
        assert row["fold"] == row["repeat"] % 4
        old = cached[pilot.reference.case_key(row)]
        assert row["reset_seed"] == old["reset_seed"] == pilot.p.reset_seed("train", seed, row["mode"], row["repeat"])
        assert row["candidate_gains"] == old["candidate_gains"]
        for name in pilot.previous.SELECTORS:
            assert all(row[key][name] == old[key][name] for key in ("scores", "choices", "selected_gains"))
        for name in pilot.SELECTORS:
            assert row["choices"][name] == pilot.p.choose(row["scores"][name])
            assert row["selected_gains"][name] == row["candidate_gains"][row["choices"][name]]
    for fold in data["folds"]:
        for name, matches in (("fit", False), ("held", True)):
            ids = {r["reset_seed"] for r in data["rows"] if (r["fold"] == fold["fold"]) == matches}
            assert fold[name + "_reset_ids"] == sorted(ids)
        assert set(fold["fit_reset_ids"]).isdisjoint(fold["held_reset_ids"])
        expected = {"updates": pilot.UPDATES, "parameters": 243, "fit_anchors": 144, "fit_nonbase_rows": 576,
            "learning_rate": .01, "penalty": 1., "downside_weight": 1., "initialization": "zeros",
            "objective": "cost_sensitive_structured_hinge", "batch": "full", "dtype": "float64"}
        assert all(fold[k] == v for k, v in expected.items())
        assert fold["train_hinge"] + 1e-10 >= fold["train_utility_regret"] >= -1e-10
    result = pilot.summarize_rows(data["rows"])
    assert result == data["summary"] and not {"margin", "cost_sensitive"}.intersection(result["calibration"])
    return result


def summarize(root, reference_root):
    runs = []
    for condition, seed in product(("v27", "v28"), (31, 33)):
        path = root / condition / f"seed{seed}.json"
        data = json.loads(path.read_text())
        previous = json.loads((reference_root / condition / f"seed{seed}.json").read_text())
        result = validate_run(data, previous, condition, seed)
        runs.append({"condition": condition, "seed": seed, "source": str(path), "summary": result,
            "fits": [{k: f[k] for k in ("fold", "train_mean_gain", "train_mean_negative_part", "losses",
                                        "train_hinge", "train_utility_regret")} for f in data["folds"]]})
    return {"protocol": pilot.PROTOCOL, "validated_runs": 4, "oof_anchors": 768,
        "new_environment_interactions": 0, "selector_updates": 16 * pilot.UPDATES,
        "runs": runs, "passing_groups": sum(r["summary"]["group_passes"] for r in runs),
        "eligible_for_fresh_validation_proposal": all(r["summary"]["group_passes"] for r in runs),
        "scope": "Repeated development CV, not fresh validation or deployment approval."}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=pilot.OUTPUT_ROOT)
    parser.add_argument("--reference-root", type=Path, default=pilot.previous.OUTPUT_ROOT)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = summarize(args.root, args.reference_root)
    args.output.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    print(json.dumps({k: v for k, v in report.items() if k != "runs"}))
