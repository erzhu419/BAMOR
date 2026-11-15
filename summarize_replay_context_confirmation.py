"""Summarize all five fixed pairs, keeping terminal and late stability separate."""

import argparse
import json
from pathlib import Path
from statistics import mean

import run_replay_context_confirmation as experiment
import summarize_replay_context_refresh as common


def validate_run(data, condition, seed):
    meta = data["replay_context_confirmation"]
    expected = {
        "protocol": experiment.PROTOCOL, "condition": condition, "complete": True,
        "seeds": list(experiment.SEEDS), "main_environment_steps": 50000,
        "total_training_interactions": 50000, "extra_environment_steps": 0,
        "extra_updates": 0, "normal_updates": 45001,
        "late_snapshot_steps": list(experiment.LATE_STEPS),
    }
    for key, value in expected.items():
        assert meta[key] == value, f"Mismatched confirmation field: {key}"
    refresh = data.get("replay_context_refresh")
    if condition == "refresh":
        assert refresh is not None and refresh["complete"]
        common.validate_trace(refresh)
    else:
        assert refresh is None
    result = common.metrics(data, seed)
    # These same diagnostics were exploratory in the pilot, fixed before this run.
    for key in tuple(result):
        if key.startswith("posthoc_"):
            result[key.removeprefix("posthoc_")] = result.pop(key)
    result["late_min_hv"] = min(result["late_snapshot_hv"])
    result["refresh_seconds"] = refresh["refresh_seconds"] if refresh else 0.
    return result


def aggregate(conditions):
    for c in conditions.values():
        assert sorted(c["per_seed"]) == list(experiment.SEEDS)
        for metric in ("final_hv", "archive_hv", "late_mean_hv", "late_min_hv"):
            for stat, fn in (("mean", mean), ("min", min)):
                c[f"{metric}_{stat}"] = fn(r[metric] for r in c["per_seed"].values())
    a, b = conditions["refresh"], conditions["stored"]
    paired = {}
    for seed in experiment.SEEDS:
        x, y = a["per_seed"][seed], b["per_seed"][seed]
        paired[seed] = {k: x[k] - y[k] for k in ("final_hv", "archive_hv", "late_mean_hv", "late_min_hv")}
        paired[seed]["final_mode_hv"] = [p - q for p, q in zip(x["final_mode_hv"], y["final_mode_hv"])]
    primary = {"final_mean_improved": a["final_hv_mean"] > b["final_hv_mean"],
               "final_minimum_improved": a["final_hv_min"] > b["final_hv_min"]}
    stability = {f"{key}_not_worse": a[key] >= b[key] for key in (
        "late_mean_hv_mean", "late_mean_hv_min", "late_min_hv_mean", "late_min_hv_min")}
    return {"conditions": conditions, "paired_refresh_minus_stored": paired,
            "final_wins": sum(p["final_hv"] > 0 for p in paired.values()),
            "primary_checks": primary, "secondary_stability_checks": stability,
            "ready_for_larger_confirmation": all(primary.values()) and all(stability.values())}


def summarize(root):
    conditions = {c: {"per_seed": {}} for c in experiment.CONDITIONS}
    validation = {}
    for seed in experiment.SEEDS:
        pair = {}
        for condition in experiment.CONDITIONS:
            path = root / condition / f"{experiment.pilot.METHOD}_seed{seed}.json"
            pair[condition] = json.loads(path.read_text())
            conditions[condition]["per_seed"][seed] = validate_run(pair[condition], condition, seed)
        validation[seed] = experiment.pilot.paired_protocol_check(pair["refresh"], pair["stored"])
    return {
        "protocol": experiment.PROTOCOL, "validated_runs": 10, "expected": 10,
        "source": str(root), "validation": validation, **aggregate(conditions),
        "new_training_interactions": 500000, "extra_environment_steps": 0, "extra_updates": 0,
        "late_snapshot_steps": list(experiment.LATE_STEPS),
        "scope": "Five independent seed pairs; checkpoints and modes are not extra replicates. "
                 "Final HV is primary; late-snapshot mean/minimum are prespecified secondary "
                 "stability diagnostics under the final encoder. All pairs must be present. "
                 "No merging with development seeds, interval retuning or best-seed selection. "
                 "Passing these small-sample checks is not a significance or cross-environment claim.",
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=experiment.OUTPUT_ROOT)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = summarize(args.root)
    args.output.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    print(json.dumps({k: report[k] for k in ("validated_runs", "primary_checks", "secondary_stability_checks")}))
