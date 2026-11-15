"""Recompute the frozen five-seed v28 gate without changing method identities."""

import argparse
import json
from pathlib import Path
from statistics import mean

import numpy as np

import run_mean_value_pilot as pilot
import summarize_replay_context_confirmation as controls
import summarize_replay_context_refresh as common


def variant_metrics(data, seed):
    meta = data["mean_value_pilot"]
    expected = {"protocol": pilot.PROTOCOL, "complete": True,
                "seeds": list(pilot.SEEDS), "mean_value_fraction": .5,
                "extra_training_updates": 0,
                "late_snapshot_steps": list(pilot.controls.LATE_STEPS)}
    for key, value in expected.items():
        assert meta[key] == value, f"Mismatched pilot field: {key}"
    actor = data["actor_update"]
    assert actor["rule"] == "half_selected_half_own_preference_mean_value"
    assert actor["mean_value_fraction"] == .5
    assert actor["updates"] == 45001
    assert actor["single_candidate_updates"] > 0 and actor["envelope_updates"] > 0
    assert actor["single_candidate_updates"] + actor["envelope_updates"] == actor["updates"]
    assert data["training_interactions"] == 50000
    assert data["evaluation_interactions"] == 19 * 4 * 21 * 3 * 1000
    assert data["environment_initialization_probe_steps"] == 2
    assert not data.get("replay_context_refresh")
    result = common.metrics(data, seed, expected_method=pilot.METHOD)
    for key in tuple(result):
        if key.startswith("posthoc_"):
            result[key.removeprefix("posthoc_")] = result.pop(key)
    result["late_min_hv"] = min(result["late_snapshot_hv"])
    result["actor_update"] = actor
    return result


def return_diagnostics(data):
    weights = np.asarray(data["evaluation_weights"])
    modes = []
    for front in data["pareto_per_mode"]:
        returns = np.asarray([p["return"] for p in front if p["archive_step"] == 50000])
        modes.append({"preference_grid_mean_return": returns.mean(axis=0).tolist(),
                      "preference_grid_min_return": returns.min(axis=0).tolist(),
                      "preference_grid_max_return": returns.max(axis=0).tolist(),
                      "own_preference_utility_mean": float((weights * returns).sum(axis=1).mean())})
    return modes


def aggregate(conditions):
    for c in conditions.values():
        assert sorted(c["per_seed"]) == list(pilot.SEEDS), "All five pairs are required"
        for metric in ("final_hv", "archive_hv", "late_mean_hv", "late_min_hv"):
            for stat, fn in (("mean", mean), ("min", min)):
                c[f"{metric}_{stat}"] = fn(r[metric] for r in c["per_seed"].values())
    a, b = conditions["v28"], conditions["v27"]
    paired = {}
    for seed in pilot.SEEDS:
        x, y = a["per_seed"][seed], b["per_seed"][seed]
        paired[seed] = {k: x[k] - y[k] for k in ("final_hv", "archive_hv", "late_mean_hv", "late_min_hv")}
        paired[seed]["final_mode_hv"] = [p - q for p, q in zip(x["final_mode_hv"], y["final_mode_hv"])]
    primary = {"final_mean_improved": a["final_hv_mean"] > b["final_hv_mean"],
               "final_minimum_improved": a["final_hv_min"] > b["final_hv_min"]}
    stability = {f"{key}_not_worse": a[key] >= b[key] for key in (
        "late_mean_hv_mean", "late_mean_hv_min", "late_min_hv_mean", "late_min_hv_min")}
    return {"conditions": conditions, "paired_v28_minus_v27": paired,
            "final_wins": sum(p["final_hv"] > 0 for p in paired.values()),
            "primary_checks": primary, "secondary_stability_checks": stability,
            "advance_to_fresh_confirmation": all(primary.values()) and all(stability.values())}


def summarize(root, reference_root):
    conditions = {c: {"per_seed": {}} for c in ("v27", "v28")}
    validation = {}
    budget = {k: 0 for k in ("training_interactions", "evaluation_interactions",
                             "environment_initialization_probe_steps")}
    for seed in pilot.SEEDS:
        path = root / f"{pilot.METHOD}_seed{seed}.json"
        reference_path = reference_root / f"{pilot.BASE}_seed{seed}.json"
        data, reference = [json.loads(p.read_text()) for p in (path, reference_path)]
        b, a = controls.validate_run(reference, "stored", seed), variant_metrics(data, seed)
        checks = pilot.paired_protocol_check(data, reference)
        assert data["mean_value_pilot"]["paired_protocol_checks"] == checks
        validation[seed] = {"paired_protocol_checks": checks, "hv_recomputed": True}
        for condition, metrics, raw in (("v27", b, reference), ("v28", a, data)):
            metrics["final_mode_returns"] = return_diagnostics(raw)
            conditions[condition]["per_seed"][seed] = metrics
        for key in budget:
            budget[key] += data[key]
    compute = {c: {key: sum(r[key] for r in d["per_seed"].values())
                   for key in ("training_seconds", "evaluation_seconds")}
               for c, d in conditions.items()}
    return {"protocol": pilot.PROTOCOL, "validated_runs": 10, "expected": 10,
            "sources": {"v28": str(root), "v27": str(reference_root)},
            "validation": validation, **aggregate(conditions),
            "new_interaction_budget": budget, "compute_seconds": compute,
            "late_snapshot_steps": list(pilot.controls.LATE_STEPS),
            "scope": "Five observed development pairs, not fresh confirmation. "
                     "The six frozen gates use seed-level mean-mode HV, not modes as replicates. "
                     "All snapshots use the final encoder; archive HV is distinct from final actor HV. "
                     "Return summaries average the fixed preference grid, not a single policy. "
                     "Controls are reused, so their steps are not new interactions. "
                     "Historical wall times are descriptive, not a matched compute benchmark."}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=pilot.OUTPUT_ROOT)
    parser.add_argument("--reference-root", type=Path, default=pilot.REFERENCE_ROOT)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = summarize(args.root, args.reference_root)
    args.output.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    print(json.dumps({k: report[k] for k in ("validated_runs", "primary_checks",
          "secondary_stability_checks", "final_wins", "advance_to_fresh_confirmation")}))
