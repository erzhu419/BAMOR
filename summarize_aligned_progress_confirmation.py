"""Keep selector replication distinct from regression against the legacy learner."""

import argparse
import json
from pathlib import Path
from statistics import mean

import run_aligned_progress_confirmation as experiment
import summarize_aligned_progress_pilot as common


METRICS = ("final_hv", "archive_hv", "late_mean_hv", "late_min_hv")


def validate_run(data, condition, seed):
    assert seed in experiment.SEEDS and condition in experiment.METHODS
    assert data["aligned_progress_confirmation"] == experiment.metadata(condition)
    assert data["critic_candidate_context_rule"] == experiment.metadata(condition)["critic_candidate_context_rule"]
    assert data["critic_target_rule"] == "scalar_projection_of_soft_vector"
    assert data["bootstrap_terminal_rule"] == "terminated_only"
    assert not data.get("pareto_evals")
    assert "update_routing" not in data and "replay_context_refresh" not in data
    assert data["training_interactions"] == 50000
    assert data["evaluation_interactions"] == 4788000
    assert data["environment_initialization_probe_steps"] == 2
    result = common.metrics(data, seed, expected_method=experiment.METHODS[condition])
    result["late_snapshot_hv"] = result.pop("posthoc_late_snapshot_hv")
    result["late_mean_hv"] = result.pop("posthoc_late_mean_hv")
    result["late_min_hv"] = min(result["late_snapshot_hv"])
    result["late_mode_mean_hv"] = result.pop("posthoc_late_mode_mean_hv")
    result["selector"] = common.validate_selector(data)
    return result


def validate_triplet(runs):
    pair = common.validate_pair(runs["adaptive"], runs["blind"])
    a, old = runs["adaptive"], runs["legacy"]
    for key in ("evaluation_weights", "episode_seeds_per_mode", "ref_point", "policy_archive_snapshots"):
        assert a[key] == old[key], key
    assert a["ground_truth"]["mode_history"] == old["ground_truth"]["mode_history"]
    # Target conditioning changes learning from step 5000, not the preceding rollout.
    prefix = [r for r in a["episode_rewards"] if r["step"] <= 5000]
    assert prefix and prefix == [r for r in old["episode_rewards"] if r["step"] <= 5000]
    return {"aligned_pair": pair, "legacy_prelearning_episodes_matched": len(prefix)}


def aggregate(conditions):
    assert set(conditions) == set(experiment.METHODS)
    for c in conditions.values():
        assert sorted(c["per_seed"]) == list(experiment.SEEDS)
        for metric in METRICS:
            for stat, fn in (("mean", mean), ("min", min)):
                c[f"{metric}_{stat}"] = fn(r[metric] for r in c["per_seed"].values())
    comparisons = {}
    a = conditions["adaptive"]
    for comparator in ("blind", "legacy"):
        b = conditions[comparator]
        paired = {}
        for seed in experiment.SEEDS:
            x, y = a["per_seed"][seed], b["per_seed"][seed]
            paired[seed] = {metric: x[metric] - y[metric] for metric in METRICS}
            paired[seed]["final_mode_hv"] = [p - q for p, q in zip(x["final_mode_hv"], y["final_mode_hv"])]
        comparisons[comparator] = {"paired_adaptive_minus_control": paired,
            "mean_final_difference": mean(row["final_hv"] for row in paired.values()),
            "final_wins": sum(row["final_hv"] > 0 for row in paired.values())}
    b, old = conditions["blind"], conditions["legacy"]
    selector = {"final_mean_improved": a["final_hv_mean"] > b["final_hv_mean"],
        "final_minimum_not_worse": a["final_hv_min"] >= b["final_hv_min"],
        "late_mean_not_worse": a["late_mean_hv_mean"] >= b["late_mean_hv_mean"]}
    legacy = {f"{k}_not_worse": a[k] >= old[k] for k in ("final_hv_mean", "final_hv_min", "late_mean_hv_mean")}
    return {"conditions": conditions, "comparisons": comparisons,
        "selector_replication_checks": selector, "legacy_regression_checks": legacy,
        "selector_signal_replicated": all(selector.values()),
        "legacy_regression_not_observed": all(legacy.values()),
        "advance_to_larger_confirmation": all(selector.values()) and all(legacy.values())}


def summarize(root):
    conditions = {c: {"per_seed": {}} for c in experiment.METHODS}
    validation = {}
    budget = dict(training_interactions=0, evaluation_interactions=0, environment_initialization_probe_steps=0)
    for seed in experiment.SEEDS:
        runs = {}
        for condition, method in experiment.METHODS.items():
            data = json.loads((root / f"{method}_seed{seed}.json").read_text())
            runs[condition] = data
            conditions[condition]["per_seed"][seed] = validate_run(data, condition, seed)
            for key in budget:
                budget[key] += data[key]
        validation[seed] = validate_triplet(runs)
    return {"protocol": experiment.PROTOCOL, "validated_runs": 15, "expected": 15,
        "seeds": list(experiment.SEEDS), "validation": validation, **aggregate(conditions),
        "completed_run_budget": budget, "late_snapshot_steps": list(experiment.pilot.LATE_STEPS),
        "scope": "Five fresh training seeds, no pooling with development data. Final four-mode HV is primary. "
                 "Archive and final-encoder late snapshots are separate diagnostics, not online recovery. "
                 "The observed minimum is not a population guarantee; passing is not statistical significance, "
                 "cross-environment superiority or immediate adoption. Audit failed/restarted scheduler attempts "
                 "separately from the completed-run budget."}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=experiment.OUTPUT_ROOT)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = summarize(args.root)
    args.output.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    print(json.dumps({k: report[k] for k in ("validated_runs", "selector_replication_checks", "legacy_regression_checks", "advance_to_larger_confirmation")}))
