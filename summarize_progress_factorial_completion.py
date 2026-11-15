"""Four-arm attribution with the already failed v27 advancement gate preserved."""

import argparse
import json
from pathlib import Path
from statistics import mean

import run_progress_factorial_completion as experiment
import summarize_aligned_progress_confirmation as previous


ARMS = ("aligned_adaptive", "aligned_blind", "legacy_adaptive", "legacy_blind")
BUDGET_KEYS = ("training_interactions", "evaluation_interactions", "environment_initialization_probe_steps")


def validate_new_run(data, seed):
    assert seed in experiment.SEEDS
    assert data["progress_factorial_completion"] == experiment.metadata()
    assert data["critic_candidate_context_rule"] == "candidate_preference"
    assert tuple(data[k] for k in BUDGET_KEYS) == (50000, 4788000, 2)
    result = previous.common.validate_run(data, "blind", seed, aligned=False)
    result["late_mode_mean_hv"] = result.pop("posthoc_late_mode_mean_hv")
    return result


def aggregate(conditions):
    assert set(conditions) == set(ARMS)
    for condition in conditions.values():
        assert sorted(condition["per_seed"]) == list(experiment.SEEDS)
        for metric in previous.METRICS:
            for stat, fn in (("mean", mean), ("min", min)):
                condition[f"{metric}_{stat}"] = fn(r[metric] for r in condition["per_seed"].values())
    coefficients = {
        "aligned_selector": (1, -1, 0, 0),
        "legacy_selector": (0, 0, 1, -1),
        "target_with_adaptive": (1, 0, -1, 0),
        "target_with_blind": (0, 1, 0, -1),
        "selector_by_target_interaction": (1, -1, -1, 1),
    }
    contrasts = {}
    for name, coefs in coefficients.items():
        paired = {}
        for seed in experiment.SEEDS:
            rows = [conditions[arm]["per_seed"][seed] for arm in ARMS]
            paired[seed] = {k: sum(c * r[k] for c, r in zip(coefs, rows)) for k in previous.METRICS}
            paired[seed]["final_mode_hv"] = [sum(c * r["final_mode_hv"][m] for c, r in zip(coefs, rows)) for m in range(4)]
        contrasts[name] = {"per_seed": paired,
            "mean": {k: mean(r[k] for r in paired.values()) for k in previous.METRICS},
            "final_sign_counts": {"positive": sum(r["final_hv"] > 0 for r in paired.values()),
                                  "zero": sum(r["final_hv"] == 0 for r in paired.values()),
                                  "negative": sum(r["final_hv"] < 0 for r in paired.values())}}
    a, b = conditions["legacy_adaptive"], conditions["legacy_blind"]
    checks = {"final_mean_improved": a["final_hv_mean"] > b["final_hv_mean"],
              "final_minimum_not_worse": a["final_hv_min"] >= b["final_hv_min"],
              "late_mean_not_worse": a["late_mean_hv_mean"] >= b["late_mean_hv_mean"]}
    return {"conditions": conditions, "contrasts": contrasts,
            "legacy_selector_checks": checks, "legacy_selector_checks_passed": all(checks.values())}


def summarize(root, reference_root, reference_summary):
    reference = json.loads(reference_summary.read_text())
    assert reference["protocol"] == experiment.previous.PROTOCOL
    assert reference["seeds"] == list(experiment.SEEDS)
    assert reference["validated_runs"] == reference["expected"] == 15
    assert reference["completed_run_budget"] == dict(zip(BUDGET_KEYS, (750000, 71820000, 30)))
    conditions = {arm: {"per_seed": {int(s): r for s, r in reference["conditions"][old]["per_seed"].items()}}
                  for arm, old in zip(ARMS[:3], ("adaptive", "blind", "legacy"))}
    conditions["legacy_blind"] = {"per_seed": {}}
    validation, budget = {}, dict.fromkeys(BUDGET_KEYS, 0)
    for seed in experiment.SEEDS:
        data = json.loads((root / f"{experiment.METHOD}_seed{seed}.json").read_text())
        conditions["legacy_blind"]["per_seed"][seed] = validate_new_run(data, seed)
        legacy = json.loads((reference_root / f"{experiment.previous.METHODS['legacy']}_seed{seed}.json").read_text())
        assert (legacy["method"], legacy["seed"]) == (experiment.previous.METHODS["legacy"], seed)
        validation[seed] = previous.common.validate_pair(legacy, data)
        for key in budget:
            budget[key] += data[key]
    return {"protocol": experiment.PROTOCOL, "seeds": list(experiment.SEEDS),
        "validated_new_runs": 5, "reused_validated_runs": 15, "expected_total": 20,
        "sources": {"new_root": str(root), "reference_root": str(reference_root),
                    "reference_summary": str(reference_summary)},
        "new_legacy_pair_validation": validation, **aggregate(conditions),
        "previous_decision": {k: reference[k] for k in ("selector_replication_checks", "legacy_regression_checks",
                              "selector_signal_replicated", "legacy_regression_not_observed", "advance_to_larger_confirmation")},
        "new_completed_run_budget": budget,
        "reused_completed_run_budget": reference["completed_run_budget"],
        "all_completed_run_budget": {k: budget[k] + reference["completed_run_budget"][k] for k in budget},
        "scope": "Completion was chosen after observing three arms on these five seeds: attribution, not a new "
                 "independent confirmation. Preserve the previous v27 decision. A positive interaction cannot "
                 "repair absolute regression. Legacy selector checks support only proposing a fresh paired "
                 "confirmation, not adoption. Report every seed, observed minima and late snapshots; archive "
                 "is secondary. Fixed final encoders are not online recovery. Audit failed/restarted attempts "
                 "separately; terminal budgets and descriptive signs are not population guarantees."}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=experiment.OUTPUT_ROOT)
    parser.add_argument("--reference-root", type=Path, default=experiment.previous.OUTPUT_ROOT)
    parser.add_argument("--reference-summary", type=Path, default=experiment.REFERENCE_SUMMARY)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = summarize(args.root, args.reference_root, args.reference_summary)
    args.output.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    print(json.dumps({k: report[k] for k in ("validated_new_runs", "reused_validated_runs", "legacy_selector_checks", "previous_decision")}))
