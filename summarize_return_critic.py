"""Validate shadow-critic terminal artifacts and apply the frozen advance rule."""

import argparse
import json
from itertools import product
from pathlib import Path

import numpy as np

import diagnose_return_critic as pilot


g = pilot.gradient
GROUPS = tuple(product(("v27", "v28"), (31, 33)))
BUDGET_KEYS = ("critic_data_interactions", "evaluation_interactions",
               "environment_initialization_probe_steps", "policy_training_interactions",
               "policy_updates")


def validate_run(data, condition, seed):
    expected = {"protocol": pilot.PROTOCOL, "complete": True,
                "condition": condition, "seed": seed, "method": g.METHODS[condition],
                "weights": g.WEIGHTS.tolist(), "horizons": [32, 128, 512],
                "warmup": 128, "epsilon": .05, "gamma": .99, "collection_noise_std": .1,
                "critic_data_interactions": 60000, "evaluation_interactions": 115200,
                "environment_initialization_probe_steps": 2,
                "policy_training_interactions": 0, "policy_updates": 0}
    for key, value in expected.items():
        assert data[key] == value, f"Mismatched {condition}/{seed} field: {key}"
    for key, value in {"updates": 20000, "data_rows": 60000, "batch_size": 256,
                       "learning_rate": .0003, "tau": .005}.items():
        assert data["fit"][key] == value, f"Mismatched fit field: {key}"
    episodes = data["collection_episodes"]
    expected_episodes = [{"mode": m, "weight_index": w, "repeat": r,
                          "reset_seed": pilot.reset_seed("train", seed, m, r),
                          "noise_seed": pilot.noise_seed(seed, m, w, r), "steps": 1000}
                         for m, w, r in product(range(4), range(3), range(5))]
    assert episodes == expected_episodes, "Incomplete or mismatched collection episodes"
    train_seeds = {e["reset_seed"] for e in episodes}
    cases = data["cases"]
    assert len(cases) == 36
    assert {(c["mode"], c["weight_index"], c["repeat"]) for c in cases} == set(
        product(range(4), range(3), range(3))), "Incomplete held-out case grid"
    for case in cases:
        assert case["reset_seed"] == pilot.reset_seed("test", seed, case["mode"], case["repeat"])
        assert case["reset_seed"] not in train_seeds
        assert case["weight"] == g.WEIGHTS[case["weight_index"]].tolist()
        assert case["evaluation_interactions"] == 3200
        assert case["identity_matched"] and case["branches"]["base"] == case["branches"]["repeat"]
        assert set(case["branches"]) == set(pilot.BRANCHES)
        for branch in case["branches"].values():
            assert branch["steps"] == 512
            for key in ("discounted_returns", "raw_returns"):
                assert set(branch[key]) == {"32", "128", "512"}
                values = np.asarray(list(branch[key].values()))
                assert values.shape == (3, 2) and np.isfinite(values).all()
        for scores in case["predictions"].values():
            assert set(scores) == set(pilot.BRANCHES) - {"repeat"}
            assert np.isfinite(list(scores.values())).all()
    assert sum(e["steps"] for e in episodes) == data["critic_data_interactions"]
    assert sum(c["evaluation_interactions"] for c in cases) == data["evaluation_interactions"]
    summary = pilot.summarize_cases(cases)
    assert summary == data["summary"], "Saved summary differs from rollout recomputation"
    return summary


def advance_rule(runs):
    assert len(runs) == 4 and {(r["condition"], r["seed"]) for r in runs} == set(GROUPS)
    per_group = {}
    for run in runs:
        soft, raw = (run["summary"]["aggregate"][name] for name in ("soft", "raw"))
        per_group[f"{run['condition']}/seed{run['seed']}"] = {
            "raw_plus_base_positive": raw["mean_plus_base"] > 0,
            "raw_plus_base_not_worse": raw["mean_plus_base"] >= soft["mean_plus_base"]}
    counts = {name: sum(r["summary"]["aggregate"][name]["positive_plus_minus_cells"] for r in runs)
              for name in ("soft", "raw")}
    checks = {key: all(row[key] for row in per_group.values()) for key in (
        "raw_plus_base_positive", "raw_plus_base_not_worse")}
    checks["positive_plus_minus_cells_increased"] = counts["raw"] > counts["soft"]
    return {"per_group_checks": per_group, "primary_checks": checks,
            "positive_plus_minus_cells": counts, "total_cells": 48,
            "advance_to_shadow_replication": all(checks.values())}


def summarize(root):
    runs, budget = [], dict.fromkeys(BUDGET_KEYS, 0)
    for condition, seed in GROUPS:
        path = root / condition / f"seed{seed}.json"
        data = json.loads(path.read_text())
        summary = validate_run(data, condition, seed)
        secondary = {}
        for horizon in (32, 128):
            # Reuse the identical cell weighting; prefixes never alter the gate.
            cases = [{**c, "branches": {
                name: {"discounted_returns": {"512": b["discounted_returns"][str(horizon)]}}
                for name, b in c["branches"].items()}} for c in data["cases"]]
            secondary[str(horizon)] = pilot.summarize_cases(cases)["aggregate"]
        runs.append({"condition": condition, "seed": seed, "source": str(path),
                     "summary": summary, "fit": data["fit"],
                     "secondary_horizon_aggregates": secondary})
        for key in budget:
            budget[key] += data[key]
    budget["critic_updates"] = sum(r["fit"]["updates"] for r in runs)
    return {"protocol": pilot.PROTOCOL, "validated_runs": len(runs),
            "validated_cases": 144, "identity_matches": 144,
            "new_budget": budget, "runs": runs, **advance_rule(runs),
            "scope": "Observed development models; within-model paired action interventions. "
                     "Three resets per mode/preference cell, primary horizon 512. "
                     "Raw critics have extra balanced data and train-only normalization. "
                     "Continuing TD targets versus finite-horizon rollout contrasts. "
                     "Counts are descriptive, not significance tests or deployment approval."}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=pilot.OUTPUT_ROOT)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = summarize(args.root)
    args.output.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    print(json.dumps({k: v for k, v in report.items() if k != "runs"}))
