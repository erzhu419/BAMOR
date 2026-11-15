"""Frozen shadow-pilot gates and descriptive, non-selecting fit diagnostics."""

import argparse
import json
from itertools import product
from pathlib import Path

import numpy as np
import torch

import diagnose_paired_advantage as pilot


GROUPS = tuple(product(("v27", "v28"), (31, 33)))
BUDGET_KEYS = ("label_collection_interactions", "evaluation_interactions",
               "environment_initialization_probe_steps", "policy_training_interactions",
               "policy_updates", "critic_updates")


def validate_run(data, condition, seed):
    expected = {"protocol": pilot.PROTOCOL, "complete": True, "condition": condition,
                "seed": seed, "method": pilot.g.METHODS[condition], "epsilon": .05,
                "weights": pilot.g.WEIGHTS.tolist(), "horizons": [32, 128, 512],
                "gamma": .99, "warmup": 128, "train_repeats": 16, "validation_repeats": 8,
                "candidates": list(pilot.CANDIDATES), "label_collection_interactions": 614400,
                "evaluation_interactions": 307200, "environment_initialization_probe_steps": 2,
                "policy_training_interactions": 0, "policy_updates": 0, "critic_updates": 0}
    for key, value in expected.items():
        assert data[key] == value, f"Mismatched {condition}/{seed} field: {key}"
    for key, value in {"updates": 2000, "data_rows": 768, "batch_size": 128,
                       "learning_rate": .0003, "hidden": 64}.items():
        assert data["fit"][key] == value, f"Mismatched fitting field: {key}"
    assert [r["update"] for r in data["fit"]["losses"]] == [500, 1000, 1500, 2000]
    reset_sets = []
    for split, repeats, budget_key in (("train", 16, "label_collection_interactions"),
                                       ("validation", 8, "evaluation_interactions")):
        cases = data[split + "_cases"]
        grid = set(product(range(4), range(3), range(repeats)))
        assert len(cases) == len(grid)
        assert {(c["mode"], c["weight_index"], c["repeat"]) for c in cases} == grid
        reset_sets.append({c["reset_seed"] for c in cases})
        for case in cases:
            assert case["reset_seed"] == pilot.reset_seed(split, seed, case["mode"], case["repeat"])
            assert case["weight"] == pilot.g.WEIGHTS[case["weight_index"]].tolist()
            assert ("external_scores" in case) == (split == "validation")
            for key, size in (("anchor_observation", 17), ("anchor_context", 13)):
                assert np.shape(case[key]) == (size,) and np.isfinite(case[key]).all()
            assert case["identity_matched"] and case["branches"]["base"] == case["branches"]["repeat"]
            assert set(case["branches"]) == set(pilot.p.BRANCHES)
            for branch in case["branches"].values():
                assert branch["steps"] == 512
                for key in ("discounted_returns", "raw_returns"):
                    assert set(branch[key]) == {"32", "128", "512"}
                    values = np.asarray(list(branch[key].values()))
                    assert values.shape == (3, 2) and np.isfinite(values).all()
            assert case["evaluation_interactions"] == 128 + sum(b["steps"] for b in case["branches"].values())
            scores = list(case["predictions"].values())
            assert set(case["predictions"]) == {"soft", "raw"}
            if split == "validation":
                assert case["external_scores"]["base"] == 0.
                scores.append(case["external_scores"])
            for score in scores:
                assert set(score) == set(pilot.CANDIDATES) and np.isfinite(list(score.values())).all()
        assert sum(c["evaluation_interactions"] for c in cases) == data[budget_key]
    assert reset_sets[0].isdisjoint(reset_sets[1])
    fixed = pilot.fixed_preference_scores(data["train_cases"])
    assert fixed == data["fixed_preference_scores"], "Fixed control must use training labels only"
    result = pilot.summarize_cases(data["validation_cases"], fixed)
    assert result == data["summary"], "Saved decisions/gates differ from recomputation"
    return result


@torch.no_grad()
def fit_diagnostics(data, checkpoint):
    tensors = pilot.training_data(data["train_cases"])
    saved = torch.load(checkpoint, map_location="cpu", weights_only=True)
    assert saved["action_dim"] == tensors["action_dim"] == 6
    with torch.random.fork_rng(devices=[]):
        model = pilot.PairedAdvantage(tensors)
    model.load_state_dict(saved["state_dict"])
    model.eval().requires_grad_(False)
    predictions = model(tensors["inputs"]).double().numpy()
    targets = np.asarray([pilot.realized_gains(c)[name] for c in data["train_cases"]
                          for name in pilot.CANDIDATES[1:]])
    return {"train_rmse": float(np.sqrt(np.mean((predictions - targets) ** 2))),
            "validation_rmse": data["summary"]["calibration"]["learned"]["rmse"],
            "train_targets_rms": float(np.sqrt(np.mean(targets ** 2))),
            "train_rows": len(targets), "validation_rows": 384,
            "scope": "Final saved evaluator on its own training labels; descriptive, not a new fit or gate."}


def compact_result(data, result):
    rows = result["cases"]
    selected = [c["selected_gains"]["learned"] for c in rows]
    best = [max(c["candidate_gains"].values()) for c in rows]
    worst = rows[int(np.argmin(selected))]
    cells = [{"mode": c["mode"], "weight_index": c["weight_index"],
              "selectors": {name: {k: v for k, v in stats.items() if k != "per_reset_gains"}
                            for name, stats in c["selectors"].items()}} for c in result["cells"]]
    prefixes = {}
    for h in data["horizons"]:
        gains = [float(np.dot(c["weight"],
                    np.asarray(c["branches"][r["choices"]["learned"]]["discounted_returns"][str(h)]) -
                    c["branches"]["base"]["discounted_returns"][str(h)]))
                 for c, r in zip(data["validation_cases"], rows)]
        prefixes[str(h)] = {"mean_gain": float(np.mean(gains)),
                            "mean_negative_part": float(np.mean(np.minimum(gains, 0.)))}
    return {**{k: result[k] for k in ("primary_horizon", "aggregate", "calibration", "checks", "group_passes")},
            "cells": cells,
            "selection_counts": {name: {label: sum(c["choices"][name] == label for c in rows)
                                         for label in pilot.CANDIDATES} for name in result["aggregate"]},
            "descriptive": {"selected_losses": sum(x < 0 for x in selected),
                "positive_candidate_anchors": sum(x > 0 for x in best),
                "loss_despite_positive_candidate": sum(x < 0 and b > 0 for x, b in zip(selected, best)),
                "oracle_mean_gain_upper_bound": float(np.mean(best)),
                "worst_selected_anchor": {k: worst[k] for k in ("mode", "weight_index", "reset_seed",
                                                               "choices", "selected_gains", "candidate_gains")},
                "fixed_512_choices_at_all_horizons": prefixes,
                "scope": "Label-dependent opportunity bound and fixed-choice prefixes only; no oracle selector or horizon tuning."}}


def summarize(root):
    runs, budget = [], dict.fromkeys((*BUDGET_KEYS, "evaluator_updates"), 0)
    for condition, seed in GROUPS:
        path = root / condition / f"seed{seed}.json"
        data = json.loads(path.read_text())
        result = validate_run(data, condition, seed)
        runs.append({"condition": condition, "seed": seed, "source": str(path),
                     "summary": compact_result(data, result),
                     "fit_diagnostics": fit_diagnostics(data, path.with_name(f"seed{seed}_advantage.pt"))})
        for key in BUDGET_KEYS:
            budget[key] += data[key]
        budget["evaluator_updates"] += data["fit"]["updates"]
    return {"protocol": pilot.PROTOCOL, "validated_runs": len(runs),
            "train_anchors": 768, "validation_anchors": 384, "identity_matches": 1152,
            "budget": budget, "passing_groups": sum(r["summary"]["group_passes"] for r in runs),
            "eligible_for_shadow_replication": all(r["summary"]["group_passes"] for r in runs),
            "runs": runs,
            "scope": "Frozen all-four-group rule, no pooled override. Observed policies and extra paired labels; "
                     "not an online operator router or equal-budget policy-improvement comparison."}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=pilot.OUTPUT_ROOT)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = summarize(args.root)
    args.output.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    print(json.dumps({k: v for k, v in report.items() if k != "runs"}))
