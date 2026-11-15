"""Prefix stability and reset-level influence of existing paired-return labels."""

import argparse
import json
from itertools import combinations, product
from pathlib import Path

import numpy as np

import diagnose_margin_selection as margin


p, cv = margin.p, margin.cv
PROTOCOL = "paired_horizon_attribution_v1"
HORIZONS = (32, 128, 512)
TOLERANCE = 1e-6
OUTPUT = Path("results/iclr_horizon_attribution_20260921_summary.json")
CONTRASTS = (("margin", "ridge"), ("ridge", "soft"), ("ridge", "raw"))


def paired_gains(case, horizon):
    returns = [np.asarray(case["branches"][a]["discounted_returns"][str(horizon)], dtype=np.float64)
               for a in p.CANDIDATES]
    gains = np.asarray([np.dot(case["weight"], value - returns[0]) for value in returns])
    assert gains.shape == (5,) and np.isfinite(gains).all() and gains[0] == 0.
    return gains


def balanced_mean(values, cases):
    values = np.asarray(values)
    cells = sorted({(c["mode"], c["weight_index"]) for c in cases})
    return float(np.mean([values[[i for i, c in enumerate(cases)
        if (c["mode"], c["weight_index"]) == cell]].mean() for cell in cells]))


def outcome(values, cases):
    return {"mean_gain": balanced_mean(values, cases),
            "mean_negative_part": balanced_mean(np.minimum(values, 0.), cases)}


def reset_mass(mass, cases):
    mass = np.asarray(mass, dtype=np.float64)
    assert mass.shape == (len(cases),) and np.isfinite(mass).all() and (mass >= 0.).all()
    groups = []
    for reset in sorted({c["reset_seed"] for c in cases}):
        ids = [i for i, c in enumerate(cases) if c["reset_seed"] == reset]
        first = cases[ids[0]]
        groups.append({"reset_seed": reset, "mode": first["mode"], "repeat": first["repeat"],
                       "anchors": len(ids), "mass": float(mass[ids].sum())})
    groups.sort(key=lambda g: (-g["mass"], g["reset_seed"]))
    total = float(mass.sum())
    return {"total": total, "reset_groups": len(groups),
            "top1_share": groups[0]["mass"] / total if total else None,
            "top4_share": sum(g["mass"] for g in groups[:4]) / total if total else None,
            "top_groups": groups[:4]}


def reversal_counts(before, after):
    eligible = (np.abs(before) > TOLERANCE) & (np.abs(after) > TOLERANCE)
    flips = int(np.sum(eligible & (np.sign(before) != np.sign(after))))
    n = int(eligible.sum())
    return {"reversals": flips, "eligible": n, "total": before.size,
            "rate": flips / n if n else None}


def stability(before, after, cases):
    i, j = np.triu_indices(5, 1)
    early_choice, late_choice = before.argmax(-1), after.argmax(-1)
    selected_late = after[np.arange(len(after)), early_choice]
    regret = after.max(-1) - selected_late
    return {"nonbase_sign": reversal_counts(before[:, 1:], after[:, 1:]),
            "pairwise_order": reversal_counts(before[:, i] - before[:, j], after[:, i] - after[:, j]),
            "best_action_exact_matches": int(np.sum(early_choice == late_choice)),
            "best_action_within_tolerance": int(np.sum(regret <= TOLERANCE)), "anchors": len(cases),
            "prefix_oracle_at_later_horizon": outcome(selected_late, cases),
            "mean_prefix_oracle_regret": balanced_mean(regret, cases)}


def influence(gap, cases):
    rows = []
    for reset in sorted({c["reset_seed"] for c in cases}):
        keep = [i for i, c in enumerate(cases) if c["reset_seed"] != reset]
        value = balanced_mean(gap[keep], [cases[i] for i in keep])
        rows.append({"removed_reset": reset, "mean_gap": value})
    rows.sort(key=lambda r: (r["mean_gap"], r["removed_reset"]))
    return {"all_sample_mean_gap": balanced_mean(gap, cases), "leave_one_reset_min": rows[0],
            "leave_one_reset_max": rows[-1], "adverse_gap_mass": reset_mass(np.maximum(-gap, 0.), cases)}


def analyze_run(source, decisions, condition, seed):
    cases = cv.training_cases(source, condition, seed)
    assert decisions["protocol"] == margin.PROTOCOL and decisions["complete"]
    assert (decisions["condition"], decisions["seed"], decisions["source_split"]) == (condition, seed, "train_cases")
    by_key = {margin.reference.case_key(r): r for r in decisions["rows"]}
    assert len(decisions["rows"]) == len(by_key) == len(cases)
    gains = {h: np.stack([paired_gains(c, h) for c in cases]) for h in HORIZONS}
    choices = {name: [] for name in margin.SELECTORS}
    for index, case in enumerate(cases):
        row = by_key[margin.reference.case_key(case)]
        assert row["reset_seed"] == case["reset_seed"] and row["fold"] == case["repeat"] % 4
        assert row["candidate_gains"] == dict(zip(p.CANDIDATES, gains[512][index]))
        for name in margin.SELECTORS:
            choice = row["choices"][name]
            assert choice == p.choose(row["scores"][name])
            assert row["selected_gains"][name] == row["candidate_gains"][choice]
            choices[name].append(p.CANDIDATES.index(choice))
    selected = {h: {name: values[np.arange(len(cases)), choice] for name, choice in choices.items()}
                for h, values in gains.items()}
    horizons = {}
    for h, values in gains.items():
        magnitude = np.abs(values[:, 1:])
        horizons[str(h)] = {"mean_absolute_nonbase_gain": float(magnitude.mean()),
            "median_absolute_nonbase_gain": float(np.median(magnitude)),
            "p90_absolute_nonbase_gain": float(np.quantile(magnitude, .9)),
            "max_absolute_nonbase_gain": float(magnitude.max()),
            "near_zero_nonbase_gains": int(np.sum(magnitude <= TOLERANCE)),
            "absolute_label_mass": reset_mass(magnitude.sum(-1), cases),
            "oracle_gain_upper_bound": balanced_mean(values.max(-1), cases),
            "selectors": {name: outcome(v, cases) for name, v in selected[h].items()}}
    for name in margin.SELECTORS:
        for metric, value in horizons["512"]["selectors"][name].items():
            np.testing.assert_allclose(value, decisions["summary"]["aggregate"][name][metric], rtol=1e-12, atol=1e-12)
    contrasts = {}
    for left, right in CONTRASTS:
        gaps = {h: selected[h][left] - selected[h][right] for h in HORIZONS}
        segments = {"0:32": gaps[32], "32:128": gaps[128] - gaps[32], "128:512": gaps[512] - gaps[128]}
        contrasts[f"{left}_minus_{right}"] = {**influence(gaps[512], cases),
            "horizon_mean_gaps": {str(h): balanced_mean(g, cases) for h, g in gaps.items()},
            "segment_mean_gaps": {name: balanced_mean(g, cases) for name, g in segments.items()}}
    cells = []
    for mode, wi in product(range(4), range(3)):
        ids = [i for i, c in enumerate(cases) if (c["mode"], c["weight_index"]) == (mode, wi)]
        group = [cases[i] for i in ids]
        cells.append({"mode": mode, "weight_index": wi,
            "stability32_512": stability(gains[32][ids], gains[512][ids], group),
            "margin_minus_ridge": {str(h): float((selected[h]["margin"] - selected[h]["ridge"])[ids].mean())
                                   for h in HORIZONS}})
    return {"condition": condition, "seed": seed, "anchors": len(cases), "reset_groups": 64,
            "horizons": horizons, "stability": {f"{a}:{b}": stability(gains[a], gains[b], cases)
                                                for a, b in combinations(HORIZONS, 2)},
            "contrasts": contrasts, "cells": cells,
            "unchanged_margin_gate": decisions["summary"]["group_passes"]}


def run(source_root=p.OUTPUT_ROOT, decisions_root=margin.OUTPUT_ROOT):
    runs = []
    for condition, seed in product(("v27", "v28"), (31, 33)):
        relative = Path(condition) / f"seed{seed}.json"
        source = json.loads((Path(source_root) / relative).read_text())
        decisions = json.loads((Path(decisions_root) / relative).read_text())
        runs.append(analyze_run(source, decisions, condition, seed))
    return {"protocol": PROTOCOL, "source_root": str(source_root), "decisions_root": str(decisions_root),
            "source_split": "train_cases", "horizons": list(HORIZONS), "primary_horizon": 512,
            "sign_and_tie_tolerance": TOLERANCE, "new_environment_interactions": 0,
            "model_updates": 0, "reused_anchors": 768, "runs": runs,
            "scope": "Descriptive development-label diagnostic with frozen OOF choices. Prefix oracles use hindsight; "
                     "leave-one-reset values are influence diagnostics without refitting, not CV or new gates."}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-root", type=Path, default=p.OUTPUT_ROOT)
    parser.add_argument("--decisions-root", type=Path, default=margin.OUTPUT_ROOT)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    args = parser.parse_args()
    report = run(args.source_root, args.decisions_root)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    print(json.dumps({"output": str(args.output), "runs": len(report["runs"]),
                      "new_environment_interactions": 0, "model_updates": 0}))
