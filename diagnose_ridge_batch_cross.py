"""Cross-evaluate fixed ridge fits between two existing training-label batches."""

import argparse
import json
from itertools import product
from pathlib import Path

import numpy as np
import torch

import diagnose_ridge_expansion as previous


cv, p, reference, curve = previous.cv, previous.p, previous.reference, previous.curve
PROTOCOL = "ridge_batch_cross_grouped_cv_v1"
OUTPUT_ROOT = Path("results/iclr_ridge_batch_cross_20260921")
MODELS = ("old48", "new48", "combined96")
SELECTORS = (*MODELS, "soft", "raw", "no_change")
BATCHES = ("old", "new")


def split_batches(old, new, fold):
    old_fit, old_held = cv.split_fold(old, fold)
    new_fit, new_held = cv.split_fold(new, fold)
    fits = dict(zip(MODELS, (old_fit, new_fit, old_fit + new_fit)))
    held = {"old": old_held, "new": new_held}
    held_ids = {c["reset_seed"] for batch in held.values() for c in batch}
    assert len(held_ids) == 32
    for name, cases in fits.items():
        assert len({c["reset_seed"] for c in cases}) == (96 if name == "combined96" else 48)
        assert held_ids.isdisjoint(c["reset_seed"] for c in cases)
    return fits, held


def fit_models(fits, held_features):
    predictions, metadata = {}, {}
    for name in MODELS:
        predictions[name], metadata[name] = curve.fit_predict(fits[name], held_features)
    return predictions, metadata


def decision_row(case, batch, predictions):
    scores = {name: dict(zip(p.CANDIDATES, map(float, values))) for name, values in predictions.items()}
    scores.update({name: {label: case["predictions"][name][label] - case["predictions"][name]["base"]
                         for label in p.CANDIDATES} for name in ("soft", "raw")})
    scores["no_change"] = dict.fromkeys(p.CANDIDATES, 0.)
    choices = {name: p.choose(scores[name]) for name in SELECTORS}
    gains = p.realized_gains(case)
    return {**{k: case[k] for k in ("mode", "weight_index", "repeat", "reset_seed")},
        "batch": batch, "fold": case["repeat"] % 4, "scores": scores, "choices": choices,
        "candidate_gains": gains, "selected_gains": {name: gains[choices[name]] for name in SELECTORS}}


def check_reproduction(rows, cached):
    old_rows = [r for r in rows if r["batch"] == "old"]
    errors = {}
    for name, original in (("old48", "ridge"), ("combined96", "ridge96")):
        current = np.asarray([[r["scores"][name][label] for label in p.CANDIDATES] for r in old_rows])
        expected = np.asarray([[cached[reference.case_key(r)]["scores"][original][label]
                                for label in p.CANDIDATES] for r in old_rows])
        np.testing.assert_allclose(current, expected, rtol=1e-10, atol=1e-10)
        assert all(r["choices"][name] == cached[reference.case_key(r)]["choices"][original] for r in old_rows)
        errors[name] = float(np.max(np.abs(current - expected)))
    for row in old_rows:
        original = cached[reference.case_key(row)]
        assert row["reset_seed"] == original["reset_seed"] and row["candidate_gains"] == original["candidate_gains"]
        for name in ("soft", "raw", "no_change"):
            assert all(row[k][name] == original[k][name] for k in ("scores", "choices", "selected_gains"))
    return errors


def metrics(rows, selectors=SELECTORS):
    truth = np.asarray([[r["candidate_gains"][label] for label in p.CANDIDATES[1:]] for r in rows])
    result = {}
    for name in selectors:
        gains = np.asarray([r["selected_gains"][name] for r in rows])
        predictions = np.asarray([[r["scores"][name][label] for label in p.CANDIDATES[1:]] for r in rows])
        result[name] = {"mean_gain": float(gains.mean()), "mean_negative_part": float(np.minimum(gains, 0.).mean()),
            "rmse512": float(np.sqrt(np.mean((predictions - truth) ** 2))),
            "selection_counts": {label: sum(r["choices"][name] == label for r in rows) for label in p.CANDIDATES}}
    return result


def summarize_rows(rows):
    assert len(rows) == 384
    expected = {(batch, mode, wi, repeat) for batch, start in (("old", 0), ("new", 16))
                for mode, wi, repeat in product(range(4), range(3), range(start, start + 16))}
    assert {(r["batch"], *reference.case_key(r)) for r in rows} == expected
    batches, comparisons = {}, {}
    for batch in BATCHES:
        group = [r for r in rows if r["batch"] == batch]
        aggregate = metrics(group)
        cells = [{"mode": mode, "weight_index": wi,
            "metrics": metrics([r for r in group if (r["mode"], r["weight_index"]) == (mode, wi)])}
            for mode, wi in product(range(4), range(3))]
        batches[batch] = {"anchors": len(group), "aggregate": aggregate, "cells": cells}
        comparisons[batch] = []
        for before, after in (("old48", "new48"), ("old48", "combined96"), ("new48", "combined96")):
            a, b = aggregate[before], aggregate[after]
            comparisons[batch].append({"before": before, "after": after,
                **{key: b[key] - a[key] for key in ("mean_gain", "mean_negative_part")},
                "rmse_reduction": a["rmse512"] - b["rmse512"],
                "changed_choices": sum(r["choices"][before] != r["choices"][after] for r in group)})
    return {"batches": batches, "comparisons": comparisons,
        "same_batch_gain_advantages": {
            "old48_minus_new48_on_old": batches["old"]["aggregate"]["old48"]["mean_gain"] - batches["old"]["aggregate"]["new48"]["mean_gain"],
            "new48_minus_old48_on_new": batches["new"]["aggregate"]["new48"]["mean_gain"] - batches["new"]["aggregate"]["old48"]["mean_gain"]}}


def run(condition, seed, source_root=p.OUTPUT_ROOT, expansion_root=previous.OUTPUT_ROOT, output_root=OUTPUT_ROOT):
    relative = Path(condition) / f"seed{seed}.json"
    source = json.loads((Path(source_root) / relative).read_text())
    old = cv.training_cases(source, condition, seed)
    labels = json.loads((Path(expansion_root) / condition / f"seed{seed}_labels.json").read_text())
    new = previous.validate_labels(labels, condition, seed, source["sources"])
    expanded = json.loads((Path(expansion_root) / relative).read_text())
    assert expanded["complete"] and expanded["protocol"] == previous.PROTOCOL
    assert (expanded["condition"], expanded["seed"], expanded["source_split"]) == (condition, seed, "train_cases")
    cached = {reference.case_key(r): r for r in expanded["rows"]}
    assert len(expanded["rows"]) == len(cached) == 192
    assert set(cached) == {reference.case_key(c) for c in old}
    path = Path(output_root) / relative
    path.parent.mkdir(parents=True, exist_ok=True)
    report = {"protocol": PROTOCOL, "condition": condition, "seed": seed, "complete": False,
        "source_split": "train_cases", "fold_rule": "repeat % 4", "train_horizon": 512, "penalty": 1.,
        "source": str(Path(source_root) / relative), "expansion": str(Path(expansion_root) / relative),
        "new_environment_interactions": 0, "selector_updates": 0, "policy_updates": 0, "critic_updates": 0,
        "mlp_updates": 0, "ridge_solves": 0, "reused_old_anchors": 192, "reused_new_anchors": 192,
        "previous_all_control_pass": expanded["summary"]["group_passes"], "folds": [], "rows": [],
        "scope": "Training-batch sensitivity diagnostic on reused labels; no new acceptance gate, "
                 "batch-specific model selection, physical-distribution-shift inference or fresh confirmation."}

    def save():
        path.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")

    save()
    for fold in range(4):
        fits, held = split_batches(old, new, fold)
        held_cases = [(batch, c) for batch in BATCHES for c in held[batch]]
        predictions, metadata = fit_models(fits, torch.stack([cv.features(c) for _, c in held_cases]))
        rows = [decision_row(c, batch, {name: scores[i] for name, scores in predictions.items()})
                for i, (batch, c) in enumerate(held_cases)]
        errors = check_reproduction(rows, cached)
        report["rows"].extend(rows)
        report["folds"].append({"fold": fold, "reproduction_errors": errors,
            "held_reset_ids": {batch: sorted({c["reset_seed"] for c in cases}) for batch, cases in held.items()},
            "fits": {name: {"fit_reset_ids": sorted({c["reset_seed"] for c in fits[name]}), **meta}
                     for name, meta in metadata.items()}})
        report["ridge_solves"] += 3
        save()
        print(f"BATCH_CROSS condition={condition} seed={seed} folds={fold + 1}/4 interactions=0", flush=True)
    report["summary"] = summarize_rows(report["rows"])
    report["complete"] = True
    save()
    print(f"DONE batch_cross condition={condition} seed={seed} anchors={len(report['rows'])}", flush=True)
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--condition", choices=p.g.METHODS, required=True)
    parser.add_argument("--seed", type=int, choices=p.g.SEEDS, required=True)
    parser.add_argument("--source-root", type=Path, default=p.OUTPUT_ROOT)
    parser.add_argument("--expansion-root", type=Path, default=previous.OUTPUT_ROOT)
    parser.add_argument("--output-root", type=Path, default=OUTPUT_ROOT)
    args = parser.parse_args()
    run(args.condition, args.seed, args.source_root, args.expansion_root, args.output_root)
