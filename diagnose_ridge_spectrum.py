"""Separate finite-feature projection error from ridge shrinkage on fixed folds."""

import argparse
import json
from pathlib import Path

import numpy as np
import torch

import diagnose_advantage_cv as cv


p = cv.pilot
PROTOCOL = "ridge512_spectral_diagnosis_v1"
OUTPUT_ROOT = Path("results/iclr_ridge_spectrum_20260921")
PENALTY = 1.
RANK_RULE = "eps64 * max(shape) * largest_singular_value"


def decompose(phi, targets):
    phi, targets = np.asarray(phi, dtype=np.float64), np.asarray(targets, dtype=np.float64)
    n = len(targets)
    a, b = phi / np.sqrt(n), targets / np.sqrt(n)
    u, singular, vh = np.linalg.svd(a, full_matrices=False)
    cutoff = np.finfo(np.float64).eps * max(a.shape) * singular[0]
    retained = singular > cutoff
    coordinates = u.T @ b
    factors = singular ** 2 / (singular ** 2 + PENALTY)
    beta = vh.T @ (singular / (singular ** 2 + PENALTY) * coordinates)
    projection = u[:, retained] @ coordinates[retained]
    orthogonal, shrinkage = b - projection, projection - a @ beta
    projection_mse = float(orthogonal @ orthogonal)
    shrinkage_mse = float(shrinkage @ shrinkage)
    residual = b - a @ beta
    mse = float(residual @ residual)
    cross = float(2 * orthogonal @ shrinkage)
    np.testing.assert_allclose(cross, 0., rtol=0., atol=1e-10 * (1. + float(b @ b)))
    np.testing.assert_allclose(projection_mse + shrinkage_mse, mse, rtol=1e-10, atol=1e-12)
    kept = singular[retained]
    return beta, {"rank": int(retained.sum()), "rank_rule": RANK_RULE, "rank_cutoff": float(cutoff),
        "largest_singular_value": float(singular[0]),
        "smallest_retained_singular_value": float(kept[-1]) if len(kept) else None,
        "effective_df": float(factors.sum()), "zero_prediction_mse": float(b @ b),
        "ridge_mse": mse, "projection_mse": projection_mse, "shrinkage_mse": shrinkage_mse,
        "cross_term": cross, "shrinkage_share": shrinkage_mse / mse if mse else None}


def inspect_fold(fit_cases, held_inputs):
    data = p.training_data(fit_cases)
    features = cv.QuadraticFeatures(data["inputs"], data["action_dim"])
    targets = data["targets"].double()
    scale = targets.std(unbiased=False).clamp_min(1e-3)
    beta, stats = decompose(features.transform(data["inputs"]).numpy(), (targets / scale).numpy())
    held_scores = features.transform(held_inputs).numpy() @ beta * float(scale)
    assert np.isfinite(held_scores).all() and (held_scores[:, 0] == 0.).all()
    return held_scores, {**stats, "fit_anchors": len(fit_cases), "fit_nonbase_rows": len(targets),
        "parameters": len(beta), "penalty": PENALTY, "svd_decompositions": 1, "ridge_reconstructions": 1,
        "horizon": 512, "target_dtype": "float32", "solve_dtype": "float64", "target_scale": float(scale),
        "reconstructed_train_rmse": float(np.sqrt(stats["ridge_mse"]) * float(scale))}


def summarize_folds(folds):
    mse = float(np.mean([f["ridge_mse"] for f in folds]))
    projection = float(np.mean([f["projection_mse"] for f in folds]))
    shrinkage = float(np.mean([f["shrinkage_mse"] for f in folds]))
    np.testing.assert_allclose(projection + shrinkage, mse, rtol=1e-10, atol=1e-12)
    return {"mean_normalized_ridge_mse": mse, "mean_normalized_projection_mse": projection,
        "mean_normalized_shrinkage_mse": shrinkage, "shrinkage_share": shrinkage / mse if mse else None,
        "projection_share": projection / mse if mse else None,
        "mean_zero_prediction_mse": float(np.mean([f["zero_prediction_mse"] for f in folds])),
        "rank_range": [min(f["rank"] for f in folds), max(f["rank"] for f in folds)],
        "mean_effective_df": float(np.mean([f["effective_df"] for f in folds])),
        "max_cached_score_error": max(f["max_cached_score_error"] for f in folds)}


def run(condition, seed, source_root=p.OUTPUT_ROOT, reference_root=cv.OUTPUT_ROOT, output_root=OUTPUT_ROOT):
    source = Path(source_root) / condition / f"seed{seed}.json"
    reference = Path(reference_root) / condition / f"seed{seed}.json"
    cases = cv.training_cases(json.loads(source.read_text()), condition, seed)
    old = json.loads(reference.read_text())
    assert old["complete"] and old["protocol"] == cv.PROTOCOL
    assert (old["condition"], old["seed"], old["source_split"]) == (condition, seed, "train_cases")
    by_key = {(r["mode"], r["weight_index"], r["repeat"]): r for r in old["rows"]}
    assert len(old["rows"]) == len(by_key) == len(cases)
    old_folds = {f["fold"]: f for f in old["folds"]}
    assert len(old["folds"]) == len(old_folds) == 4
    report = {"protocol": PROTOCOL, "condition": condition, "seed": seed, "complete": False,
        "source": str(source), "reference": str(reference), "source_split": "train_cases", "horizon": 512,
        "new_environment_interactions": 0, "optimizer_updates": 0, "new_selector_variants": 0,
        "svd_decompositions": 0, "ridge_reconstructions": 0, "folds": [],
        "scope": "Training-error decomposition of the original ridge. Overlapping fitting folds, "
                 "not fresh performance evidence. Cached selectors and gates are unchanged."}
    for fold in range(4):
        fit, held = cv.split_fold(cases, fold)
        scores, stats = inspect_fold(fit, torch.stack([cv.features(c) for c in held]))
        for name, group in (("fit", fit), ("held", held)):
            stats[name + "_reset_ids"] = sorted({c["reset_seed"] for c in group})
            assert stats[name + "_reset_ids"] == old_folds[fold][name + "_reset_ids"]
        saved = []
        for c in held:
            row = by_key[(c["mode"], c["weight_index"], c["repeat"])]
            assert (row["reset_seed"], row["fold"]) == (c["reset_seed"], fold)
            saved.append([row["scores"]["ridge"][label] for label in p.CANDIDATES])
        np.testing.assert_allclose(scores, saved, rtol=1e-8, atol=1e-8)
        original_rmse = old_folds[fold]["train_rmse"]["ridge"]
        np.testing.assert_allclose(stats["reconstructed_train_rmse"], original_rmse, rtol=1e-8, atol=1e-8)
        stats.update(fold=fold, original_train_rmse=original_rmse,
                     max_cached_score_error=float(np.max(np.abs(scores - saved))))
        report["folds"].append(stats)
        report["svd_decompositions"] += 1
        report["ridge_reconstructions"] += 1
        print(f"SPECTRUM condition={condition} seed={seed} fold={fold} "
              f"projection={stats['projection_mse']:.6f} shrinkage={stats['shrinkage_mse']:.6f}", flush=True)
    report["summary"] = summarize_folds(report["folds"])
    report["complete"] = True
    path = Path(output_root) / condition / f"seed{seed}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--condition", choices=p.g.METHODS, required=True)
    parser.add_argument("--seed", type=int, choices=p.g.SEEDS, required=True)
    parser.add_argument("--source-root", type=Path, default=p.OUTPUT_ROOT)
    parser.add_argument("--reference-root", type=Path, default=cv.OUTPUT_ROOT)
    parser.add_argument("--output-root", type=Path, default=OUTPUT_ROOT)
    args = parser.parse_args()
    run(args.condition, args.seed, args.source_root, args.reference_root, args.output_root)
