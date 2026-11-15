"""One fixed weaker ridge penalty on original512-step paired labels."""

import argparse
import json
from pathlib import Path

import numpy as np
import torch

import diagnose_prefix_ridge as previous


cv, p, reference = previous.cv, previous.p, previous.reference
PROTOCOL = "weak_ridge01_grouped_cv_v1"
OUTPUT_ROOT = Path("results/iclr_weak_ridge_20260921")
PENALTY = .1
SELECTORS = ("weak_ridge", *previous.SELECTORS)
CONTROLS = tuple(name for name in previous.SELECTORS if name != "no_change")


def fit_predict(fit_cases, held_inputs):
    data = p.training_data(fit_cases)
    model = cv.QuadraticRidge(data, penalty=PENALTY)
    predictions = model(held_inputs).numpy()
    assert np.isfinite(predictions).all() and (predictions[:, 0] == 0.).all()
    mse = float((model(data["inputs"]) - data["targets"].double()).square().mean())
    scale = float(model.target_scale)
    return predictions, {"train_horizon": 512, "ridge_solves": 1, "parameters": len(model.beta),
        "penalty": PENALTY, "fit_anchors": len(fit_cases), "fit_nonbase_rows": len(data["targets"]),
        "target_dtype": "float32", "solve_dtype": "float64", "target_scale": scale,
        "train_rmse512": float(np.sqrt(mse)), "train_normalized_mse": mse / scale ** 2}


def summarize_rows(rows):
    result = cv.summarize_rows(rows, selectors=SELECTORS, controls=CONTROLS,
        target="weak_ridge", value_selectors=("weak_ridge", *reference.SELECTORS))
    result["score_semantics"] = {"weak_ridge": "512-step gain predictions with fixed penalty0.1",
        "ridge128": "128-step gain predictions, not512-step value estimates",
        **{name: "Dimensionless decision logits, not return predictions" for name in ("margin", "cost_sensitive")}}
    return result


def run(condition, seed, source_root=p.OUTPUT_ROOT, reference_root=previous.OUTPUT_ROOT, output_root=OUTPUT_ROOT):
    source = Path(source_root) / condition / f"seed{seed}.json"
    ref_path = Path(reference_root) / condition / f"seed{seed}.json"
    cases = cv.training_cases(json.loads(source.read_text()), condition, seed)
    old = json.loads(ref_path.read_text())
    assert old["complete"] and old["protocol"] == previous.PROTOCOL
    assert (old["condition"], old["seed"], old["source_split"]) == (condition, seed, "train_cases")
    cached = {reference.case_key(r): r for r in old["rows"]}
    assert len(old["rows"]) == len(cached) == len(cases) and set(cached) == {reference.case_key(c) for c in cases}
    path = Path(output_root) / condition / f"seed{seed}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    report = {"protocol": PROTOCOL, "condition": condition, "seed": seed, "complete": False,
        "source": str(source), "reference": str(ref_path), "source_split": "train_cases",
        "train_horizon": 512, "primary_horizon": 512, "penalty": PENALTY,
        "new_environment_interactions": 0, "mlp_updates": 0, "policy_updates": 0, "critic_updates": 0,
        "selector_updates": 0, "ridge_solves": 0, "reused_anchors": 192, "reused_nonbase_labels": 768,
        "folds": [], "rows": [], "scope": "Fixed penalty0.1 training-only ridge development CV. "
        "Original512-step gates; all cached controls retained. No simulator, checkpoints or old validation inputs."}

    def save():
        path.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")

    save()
    for fold in range(4):
        fit, held = cv.split_fold(cases, fold)
        predictions, meta = fit_predict(fit, torch.stack([cv.features(c) for c in held]))
        report["rows"].extend(reference.join_row(c, cached[reference.case_key(c)],
            {"weak_ridge": predictions[i]}, selectors=SELECTORS) for i, c in enumerate(held))
        report["folds"].append({"fold": fold, "fit_reset_ids": sorted({c["reset_seed"] for c in fit}),
            "held_reset_ids": sorted({c["reset_seed"] for c in held}), **meta})
        report["ridge_solves"] += meta["ridge_solves"]
        save()
        print(f"WEAK_RIDGE condition={condition} seed={seed} folds={fold + 1}/4 interactions=0", flush=True)
    report["summary"] = summarize_rows(report["rows"])
    report["complete"] = True
    save()
    print(f"DONE weak_ridge condition={condition} seed={seed} passes={report['summary']['group_passes']}", flush=True)
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--condition", choices=p.g.METHODS, required=True)
    parser.add_argument("--seed", type=int, choices=p.g.SEEDS, required=True)
    parser.add_argument("--source-root", type=Path, default=p.OUTPUT_ROOT)
    parser.add_argument("--reference-root", type=Path, default=previous.OUTPUT_ROOT)
    parser.add_argument("--output-root", type=Path, default=OUTPUT_ROOT)
    args = parser.parse_args()
    run(args.condition, args.seed, args.source_root, args.reference_root, args.output_root)
