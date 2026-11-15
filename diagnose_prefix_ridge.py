"""Fixed128-step ridge targets, evaluated with the original512-step gates."""

import argparse
import json
from pathlib import Path

import numpy as np
import torch

import diagnose_horizon_attribution as horizon


previous, cv, p = horizon.margin, horizon.cv, horizon.p
reference = previous.reference
PROTOCOL = "prefix128_ridge_grouped_cv_v1"
OUTPUT_ROOT = Path("results/iclr_prefix_ridge_20260921")
TRAIN_HORIZON = 128
SELECTORS = ("ridge128", *previous.SELECTORS)
CONTROLS = tuple(name for name in previous.SELECTORS if name != "no_change")


def training_data(cases):
    return {"inputs": torch.cat([cv.features(c)[1:] for c in cases]),
            "targets": torch.tensor(np.concatenate([horizon.paired_gains(c, TRAIN_HORIZON)[1:]
                                                      for c in cases]), dtype=torch.float32),
            "action_dim": 6}


def fit_predict(fit_cases, held_inputs):
    data = training_data(fit_cases)
    model = cv.QuadraticRidge(data)
    predictions = model(held_inputs).numpy()
    assert np.isfinite(predictions).all() and (predictions[:, 0] == 0.).all()
    rmse = float((model(data["inputs"]) - data["targets"].double()).square().mean().sqrt())
    return predictions, {"train_horizon": TRAIN_HORIZON, "ridge_solves": 1,
        "parameters": len(model.beta), "penalty": cv.RIDGE_PENALTY, "fit_anchors": len(fit_cases),
        "fit_nonbase_rows": len(data["targets"]), "target_dtype": "float32", "solve_dtype": "float64",
        "target_scale": float(model.target_scale), "train_rmse128": rmse}


def decision_row(case, old, scores):
    row = reference.join_row(case, old, {"ridge128": scores}, selectors=SELECTORS)
    row["prefix_candidate_gains"] = dict(zip(p.CANDIDATES, map(float, horizon.paired_gains(case, TRAIN_HORIZON))))
    return row


def summarize_rows(rows):
    result = cv.summarize_rows(rows, selectors=SELECTORS, controls=CONTROLS,
                               target="ridge128", value_selectors=reference.SELECTORS)
    result["score_semantics"] = {"ridge128": "128-step gain predictions, not512-step value estimates",
        **{name: "Dimensionless decision logits, not return predictions" for name in ("margin", "cost_sensitive")}}
    truth = np.asarray([r["prefix_candidate_gains"][a] for r in rows for a in p.CANDIDATES[1:]])
    pred = np.asarray([r["scores"]["ridge128"][a] for r in rows for a in p.CANDIDATES[1:]])
    result["prefix_diagnostic"] = {"horizon": TRAIN_HORIZON,
        "ridge128_calibration": {"rmse": float(np.sqrt(np.mean((pred - truth) ** 2))),
            "sign_matches": int(np.sum(np.sign(pred) == np.sign(truth))), "comparisons": len(truth)},
        "oracle_mean_gain_upper_bound": horizon.balanced_mean(
            [max(r["prefix_candidate_gains"].values()) for r in rows], rows),
        "selectors": {name: horizon.outcome(np.asarray([r["prefix_candidate_gains"][r["choices"][name]]
                                                       for r in rows]), rows) for name in SELECTORS}}
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
        "train_horizon": TRAIN_HORIZON, "primary_horizon": 512,
        "new_environment_interactions": 0, "mlp_updates": 0, "policy_updates": 0, "critic_updates": 0,
        "selector_updates": 0, "ridge_solves": 0, "reused_anchors": 192, "reused_nonbase_labels": 768,
        "folds": [], "rows": [], "scope": "Fixed128-step training-only ridge development CV. "
        "Original512-step gates; cached controls. No simulator, checkpoints or old validation inputs."}

    def save():
        path.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")

    save()
    for fold in range(4):
        fit, held = cv.split_fold(cases, fold)
        predictions, meta = fit_predict(fit, torch.stack([cv.features(c) for c in held]))
        report["rows"].extend(decision_row(c, cached[reference.case_key(c)], predictions[i])
                              for i, c in enumerate(held))
        report["folds"].append({"fold": fold, "fit_reset_ids": sorted({c["reset_seed"] for c in fit}),
            "held_reset_ids": sorted({c["reset_seed"] for c in held}), **meta})
        report["ridge_solves"] += meta["ridge_solves"]
        save()
        print(f"PREFIX_CV condition={condition} seed={seed} folds={fold + 1}/4 interactions=0", flush=True)
    report["summary"] = summarize_rows(report["rows"])
    report["complete"] = True
    save()
    print(f"DONE prefix_cv condition={condition} seed={seed} passes={report['summary']['group_passes']}", flush=True)
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
