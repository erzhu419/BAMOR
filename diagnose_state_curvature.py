"""Add a state-dependent even response to the fixed paired-gain ridge."""

import argparse
import json
from pathlib import Path

import numpy as np
import torch

import diagnose_weak_ridge as previous


cv, p, reference = previous.cv, previous.p, previous.reference
PROTOCOL = "state_curvature_ridge_grouped_cv_v1"
OUTPUT_ROOT = Path("results/iclr_state_curvature_20260921")
PENALTY = 1.
FEATURE_EXTENSION = "z*sum(d**2)"
SELECTORS = ("state_curvature", *previous.SELECTORS)
CONTROLS = tuple(name for name in previous.SELECTORS if name != "no_change")


class StateCurvatureRidge(cv.QuadraticRidge):
    def basis(self, inputs):
        old = super().basis(inputs)
        x = inputs.double()
        state = (x[..., :-self.action_dim] - self.state_mean) / self.state_scale
        delta = x[..., -self.action_dim:] / self.delta_scale
        return torch.cat((old, state * delta.square().sum(-1, keepdim=True)), dim=-1)


def fit_predict(fit_cases, held_inputs):
    data = p.training_data(fit_cases)
    model = StateCurvatureRidge(data, penalty=PENALTY)
    predictions = model(held_inputs).numpy()
    assert np.isfinite(predictions).all() and (predictions[:, 0] == 0.).all()
    mse = float((model(data["inputs"]) - data["targets"].double()).square().mean())
    scale = float(model.target_scale)
    return predictions, {"train_horizon": 512, "ridge_solves": 1, "parameters": len(model.beta),
        "feature_extension": FEATURE_EXTENSION, "extra_parameters": len(model.state_mean),
        "penalty": PENALTY, "fit_anchors": len(fit_cases), "fit_nonbase_rows": len(data["targets"]),
        "target_dtype": "float32", "solve_dtype": "float64", "target_scale": scale,
        "train_rmse512": float(np.sqrt(mse)), "train_normalized_mse": mse / scale ** 2}


def summarize_rows(rows):
    result = cv.summarize_rows(rows, selectors=SELECTORS, controls=CONTROLS,
        target="state_curvature", value_selectors=("state_curvature", "weak_ridge", *reference.SELECTORS))
    result["score_semantics"] = {"state_curvature": "512-step gain predictions with state-dependent radial curvature",
        "weak_ridge": "512-step gain predictions with fixed penalty0.1",
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
        "train_horizon": 512, "primary_horizon": 512, "penalty": PENALTY, "feature_extension": FEATURE_EXTENSION,
        "new_environment_interactions": 0, "mlp_updates": 0, "policy_updates": 0, "critic_updates": 0,
        "selector_updates": 0, "ridge_solves": 0, "reused_anchors": 192, "reused_nonbase_labels": 768,
        "folds": [], "rows": [], "scope": "Training-only state-curvature ridge development CV. "
        "Original512-step gates; all cached controls. No simulator, checkpoints or old validation inputs."}

    def save():
        path.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")

    save()
    for fold in range(4):
        fit, held = cv.split_fold(cases, fold)
        predictions, meta = fit_predict(fit, torch.stack([cv.features(c) for c in held]))
        report["rows"].extend(reference.join_row(c, cached[reference.case_key(c)],
            {"state_curvature": predictions[i]}, selectors=SELECTORS) for i, c in enumerate(held))
        report["folds"].append({"fold": fold, "fit_reset_ids": sorted({c["reset_seed"] for c in fit}),
            "held_reset_ids": sorted({c["reset_seed"] for c in held}), **meta})
        report["ridge_solves"] += meta["ridge_solves"]
        save()
        print(f"CURVATURE condition={condition} seed={seed} folds={fold + 1}/4 interactions=0", flush=True)
    report["summary"] = summarize_rows(report["rows"])
    report["complete"] = True
    save()
    print(f"DONE curvature condition={condition} seed={seed} passes={report['summary']['group_passes']}", flush=True)
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
