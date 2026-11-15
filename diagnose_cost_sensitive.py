"""Direct risk-adjusted selection objective on fixed paired-return labels."""

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from torch import nn

import diagnose_residual_advantage as reference


cv, p = reference.cv, reference.p
PROTOCOL = "cost_sensitive_grouped_cv_v1"
OUTPUT_ROOT = Path("results/iclr_cost_sensitive_20260921")
UPDATES = 2000
LEARNING_RATE = .01
PENALTY = 1.
DOWNSIDE_WEIGHT = 1.
SELECTORS = ("cost_sensitive", *reference.SELECTORS)
CONTROLS = tuple(name for name in reference.SELECTORS if name != "no_change")


def model_inputs(cases):
    return torch.stack([cv.features(c) for c in cases])


def utility(gains):
    return gains + DOWNSIDE_WEIGHT * gains.clamp_max(0.)


def objective(logits, normalized_utility, coefficients):
    return -(logits.softmax(-1) * normalized_utility).sum(-1).mean() + PENALTY * coefficients.square().sum()


class CostSensitiveSelector(nn.Module):
    def __init__(self, inputs, action_dim):
        super().__init__()
        nonbase = inputs[:, 1:].reshape(-1, inputs.shape[-1])
        self.features = cv.QuadraticFeatures(nonbase, action_dim)
        self.beta = nn.Parameter(torch.zeros(len(self.features.feature_scale), dtype=torch.float64))

    def forward(self, inputs):
        return self.features.transform(inputs) @ self.beta


def fit_selector(inputs, gains, action_dim, updates=UPDATES):
    model = CostSensitiveSelector(inputs, action_dim)
    scale = gains[:, 1:].std(unbiased=False).clamp_min(1e-3)
    adjusted = utility(gains) / scale
    phi = model.features.transform(inputs)
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    logs = []
    for update in range(1, updates + 1):
        loss = objective(phi @ model.beta, adjusted, model.beta)
        assert torch.isfinite(loss)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if update % 500 == 0 or update == updates:
            with torch.no_grad():
                value = float(objective(phi @ model.beta, adjusted, model.beta))
            logs.append({"update": update, "objective": value})
            print(f"COST_FIT update={update}/{updates} objective={value:.6f}", flush=True)
    model.zero_grad(set_to_none=True)
    model.eval().requires_grad_(False)
    with torch.no_grad():
        chosen = model(inputs).argmax(-1)
        selected = gains[torch.arange(len(gains)), chosen]
    return model, {"updates": updates, "parameters": len(model.beta), "fit_anchors": len(inputs),
        "fit_nonbase_rows": len(inputs) * 4, "learning_rate": LEARNING_RATE, "penalty": PENALTY,
        "downside_weight": DOWNSIDE_WEIGHT, "initialization": "zeros", "temperature": 1.,
        "batch": "full", "dtype": "float64", "target_scale": float(scale), "losses": logs,
        "train_mean_gain": float(selected.mean()), "train_mean_negative_part": float(selected.clamp_max(0).mean())}


def fit_predict(fit_cases, held_inputs):
    inputs = model_inputs(fit_cases)
    gains = torch.tensor([[p.realized_gains(c)[label] for label in p.CANDIDATES] for c in fit_cases], dtype=torch.float64)
    model, meta = fit_selector(inputs, gains, action_dim=6, updates=UPDATES)
    with torch.no_grad():
        predictions = model(held_inputs).numpy()
    assert np.isfinite(predictions).all() and (predictions[:, 0] == 0.).all()
    return predictions, meta


def summarize_rows(rows):
    result = cv.summarize_rows(rows, selectors=SELECTORS, controls=CONTROLS,
                               target="cost_sensitive", value_selectors=reference.SELECTORS)
    result["score_semantics"] = {"cost_sensitive": "Dimensionless decision logits, not return predictions"}
    return result


def run(condition, seed, source_root=p.OUTPUT_ROOT, reference_root=reference.OUTPUT_ROOT, output_root=OUTPUT_ROOT):
    source = Path(source_root) / condition / f"seed{seed}.json"
    ref_path = Path(reference_root) / condition / f"seed{seed}.json"
    cases = cv.training_cases(json.loads(source.read_text()), condition, seed)
    old = json.loads(ref_path.read_text())
    assert old["complete"] and old["protocol"] == reference.PROTOCOL
    assert (old["condition"], old["seed"], old["source_split"]) == (condition, seed, "train_cases")
    cached = {reference.case_key(r): r for r in old["rows"]}
    assert len(old["rows"]) == len(cached) == len(cases) and set(cached) == {reference.case_key(c) for c in cases}
    path = Path(output_root) / condition / f"seed{seed}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    report = {"protocol": PROTOCOL, "condition": condition, "seed": seed, "complete": False,
        "source": str(source), "reference": str(ref_path), "source_split": "train_cases",
        "new_environment_interactions": 0, "mlp_updates": 0, "policy_updates": 0, "critic_updates": 0,
        "selector_updates": 0, "reused_anchors": 192, "reused_nonbase_labels": 768,
        "folds": [], "rows": [], "scope": "Training-only development CV on paired labels. "
        "Direct cost-sensitive logits; all old OOF controls reused. No simulator, checkpoints or old validation inputs."}

    def save():
        path.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")

    save()
    for fold in range(4):
        fit, held = cv.split_fold(cases, fold)
        predictions, meta = fit_predict(fit, model_inputs(held))
        report["rows"].extend(reference.join_row(c, cached[reference.case_key(c)],
            {"cost_sensitive": predictions[i]}, selectors=SELECTORS) for i, c in enumerate(held))
        report["folds"].append({"fold": fold, "fit_reset_ids": sorted({c["reset_seed"] for c in fit}),
            "held_reset_ids": sorted({c["reset_seed"] for c in held}), **meta})
        report["selector_updates"] += meta["updates"]
        save()
        print(f"COST_CV condition={condition} seed={seed} folds={fold + 1}/4 interactions=0", flush=True)
    report["summary"] = summarize_rows(report["rows"])
    report["complete"] = True
    save()
    print(f"DONE cost_cv condition={condition} seed={seed} passes={report['summary']['group_passes']}", flush=True)
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--condition", choices=p.g.METHODS, required=True)
    parser.add_argument("--seed", type=int, choices=p.g.SEEDS, required=True)
    parser.add_argument("--source-root", type=Path, default=p.OUTPUT_ROOT)
    parser.add_argument("--reference-root", type=Path, default=reference.OUTPUT_ROOT)
    parser.add_argument("--output-root", type=Path, default=OUTPUT_ROOT)
    args = parser.parse_args()
    run(args.condition, args.seed, args.source_root, args.reference_root, args.output_root)
