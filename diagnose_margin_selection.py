"""Deterministic cost-sensitive ranking on existing paired-return labels."""

import argparse
import json
from pathlib import Path

import numpy as np
import torch

import diagnose_cost_sensitive as previous


reference, cv, p = previous.reference, previous.cv, previous.p
PROTOCOL = "margin_selection_grouped_cv_v1"
OUTPUT_ROOT = Path("results/iclr_margin_selection_20260921")
UPDATES = 2000
LEARNING_RATE = .01
PENALTY = 1.
SELECTORS = ("margin", *previous.SELECTORS)
CONTROLS = tuple(name for name in previous.SELECTORS if name != "no_change")


def margin_losses(scores, utility):
    best = utility.argmax(-1, keepdim=True)
    regret = utility.gather(-1, best) - utility
    return (scores + regret).max(-1).values - scores.gather(-1, best).squeeze(-1)


def objective(scores, utility, coefficients):
    return margin_losses(scores, utility).mean() + PENALTY * coefficients.square().sum()


def fit_selector(inputs, gains, action_dim, updates=UPDATES):
    model = previous.CostSensitiveSelector(inputs, action_dim)
    scale = gains[:, 1:].std(unbiased=False).clamp_min(1e-3)
    utility = previous.utility(gains) / scale
    phi = model.features.transform(inputs)
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    logs = []
    for update in range(1, updates + 1):
        loss = objective(phi @ model.beta, utility, model.beta)
        assert torch.isfinite(loss)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if update % 500 == 0 or update == updates:
            with torch.no_grad():
                value = float(objective(phi @ model.beta, utility, model.beta))
            logs.append({"update": update, "objective": value})
            print(f"MARGIN_FIT update={update}/{updates} objective={value:.6f}", flush=True)
    model.zero_grad(set_to_none=True)
    model.eval().requires_grad_(False)
    with torch.no_grad():
        scores = model(inputs)
        chosen = scores.argmax(-1)
        selected = gains[torch.arange(len(gains)), chosen]
        hinge = float(margin_losses(scores, utility).mean())
        regret = float((utility.max(-1).values - utility.gather(-1, chosen[:, None]).squeeze(-1)).mean())
    return model, {"updates": updates, "parameters": len(model.beta), "fit_anchors": len(inputs),
        "fit_nonbase_rows": len(inputs) * 4, "learning_rate": LEARNING_RATE, "penalty": PENALTY,
        "downside_weight": previous.DOWNSIDE_WEIGHT, "initialization": "zeros",
        "objective": "cost_sensitive_structured_hinge", "batch": "full", "dtype": "float64",
        "target_scale": float(scale), "losses": logs, "train_hinge": hinge, "train_utility_regret": regret,
        "train_mean_gain": float(selected.mean()), "train_mean_negative_part": float(selected.clamp_max(0).mean())}


def fit_predict(fit_cases, held_inputs):
    inputs = previous.model_inputs(fit_cases)
    gains = torch.tensor([[p.realized_gains(c)[label] for label in p.CANDIDATES] for c in fit_cases], dtype=torch.float64)
    model, meta = fit_selector(inputs, gains, action_dim=6, updates=UPDATES)
    with torch.no_grad():
        predictions = model(held_inputs).numpy()
    assert np.isfinite(predictions).all() and (predictions[:, 0] == 0.).all()
    return predictions, meta


def summarize_rows(rows):
    result = cv.summarize_rows(rows, selectors=SELECTORS, controls=CONTROLS,
                               target="margin", value_selectors=reference.SELECTORS)
    result["score_semantics"] = {name: "Dimensionless decision logits, not return predictions"
                                 for name in ("margin", "cost_sensitive")}
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
        "new_environment_interactions": 0, "mlp_updates": 0, "policy_updates": 0, "critic_updates": 0,
        "selector_updates": 0, "reused_anchors": 192, "reused_nonbase_labels": 768,
        "folds": [], "rows": [], "scope": "Training-only margin development CV on paired labels. "
        "Cached controls retained. No simulator, checkpoints or old validation inputs."}

    def save():
        path.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")

    save()
    for fold in range(4):
        fit, held = cv.split_fold(cases, fold)
        predictions, meta = fit_predict(fit, previous.model_inputs(held))
        report["rows"].extend(reference.join_row(c, cached[reference.case_key(c)],
            {"margin": predictions[i]}, selectors=SELECTORS) for i, c in enumerate(held))
        report["folds"].append({"fold": fold, "fit_reset_ids": sorted({c["reset_seed"] for c in fit}),
            "held_reset_ids": sorted({c["reset_seed"] for c in held}), **meta})
        report["selector_updates"] += meta["updates"]
        save()
        print(f"MARGIN_CV condition={condition} seed={seed} folds={fold + 1}/4 interactions=0", flush=True)
    report["summary"] = summarize_rows(report["rows"])
    report["complete"] = True
    save()
    print(f"DONE margin_cv condition={condition} seed={seed} passes={report['summary']['group_passes']}", flush=True)
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
