"""Training-only reset-grouped OOF comparison of MLP and quadratic ridge."""

import argparse
import json
from itertools import product
from pathlib import Path

import numpy as np
import torch

import diagnose_paired_advantage as pilot


PROTOCOL = "paired_advantage_grouped_cv_v1"
OUTPUT_ROOT = Path("results/iclr_advantage_cv_20260921")
FOLDS = 4
MLP_UPDATES = 2000
RIDGE_PENALTY = 1.
SELECTORS = ("ridge", "mlp", "fixed_preference", "soft", "raw", "no_change")
CONTROLS = ("mlp", "fixed_preference", "soft", "raw")


def training_cases(report, condition, seed):
    assert report["protocol"] == pilot.PROTOCOL and report["complete"]
    assert (report["condition"], report["seed"]) == (condition, seed)
    assert (report["epsilon"], report["gamma"], report["horizons"]) == (.05, .99, [32, 128, 512])
    cases = report["train_cases"]
    assert len(cases) == 192
    assert {(c["mode"], c["weight_index"], c["repeat"]) for c in cases} == set(product(range(4), range(3), range(16)))
    for c in cases:
        assert c["reset_seed"] == pilot.reset_seed("train", seed, c["mode"], c["repeat"])
        assert c["weight"] == pilot.g.WEIGHTS[c["weight_index"]].tolist()
        assert "external_scores" not in c
    return cases


def split_fold(cases, fold):
    fit = [c for c in cases if c["repeat"] % FOLDS != fold]
    held = [c for c in cases if c["repeat"] % FOLDS == fold]
    assert (len(fit), len(held)) == (144, 48)
    assert {c["reset_seed"] for c in fit}.isdisjoint(c["reset_seed"] for c in held)
    return fit, held


def features(case):
    actions = {"base": case["diagnostics"]["soft"]["actions"]["base"]}
    actions.update({name + "_" + sign: case["diagnostics"][name]["actions"][sign]
                    for name in ("soft", "raw") for sign in ("plus", "minus")})
    return pilot.action_features(case["anchor_observation"], case["anchor_context"], actions)


class QuadraticFeatures:
    @torch.no_grad()
    def __init__(self, inputs, action_dim):
        self.action_dim = action_dim
        x = inputs.double()
        state, delta = x[:, :-self.action_dim], x[:, -self.action_dim:]
        self.state_mean = state.mean(0)
        self.state_scale = state.std(0, unbiased=False).clamp_min(1e-3)
        self.delta_scale = delta.square().mean(0).sqrt().clamp_min(1e-3)
        phi = self.basis(x)
        self.feature_scale = phi.square().mean(0).sqrt().clamp_min(1e-3)

    def basis(self, inputs):
        x = inputs.double()
        state = (x[..., :-self.action_dim] - self.state_mean) / self.state_scale
        delta = x[..., -self.action_dim:] / self.delta_scale
        i, j = torch.triu_indices(self.action_dim, self.action_dim)
        return torch.cat((delta, (state.unsqueeze(-1) * delta.unsqueeze(-2)).flatten(-2),
                          delta[..., i] * delta[..., j]), dim=-1)

    def transform(self, inputs):
        return self.basis(inputs) / self.feature_scale


class QuadraticRidge(QuadraticFeatures):
    @torch.no_grad()
    def __init__(self, data, penalty=RIDGE_PENALTY):
        super().__init__(data["inputs"], data["action_dim"])
        y = data["targets"].double()
        self.target_scale = y.std(unbiased=False).clamp_min(1e-3)
        phi = self.transform(data["inputs"])
        gram = phi.T @ phi / len(phi) + penalty * torch.eye(phi.shape[1], dtype=torch.float64)
        self.beta = torch.linalg.solve(gram, phi.T @ (y / self.target_scale) / len(phi))
        assert torch.isfinite(self.beta).all()

    def __call__(self, inputs):
        return (self.transform(inputs) @ self.beta) * self.target_scale


def fit_fold(fit_cases, held_features, seed, fold):
    data = pilot.training_data(fit_cases)
    fit_seed = seed + 1000 * (fold + 1)
    mlp, metadata = pilot.fit_evaluator(data, fit_seed, updates=MLP_UPDATES)
    ridge = QuadraticRidge(data)
    predictions, train_rmse = {}, {}
    with torch.no_grad():
        for name, model in (("mlp", mlp), ("ridge", ridge)):
            predictions[name] = model(held_features).reshape(-1, len(pilot.CANDIDATES)).numpy()
            assert np.isfinite(predictions[name]).all() and (predictions[name][:, 0] == 0.).all()
            train_rmse[name] = float((model(data["inputs"]).double() - data["targets"].double()).square().mean().sqrt())
    return predictions, {"fit_seed": fit_seed, "mlp": metadata,
        "mlp_parameters": sum(p.numel() for p in mlp.parameters()),
        "ridge_coefficients": len(ridge.beta), "ridge_penalty": RIDGE_PENALTY,
        "ridge_solves": 1, "train_rmse": train_rmse}


def decision_row(case, fold, predictions, fixed):
    scores = {name: {label: float(v) for label, v in zip(pilot.CANDIDATES, values)}
              for name, values in predictions.items()}
    scores.update({name: {label: case["predictions"][name][label] - case["predictions"][name]["base"]
                         for label in pilot.CANDIDATES} for name in ("soft", "raw")})
    scores.update(fixed_preference=fixed[str(case["weight_index"])], no_change=dict.fromkeys(pilot.CANDIDATES, 0.))
    choices = {name: pilot.choose(scores[name]) for name in SELECTORS}
    gains = pilot.realized_gains(case)
    return {**{k: case[k] for k in ("mode", "weight_index", "repeat", "reset_seed")},
            "fold": fold, "scores": scores, "choices": choices, "candidate_gains": gains,
            "selected_gains": {name: gains[choices[name]] for name in SELECTORS}}


def summarize_rows(rows, selectors=SELECTORS, controls=CONTROLS, target="ridge", value_selectors=None):
    assert len(rows) == 192
    assert {(r["mode"], r["weight_index"], r["repeat"]) for r in rows} == set(product(range(4), range(3), range(16)))
    cells = []
    for mode, wi in product(range(4), range(3)):
        group = [r for r in rows if (r["mode"], r["weight_index"]) == (mode, wi)]
        cell = {"mode": mode, "weight_index": wi, "selectors": {}}
        for name in selectors:
            values = [r["selected_gains"][name] for r in group]
            cell["selectors"][name] = {"mean_gain": float(np.mean(values)),
                "mean_negative_part": float(np.mean(np.minimum(values, 0.)))}
        cells.append(cell)
    aggregate = {name: {k: float(np.mean([c["selectors"][name][k] for c in cells]))
                        for k in ("mean_gain", "mean_negative_part")} for name in selectors}
    calibration = {}
    for name in selectors if value_selectors is None else value_selectors:
        pred = np.asarray([r["scores"][name][label] for r in rows for label in pilot.CANDIDATES[1:]])
        truth = np.asarray([r["candidate_gains"][label] for r in rows for label in pilot.CANDIDATES[1:]])
        calibration[name] = {"rmse": float(np.sqrt(np.mean((pred - truth) ** 2))),
                             "sign_matches": int(np.sum(np.sign(pred) == np.sign(truth))), "comparisons": len(truth)}
    selected = aggregate[target]
    checks = {"positive_mean_gain": selected["mean_gain"] > 0,
              "mean_not_worse_than_controls": all(selected["mean_gain"] >= aggregate[c]["mean_gain"] for c in controls),
              "downside_not_worse_than_controls": all(selected["mean_negative_part"] >= aggregate[c]["mean_negative_part"] for c in controls)}
    return {"primary_horizon": 512, "cells": cells, "aggregate": aggregate,
            "calibration": calibration,
            "selection_counts": {name: {label: sum(r["choices"][name] == label for r in rows)
                                          for label in pilot.CANDIDATES} for name in selectors},
            "checks": checks, "group_passes": all(checks.values())}


def run(condition, seed, source_root=pilot.OUTPUT_ROOT, output_root=OUTPUT_ROOT):
    source = Path(source_root) / condition / f"seed{seed}.json"
    # The source also contains the old validation split; nothing below receives it.
    cases = training_cases(json.loads(source.read_text()), condition, seed)
    path = Path(output_root) / condition / f"seed{seed}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    report = {"protocol": PROTOCOL, "condition": condition, "seed": seed, "source": str(source),
              "source_split": "train_cases", "complete": False, "fold_rule": "repeat % 4",
              "new_environment_interactions": 0, "policy_updates": 0, "critic_updates": 0,
              "mlp_updates": 0, "ridge_solves": 0, "reused_anchors": len(cases),
              "reused_nonbase_labels": len(cases) * 4, "folds": [], "rows": [],
              "scope": "Development OOF diagnostic on existing training labels, not fresh validation. "
                       "No simulator, policy checkpoints, original validation cases or full-data evaluator inputs."}

    def save():
        path.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")

    save()
    for fold in range(FOLDS):
        fit, held = split_fold(cases, fold)
        fixed = pilot.fixed_preference_scores(fit)
        predictions, meta = fit_fold(fit, torch.cat([features(c) for c in held]), seed, fold)
        # Held-out labels are first used here, after all fold predictions exist.
        rows = [decision_row(c, fold, {name: values[i] for name, values in predictions.items()}, fixed)
                for i, c in enumerate(held)]
        report["rows"].extend(rows)
        report["folds"].append({"fold": fold, "fit_reset_ids": sorted({c["reset_seed"] for c in fit}),
            "held_reset_ids": sorted({c["reset_seed"] for c in held}),
            "fit_anchors": len(fit), "held_anchors": len(held), "fixed_preference_scores": fixed, **meta})
        report["mlp_updates"] += meta["mlp"]["updates"]
        report["ridge_solves"] += meta["ridge_solves"]
        save()
        print(f"ADVANTAGE_CV condition={condition} seed={seed} folds={fold + 1}/{FOLDS} "
              f"oof_anchors={len(report['rows'])} interactions=0", flush=True)
    report["summary"] = summarize_rows(report["rows"])
    report["complete"] = True
    save()
    print(f"DONE advantage_cv condition={condition} seed={seed} "
          f"group_passes={report['summary']['group_passes']}", flush=True)
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--condition", choices=pilot.g.METHODS, required=True)
    parser.add_argument("--seed", type=int, choices=pilot.g.SEEDS, required=True)
    parser.add_argument("--source-root", type=Path, default=pilot.OUTPUT_ROOT)
    parser.add_argument("--output-root", type=Path, default=OUTPUT_ROOT)
    args = parser.parse_args()
    run(args.condition, args.seed, args.source_root, args.output_root)
