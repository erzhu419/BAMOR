"""Train-only paired-return supervision and pre-rollout shadow decisions."""

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from torch import nn

import diagnose_action_radius as sources


p = sources.pilot
g = sources.g
training = sources.training
PROTOCOL = "paired_mc_advantage_shadow_v1"
OUTPUT_ROOT = Path("results/iclr_paired_advantage_20260921")
CANDIDATES = ("base", "soft_plus", "soft_minus", "raw_plus", "raw_minus")
TRAIN_REPEATS = 16
VALIDATION_REPEATS = 8
FIT_UPDATES = 2000
BATCH_SIZE = 128
LEARNING_RATE = 3e-4
EPSILON = .05
CONTROLS = ("soft", "raw", "fixed_preference")


def reset_seed(split, seed, mode, repeat):
    return {"train": 47000000, "validation": 48000000}[split] + seed * 10000 + mode * 100 + repeat


def action_features(obs, context, actions):
    base = np.asarray(actions["base"], dtype=np.float32)
    return torch.as_tensor(np.stack([np.concatenate((obs, context, base,
        np.asarray(actions[name], dtype=np.float32) - base)) for name in CANDIDATES]), dtype=torch.float32)


def realized_gains(case):
    horizon = str(max(g.HORIZONS))
    base = np.asarray(case["branches"]["base"]["discounted_returns"][horizon])
    return {name: float(np.dot(case["weight"],
        np.asarray(case["branches"][name]["discounted_returns"][horizon]) - base)) for name in CANDIDATES}


def training_data(cases):
    features, targets = [], []
    for case in cases:
        actions = {"base": case["diagnostics"]["soft"]["actions"]["base"]}
        actions.update({name + "_" + sign: case["diagnostics"][name]["actions"][sign]
                        for name in ("soft", "raw") for sign in ("plus", "minus")})
        features.append(action_features(case["anchor_observation"], case["anchor_context"], actions)[1:])
        gains = realized_gains(case)
        targets.extend(gains[name] for name in CANDIDATES[1:])
    return {"inputs": torch.cat(features), "targets": torch.tensor(targets, dtype=torch.float32),
            "action_dim": len(actions["base"])}


class PairedAdvantage(nn.Module):
    def __init__(self, data, hidden=64):
        super().__init__()
        self.action_dim = data["action_dim"]
        self.register_buffer("input_mean", data["inputs"].mean(0))
        self.register_buffer("input_scale", data["inputs"].std(0, unbiased=False).clamp_min(1e-3))
        self.register_buffer("return_scale", data["targets"].std(unbiased=False).clamp_min(1e-3))
        self.net = nn.Sequential(nn.Linear(data["inputs"].shape[1], hidden), nn.ReLU(),
                                 nn.Linear(hidden, hidden), nn.ReLU(), nn.Linear(hidden, 1))

    def forward(self, inputs):
        # Anchor no-change at exactly zero without forcing opposite actions to be antisymmetric.
        base = torch.cat((inputs[..., :-self.action_dim], torch.zeros_like(inputs[..., -self.action_dim:])), -1)
        return ((self.net((inputs - self.input_mean) / self.input_scale) -
                 self.net((base - self.input_mean) / self.input_scale)) * self.return_scale).squeeze(-1)


def fit_evaluator(data, seed, updates=FIT_UPDATES):
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(49000000 + seed)
        model = PairedAdvantage(data)
        optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
        rng = torch.Generator().manual_seed(50000000 + seed)
        logs, losses = [], []
        for update in range(1, updates + 1):
            index = torch.randint(len(data["inputs"]), (BATCH_SIZE,), generator=rng)
            loss = ((model(data["inputs"][index]) - data["targets"][index]) / model.return_scale).square().mean()
            assert torch.isfinite(loss), "Nonfinite paired-advantage loss"
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            losses.append(float(loss.detach()))
            if update % 500 == 0 or update == updates:
                logs.append({"update": update, "mean_normalized_mse": float(np.mean(losses))})
                losses.clear()
                print(f"ADVANTAGE_FIT update={update}/{updates} loss={logs[-1]['mean_normalized_mse']:.6f}", flush=True)
    model.zero_grad(set_to_none=True)
    model.eval().requires_grad_(False)
    return model, {"updates": updates, "data_rows": len(data["inputs"]), "batch_size": BATCH_SIZE,
                   "learning_rate": LEARNING_RATE, "hidden": 64, "losses": logs,
                   "return_scale": float(model.return_scale)}


def fixed_preference_scores(cases):
    return {str(wi): {name: float(np.mean([realized_gains(c)[name] for c in cases if c["weight_index"] == wi]))
                     for name in CANDIDATES} for wi in range(len(g.WEIGHTS))}


@torch.no_grad()
def predict_actions(model, obs, context, actions):
    values = model(action_features(obs, context, actions)).tolist()
    assert values[0] == 0. and np.isfinite(values).all()
    return dict(zip(CANDIDATES, values))


def choose(scores):
    # Base is first, so a zero tie retains the unchanged action.
    assert scores["base"] == 0.
    return CANDIDATES[int(np.argmax([scores[name] for name in CANDIDATES]))]


def summarize_cases(cases, fixed_scores):
    rows = []
    for case in cases:
        scores = {name: {label: case["predictions"][name][label] - case["predictions"][name]["base"]
                         for label in CANDIDATES} for name in ("soft", "raw")}
        scores.update(learned=case["external_scores"], fixed_preference=fixed_scores[str(case["weight_index"])],
                      no_change=dict.fromkeys(CANDIDATES, 0.))
        choices = {name: choose(value) for name, value in scores.items()}
        gains = realized_gains(case)
        rows.append({"mode": case["mode"], "weight_index": case["weight_index"],
                     "reset_seed": case["reset_seed"], "choices": choices,
                     "selected_gains": {name: gains[label] for name, label in choices.items()},
                     "scores": scores, "candidate_gains": gains})
    cells = []
    for mode, wi in sorted({(c["mode"], c["weight_index"]) for c in rows}):
        group = [c for c in rows if (c["mode"], c["weight_index"]) == (mode, wi)]
        assert len(group) == VALIDATION_REPEATS and len({c["reset_seed"] for c in group}) == VALIDATION_REPEATS
        cell = {"mode": mode, "weight_index": wi, "selectors": {}}
        for name in (*CONTROLS, "learned", "no_change"):
            values = [c["selected_gains"][name] for c in group]
            cell["selectors"][name] = {"per_reset_gains": values, "mean_gain": float(np.mean(values)),
                "mean_negative_part": float(np.mean(np.minimum(values, 0.))),
                "selected_nonbase": sum(c["choices"][name] != "base" for c in group)}
        cells.append(cell)
    aggregate = {name: {key: float(np.mean([c["selectors"][name][key] for c in cells]))
                        for key in ("mean_gain", "mean_negative_part")}
                 for name in (*CONTROLS, "learned", "no_change")}
    calibration = {}
    for name in (*CONTROLS, "learned"):
        predicted = [c["scores"][name][label] for c in rows for label in CANDIDATES[1:]]
        actual = [c["candidate_gains"][label] for c in rows for label in CANDIDATES[1:]]
        calibration[name] = {"rmse": float(np.sqrt(np.mean((np.asarray(predicted) - actual) ** 2))),
            "sign_matches": int(np.sum(np.sign(predicted) == np.sign(actual))), "comparisons": len(actual)}
    learned = aggregate["learned"]
    checks = {"positive_mean_gain": learned["mean_gain"] > 0,
              "mean_not_worse_than_controls": all(learned["mean_gain"] >= aggregate[c]["mean_gain"] for c in CONTROLS),
              "downside_not_worse_than_controls": all(learned["mean_negative_part"] >= aggregate[c]["mean_negative_part"] for c in CONTROLS)}
    return {"primary_horizon": max(g.HORIZONS), "cases": rows, "cells": cells,
            "aggregate": aggregate, "calibration": calibration, "checks": checks,
            "group_passes": all(checks.values())}


def run(condition, seed, output_root=OUTPUT_ROOT):
    networks, critics, _, paths = sources.load_inputs(condition, seed)
    method = g.METHODS[condition]
    directory = Path(output_root) / condition
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / f"seed{seed}.json"
    report = {"protocol": PROTOCOL, "condition": condition, "seed": seed, "method": method,
              "sources": paths, "complete": False, "epsilon": EPSILON,
              "weights": g.WEIGHTS.tolist(), "horizons": list(g.HORIZONS), "gamma": g.GAMMA,
              "warmup": g.WARMUP, "train_repeats": TRAIN_REPEATS, "validation_repeats": VALIDATION_REPEATS,
              "candidates": list(CANDIDATES), "train_cases": [], "validation_cases": [],
              "label_collection_interactions": 0, "evaluation_interactions": 0,
              "environment_initialization_probe_steps": 0, "policy_training_interactions": 0,
              "policy_updates": 0, "critic_updates": 0,
              "scope": "Observed fixed policies; new disjoint train/validation resets. "
                       "Paired 512-step raw targets, no entropy/bootstrap. Predictions precede labels. "
                       "Every rollout is charged; simulator-copy access and extra balanced supervision. "
                       "One-action selection is not operator routing or equal-budget policy improvement."}

    def save():
        path.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")

    save()
    model = None
    for split, repeats, case_key, budget_key in (
            ("train", TRAIN_REPEATS, "train_cases", "label_collection_interactions"),
            ("validation", VALIDATION_REPEATS, "validation_cases", "evaluation_interactions")):
        if split == "validation":
            data = training_data(report["train_cases"])
            report["fixed_preference_scores"] = fixed_preference_scores(report["train_cases"])
            model, report["fit"] = fit_evaluator(data, seed, FIT_UPDATES)
            torch.save({"state_dict": model.state_dict(), "action_dim": model.action_dim},
                       directory / f"seed{seed}_advantage.pt")
            del data
            save()
        # Validation environment and outcomes do not exist until the fixed fit has finished.
        env = training.make_piecewise_mo_env("mo-halfcheetah-v5", switch_interval=25000,
                                            seed=seed + (3000 if split == "train" else 4000))
        report["environment_initialization_probe_steps"] += 1
        env._lock_mode = True
        try:
            for mode in range(env.num_modes):
                env._apply_mode(mode)
                for wi, weight in enumerate(g.WEIGHTS):
                    for repeat in range(repeats):
                        scorer = None if model is None else lambda obs, ctx, actions: predict_actions(model, obs, ctx, actions)
                        case = p.evaluate_case(env, networks, critics, method, weight,
                            reset_seed(split, seed, mode, repeat), epsilon=EPSILON, score_actions=scorer)
                        case.update(mode=mode, weight_index=wi, repeat=repeat)
                        report[case_key].append(case)
                        report[budget_key] += case["evaluation_interactions"]
                    save()
                    print(f"PAIRED_ADVANTAGE split={split} condition={condition} seed={seed} "
                          f"cases={len(report[case_key])} interactions={report[budget_key]}", flush=True)
            assert env._total_steps == len(report[case_key]) * (g.WARMUP + max(g.HORIZONS))
        finally:
            env.close()
    report["summary"] = summarize_cases(report["validation_cases"], report["fixed_preference_scores"])
    report["complete"] = True
    save()
    print(f"DONE paired_advantage condition={condition} seed={seed} "
          f"collection={report['label_collection_interactions']} evaluation={report['evaluation_interactions']}", flush=True)
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--condition", choices=g.METHODS, required=True)
    parser.add_argument("--seed", type=int, choices=g.SEEDS, required=True)
    parser.add_argument("--output-root", type=Path, default=OUTPUT_ROOT)
    args = parser.parse_args()
    run(args.condition, args.seed, args.output_root)
