"""Expand frozen paired-return training labels and compare ridge48 with ridge96."""

import argparse
import json
from itertools import product
from pathlib import Path

import numpy as np
import torch

import diagnose_ridge_learning_curve as curve


cv, p, reference, previous = curve.cv, curve.p, curve.reference, curve.previous
PROTOCOL = "ridge_label_expansion_grouped_cv_v1"
LABEL_PROTOCOL = "paired_gain_training_repeats16_31_v1"
OUTPUT_ROOT = Path("results/iclr_ridge_expansion_20260921")
REPEATS = range(16, 32)
SELECTORS = ("ridge96", *previous.SELECTORS)
CONTROLS = tuple(name for name in previous.SELECTORS if name != "no_change")


def validate_labels(data, condition, seed, sources):
    expected = {"protocol": LABEL_PROTOCOL, "complete": True, "condition": condition, "seed": seed,
        "method": p.g.METHODS[condition], "sources": sources, "repeats": list(REPEATS),
        "epsilon": .05, "warmup": 128, "horizons": [32, 128, 512], "gamma": .99,
        "weights": p.g.WEIGHTS.tolist(), "candidates": list(p.CANDIDATES),
        "label_collection_interactions": 614400, "environment_initialization_probe_steps": 1,
        "new_environment_interactions": 614401, "policy_updates": 0, "critic_updates": 0,
        "evaluation_interactions": 0, "policy_training_interactions": 0}
    for key, value in expected.items():
        assert data[key] == value, f"Label field mismatch: {key}"
    cases = data["train_cases"]
    assert len(cases) == 192
    assert {reference.case_key(c) for c in cases} == set(product(range(4), range(3), REPEATS))
    old_ids = {p.reset_seed(split, seed, mode, repeat) for split, repeats in
               (("train", range(16)), ("validation", range(8))) for mode in range(4) for repeat in repeats}
    assert len({c["reset_seed"] for c in cases}) == 64
    assert old_ids.isdisjoint(c["reset_seed"] for c in cases)
    for c in cases:
        assert c["reset_seed"] == p.reset_seed("train", seed, c["mode"], c["repeat"])
        assert c["weight"] == p.g.WEIGHTS[c["weight_index"]].tolist() and "external_scores" not in c
        assert c["identity_matched"] and c["branches"]["base"] == c["branches"]["repeat"]
        assert set(c["branches"]) == set(p.p.BRANCHES)
        for branch in c["branches"].values():
            assert branch["steps"] == 512
            for key in ("discounted_returns", "raw_returns"):
                assert set(branch[key]) == {"32", "128", "512"}
                values = np.asarray(list(branch[key].values()))
                assert values.shape == (3, 2) and np.isfinite(values).all()
        assert c["evaluation_interactions"] == 128 + sum(b["steps"] for b in c["branches"].values())
        inputs = cv.features(c)
        assert inputs.shape == (5, 42) and torch.isfinite(inputs).all()
    assert sum(c["evaluation_interactions"] for c in cases) == data["label_collection_interactions"]
    return cases


def collect_labels(condition, seed, sources, path):
    path = Path(path)
    if path.exists():
        data = json.loads(path.read_text())
        assert data["complete"], "Incomplete collection exists; inspect its budget before any recollection"
        validate_labels(data, condition, seed, sources)
        return data
    networks, critics, _, loaded = p.sources.load_inputs(condition, seed)
    assert loaded == sources, "Expansion must use the original checkpoint paths"
    method = p.g.METHODS[condition]
    data = {"protocol": LABEL_PROTOCOL, "condition": condition, "seed": seed, "method": method,
        "complete": False, "sources": sources, "repeats": list(REPEATS), "epsilon": p.EPSILON,
        "warmup": p.g.WARMUP, "horizons": list(p.g.HORIZONS), "gamma": p.g.GAMMA,
        "weights": p.g.WEIGHTS.tolist(), "candidates": list(p.CANDIDATES), "train_cases": [],
        "label_collection_interactions": 0, "environment_initialization_probe_steps": 0,
        "new_environment_interactions": 0, "policy_updates": 0, "critic_updates": 0,
        "evaluation_interactions": 0, "policy_training_interactions": 0}
    path.parent.mkdir(parents=True, exist_ok=True)

    def save():
        path.write_text(json.dumps(data, indent=2, allow_nan=False) + "\n")

    save()
    env = p.training.make_piecewise_mo_env("mo-halfcheetah-v5", switch_interval=25000, seed=seed + 3000)
    data["environment_initialization_probe_steps"] = data["new_environment_interactions"] = 1
    save()
    env._lock_mode = True
    try:
        assert env.num_modes == 4
        for mode in range(4):
            env._apply_mode(mode)
            for wi, weight in enumerate(p.g.WEIGHTS):
                for repeat in REPEATS:
                    case = p.p.evaluate_case(env, networks, critics, method, weight,
                        p.reset_seed("train", seed, mode, repeat), epsilon=p.EPSILON)
                    case.update(mode=mode, weight_index=wi, repeat=repeat)
                    data["train_cases"].append(case)
                    data["label_collection_interactions"] += case["evaluation_interactions"]
                    data["new_environment_interactions"] += case["evaluation_interactions"]
                    save()
                print(f"EXPANSION_LABELS condition={condition} seed={seed} mode={mode} weight={wi} "
                      f"anchors={len(data['train_cases'])}/192 steps={data['new_environment_interactions']}", flush=True)
        assert env._total_steps == len(data["train_cases"]) * (p.g.WARMUP + max(p.g.HORIZONS))
    finally:
        env.close()
    data["complete"] = True
    validate_labels(data, condition, seed, sources)
    save()
    return data


def split_expanded(old_cases, new_cases, fold):
    old_fit, held = cv.split_fold(old_cases, fold)
    new_fit = [c for c in new_cases if c["repeat"] % 4 != fold]
    assert len(new_fit) == 144 and len({c["reset_seed"] for c in new_fit}) == 48
    fit = old_fit + new_fit
    assert len(fit) == 288 and len({c["reset_seed"] for c in fit}) == 96
    assert {c["reset_seed"] for c in fit}.isdisjoint(c["reset_seed"] for c in held)
    return fit, held


def summarize_rows(rows):
    summary = cv.summarize_rows(rows, selectors=SELECTORS, controls=CONTROLS, target="ridge96",
        value_selectors=("ridge96", "state_curvature", "weak_ridge", *reference.SELECTORS))
    old, new = (summary["aggregate"][name] for name in ("ridge", "ridge96"))
    change = {key: new[key] - old[key] for key in ("mean_gain", "mean_negative_part")}
    change["rmse_reduction"] = summary["calibration"]["ridge"]["rmse"] - summary["calibration"]["ridge96"]["rmse"]
    summary["change48_to96"] = change
    summary["scaling_checks"] = {"mean_gain_increases": change["mean_gain"] > 0,
        "downside_not_worse": change["mean_negative_part"] >= 0, "rmse_decreases": change["rmse_reduction"] > 0}
    summary["scaling_improves"] = all(summary["scaling_checks"].values())
    return summary


def run(condition, seed, source_root=p.OUTPUT_ROOT, reference_root=previous.OUTPUT_ROOT, output_root=OUTPUT_ROOT):
    source = Path(source_root) / condition / f"seed{seed}.json"
    original = json.loads(source.read_text())
    cases = cv.training_cases(original, condition, seed)
    ref_path = Path(reference_root) / condition / f"seed{seed}.json"
    old = json.loads(ref_path.read_text())
    assert old["complete"] and old["protocol"] == previous.PROTOCOL
    assert (old["condition"], old["seed"], old["source_split"]) == (condition, seed, "train_cases")
    cached = {reference.case_key(r): r for r in old["rows"]}
    assert len(old["rows"]) == len(cached) == 192
    assert set(cached) == {reference.case_key(c) for c in cases}
    path = Path(output_root) / condition / f"seed{seed}.json"
    label_path = path.with_name(f"seed{seed}_labels.json")
    labels = collect_labels(condition, seed, original["sources"], label_path)
    new_cases = labels["train_cases"]
    report = {"protocol": PROTOCOL, "condition": condition, "seed": seed, "complete": False,
        "source": str(source), "reference": str(ref_path), "labels": str(label_path),
        "source_split": "train_cases", "fold_rule": "repeat % 4", "new_repeats": list(REPEATS),
        "penalty": 1., "train_horizon": 512, "primary_horizon": 512,
        "label_collection_interactions": labels["label_collection_interactions"],
        "environment_initialization_probe_steps": labels["environment_initialization_probe_steps"],
        "new_environment_interactions": labels["new_environment_interactions"],
        "evaluation_interactions": 0, "policy_training_interactions": 0,
        "policy_updates": 0, "critic_updates": 0, "selector_updates": 0, "mlp_updates": 0,
        "ridge_solves": 0, "old_anchors": 192, "new_anchors": 192, "folds": [], "rows": [],
        "scope": "Expanded training supervision, evaluated on reused old development holdouts. "
                 "No new held-out confirmation, policy improvement or online operator-routing claim."}

    def save():
        path.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")

    save()
    for fold in range(4):
        fit, held = split_expanded(cases, new_cases, fold)
        scores, metadata = curve.fit_predict(fit, torch.stack([cv.features(c) for c in held]))
        report["rows"].extend(reference.join_row(c, cached[reference.case_key(c)],
            {"ridge96": scores[i]}, selectors=SELECTORS) for i, c in enumerate(held))
        report["folds"].append({"fold": fold, "fit_reset_ids": sorted({c["reset_seed"] for c in fit}),
            "held_reset_ids": sorted({c["reset_seed"] for c in held}), **metadata})
        report["ridge_solves"] += 1
        save()
    report["summary"] = summarize_rows(report["rows"])
    report["complete"] = True
    save()
    print(f"DONE ridge_expansion condition={condition} seed={seed} "
          f"passes={report['summary']['group_passes']} steps={report['new_environment_interactions']}", flush=True)
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
