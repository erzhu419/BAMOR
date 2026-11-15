"""Fit objective-wise paired-return heads before preference scalarization."""

import argparse
import json
from pathlib import Path

import numpy as np
import torch

import diagnose_huber_advantage as previous


cv, p, reference, expansion = previous.cv, previous.p, previous.reference, previous.expansion
cross = previous.previous
PROTOCOL = "vector_combined_advantage_grouped_cv_v1"
OUTPUT_ROOT = Path("results/iclr_vector_advantage_20260921")
TARGET = "vector_ridge"
SELECTORS = (*previous.SELECTORS, TARGET)
CONTROLS = (*previous.CONTROLS, previous.TARGET)


def vector_gains(case):
    returns = np.asarray([case["branches"][name]["discounted_returns"]["512"] for name in p.CANDIDATES])
    assert returns.shape == (5, 2) and np.isfinite(returns).all()
    return returns - returns[0]


def training_data(cases):
    inputs = torch.cat([cv.features(c)[1:] for c in cases])
    targets = np.stack([vector_gains(c)[1:] for c in cases]).reshape(-1, 2)
    return {"inputs": inputs, "targets": torch.as_tensor(targets, dtype=torch.float32),
            "action_dim": len(cases[0]["diagnostics"]["soft"]["actions"]["base"])}


class VectorRidge(cv.QuadraticRidge):
    @torch.no_grad()
    def __init__(self, data):
        cv.QuadraticFeatures.__init__(self, data["inputs"], data["action_dim"])
        y = data["targets"].double()
        assert y.ndim == 2 and y.shape[1] == 2
        self.target_scale = y.std(0, unbiased=False).clamp_min(1e-3)
        phi = self.transform(data["inputs"])
        gram = phi.T @ phi / len(phi) + torch.eye(phi.shape[1], dtype=torch.float64)
        rhs = phi.T @ (y / self.target_scale) / len(phi)
        self.beta = torch.linalg.solve(gram, rhs)
        self.solve_residual = float((gram @ self.beta - rhs).abs().max())
        assert torch.isfinite(self.beta).all() and self.solve_residual < 1e-10


def fit_predict(cases, held_features):
    data = training_data(cases)
    model = VectorRidge(data)
    vectors = model(held_features).numpy()
    assert vectors.shape == (*held_features.shape[:-1], 2)
    assert np.isfinite(vectors).all() and (vectors[:, 0] == 0.).all()
    return vectors, {"heads": 2, "coefficients_per_head": len(model.beta), "parameters": model.beta.numel(),
        "penalty": 1., "linear_solves": 1, "fit_anchors": len(cases), "fit_nonbase_rows": len(data["targets"]),
        "target_scales": model.target_scale.tolist(), "normal_equation_residual_inf": model.solve_residual,
        "train_vector_rmse512": (model(data["inputs"]) - data["targets"].double()).square().mean(0).sqrt().tolist()}


def join_row(cached, vectors, weight):
    vectors, weight = np.asarray(vectors), np.asarray(weight)
    assert vectors.shape == (5, 2) and weight.shape == (2,)
    scores = {**cached["scores"], TARGET: dict(zip(p.CANDIDATES, map(float, vectors @ weight)))}
    choices = {name: p.choose(scores[name]) for name in SELECTORS}
    gains = cached["candidate_gains"]
    return {**cached, "weight": weight.tolist(), "vector_scores": vectors.tolist(),
        "scores": scores, "choices": choices, "selected_gains": {name: gains[choices[name]] for name in SELECTORS}}


def vector_errors(rows):
    pred = np.asarray([r["vector_scores"] for r in rows])[:, 1:]
    truth = np.asarray([r["vector_gains"] for r in rows])[:, 1:]
    return {"vector_rmse512": np.sqrt(np.mean((pred - truth) ** 2, axis=(0, 1))).tolist(),
            "vector_zero_rmse512": np.sqrt(np.mean(truth ** 2, axis=(0, 1))).tolist()}


def summarize_rows(rows, legacy_aggregate):
    result = previous.summarize_rows(rows, legacy_aggregate, target=TARGET, controls=CONTROLS,
                                     extra_selectors=(previous.TARGET,))
    for batch, block in result["batches"].items():
        group = [r for r in rows if r["batch"] == batch]
        block.update(vector_errors(group))
        for cell in block["cells"]:
            cell.update(vector_errors([r for r in group if (r["mode"], r["weight_index"]) == (cell["mode"], cell["weight_index"])]))
    return result


def run(condition, seed, source_root=p.OUTPUT_ROOT, expansion_root=expansion.OUTPUT_ROOT,
        reference_root=previous.OUTPUT_ROOT, output_root=OUTPUT_ROOT):
    relative = Path(condition) / f"seed{seed}.json"
    source = json.loads((Path(source_root) / relative).read_text())
    old = cv.training_cases(source, condition, seed)
    labels = json.loads((Path(expansion_root) / condition / f"seed{seed}_labels.json").read_text())
    new = expansion.validate_labels(labels, condition, seed, source["sources"])
    cached = json.loads((Path(reference_root) / relative).read_text())
    assert cached["complete"] and cached["protocol"] == previous.PROTOCOL
    assert (cached["condition"], cached["seed"], cached["source_split"]) == (condition, seed, "train_cases")
    lookup = {reference.case_key(r): r for r in cached["rows"]}
    assert len(cached["rows"]) == len(lookup) == 384
    assert set(lookup) == {reference.case_key(c) for c in old + new}
    legacy = json.loads((Path(expansion_root) / relative).read_text())
    legacy_aggregate = previous.validate_legacy(legacy, cached, condition, seed)
    assert legacy_aggregate == cached["legacy_aggregate"]
    assert previous.summarize_rows(cached["rows"], legacy_aggregate) == cached["summary"]
    path = Path(output_root) / relative
    path.parent.mkdir(parents=True, exist_ok=True)
    report = {"protocol": PROTOCOL, "condition": condition, "seed": seed, "complete": False,
        "source_split": "train_cases", "fold_rule": "repeat % 4", "train_horizon": 512, "penalty": 1.,
        "new_environment_interactions": 0, "policy_updates": 0, "critic_updates": 0, "mlp_updates": 0,
        "vector_fits": 0, "linear_solves": 0, "legacy_aggregate": legacy_aggregate,
        "legacy_controls": list(expansion.CONTROLS), "batch_controls": list(CONTROLS), "folds": [], "rows": [],
        "scope": "Two objective heads with retained preference context; 486 coefficients versus 243. "
                 "Reused development labels, not a capacity-matched causal test or fresh confirmation."}

    def save():
        path.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")

    save()
    for fold in range(4):
        fits, held = cross.split_batches(old, new, fold)
        held_cases = [(batch, c) for batch in cross.BATCHES for c in held[batch]]
        vectors, metadata = fit_predict(fits["combined96"], torch.stack([cv.features(c) for _, c in held_cases]))
        for i, (batch, case) in enumerate(held_cases):
            original = lookup[reference.case_key(case)]
            assert (original["batch"], original["fold"], original["reset_seed"]) == (batch, fold, case["reset_seed"])
            row = join_row(original, vectors[i], case["weight"])
            row["vector_gains"] = vector_gains(case).tolist()
            assert row["candidate_gains"] == p.realized_gains(case)
            report["rows"].append(row)
        report["folds"].append({"fold": fold, "fit_reset_ids": sorted({c["reset_seed"] for c in fits["combined96"]}),
            "held_reset_ids": {batch: sorted({c["reset_seed"] for c in cases}) for batch, cases in held.items()}, **metadata})
        report["vector_fits"] += 1
        report["linear_solves"] += metadata["linear_solves"]
        save()
        print(f"VECTOR_ADVANTAGE condition={condition} seed={seed} folds={fold + 1}/4 interactions=0", flush=True)
    report["summary"] = summarize_rows(report["rows"], legacy_aggregate)
    report["complete"] = True
    save()
    print(f"DONE vector_advantage condition={condition} seed={seed} passes={report['summary']['group_passes']}", flush=True)
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--condition", choices=p.g.METHODS, required=True)
    parser.add_argument("--seed", type=int, choices=p.g.SEEDS, required=True)
    parser.add_argument("--source-root", type=Path, default=p.OUTPUT_ROOT)
    parser.add_argument("--expansion-root", type=Path, default=expansion.OUTPUT_ROOT)
    parser.add_argument("--reference-root", type=Path, default=previous.OUTPUT_ROOT)
    parser.add_argument("--output-root", type=Path, default=OUTPUT_ROOT)
    args = parser.parse_args()
    run(args.condition, args.seed, args.source_root, args.expansion_root, args.reference_root, args.output_root)
