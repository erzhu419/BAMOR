"""Capacity-matched preference-projected ridge on existing paired returns."""

import argparse
import json
from pathlib import Path

import numpy as np
import torch

import diagnose_vector_advantage as previous


cv, p, reference, expansion, cross = previous.cv, previous.p, previous.reference, previous.expansion, previous.cross
gates = previous.previous
PROTOCOL = "projected_combined_advantage_grouped_cv_v1"
OUTPUT_ROOT = Path("results/iclr_projected_advantage_20260921")
TARGET = "projected_ridge"
SELECTORS = (*previous.SELECTORS, TARGET)
CONTROLS = (*previous.CONTROLS, previous.TARGET)


def training_data(cases):
    data = p.training_data(cases)
    return {**data, "weights": torch.tensor([c["weight"] for c in cases], dtype=torch.float64).repeat_interleave(4, 0)}


class ProjectedRidge(cv.QuadraticFeatures):
    @torch.no_grad()
    def __init__(self, data):
        super().__init__(data["inputs"], data["action_dim"])
        y = data["targets"].double()
        self.target_scale = y.std(unbiased=False).clamp_min(1e-3)
        psi = self.project(data["inputs"], data["weights"])
        gram = psi.T @ psi / len(psi) + torch.eye(psi.shape[1], dtype=torch.float64)
        rhs = psi.T @ (y / self.target_scale) / len(psi)
        self.beta = torch.linalg.solve(gram, rhs)
        self.solve_residual = float((gram @ self.beta - rhs).abs().max())
        assert torch.isfinite(self.beta).all() and self.solve_residual < 1e-10

    def project(self, inputs, weights):
        phi = self.transform(inputs)
        assert weights.shape == (*phi.shape[:-1], 2)
        # Preserve Phi's fitting-only normalization; no second normalization of w tensor Phi.
        return (weights.double().unsqueeze(-1) * phi.unsqueeze(-2)).flatten(-2)

    def __call__(self, inputs, weights):
        return (self.project(inputs, weights) @ self.beta) * self.target_scale


def fit_predict(cases, held_features, held_weights):
    data = training_data(cases)
    model = ProjectedRidge(data)
    scores = model(held_features, held_weights).numpy()
    assert scores.shape == held_features.shape[:-1]
    assert np.isfinite(scores).all() and (scores[:, 0] == 0.).all()
    return scores, {"heads": 2, "coefficients_per_head": len(model.feature_scale), "parameters": model.beta.numel(),
        "penalty": 1., "linear_solves": 1, "fit_anchors": len(cases), "fit_nonbase_rows": len(data["targets"]),
        "target_scale": float(model.target_scale), "normal_equation_residual_inf": model.solve_residual,
        "train_rmse512": float((model(data["inputs"], data["weights"]) - data["targets"].double()).square().mean().sqrt())}


def join_row(cached, prediction):
    prediction = np.asarray(prediction)
    assert prediction.shape == (5,) and np.isfinite(prediction).all() and prediction[0] == 0.
    scores = {**cached["scores"], TARGET: dict(zip(p.CANDIDATES, map(float, prediction)))}
    choices = {name: p.choose(scores[name]) for name in SELECTORS}
    gains = cached["candidate_gains"]
    return {**cached, "scores": scores, "choices": choices,
            "selected_gains": {name: gains[choices[name]] for name in SELECTORS}}


def summarize_rows(rows, legacy_aggregate):
    return gates.summarize_rows(rows, legacy_aggregate, target=TARGET, controls=CONTROLS,
                                extra_selectors=(gates.TARGET, previous.TARGET))


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
    legacy_aggregate = gates.validate_legacy(legacy, cached, condition, seed)
    assert legacy_aggregate == cached["legacy_aggregate"]
    assert previous.summarize_rows(cached["rows"], legacy_aggregate) == cached["summary"]
    path = Path(output_root) / relative
    path.parent.mkdir(parents=True, exist_ok=True)
    report = {"protocol": PROTOCOL, "condition": condition, "seed": seed, "complete": False,
        "source_split": "train_cases", "fold_rule": "repeat % 4", "train_horizon": 512, "penalty": 1.,
        "new_environment_interactions": 0, "policy_updates": 0, "critic_updates": 0, "mlp_updates": 0,
        "projected_fits": 0, "linear_solves": 0, "legacy_aggregate": legacy_aggregate,
        "legacy_controls": list(expansion.CONTROLS), "batch_controls": list(CONTROLS), "folds": [], "rows": [],
        "scope": "486 coefficients, same Phi and penalty as vector heads; scalar preference-projected loss. "
                 "Reused development labels; latent heads are not objective-return estimators."}

    def save():
        path.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")

    save()
    for fold in range(4):
        fits, held = cross.split_batches(old, new, fold)
        held_cases = [(batch, c) for batch in cross.BATCHES for c in held[batch]]
        features = torch.stack([cv.features(c) for _, c in held_cases])
        weights = torch.tensor([c["weight"] for _, c in held_cases], dtype=torch.float64)[:, None, :].expand(-1, 5, -1)
        predictions, metadata = fit_predict(fits["combined96"], features, weights)
        for i, (batch, case) in enumerate(held_cases):
            original = lookup[reference.case_key(case)]
            assert (original["batch"], original["fold"], original["reset_seed"], original["weight"]) == (batch, fold, case["reset_seed"], case["weight"])
            row = join_row(original, predictions[i])
            assert row["candidate_gains"] == p.realized_gains(case)
            report["rows"].append(row)
        report["folds"].append({"fold": fold, "fit_reset_ids": sorted({c["reset_seed"] for c in fits["combined96"]}),
            "held_reset_ids": {batch: sorted({c["reset_seed"] for c in cases}) for batch, cases in held.items()}, **metadata})
        report["projected_fits"] += 1
        report["linear_solves"] += 1
        save()
        print(f"PROJECTED_ADVANTAGE condition={condition} seed={seed} folds={fold + 1}/4 interactions=0", flush=True)
    report["summary"] = summarize_rows(report["rows"], legacy_aggregate)
    report["complete"] = True
    save()
    print(f"DONE projected_advantage condition={condition} seed={seed} passes={report['summary']['group_passes']}", flush=True)
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
