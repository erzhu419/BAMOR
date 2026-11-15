"""Hold combined fitting labels fixed while changing the input-transform donor."""

import argparse
import json
from pathlib import Path

import numpy as np
import torch

import diagnose_ridge_batch_cross as previous


cv, p, reference = previous.cv, previous.p, previous.reference
PROTOCOL = "ridge_input_transform_grouped_cv_v1"
OUTPUT_ROOT = Path("results/iclr_ridge_transform_20260921")
DONORS = {"combined_old_transform": "old48", "combined_new_transform": "new48"}
SELECTORS = (*previous.SELECTORS, *DONORS)


class FixedTransformRidge(cv.QuadraticRidge):
    @torch.no_grad()
    def __init__(self, data, transform_inputs):
        # Only the basis normalization sees donor inputs; the solve uses all combined labels.
        cv.QuadraticFeatures.__init__(self, transform_inputs, data["action_dim"])
        y = data["targets"].double()
        self.target_scale = y.std(unbiased=False).clamp_min(1e-3)
        phi = self.transform(data["inputs"])
        gram = phi.T @ phi / len(phi) + torch.eye(phi.shape[1], dtype=torch.float64)
        self.beta = torch.linalg.solve(gram, phi.T @ (y / self.target_scale) / len(phi))
        assert torch.isfinite(self.beta).all()


def fit_models(fits, held_features):
    data = p.training_data(fits["combined96"])
    predictions, metadata = {}, {}
    for name, donor in DONORS.items():
        inputs = torch.cat([cv.features(c)[1:] for c in fits[donor]])
        model = FixedTransformRidge(data, inputs)
        predictions[name] = model(held_features).numpy()
        assert np.isfinite(predictions[name]).all() and (predictions[name][:, 0] == 0.).all()
        mse = float((model(data["inputs"]) - data["targets"].double()).square().mean())
        metadata[name] = {"transform_donor": donor, "parameters": len(model.beta), "penalty": 1.,
            "train_horizon": 512, "fit_anchors": len(fits["combined96"]), "fit_nonbase_rows": len(data["targets"]),
            "transform_anchors": len(fits[donor]), "transform_nonbase_rows": len(inputs),
            "target_scale": float(model.target_scale), "train_rmse512": float(np.sqrt(mse)),
            "train_normalized_mse": mse / float(model.target_scale) ** 2}
    return predictions, metadata


def join_row(cached, predictions):
    scores = {**cached["scores"], **{name: dict(zip(p.CANDIDATES, map(float, values)))
                                   for name, values in predictions.items()}}
    choices = {name: p.choose(scores[name]) for name in SELECTORS}
    gains = cached["candidate_gains"]
    return {**cached, "scores": scores, "choices": choices,
            "selected_gains": {name: gains[choices[name]] for name in SELECTORS}}


def summarize_rows(rows):
    result = previous.summarize_rows(rows)
    comparisons = {}
    for batch in previous.BATCHES:
        group = [r for r in rows if r["batch"] == batch]
        block = result["batches"][batch]
        block["aggregate"].update(previous.metrics(group, selectors=DONORS))
        for cell in block["cells"]:
            selected = [r for r in group if (r["mode"], r["weight_index"]) == (cell["mode"], cell["weight_index"])]
            cell["metrics"].update(previous.metrics(selected, selectors=DONORS))
        baseline = block["aggregate"]["combined96"]
        comparisons[batch] = {name: {
            **{key: block["aggregate"][name][key] - baseline[key] for key in ("mean_gain", "mean_negative_part")},
            "rmse_reduction": baseline["rmse512"] - block["aggregate"][name]["rmse512"],
            "rmse_minus_zero": block["aggregate"][name]["rmse512"] - block["aggregate"]["no_change"]["rmse512"],
            "changed_choices": sum(r["choices"][name] != r["choices"]["combined96"] for r in group)}
            for name in DONORS}
    result["transform_comparisons"] = comparisons
    return result


def run(condition, seed, source_root=p.OUTPUT_ROOT, expansion_root=previous.previous.OUTPUT_ROOT,
        reference_root=previous.OUTPUT_ROOT, output_root=OUTPUT_ROOT):
    relative = Path(condition) / f"seed{seed}.json"
    source = json.loads((Path(source_root) / relative).read_text())
    old = cv.training_cases(source, condition, seed)
    labels = json.loads((Path(expansion_root) / condition / f"seed{seed}_labels.json").read_text())
    new = previous.previous.validate_labels(labels, condition, seed, source["sources"])
    cached = json.loads((Path(reference_root) / relative).read_text())
    assert cached["complete"] and cached["protocol"] == previous.PROTOCOL
    assert (cached["condition"], cached["seed"], cached["source_split"]) == (condition, seed, "train_cases")
    lookup = {reference.case_key(r): r for r in cached["rows"]}
    assert len(cached["rows"]) == len(lookup) == 384
    assert set(lookup) == {reference.case_key(c) for c in old + new}
    path = Path(output_root) / relative
    path.parent.mkdir(parents=True, exist_ok=True)
    report = {"protocol": PROTOCOL, "condition": condition, "seed": seed, "complete": False,
        "source_split": "train_cases", "fold_rule": "repeat % 4", "train_horizon": 512, "penalty": 1.,
        "source": str(Path(source_root) / relative), "reference": str(Path(reference_root) / relative),
        "transform_donors": DONORS, "new_environment_interactions": 0, "selector_updates": 0,
        "policy_updates": 0, "critic_updates": 0, "mlp_updates": 0, "ridge_solves": 0,
        "reused_old_anchors": 192, "reused_new_anchors": 192,
        "previous_all_control_pass": cached["previous_all_control_pass"], "folds": [], "rows": [],
        "scope": "Fixed combined labels with donor-only input normalization; descriptive development "
                 "control, not a new acceptance gate or batch-specific transform selection."}

    def save():
        path.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")

    save()
    for fold in range(4):
        fits, held = previous.split_batches(old, new, fold)
        held_cases = [(batch, c) for batch in previous.BATCHES for c in held[batch]]
        scores, metadata = fit_models(fits, torch.stack([cv.features(c) for _, c in held_cases]))
        for i, (batch, case) in enumerate(held_cases):
            cached_row = lookup[reference.case_key(case)]
            assert (cached_row["batch"], cached_row["fold"], cached_row["reset_seed"]) == (batch, fold, case["reset_seed"])
            row = join_row(cached_row, {name: values[i] for name, values in scores.items()})
            assert row["candidate_gains"] == p.realized_gains(case)
            report["rows"].append(row)
        report["folds"].append({"fold": fold,
            "fit_reset_ids": sorted({c["reset_seed"] for c in fits["combined96"]}),
            "held_reset_ids": {batch: sorted({c["reset_seed"] for c in cases}) for batch, cases in held.items()},
            "fits": {name: {"transform_reset_ids": sorted({c["reset_seed"] for c in fits[DONORS[name]]}), **meta}
                     for name, meta in metadata.items()}})
        report["ridge_solves"] += 2
        save()
        print(f"RIDGE_TRANSFORM condition={condition} seed={seed} folds={fold + 1}/4 interactions=0", flush=True)
    report["summary"] = summarize_rows(report["rows"])
    report["complete"] = True
    save()
    print(f"DONE ridge_transform condition={condition} seed={seed} anchors={len(report['rows'])}", flush=True)
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--condition", choices=p.g.METHODS, required=True)
    parser.add_argument("--seed", type=int, choices=p.g.SEEDS, required=True)
    parser.add_argument("--source-root", type=Path, default=p.OUTPUT_ROOT)
    parser.add_argument("--expansion-root", type=Path, default=previous.previous.OUTPUT_ROOT)
    parser.add_argument("--reference-root", type=Path, default=previous.OUTPUT_ROOT)
    parser.add_argument("--output-root", type=Path, default=OUTPUT_ROOT)
    args = parser.parse_args()
    run(args.condition, args.seed, args.source_root, args.expansion_root, args.reference_root, args.output_root)
