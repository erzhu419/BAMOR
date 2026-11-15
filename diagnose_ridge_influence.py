"""Exact added-batch ridge attribution in fixed combined-input coordinates."""

import argparse
import json
from pathlib import Path

import numpy as np
import torch

import diagnose_ridge_batch_cross as previous


cv, p, reference = previous.cv, previous.p, previous.reference
PROTOCOL = "ridge_added_label_influence_v1"
OUTPUT_ROOT = Path("results/iclr_ridge_influence_20260921")
OLD = "old_labels_combined_transform"
SELECTORS = (*previous.SELECTORS, OLD)


@torch.no_grad()
def solve_decomposition(x_old, y_old, x_new, y_new):
    x, y = torch.cat((x_old, x_new)), torch.cat((y_old, y_new))
    identity = torch.eye(x.shape[1], dtype=torch.float64)
    old = torch.linalg.solve(x_old.T @ x_old / len(x_old) + identity, x_old.T @ y_old / len(x_old))
    gram = x.T @ x / len(x) + identity
    combined = torch.linalg.solve(gram, x.T @ y / len(x))
    groups = x_new.reshape(-1, 4, x.shape[1])
    label_rhs = torch.einsum("acd,ac->da", groups, y_new.reshape(-1, 4)) / len(x)
    feedback_rhs = -torch.einsum("acd,ac->da", groups, (x_new @ old).reshape(-1, 4)) / len(x)
    # Old normal equations leave -alpha * lambda * beta_old when losses are re-averaged.
    ridge_rhs = -len(x_new) / len(x) * old[:, None]
    parts = torch.linalg.solve(gram, torch.cat((label_rhs, feedback_rhs, ridge_rhs), dim=1))
    torch.testing.assert_close(parts.sum(1), combined - old, rtol=1e-9, atol=1e-10)
    k = len(groups)
    return old, combined, parts[:, :k], parts[:, k:2 * k], parts[:, -1:]


@torch.no_grad()
def decompose(fits, held_features):
    data = p.training_data(fits["combined96"])
    transform = cv.QuadraticFeatures(data["inputs"], data["action_dim"])
    x = transform.transform(data["inputs"])
    raw_y = data["targets"].double()
    scale = raw_y.std(unbiased=False).clamp_min(1e-3)
    n_old = len(fits["old48"]) * 4
    old, combined, labels, feedback, ridge = solve_decomposition(
        x[:n_old], raw_y[:n_old] / scale, x[n_old:], raw_y[n_old:] / scale)
    held = transform.transform(held_features)
    scores = {OLD: (held @ old * scale).numpy(), "combined96": (held @ combined * scale).numpy()}
    for value in scores.values():
        assert np.isfinite(value).all() and (value[:, 0] == 0.).all()
    parts = (labels + feedback) * scale
    delta = (combined - old) * scale
    reconstructed = held @ (parts.sum(1) + ridge[:, 0] * scale)
    np.testing.assert_allclose(reconstructed, scores["combined96"] - scores[OLD], rtol=1e-9, atol=1e-10)
    return {"scores": scores, "held": held, "parts": parts, "delta": delta,
        "labels": labels * scale, "feedback": feedback * scale, "ridge": ridge * scale,
        "new_targets": raw_y[n_old:].reshape(-1, 4),
        "new_residuals": (raw_y[n_old:] - x[n_old:] @ old * scale).reshape(-1, 4),
        "target_scale": float(scale), "parameters": len(old),
        "coefficient_error": float((parts.sum(1) + ridge[:, 0] * scale - delta).abs().max()),
        "score_error": float((reconstructed - torch.from_numpy(scores["combined96"] - scores[OLD])).abs().max())}


def ratio(numerator, denominator):
    return float(numerator / denominator) if denominator != 0 else None


def describe_vectors(vectors, result, batch_indices):
    delta = result["delta"]
    projection = vectors.T @ delta
    held_scores = result["held"] @ vectors
    old_choices = result["scores"][OLD].argmax(1)
    new_choices = result["scores"]["combined96"].argmax(1)
    records = [{"coefficient_norm": float(vectors[:, i].norm()),
                "coefficient_delta_projection": ratio(projection[i], delta.square().sum()), "batches": {}}
               for i in range(vectors.shape[1])]
    for batch, indices in batch_indices.items():
        rms = held_scores[indices, 1:].square().mean((0, 1)).sqrt()
        margins = (held_scores[indices, new_choices[indices]] - held_scores[indices, old_choices[indices]]).mean(0)
        for i, record in enumerate(records):
            record["batches"][batch] = {"score_rms": float(rms[i]), "mean_switch_margin": float(margins[i])}
    return records


def fold_report(fits, held_cases, result, fold):
    cases = fits["new48"]
    reset_ids = sorted({c["reset_seed"] for c in cases})
    index = {reset: [i for i, c in enumerate(cases) if c["reset_seed"] == reset] for reset in reset_ids}
    pref_index = {wi: [i for i, c in enumerate(cases) if c["weight_index"] == wi] for wi in range(3)}
    vectors = torch.cat((torch.stack([result["parts"][:, ids].sum(1) for ids in index.values()], 1),
        torch.stack([result["parts"][:, ids].sum(1) for ids in pref_index.values()], 1),
        result["labels"].sum(1, keepdim=True), result["feedback"].sum(1, keepdim=True),
        result["ridge"], result["delta"][:, None]), 1)
    batch_indices = {batch: [i for i, (b, _) in enumerate(held_cases) if b == batch] for batch in previous.BATCHES}
    stats = describe_vectors(vectors, result, batch_indices)
    resets = []
    for i, (reset, ids) in enumerate(index.items()):
        resets.append({"reset_seed": reset, "mode": cases[ids[0]]["mode"], "repeat": cases[ids[0]]["repeat"],
            "label_abs_share": ratio(result["new_targets"][ids].abs().sum(), result["new_targets"].abs().sum()),
            "residual_squared_share": ratio(result["new_residuals"][ids].square().sum(), result["new_residuals"].square().sum()),
            **stats[i]})
    k = len(reset_ids)
    components = dict(zip(("labels", "feedback", "ridge", "total"), stats[k + 3:]))
    for batch in previous.BATCHES:
        total = components["total"]["batches"][batch]["mean_switch_margin"]
        reconstructed = sum(r["batches"][batch]["mean_switch_margin"] for r in resets) + components["ridge"]["batches"][batch]["mean_switch_margin"]
        np.testing.assert_allclose(reconstructed, total, rtol=1e-9, atol=1e-10)
    return {"fold": fold, "fit_reset_ids": {name: sorted({c["reset_seed"] for c in fit}) for name, fit in fits.items()},
        "held_reset_ids": {batch: sorted({c["reset_seed"] for b, c in held_cases if b == batch}) for batch in previous.BATCHES},
        **{key: result[key] for key in ("target_scale", "parameters", "coefficient_error", "score_error")},
        "resets": resets, "preferences": [{"weight_index": wi, **stats[k + wi]} for wi in range(3)],
        "components": components,
        "changed_choices": {batch: sum(bool(result["scores"][OLD][i].argmax() != result["scores"]["combined96"][i].argmax())
                                         for i in indices) for batch, indices in batch_indices.items()}}


def summarize_rows(rows):
    result = previous.summarize_rows(rows)
    for batch, block in result["batches"].items():
        selected = [r for r in rows if r["batch"] == batch]
        block["aggregate"].update(previous.metrics(selected, selectors=(OLD,)))
        for cell in block["cells"]:
            cell["metrics"].update(previous.metrics([r for r in selected if (r["mode"], r["weight_index"]) ==
                (cell["mode"], cell["weight_index"])], selectors=(OLD,)))
    return result


def compact_fold(fold):
    result = {k: v for k, v in fold.items() if k not in ("resets", "fit_reset_ids", "held_reset_ids")}
    result["influence"] = {}
    for batch in previous.BATCHES:
        ordered = sorted(fold["resets"], key=lambda r: (-r["batches"][batch]["score_rms"], r["reset_seed"]))
        mass = sum(r["batches"][batch]["score_rms"] for r in ordered)
        result["influence"][batch] = {"top1_rms_mass_share": ratio(ordered[0]["batches"][batch]["score_rms"], mass),
            "top5_rms_mass_share": ratio(sum(r["batches"][batch]["score_rms"] for r in ordered[:5]), mass),
            "reset_rms_mass": mass, "top5_resets": ordered[:5],
            "positive_reset_margin": sum(max(r["batches"][batch]["mean_switch_margin"], 0.) for r in ordered),
            "negative_reset_margin": sum(min(r["batches"][batch]["mean_switch_margin"], 0.) for r in ordered)}
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
        "source_split": "train_cases", "fold_rule": "repeat % 4", "penalty": 1., "train_horizon": 512,
        "transform": "combined96 fitting inputs", "target_scale": "combined96 fitting labels",
        "new_environment_interactions": 0, "optimizer_updates": 0, "model_solves": 0, "attribution_solves": 0,
        "previous_all_control_pass": cached["previous_all_control_pass"], "folds": [], "rows": []}

    def save():
        path.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")

    save()
    for fold in range(4):
        fits, held = previous.split_batches(old, new, fold)
        held_cases = [(batch, c) for batch in previous.BATCHES for c in held[batch]]
        result = decompose(fits, torch.stack([cv.features(c) for _, c in held_cases]))
        errors = []
        for i, (batch, case) in enumerate(held_cases):
            original = lookup[reference.case_key(case)]
            assert (original["batch"], original["fold"], original["reset_seed"]) == (batch, fold, case["reset_seed"])
            expected = [original["scores"]["combined96"][name] for name in p.CANDIDATES]
            np.testing.assert_allclose(result["scores"]["combined96"][i], expected, rtol=1e-10, atol=1e-10)
            assert p.CANDIDATES[result["scores"]["combined96"][i].argmax()] == original["choices"]["combined96"]
            errors.append(float(np.max(np.abs(result["scores"]["combined96"][i] - expected))))
            scores = {**original["scores"], OLD: dict(zip(p.CANDIDATES, map(float, result["scores"][OLD][i])))}
            choices = {name: p.choose(scores[name]) for name in SELECTORS}
            assert original["candidate_gains"] == p.realized_gains(case)
            report["rows"].append({**original, "scores": scores, "choices": choices,
                "selected_gains": {name: original["candidate_gains"][choices[name]] for name in SELECTORS}})
        report["folds"].append({**fold_report(fits, held_cases, result, fold), "cached_score_error": max(errors)})
        report["model_solves"] += 2
        report["attribution_solves"] += 1
        save()
        print(f"RIDGE_INFLUENCE condition={condition} seed={seed} folds={fold + 1}/4 interactions=0", flush=True)
    report["summary"] = summarize_rows(report["rows"])
    report["complete"] = True
    save()
    print(f"DONE ridge_influence condition={condition} seed={seed} anchors={len(report['rows'])}", flush=True)
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
