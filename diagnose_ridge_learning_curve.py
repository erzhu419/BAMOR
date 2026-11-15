"""Fixed-ridge learning curves on nested, reset-grouped training subsets."""

import argparse
import json
from pathlib import Path

import numpy as np
import torch

import diagnose_state_curvature as previous


cv, p, reference = previous.cv, previous.p, previous.reference
PROTOCOL = "ridge_learning_curve_grouped_cv_v1"
OUTPUT_ROOT = Path("results/iclr_ridge_learning_curve_20260921")
SIZES = (16, 32, 48)
SUBSET_RULE = "repeat blocks; cyclic start (block+chain)%3; first size/16 entries"
FIT_SPECS = tuple((size, chain) for size in (16, 32) for chain in range(3)) + ((48, 0),)


def selector(size, chain):
    return f"ridge{size}_c{chain}" if size != 48 else "ridge48"


CURVE_SELECTORS = tuple(selector(*spec) for spec in FIT_SPECS)
SELECTORS = (*CURVE_SELECTORS, *previous.SELECTORS)
CONTROLS = tuple(name for name in previous.SELECTORS if name != "no_change")


def fitting_subset(fit_cases, fold, size, chain):
    assert size in SIZES and chain in range(3) and fold in range(4)
    repeats = set()
    for block in range(4):
        available = [r for r in range(4 * block, 4 * block + 4) if r % 4 != fold]
        repeats.update(available[(block + chain + j) % 3] for j in range(size // 16))
    subset = [c for c in fit_cases if c["repeat"] in repeats]
    assert len(subset) == size * 3
    assert len({c["reset_seed"] for c in subset}) == size
    assert {(c["mode"], c["weight_index"], c["repeat"]) for c in subset} == {
        (mode, wi, repeat) for mode in range(4) for wi in range(3) for repeat in repeats}
    return subset


def fit_predict(fit_cases, held_features):
    data = p.training_data(fit_cases)
    model = cv.QuadraticRidge(data, penalty=1.)
    scores = model(held_features).numpy()
    assert np.isfinite(scores).all() and (scores[:, 0] == 0.).all()
    mse = float((model(data["inputs"]) - data["targets"].double()).square().mean())
    return scores, {"parameters": len(model.beta), "penalty": 1., "train_horizon": 512,
        "fit_anchors": len(fit_cases), "fit_nonbase_rows": len(data["targets"]),
        "target_scale": float(model.target_scale), "train_rmse512": float(np.sqrt(mse)),
        "train_normalized_mse": mse / float(model.target_scale) ** 2}


def check_endpoint(scores, cached_rows):
    cached = np.asarray([[r["scores"]["ridge"][label] for label in p.CANDIDATES] for r in cached_rows])
    np.testing.assert_allclose(scores, cached, rtol=1e-10, atol=1e-10)
    assert all(p.choose(dict(zip(p.CANDIDATES, s))) == r["choices"]["ridge"]
               for s, r in zip(scores, cached_rows))
    return float(np.max(np.abs(scores - cached)))


def curve_analysis(points):
    transitions = []
    for before, after in zip(points, points[1:]):
        changes = {key: after[key] - before[key] for key in ("mean_gain", "mean_negative_part")}
        changes["rmse_reduction"] = before["rmse512"] - after["rmse512"]
        counts = {"mean_gain": 0, "mean_negative_part": 0, "rmse512": 0}
        for low, high in zip(before["chains"], after["chains"]):
            counts["mean_gain"] += high["mean_gain"] > low["mean_gain"]
            counts["mean_negative_part"] += high["mean_negative_part"] >= low["mean_negative_part"]
            counts["rmse512"] += high["rmse512"] < low["rmse512"]
        transitions.append({"from": before["fit_reset_groups"], "to": after["fit_reset_groups"],
            **changes, "favorable_chain_counts": counts})
    recent = transitions[-1]
    checks = {"mean_gain_increases": recent["mean_gain"] > 0,
        "downside_not_worse": recent["mean_negative_part"] >= 0,
        "rmse_decreases": recent["rmse_reduction"] > 0}
    return {"transitions": transitions, "recent_checks": checks,
        "recent_improvement": all(checks.values())}


def summarize_rows(rows):
    summary = cv.summarize_rows(rows, selectors=SELECTORS, controls=CONTROLS,
        target="ridge48", value_selectors=(*CURVE_SELECTORS, "ridge"))
    points = []
    for size in SIZES:
        chains = [{"chain": chain, **summary["aggregate"][selector(size, chain)],
            "rmse512": summary["calibration"][selector(size, chain)]["rmse"],
            "base_choices": summary["selection_counts"][selector(size, chain)]["base"]}
            for chain in range(3)]
        points.append({"fit_reset_groups": size,
            **{k: float(np.mean([c[k] for c in chains])) for k in ("mean_gain", "mean_negative_part")},
            "rmse512": float(np.sqrt(np.mean([c["rmse512"] ** 2 for c in chains]))),
            "chains": chains})
    return {"curve": points, **curve_analysis(points),
        "cached_controls": {name: summary["aggregate"][name] for name in previous.SELECTORS},
        "endpoint_control_checks": summary["checks"],
        "endpoint_passes_controls": summary["group_passes"]}


def run(condition, seed, source_root=p.OUTPUT_ROOT, reference_root=previous.OUTPUT_ROOT,
        output_root=OUTPUT_ROOT):
    source = Path(source_root) / condition / f"seed{seed}.json"
    ref_path = Path(reference_root) / condition / f"seed{seed}.json"
    cases = cv.training_cases(json.loads(source.read_text()), condition, seed)
    old = json.loads(ref_path.read_text())
    assert old["complete"] and old["protocol"] == previous.PROTOCOL
    assert (old["condition"], old["seed"], old["source_split"]) == (condition, seed, "train_cases")
    cached = {reference.case_key(r): r for r in old["rows"]}
    assert len(old["rows"]) == len(cached) == len(cases)
    assert set(cached) == {reference.case_key(c) for c in cases}
    path = Path(output_root) / condition / f"seed{seed}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    report = {"protocol": PROTOCOL, "condition": condition, "seed": seed, "complete": False,
        "source": str(source), "reference": str(ref_path), "source_split": "train_cases",
        "subset_rule": SUBSET_RULE, "sizes": list(SIZES), "chains": 3, "fold_rule": "repeat % 4",
        "train_horizon": 512, "primary_horizon": 512, "penalty": 1.,
        "new_environment_interactions": 0, "selector_updates": 0, "policy_updates": 0,
        "critic_updates": 0, "mlp_updates": 0, "ridge_solves": 0,
        "reused_anchors": 192, "reused_nonbase_labels": 768, "folds": [], "rows": []}

    def save():
        path.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")

    save()
    for fold in range(4):
        fit, held = cv.split_fold(cases, fold)
        held_features = torch.stack([cv.features(c) for c in held])
        predictions, fits = {}, []
        for size, chain in FIT_SPECS:
            subset = fitting_subset(fit, fold, size, chain)
            name = selector(size, chain)
            predictions[name], metadata = fit_predict(subset, held_features)
            fits.append({"size": size, "chain": chain, "fit_reset_ids": sorted({c["reset_seed"] for c in subset}),
                         **metadata})
        error = check_endpoint(predictions["ridge48"], [cached[reference.case_key(c)] for c in held])
        # All seven fits receive held features only; held gains enter after predictions.
        report["rows"].extend(reference.join_row(c, cached[reference.case_key(c)],
            {name: values[i] for name, values in predictions.items()}, selectors=SELECTORS)
            for i, c in enumerate(held))
        report["folds"].append({"fold": fold, "held_reset_ids": sorted({c["reset_seed"] for c in held}),
            "fits": fits, "endpoint_max_abs_error": error})
        report["ridge_solves"] += len(fits)
        save()
        print(f"LEARNING_CURVE condition={condition} seed={seed} folds={fold + 1}/4 interactions=0", flush=True)
    report["summary"] = summarize_rows(report["rows"])
    report["complete"] = True
    save()
    print(f"DONE learning_curve condition={condition} seed={seed} "
          f"recent_improvement={report['summary']['recent_improvement']}", flush=True)
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
