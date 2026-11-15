"""One fixed Huber-loss intervention on combined paired-return training labels."""

import argparse
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

import diagnose_ridge_batch_cross as previous


cv, p, reference, expansion = previous.cv, previous.p, previous.reference, previous.previous
PROTOCOL = "huber_combined_advantage_grouped_cv_v1"
OUTPUT_ROOT = Path("results/iclr_huber_advantage_20260921")
TARGET = "huber"
SELECTORS = (*previous.SELECTORS, TARGET)
CONTROLS = tuple(n for n in previous.SELECTORS if n != "no_change")
DELTA, PENALTY, GRAD_TOL, MAX_ITERATIONS = 1., 1., 1e-8, 100


@torch.no_grad()
def solve_huber(phi, y):
    beta = torch.zeros(phi.shape[1], dtype=torch.float64)
    identity = torch.eye(len(beta), dtype=torch.float64)

    def objective(b):
        return float(F.huber_loss(phi @ b, y, delta=DELTA) + .5 * PENALTY * b.square().sum())

    history = [objective(beta)]
    for step in range(1, MAX_ITERATIONS + 1):
        residual = y - phi @ beta
        weights = DELTA / residual.abs().clamp_min(DELTA)
        # Keep the original sample-count divisor: normalizing by sum(weights) changes regularization.
        gram = phi.T @ (weights[:, None] * phi) / len(y) + PENALTY * identity
        beta = torch.linalg.solve(gram, phi.T @ (weights * y) / len(y))
        value = objective(beta)
        assert np.isfinite(value) and value <= history[-1] + 1e-12 * (1. + abs(history[-1]))
        history.append(value)
        residual = y - phi @ beta
        gradient = PENALTY * beta - phi.T @ residual.clamp(-DELTA, DELTA) / len(y)
        stationarity = float(gradient.abs().max())
        if stationarity <= GRAD_TOL:
            weights = DELTA / residual.abs().clamp_min(DELTA)
            return beta, weights, {"converged": True, "linear_solves": step,
                "stationarity_inf": stationarity, "objective_history": history}
    raise RuntimeError(f"Huber IRLS did not converge in {MAX_ITERATIONS} solves: gradient={stationarity}")


class HuberRidge(cv.QuadraticRidge):
    @torch.no_grad()
    def __init__(self, data):
        cv.QuadraticFeatures.__init__(self, data["inputs"], data["action_dim"])
        y = data["targets"].double()
        self.target_scale = y.std(unbiased=False).clamp_min(1e-3)
        self.beta, self.weights, self.solver = solve_huber(self.transform(data["inputs"]), y / self.target_scale)


def fit_predict(cases, held_features):
    data = p.training_data(cases)
    model = HuberRidge(data)
    scores = model(held_features).numpy()
    assert np.isfinite(scores).all() and (scores[:, 0] == 0.).all()
    weights = model.weights.reshape(len(cases), 4)
    weight_groups = []
    for batch in previous.BATCHES:
        for wi in range(3):
            indices = [i for i, c in enumerate(cases) if (c["repeat"] < 16) == (batch == "old") and c["weight_index"] == wi]
            w = weights[indices].flatten()
            weight_groups.append({"batch": batch, "weight_index": wi, "rows": len(w),
                "downweighted_rows": int((w < 1.).sum()), "mean_weight": float(w.mean()), "min_weight": float(w.min())})
    return scores, {"parameters": len(model.beta), "fit_anchors": len(cases), "fit_nonbase_rows": len(data["targets"]),
        "target_scale": float(model.target_scale), "delta": DELTA, "penalty": PENALTY,
        "max_iterations": MAX_ITERATIONS, "gradient_tolerance": GRAD_TOL,
        "train_rmse512": float((model(data["inputs"]) - data["targets"].double()).square().mean().sqrt()),
        "weight_groups": weight_groups, **model.solver}


def join_row(cached, prediction):
    scores = {**cached["scores"], TARGET: dict(zip(p.CANDIDATES, map(float, prediction)))}
    choices = {name: p.choose(scores[name]) for name in SELECTORS}
    gains = cached["candidate_gains"]
    return {**cached, "scores": scores, "choices": choices,
            "selected_gains": {name: gains[choices[name]] for name in SELECTORS}}


def validate_legacy(legacy, cached, condition, seed):
    assert legacy["complete"] and legacy["protocol"] == expansion.PROTOCOL
    assert (legacy["condition"], legacy["seed"], legacy["source_split"]) == (condition, seed, "train_cases")
    lookup = {reference.case_key(r): r for r in legacy["rows"]}
    old_rows = [r for r in cached["rows"] if r["batch"] == "old"]
    assert len(legacy["rows"]) == len(lookup) == len(old_rows) == 192
    assert set(lookup) == {reference.case_key(r) for r in old_rows}
    for row in old_rows:
        old = lookup[reference.case_key(row)]
        assert row["reset_seed"] == old["reset_seed"] and row["candidate_gains"] == old["candidate_gains"]
        for name, original in (("old48", "ridge"), ("combined96", "ridge96"), ("soft", "soft"), ("raw", "raw"), ("no_change", "no_change")):
            assert all(row[key][name] == old[key][original] for key in ("scores", "choices", "selected_gains"))
    result = expansion.summarize_rows(legacy["rows"])
    assert result == legacy["summary"]
    return result["aggregate"]


def gate(selected, controls):
    return {"positive_mean_gain": selected["mean_gain"] > 0.,
        "mean_not_worse_than_controls": all(selected["mean_gain"] >= c["mean_gain"] for c in controls),
        "downside_not_worse_than_controls": all(selected["mean_negative_part"] >= c["mean_negative_part"] for c in controls)}


def summarize_rows(rows, legacy_aggregate, target=TARGET, controls=CONTROLS, extra_selectors=()):
    result = previous.summarize_rows(rows)
    checks, comparisons = {}, {}
    for batch, block in result["batches"].items():
        group = [r for r in rows if r["batch"] == batch]
        block["aggregate"].update(previous.metrics(group, selectors=(*extra_selectors, target)))
        for cell in block["cells"]:
            selected = [r for r in group if (r["mode"], r["weight_index"]) == (cell["mode"], cell["weight_index"])]
            cell["metrics"].update(previous.metrics(selected, selectors=(*extra_selectors, target)))
        a = block["aggregate"]
        checks[batch] = gate(a[target], [a[n] for n in controls])
        comparisons[batch] = {**{k: a[target][k] - a["combined96"][k] for k in ("mean_gain", "mean_negative_part")},
            "rmse_reduction": a["combined96"]["rmse512"] - a[target]["rmse512"],
            "rmse_minus_zero": a[target]["rmse512"] - a["no_change"]["rmse512"],
            "changed_choices": sum(r["choices"][target] != r["choices"]["combined96"] for r in group)}
    old = cv.summarize_rows([r for r in rows if r["batch"] == "old"], selectors=(target,), controls=(), target=target)
    result["old_primary_checks"] = gate(old["aggregate"][target], [legacy_aggregate[n] for n in expansion.CONTROLS])
    result["batch_checks"], result[f"{target}_minus_combined"] = checks, comparisons
    result["group_passes"] = all(result["old_primary_checks"].values()) and all(all(c.values()) for c in checks.values())
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
    legacy_aggregate = validate_legacy(legacy, cached, condition, seed)
    path = Path(output_root) / relative
    path.parent.mkdir(parents=True, exist_ok=True)
    report = {"protocol": PROTOCOL, "condition": condition, "seed": seed, "complete": False,
        "source_split": "train_cases", "fold_rule": "repeat % 4", "train_horizon": 512,
        "delta": DELTA, "penalty": PENALTY, "max_iterations": MAX_ITERATIONS, "gradient_tolerance": GRAD_TOL,
        "new_environment_interactions": 0, "policy_updates": 0, "critic_updates": 0, "mlp_updates": 0,
        "huber_fits": 0, "linear_solves": 0, "legacy_aggregate": legacy_aggregate,
        "legacy_controls": list(expansion.CONTROLS), "batch_controls": list(CONTROLS), "folds": [], "rows": [],
        "scope": "Fixed Huber loss on reused development labels; no sample removal, parameter search or fresh confirmation."}

    def save():
        path.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")

    save()
    for fold in range(4):
        fits, held = previous.split_batches(old, new, fold)
        held_cases = [(batch, c) for batch in previous.BATCHES for c in held[batch]]
        scores, metadata = fit_predict(fits["combined96"], torch.stack([cv.features(c) for _, c in held_cases]))
        for i, (batch, case) in enumerate(held_cases):
            original = lookup[reference.case_key(case)]
            assert (original["batch"], original["fold"], original["reset_seed"]) == (batch, fold, case["reset_seed"])
            row = join_row(original, scores[i])
            assert row["candidate_gains"] == p.realized_gains(case)
            report["rows"].append(row)
        report["folds"].append({"fold": fold, "fit_reset_ids": sorted({c["reset_seed"] for c in fits["combined96"]}),
            "held_reset_ids": {batch: sorted({c["reset_seed"] for c in cases}) for batch, cases in held.items()}, **metadata})
        report["huber_fits"] += 1
        report["linear_solves"] += metadata["linear_solves"]
        save()
        print(f"HUBER condition={condition} seed={seed} folds={fold + 1}/4 solves={report['linear_solves']} interactions=0", flush=True)
    report["summary"] = summarize_rows(report["rows"], legacy_aggregate)
    report["complete"] = True
    save()
    print(f"DONE huber condition={condition} seed={seed} passes={report['summary']['group_passes']}", flush=True)
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
