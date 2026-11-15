"""Decompose stored episode credit associations without new interactions."""

import argparse
import json
from pathlib import Path

import numpy as np

import run_episode_forecast_pilot as source


PROTOCOL = "stored_episode_credit_signal_v1"
ROOT = source.ROOT / "credit_signal_diagnostic"
OPERATORS = ("uniform", "stratified", "mixed")
RULES = ("ema", "equal_mix", "past_mix")


def forecast_scores(record):
    before = record["forecast_before"]
    if before["scale"] is None or before["trend_prediction"] is None:
        return {}
    return {rule: float(np.tanh((record["utility"] - (
        (1 - weight) * before["ema_prediction"] + weight * before["trend_prediction"])) / before["scale"]))
        for rule, weight in zip(RULES, (0., .5, before["trend_weight"]))}


def extract(allocation, method):
    rows = []
    for record in allocation["episode_history"]:
        if record["selection_phase"] != "free":
            continue
        scores = forecast_scores(record)
        if not scores:
            continue
        if record["credited"]:
            rule = "ema" if method == "fixed" else method
            if not np.isclose(record["score"], scores[rule], rtol=0, atol=1e-12):
                raise ValueError("diagnostic credit differs from logged training credit")
        rows.append(dict(start_step=record["start_step"], epoch=record["epoch_start"],
            operator=record["operator"], length=record["length"], utility=record["utility"],
            credited=record["credited"], crossing=record["mixed_epoch"], scores=scores))
    return rows


def fit(target, design):
    coef, _, rank, _ = np.linalg.lstsq(design, target, rcond=None)
    residual = target - design @ coef
    return coef, float(residual @ residual), int(rank)


def partial_r2(before, after, target):
    if before <= 1e-12 * max(1., float(target @ target)):
        return None
    return float(np.clip(1 - after / before, 0., 1.))


def centered_effects(coef):
    effects = np.asarray([0., *coef[-2:]])
    return dict(zip(OPERATORS, (effects - effects.mean()).tolist()))


def decompose(target, progress, operators, length):
    designs = (progress, np.column_stack((progress, operators)),
        np.column_stack((progress, length)), np.column_stack((progress, length, operators)))
    fitted = [fit(target, design) for design in designs]
    total = float(np.sum((target - target.mean()) ** 2))
    return dict(progress_reset_r2=partial_r2(total, fitted[0][1], target),
        length_partial_r2=partial_r2(fitted[0][1], fitted[2][1], target),
        operator_partial_r2=partial_r2(fitted[0][1], fitted[1][1], target),
        operator_partial_r2_after_length=partial_r2(fitted[2][1], fitted[3][1], target),
        length_partial_r2_after_operator=partial_r2(fitted[1][1], fitted[3][1], target),
        design_ranks=[item[2] for item in fitted],
        operator_contrasts=centered_effects(fitted[1][0]),
        length_adjusted_operator_contrasts=centered_effects(fitted[3][0]))


def correlation(a, b):
    a, b = a - a.mean(), b - b.mean()
    norm = float(np.linalg.norm(a) * np.linalg.norm(b))
    return float(a @ b / norm) if norm > 0 else None


def analyze(rows, rule):
    if not rows:
        return None
    step = np.asarray([r["start_step"] for r in rows]) / 75000.
    epoch = np.asarray([r["epoch"] for r in rows])
    progress = np.column_stack((np.ones(len(rows)), step,
        *[epoch == value for value in np.unique(epoch)[1:]]))
    operators = np.column_stack([[r["operator"] == op for r in rows] for op in OPERATORS[1:]])
    length = np.asarray([r["length"] for r in rows], dtype=float) / 1000.
    score = np.asarray([r["scores"][rule] for r in rows])
    utility = np.asarray([r["utility"] for r in rows])
    rate = utility / (length * 1000.)
    result = decompose(score, progress, operators, length)
    result.update(count=len(rows), crossing_count=sum(r["crossing"] for r in rows),
        score_length_correlation=correlation(score, length),
        return_length_correlation=correlation(utility, length),
        positive_score_fraction=float(np.mean(score > 0)),
        saturated_score_fraction=float(np.mean(np.abs(score) >= .95)),
        mean_length=float(length.mean() * 1000.),
        progress_adjusted_return_contrasts=centered_effects(
            fit(utility, np.column_stack((progress, operators)))[0]),
        progress_adjusted_return_rate_contrasts=centered_effects(
            fit(rate, np.column_stack((progress, operators)))[0]))
    return result


def aggregate(groups):
    groups = [group for group in groups if group is not None]
    if not groups:
        return None
    numeric = ("progress_reset_r2", "length_partial_r2", "operator_partial_r2",
        "operator_partial_r2_after_length", "length_partial_r2_after_operator",
        "score_length_correlation", "return_length_correlation", "positive_score_fraction",
        "saturated_score_fraction", "mean_length")
    mean = lambda values: float(np.mean(values)) if values else None
    result = dict(seed_count=len(groups), episode_count=sum(g["count"] for g in groups),
        crossing_count=sum(g["crossing_count"] for g in groups),
        equal_seed_means={key: mean([g[key] for g in groups if g[key] is not None]) for key in numeric},
        valid_seed_counts={key: sum(g[key] is not None for g in groups) for key in numeric})
    for key in ("operator_contrasts", "length_adjusted_operator_contrasts",
                "progress_adjusted_return_contrasts", "progress_adjusted_return_rate_contrasts"):
        result[key] = {op: mean([g[key][op] for g in groups]) for op in OPERATORS}
    return result


def run(source_root, root):
    frozen = json.loads((root / "protocol.json").read_text())
    readout = json.loads((source_root / "compact_readout.json").read_text())
    if (frozen["protocol"] != PROTOCOL or readout["protocol"] != source.PROTOCOL
            or readout["seeds"] != frozen["seeds"] or readout["seeds"] != list(source.SEEDS)
            or sorted(readout["environments"]) != sorted(frozen["environments"])
            or readout["decision"] != "hold_online_forecast_allocation"):
        raise ValueError("source differs from the frozen completed HOLD cohort")
    result = dict(protocol=PROTOCOL, source_protocol=readout["protocol"],
        seeds=readout["seeds"], source_decision=readout["decision"],
        decision="hold_online_forecast_allocation", additional_environment_interactions=0,
        additional_probe_interactions=0, optimizer_updates=0, v25_changed=False,
        statistical_regression_fits=0, source_runs_checked=0, environments={})
    for env in frozen["environments"]:
        runs = []
        for seed in frozen["seeds"]:
            for method in source.METHODS:
                record = json.loads((source_root / env.replace("-", "_") / method / f"seed{seed}.json").read_text())
                if (record["protocol"], record["env"], record["method"], record["seed"], record["complete"]) != (
                        source.PROTOCOL, env, method, seed, True):
                    raise ValueError("source run identity differs from diagnostic roster")
                rows = extract(record["allocation"], method)
                cohorts = {"credited_free": [r for r in rows if r["credited"]]}
                if method == "fixed":
                    cohorts["all_ready_free"] = rows
                rules = RULES if method == "fixed" else (method,)
                measured = {name: {rule: analyze(group, rule) for rule in rules}
                            for name, group in cohorts.items()}
                result["statistical_regression_fits"] += 6 * sum(
                    value is not None for group in measured.values() for value in group.values())
                runs.append(dict(seed=seed, method=method, reset_count=len(record["allocation"]["reset_history"]),
                    excluded_warmup_or_forced=sum(r["selection_phase"] != "free"
                        for r in record["allocation"]["episode_history"]),
                    final_partial_length=record["allocation"]["partial_episode_length"], cohorts=measured))
                result["source_runs_checked"] += 1
        primary = {name: {rule: aggregate([r["cohorts"][name][rule] for r in runs if r["method"] == "fixed"])
            for rule in RULES} for name in ("credited_free", "all_ready_free")}
        adaptive = {method: aggregate([r["cohorts"]["credited_free"][method]
            for r in runs if r["method"] == method]) for method in RULES}
        result["environments"][env] = dict(runs=runs, primary_fixed_route=primary,
            adaptive_route_associations=adaptive)
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, default=source.ROOT)
    parser.add_argument("--root", type=Path, default=ROOT)
    args = parser.parse_args()
    report = run(args.source_root, args.root)
    (args.root / "readout.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    print(json.dumps(dict(decision=report["decision"], source_runs_checked=report["source_runs_checked"],
        regression_fits=report["statistical_regression_fits"],
        fixed_route={env: v["primary_fixed_route"] for env, v in report["environments"].items()})))
