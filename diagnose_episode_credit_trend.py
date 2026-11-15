"""Test causal return-trend forecasts on stored fixed-random episode routes."""

import argparse
import json
from pathlib import Path

import numpy as np

from diagnose_episode_allocation import METHODS, ROOT as SOURCE_ROOT, diagnose_run
from diagnose_episode_credit_warmup import validate_legacy_baseline
from episode_credit_trend import EpisodeCreditTrend, TREND_WINDOW


ROOT = Path("results/iclr_episode_credit_trend_20261001")
PROTOCOL = "stored_past_only_episode_trend_v1"
OPERATORS = ("uniform", "stratified", "mixed")
CONTROL_REPLICATES = 20
CONTROL_RNG = 1729
KNOWN_EFFECT = dict(zip(OPERATORS, (-1., 0., 1.)))


def rescore(episodes, rate):
    baseline = EpisodeCreditTrend(rate)
    rows, fits, free_count = [], 0, 0
    for episode in episodes:
        if episode["phase"] == "free":
            free_count += 1
            before = baseline.predict(episode["start_step"])
            fits += before["trend_prediction"] is not None
            if before["trend_prediction"] is not None and before["scale"] is not None:
                item = dict(operator=episode["operator"], utility=episode["utility"],
                    detected_mixed=episode["detected_mixed"], **before)
                for name in ("ema", "trend"):
                    error = episode["utility"] - before[f"{name}_prediction"]
                    item[f"{name}_error"] = error
                    item[f"{name}_score"] = float(np.tanh(error / before["scale"]))
                rows.append(item)
        if episode["detected_mixed"]:
            baseline.reset()
        else:
            baseline.observe(episode["start_step"], episode["utility"])
    return rows, dict(source_random_free_episodes=free_count,
        paired_forecast_episodes=len(rows), unavailable_forecast_episodes=free_count - len(rows),
        retained_reset_crossing_forecast_episodes=sum(row["detected_mixed"] for row in rows),
        statistical_baseline_fits=int(fits))


def metrics(rows, names=("ema", "trend")):
    result = dict(count=len(rows))
    for name in names:
        errors = np.asarray([row[f"{name}_error"] for row in rows])
        scores = np.asarray([row[f"{name}_score"] for row in rows])
        mean = lambda values: float(np.mean(values)) if len(values) else None
        result[name] = dict(rmse=float(np.sqrt(np.mean(errors ** 2))) if len(errors) else None,
            mae=mean(np.abs(errors)), mean_error=mean(errors), mean_score=mean(scores),
            positive_score_fraction=mean(scores > 0),
            centered_raw_contrasts={}, centered_score_contrasts={})
        for operator in OPERATORS:
            # p=1/3 is the recorded fixed route; these are diagnostic contrasts.
            weight = np.asarray([3 * (row["operator"] == operator) - 1 for row in rows])
            result[name]["centered_raw_contrasts"][operator] = mean(weight * errors)
            result[name]["centered_score_contrasts"][operator] = mean(weight * scores)
    return result


def diagnose(data):
    allocation = data["adaptive_operator_final"]
    if allocation["selection_uses_scores"] or allocation["operators"] != list(OPERATORS):
        raise ValueError("trend diagnostic requires the original fixed three-arm random route")
    validate_legacy_baseline(allocation["episode_history"], allocation["baseline_rate"],
                             allocation["learning_starts"])
    episodes, _ = diagnose_run(data)
    rows, counts = rescore(episodes, allocation["baseline_rate"])
    return episodes, rows, counts


def constructed_control(episodes, seed, replicate, effects, noise):
    rng = np.random.default_rng(CONTROL_RNG + seed * 100 + replicate)
    errors = rng.normal(0., noise, len(episodes))
    return [dict(episode, utility=float(3 + episode["start_step"] / 1000
                 + effects[episode["operator"]] + error))
            for episode, error in zip(episodes, errors)]


def controls(templates):
    zero_effect = dict.fromkeys(OPERATORS, 0.)
    deterministic_rows, all_fits = [], 0
    for seed, episodes, rate in templates:
        rows, counts = rescore(constructed_control(episodes, seed, 0, zero_effect, 0.), rate)
        deterministic_rows.extend(rows)
        all_fits += counts["statistical_baseline_fits"]
    maximum_error = max(abs(row["trend_error"]) for row in deterministic_rows)
    result = dict(deterministic_growth_null=metrics(deterministic_rows),
        deterministic_growth_null_maximum_trend_error=maximum_error,
        deterministic_growth_null_passed=maximum_error <= 1e-9)
    for name, effects in (("noisy_growth_null", zero_effect), ("known_arm_effect", KNOWN_EFFECT)):
        pooled, replicates = [], []
        for replicate in range(CONTROL_REPLICATES):
            group = []
            for seed, episodes, rate in templates:
                rows, counts = rescore(constructed_control(episodes, seed, replicate, effects, 1.), rate)
                group.extend(rows)
                all_fits += counts["statistical_baseline_fits"]
            pooled.extend(group)
            replicates.append(metrics(group)["trend"]["centered_raw_contrasts"])
        result[name] = dict(metrics=metrics(pooled), expected_centered_raw_contrasts=effects,
            replicate_trend_raw_contrasts=replicates)
    contrasts = result["known_arm_effect"]["metrics"]["trend"]["centered_raw_contrasts"]
    result["known_effect_order_recovered"] = contrasts["uniform"] < contrasts["stratified"] < contrasts["mixed"]
    result["statistical_baseline_fits"] = all_fits
    return result


def run(source_root, root):
    frozen = json.loads((root / "protocol.json").read_text())
    source = json.loads((source_root / "compact_readout.json").read_text())
    if (frozen["protocol"] != PROTOCOL or frozen["trend_window"] != TREND_WINDOW
            or frozen["control_replicates"] != CONTROL_REPLICATES
            or frozen["control_rng"] != CONTROL_RNG
            or source["protocol"] != frozen["source_protocol"]
            or source["seeds"] != frozen["seeds"]
            or sorted(source["environments"]) != sorted(frozen["environments"])):
        raise ValueError("frozen trend protocol differs from source or implementation")
    result = dict(protocol=PROTOCOL, source_protocol=source["protocol"], seeds=source["seeds"],
        trend_window=TREND_WINDOW, additional_environment_interactions=0,
        additional_probe_interactions=0, optimizer_updates=0, v25_changed=False,
        source_method="fixed", statistical_baseline_fits=dict(recorded=0, constructed_controls=0),
        environments={})
    for env in frozen["environments"]:
        rows, runs, templates = [], [], []
        for seed in source["seeds"]:
            path = source_root / env.replace("-", "_") / "fixed" / f"{METHODS['fixed']}_seed{seed}.json"
            data = json.loads(path.read_text())
            if (data["env"], data["method"], data["seed"]) != (env, METHODS["fixed"], seed):
                raise ValueError("source identity differs from fixed-route readout")
            episodes, group, counts = diagnose(data)
            rows.extend(group)
            templates.append((seed, episodes, data["adaptive_operator_final"]["baseline_rate"]))
            runs.append(dict(seed=seed, counts=counts, metrics=metrics(group)))
            result["statistical_baseline_fits"]["recorded"] += counts["statistical_baseline_fits"]
        synthetic = controls(templates)
        result["statistical_baseline_fits"]["constructed_controls"] += synthetic["statistical_baseline_fits"]
        deltas = [record["metrics"]["trend"]["rmse"] - record["metrics"]["ema"]["rmse"] for record in runs]
        result["environments"][env] = dict(runs=runs, pooled=metrics(rows), controls=synthetic,
            paired_seed_rmse_deltas=deltas, mean_paired_seed_rmse_delta=float(np.mean(deltas)),
            seed_rmse_improvements=sum(delta < 0 for delta in deltas),
            mean_seed_rmse={name: float(np.mean([record["metrics"][name]["rmse"] for record in runs]))
                            for name in ("ema", "trend")})
    result["source_runs_checked"] = sum(len(values["runs"]) for values in result["environments"].values())
    result["source_baseline_round_trip_passed"] = True
    result["controls_passed"] = all(values["controls"]["deterministic_growth_null_passed"]
        and values["controls"]["known_effect_order_recovered"] for values in result["environments"].values())
    result["forecast_comparison_passed"] = all(values["mean_paired_seed_rmse_delta"] < 0
                                             for values in result["environments"].values())
    result["decision"] = ("forecast_supported_online_benefit_unproven"
        if result["controls_passed"] and result["forecast_comparison_passed"] else "hold_trend_baseline")
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, default=SOURCE_ROOT)
    parser.add_argument("--root", type=Path, default=ROOT)
    args = parser.parse_args()
    result = run(args.source_root, args.root)
    (args.root / "readout.json").write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    print(json.dumps(dict(decision=result["decision"],
        source_runs_checked=result["source_runs_checked"],
        environments={env: dict(mean_seed_rmse=values["mean_seed_rmse"],
            seed_rmse_improvements=values["seed_rmse_improvements"],
            mean_paired_seed_rmse_delta=values["mean_paired_seed_rmse_delta"],
            deterministic_growth_null_passed=values["controls"]["deterministic_growth_null_passed"],
            known_effect_order_recovered=values["controls"]["known_effect_order_recovered"])
            for env, values in result["environments"].items()}), indent=2))
