"""Replay a past-error-only EMA/trend selection rule on fixed recorded routes."""

import argparse
from collections import Counter
import json
from pathlib import Path

import numpy as np

import diagnose_episode_credit_trend as trend
from diagnose_episode_allocation import METHODS, ROOT as SOURCE_ROOT
from episode_credit_selector import EpisodeCreditSelector
from episode_credit_trend import TREND_WINDOW


ROOT = Path("results/iclr_episode_credit_selector_20261001")
PROTOCOL = "stored_past_error_episode_credit_selector_v1"
NAMES = ("ema", "trend", "selected")


def rescore(episodes, rate):
    selector = EpisodeCreditSelector(rate)
    rows, fits, free_count, switches = [], 0, 0, 0
    previous = None
    for episode in episodes:
        before = selector.predict(episode["start_step"])
        fits += before["trend_prediction"] is not None
        if episode["phase"] == "free":
            free_count += 1
            if before["trend_prediction"] is not None and before["scale"] is not None:
                item = dict(operator=episode["operator"], utility=episode["utility"],
                    detected_mixed=episode["detected_mixed"], **before)
                for name in NAMES:
                    error = episode["utility"] - before[f"{name}_prediction"]
                    item[f"{name}_error"] = error
                    item[f"{name}_score"] = float(np.tanh(error / before["scale"]))
                rows.append(item)
                switches += previous is not None and previous != before["selected"]
                previous = before["selected"]
        selector.observe(episode["start_step"], episode["utility"], before, episode["detected_mixed"])
        if episode["detected_mixed"]:
            previous = None
    return rows, dict(source_random_free_episodes=free_count, paired_forecast_episodes=len(rows),
        unavailable_forecast_episodes=free_count - len(rows),
        retained_reset_crossing_forecast_episodes=sum(row["detected_mixed"] for row in rows),
        selected_models=dict(Counter(row["selected"] for row in rows)),
        within_epoch_free_forecast_switches=int(switches), statistical_baseline_fits=int(fits))


def metrics(rows):
    result = trend.metrics(rows, names=NAMES)
    losses = {name: sum(row[f"{name}_error"] ** 2 for row in rows) for name in NAMES}
    best_fixed = "trend" if losses["trend"] < losses["ema"] else "ema"
    result["hindsight_best_fixed"] = best_fixed
    result["selected_squared_loss_minus_best_fixed"] = losses["selected"] - losses[best_fixed]
    return result


def controls(templates):
    zero = dict.fromkeys(trend.OPERATORS, 0.)
    deterministic, fits = [], 0
    for seed, episodes, rate in templates:
        rows, counts = rescore(trend.constructed_control(episodes, seed, 0, zero, 0.), rate)
        deterministic.extend(rows)
        fits += counts["statistical_baseline_fits"]
    maximum_error = max(abs(row["selected_error"]) for row in deterministic)
    result = dict(deterministic_growth_null=metrics(deterministic),
        deterministic_selected_maximum_error=maximum_error,
        deterministic_growth_null_passed=maximum_error <= 1e-9)
    for name, effects in (("noisy_growth_null", zero), ("known_arm_effect", trend.KNOWN_EFFECT)):
        pooled = []
        for replicate in range(trend.CONTROL_REPLICATES):
            for seed, episodes, rate in templates:
                rows, counts = rescore(trend.constructed_control(episodes, seed, replicate, effects, 1.), rate)
                pooled.extend(rows)
                fits += counts["statistical_baseline_fits"]
        result[name] = dict(metrics=metrics(pooled), expected_centered_raw_contrasts=effects)
    contrasts = result["known_arm_effect"]["metrics"]["selected"]["centered_raw_contrasts"]
    result["known_effect_order_recovered"] = contrasts["uniform"] < contrasts["stratified"] < contrasts["mixed"]
    result["statistical_baseline_fits"] = fits
    return result


def run(source_root, root):
    frozen = json.loads((root / "protocol.json").read_text())
    source = json.loads((source_root / "compact_readout.json").read_text())
    previous = json.loads((trend.ROOT / "readout.json").read_text())
    if (frozen["protocol"] != PROTOCOL or frozen["trend_window"] != TREND_WINDOW
            or frozen["control_replicates"] != trend.CONTROL_REPLICATES
            or frozen["control_rng"] != trend.CONTROL_RNG
            or source["protocol"] != frozen["source_protocol"]
            or source["seeds"] != frozen["seeds"]
            or sorted(source["environments"]) != sorted(frozen["environments"])
            or previous["protocol"] != trend.PROTOCOL):
        raise ValueError("frozen selector protocol differs from source or implementation")
    result = dict(protocol=PROTOCOL, source_protocol=source["protocol"], seeds=source["seeds"],
        trend_window=TREND_WINDOW, evidence_role="Reused-log development diagnostic",
        additional_environment_interactions=0, additional_probe_interactions=0,
        optimizer_updates=0, v25_changed=False, source_method="fixed",
        statistical_baseline_fits=dict(recorded=0, constructed_controls=0), environments={})
    for env in frozen["environments"]:
        rows, runs, templates = [], [], []
        previous_runs = {record["seed"]: record for record in previous["environments"][env]["runs"]}
        for seed in source["seeds"]:
            path = source_root / env.replace("-", "_") / "fixed" / f"{METHODS['fixed']}_seed{seed}.json"
            data = json.loads(path.read_text())
            if (data["env"], data["method"], data["seed"]) != (env, METHODS["fixed"], seed):
                raise ValueError("source identity differs from fixed-route readout")
            allocation = data["adaptive_operator_final"]
            if allocation["selection_uses_scores"] or allocation["operators"] != list(trend.OPERATORS):
                raise ValueError("selector diagnostic requires the original fixed random route")
            trend.validate_legacy_baseline(allocation["episode_history"], allocation["baseline_rate"],
                                           allocation["learning_starts"])
            episodes, _ = trend.diagnose_run(data)
            group, counts = rescore(episodes, allocation["baseline_rate"])
            measured = metrics(group)
            original = previous_runs[seed]["metrics"]
            if (measured["count"] != original["count"] or
                    any(measured[name] != original[name] for name in ("ema", "trend"))):
                raise ValueError("reference forecasts or evaluation cohort changed")
            rows.extend(group)
            templates.append((seed, episodes, allocation["baseline_rate"]))
            runs.append(dict(seed=seed, counts=counts, metrics=measured))
            result["statistical_baseline_fits"]["recorded"] += counts["statistical_baseline_fits"]
        synthetic = controls(templates)
        result["statistical_baseline_fits"]["constructed_controls"] += synthetic["statistical_baseline_fits"]
        deltas = [record["metrics"]["selected"]["rmse"] - record["metrics"]["ema"]["rmse"] for record in runs]
        result["environments"][env] = dict(runs=runs, pooled=metrics(rows), controls=synthetic,
            paired_seed_rmse_deltas=deltas, mean_paired_seed_rmse_delta=float(np.mean(deltas)),
            seed_rmse_improvements=sum(delta < 0 for delta in deltas),
            mean_seed_rmse={name: float(np.mean([record["metrics"][name]["rmse"] for record in runs]))
                            for name in NAMES})
    result["source_runs_checked"] = sum(len(values["runs"]) for values in result["environments"].values())
    result["source_baseline_round_trip_passed"] = True
    result["reference_forecasts_and_cohort_unchanged"] = True
    result["controls_passed"] = all(values["controls"]["deterministic_growth_null_passed"]
        and values["controls"]["known_effect_order_recovered"] for values in result["environments"].values())
    result["forecast_comparison_passed"] = (
        result["environments"]["mo-halfcheetah-v5"]["mean_paired_seed_rmse_delta"] < 0
        and result["environments"]["mo-hopper-v5"]["mean_paired_seed_rmse_delta"] <= 0)
    result["decision"] = ("development_forecast_supported_fresh_online_test_unproven"
        if result["controls_passed"] and result["forecast_comparison_passed"] else "hold_past_error_selector")
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, default=SOURCE_ROOT)
    parser.add_argument("--root", type=Path, default=ROOT)
    args = parser.parse_args()
    result = run(args.source_root, args.root)
    (args.root / "readout.json").write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    print(json.dumps(dict(decision=result["decision"], source_runs_checked=result["source_runs_checked"],
        environments={env: dict(mean_seed_rmse=values["mean_seed_rmse"],
            seed_rmse_improvements=values["seed_rmse_improvements"],
            mean_paired_seed_rmse_delta=values["mean_paired_seed_rmse_delta"])
            for env, values in result["environments"].items()}), indent=2))
