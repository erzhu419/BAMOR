"""Replay a past-only convex forecast combination on fixed recorded routes."""

import argparse
import json
from pathlib import Path

import numpy as np

import diagnose_episode_credit_selector as selector
from diagnose_episode_allocation import METHODS, ROOT as SOURCE_ROOT
from episode_credit_blend import EpisodeCreditBlend
from episode_credit_trend import TREND_WINDOW


ROOT = Path("results/iclr_episode_credit_blend_20261001")
PROTOCOL = "stored_past_error_episode_credit_blend_v1"
NAMES = ("ema", "trend", "equal_mixture", "blend")
REFERENCE_FIELDS = ("operator", "utility", "detected_mixed", "ema_prediction", "trend_prediction",
                    "scale", "fit_samples", "ema_error", "trend_error", "ema_score", "trend_score",
                    "past_comparisons")


def rescore(episodes, rate):
    baseline = EpisodeCreditBlend(rate)
    rows, fits, free_count = [], 0, 0
    for episode in episodes:
        before = baseline.predict(episode["start_step"])
        fits += before["trend_prediction"] is not None
        if episode["phase"] == "free":
            free_count += 1
            if before["trend_prediction"] is not None and before["scale"] is not None:
                item = dict(operator=episode["operator"], utility=episode["utility"],
                    start_step=episode["start_step"], detected_mixed=episode["detected_mixed"], **before)
                item["equal_mixture_prediction"] = (before["ema_prediction"] + before["trend_prediction"]) / 2.
                for name in NAMES:
                    error = episode["utility"] - item[f"{name}_prediction"]
                    item[f"{name}_error"] = error
                    item[f"{name}_score"] = float(np.tanh(error / before["scale"]))
                rows.append(item)
        baseline.observe(episode["start_step"], episode["utility"], before, episode["detected_mixed"])
    weights = np.asarray([row["trend_weight"] for row in rows])
    return rows, dict(source_random_free_episodes=free_count, paired_forecast_episodes=len(rows),
        unavailable_forecast_episodes=free_count - len(rows),
        retained_reset_crossing_forecast_episodes=sum(row["detected_mixed"] for row in rows),
        statistical_baseline_fits=int(fits),
        trend_weight=dict(mean=float(weights.mean()) if len(weights) else None,
            minimum=float(weights.min()) if len(weights) else None,
            maximum=float(weights.max()) if len(weights) else None,
            zero_count=int(np.sum(weights == 0.)), one_count=int(np.sum(weights == 1.))))


def metrics(rows):
    return selector.trend.metrics(rows, names=NAMES)


def validate_reference(rows, reference_rows, reference_metrics):
    if (len(rows) != len(reference_rows) or selector.metrics(reference_rows) != reference_metrics
            or any(row[key] != reference[key] for row, reference in zip(rows, reference_rows)
                   for key in REFERENCE_FIELDS)):
        raise ValueError("reference forecasts or evaluation cohort changed")


def controls(templates):
    zero = dict.fromkeys(selector.trend.OPERATORS, 0.)
    deterministic, fits = [], 0
    for seed, episodes, rate in templates:
        rows, counts = rescore(selector.trend.constructed_control(episodes, seed, 0, zero, 0.), rate)
        deterministic.extend(rows)
        fits += counts["statistical_baseline_fits"]
    maximum_error = max(abs(row["blend_error"]) for row in deterministic)
    result = dict(deterministic_growth_null=metrics(deterministic),
        deterministic_blend_maximum_error=maximum_error,
        deterministic_growth_null_passed=maximum_error <= 1e-9)
    for name, effects in (("noisy_growth_null", zero), ("known_arm_effect", selector.trend.KNOWN_EFFECT)):
        pooled = []
        for replicate in range(selector.trend.CONTROL_REPLICATES):
            for seed, episodes, rate in templates:
                rows, counts = rescore(selector.trend.constructed_control(episodes, seed, replicate, effects, 1.), rate)
                pooled.extend(rows)
                fits += counts["statistical_baseline_fits"]
        result[name] = dict(metrics=metrics(pooled), expected_centered_raw_contrasts=effects)
    contrasts = result["known_arm_effect"]["metrics"]["blend"]["centered_raw_contrasts"]
    result["known_effect_order_recovered"] = contrasts["uniform"] < contrasts["stratified"] < contrasts["mixed"]
    result["statistical_baseline_fits"] = fits
    return result


def run(source_root, root):
    frozen = json.loads((root / "protocol.json").read_text())
    source = json.loads((source_root / "compact_readout.json").read_text())
    previous = json.loads((selector.ROOT / "readout.json").read_text())
    if (frozen["protocol"] != PROTOCOL or frozen["trend_window"] != TREND_WINDOW
            or frozen["control_replicates"] != selector.trend.CONTROL_REPLICATES
            or frozen["control_rng"] != selector.trend.CONTROL_RNG
            or source["protocol"] != frozen["source_protocol"]
            or source["seeds"] != frozen["seeds"]
            or sorted(source["environments"]) != sorted(frozen["environments"])
            or previous["protocol"] != selector.PROTOCOL):
        raise ValueError("frozen blend protocol differs from source or implementation")
    result = dict(protocol=PROTOCOL, source_protocol=source["protocol"], seeds=source["seeds"],
        trend_window=TREND_WINDOW, evidence_role="Reused-log development diagnostic",
        additional_environment_interactions=0, additional_probe_interactions=0,
        optimizer_updates=0, v25_changed=False, source_method="fixed",
        statistical_baseline_fits=dict(recorded_blend=0, reference_replay=0, constructed_controls=0),
        environments={}, prior_failure_cases=[])
    for env in frozen["environments"]:
        rows, runs, templates, reference_pooled = [], [], [], []
        previous_runs = {record["seed"]: record for record in previous["environments"][env]["runs"]}
        for seed in source["seeds"]:
            path = source_root / env.replace("-", "_") / "fixed" / f"{METHODS['fixed']}_seed{seed}.json"
            data = json.loads(path.read_text())
            allocation = data["adaptive_operator_final"]
            if ((data["env"], data["method"], data["seed"]) != (env, METHODS["fixed"], seed)
                    or allocation["selection_uses_scores"]
                    or allocation["operators"] != list(selector.trend.OPERATORS)):
                raise ValueError("blend diagnostic requires the original fixed random route")
            episodes, _ = selector.trend.diagnose_run(data)
            group, counts = rescore(episodes, allocation["baseline_rate"])
            reference, reference_counts = selector.rescore(episodes, allocation["baseline_rate"])
            validate_reference(group, reference, previous_runs[seed]["metrics"])
            measured = metrics(group)
            measured["hard_selector"] = selector.metrics(reference)["selected"]
            rows.extend(group)
            reference_pooled.extend(reference)
            templates.append((seed, episodes, allocation["baseline_rate"]))
            runs.append(dict(seed=seed, counts=counts, metrics=measured))
            result["statistical_baseline_fits"]["recorded_blend"] += counts["statistical_baseline_fits"]
            result["statistical_baseline_fits"]["reference_replay"] += reference_counts["statistical_baseline_fits"]
            starts = [case["start_step"] for case in frozen["prior_failure_cases"]
                      if case["env"] == env and case["seed"] == seed]
            for row, old in zip(group, reference):
                if row["start_step"] in starts:
                    result["prior_failure_cases"].append(dict(env=env, seed=seed, **row,
                        hard_selector_prediction=old["selected_prediction"], hard_selector_error=old["selected_error"]))
        synthetic = controls(templates)
        result["statistical_baseline_fits"]["constructed_controls"] += synthetic["statistical_baseline_fits"]
        deltas = [record["metrics"]["blend"]["rmse"] - record["metrics"]["ema"]["rmse"] for record in runs]
        pooled = metrics(rows)
        pooled["hard_selector"] = selector.metrics(reference_pooled)["selected"]
        result["environments"][env] = dict(runs=runs, pooled=pooled, controls=synthetic,
            paired_seed_rmse_deltas=deltas, mean_paired_seed_rmse_delta=float(np.mean(deltas)),
            seed_rmse_improvements=sum(delta < 0 for delta in deltas),
            mean_seed_rmse={name: float(np.mean([record["metrics"][name]["rmse"] for record in runs]))
                            for name in (*NAMES, "hard_selector")})
    result["source_runs_checked"] = sum(len(values["runs"]) for values in result["environments"].values())
    result["reference_forecasts_and_cohort_unchanged"] = True
    result["controls_passed"] = all(values["controls"]["deterministic_growth_null_passed"]
        and values["controls"]["known_effect_order_recovered"] for values in result["environments"].values())
    result["forecast_comparison_passed"] = (
        result["environments"]["mo-halfcheetah-v5"]["mean_paired_seed_rmse_delta"] < 0
        and result["environments"]["mo-hopper-v5"]["mean_paired_seed_rmse_delta"] <= 0)
    result["decision"] = ("development_blend_supported_fresh_online_test_unproven"
        if result["controls_passed"] and result["forecast_comparison_passed"] else "hold_past_error_blend")
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, default=SOURCE_ROOT)
    parser.add_argument("--root", type=Path, default=ROOT)
    args = parser.parse_args()
    result = run(args.source_root, args.root)
    (args.root / "readout.json").write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    print(json.dumps(dict(decision=result["decision"], controls_passed=result["controls_passed"],
        reference_forecasts_and_cohort_unchanged=result["reference_forecasts_and_cohort_unchanged"],
        environments={env: dict(mean_seed_rmse=values["mean_seed_rmse"],
            seed_rmse_improvements=values["seed_rmse_improvements"],
            mean_paired_seed_rmse_delta=values["mean_paired_seed_rmse_delta"])
            for env, values in result["environments"].items()}), indent=2))
