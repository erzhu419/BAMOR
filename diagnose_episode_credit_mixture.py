"""Diagnose paired forecast complementarity without a deployable mixture rule."""

import argparse
import json
from pathlib import Path

import numpy as np

import diagnose_episode_credit_selector as selector
from diagnose_episode_allocation import METHODS, ROOT as SOURCE_ROOT, diagnose_run
from episode_credit_trend import TREND_WINDOW


ROOT = Path("results/iclr_episode_credit_mixture_room_20261001")
PROTOCOL = "stored_episode_credit_convex_room_v1"
TOP_EPISODES = 5


def paired_moments(rows):
    ema = np.asarray([row["ema_error"] for row in rows])
    trend = np.asarray([row["trend_error"] for row in rows])
    difference = ema - trend
    energy = float(difference @ difference)
    weight = float(np.clip((ema @ difference) / energy, 0., 1.)) if energy > 0 else 0.
    mse = lambda values: float(np.mean(values ** 2))
    losses = dict(ema=mse(ema), trend=mse(trend), equal_mixture=mse((ema + trend) / 2),
                  hindsight_mixture=mse(ema - weight * difference))
    return dict(count=len(rows), ema_bias=float(ema.mean()), trend_bias=float(trend.mean()),
        error_covariance=float(np.mean((ema - ema.mean()) * (trend - trend.mean()))),
        error_cross_moment=float(np.mean(ema * trend)),
        error_difference_energy=energy / len(rows),
        opposite_sign_error_fraction=float(np.mean(ema * trend < 0)),
        hindsight_trend_weight=weight, mse=losses,
        rmse={name: float(np.sqrt(value)) for name, value in losses.items()},
        hindsight_mse_improvement_over_best_endpoint=min(losses["ema"], losses["trend"])
                                                     - losses["hindsight_mixture"])


def loss_budget(rows):
    excess = [row["selected_error"] ** 2 - row["ema_error"] ** 2 for row in rows]
    positive = sum(max(value, 0.) for value in excess)
    negative = sum(min(value, 0.) for value in excess)
    ranked = sorted(((value, row) for value, row in zip(excess, rows) if value > 0),
                    key=lambda pair: pair[0], reverse=True)[:TOP_EPISODES]
    keys = ("seed", "start_step", "end_step", "length", "epoch_start", "operator",
            "detected_mixed", "true_mixed", "true_modes", "selected", "utility",
            "ema_prediction", "trend_prediction", "ema_error", "trend_error",
            "past_comparisons", "past_squared_errors", "fit_samples")
    top = [dict({key: row[key] for key in keys}, excess_squared_error=value) for value, row in ranked]
    group = lambda field, state: dict(count=sum(row[field] == state for row in rows),
        net_excess_squared_error=sum(value for value, row in zip(excess, rows) if row[field] == state))
    return dict(count=len(rows), net_excess_squared_error=sum(excess),
        gross_positive_excess_squared_error=positive, helpful_squared_error_delta=negative,
        harmful_episodes=sum(value > 0 for value in excess), helpful_episodes=sum(value < 0 for value in excess),
        top_positive_excess_fraction=sum(value for value, _ in ranked) / positive if positive > 0 else None,
        by_detected_mixed={str(state): group("detected_mixed", state) for state in (False, True)},
        by_true_mixed={str(state): group("true_mixed", state) for state in (False, True)},
        top_episodes=top)


def run(source_root, root):
    frozen = json.loads((root / "protocol.json").read_text())
    source = json.loads((source_root / "compact_readout.json").read_text())
    previous = json.loads((selector.ROOT / "readout.json").read_text())
    if (frozen["protocol"] != PROTOCOL or frozen["trend_window"] != TREND_WINDOW
            or frozen["top_episodes"] != TOP_EPISODES
            or source["protocol"] != frozen["source_protocol"]
            or source["seeds"] != frozen["seeds"]
            or sorted(source["environments"]) != sorted(frozen["environments"])
            or previous["protocol"] != selector.PROTOCOL):
        raise ValueError("paired error diagnostic differs from frozen source protocol")
    result = dict(protocol=PROTOCOL, source_protocol=source["protocol"], seeds=source["seeds"],
        evidence_role="Post-hoc development diagnostic; hindsight weights are not operational",
        additional_environment_interactions=0, additional_probe_interactions=0, optimizer_updates=0,
        statistical_baseline_fits=0, v25_changed=False,
        decision="diagnostic_only_keep_hard_selector_hold", environments={})
    for env in frozen["environments"]:
        pooled, runs = [], []
        references = {record["seed"]: record for record in previous["environments"][env]["runs"]}
        for seed in source["seeds"]:
            path = source_root / env.replace("-", "_") / "fixed" / f"{METHODS['fixed']}_seed{seed}.json"
            data = json.loads(path.read_text())
            allocation = data["adaptive_operator_final"]
            if ((data["env"], data["method"], data["seed"]) != (env, METHODS["fixed"], seed)
                    or allocation["selection_uses_scores"]
                    or allocation["operators"] != list(selector.trend.OPERATORS)):
                raise ValueError("paired error diagnostic requires the original fixed random route")
            episodes, _ = diagnose_run(data)
            rows, counts = selector.rescore(episodes, allocation["baseline_rate"])
            free = [episode for episode in episodes if episode["phase"] == "free"]
            if len(free) != len(rows) or selector.metrics(rows) != references[seed]["metrics"]:
                raise ValueError("reference forecasts or complete free-episode cohort changed")
            for row, episode in zip(rows, free):
                row.update(seed=seed, **{key: episode[key] for key in (
                    "start_step", "end_step", "length", "epoch_start", "true_mixed", "true_modes")})
            pooled.extend(rows)
            result["statistical_baseline_fits"] += counts["statistical_baseline_fits"]
            runs.append(dict(seed=seed, moments=paired_moments(rows),
                loss_budget={key: value for key, value in loss_budget(rows).items() if key != "top_episodes"}))
        mean_rmse = {name: float(np.mean([record["moments"]["rmse"][name] for record in runs]))
                     for name in ("ema", "trend", "equal_mixture", "hindsight_mixture")}
        result["environments"][env] = dict(runs=runs, pooled_moments=paired_moments(pooled),
            mean_seed_rmse=mean_rmse, loss_budget=loss_budget(pooled),
            mean_seed_best_endpoint_rmse=float(np.mean([
                min(record["moments"]["rmse"]["ema"], record["moments"]["rmse"]["trend"])
                for record in runs])),
            seeds_with_interior_hindsight_weight=sum(0 < record["moments"]["hindsight_trend_weight"] < 1
                                                      for record in runs))
    result["source_runs_checked"] = sum(len(values["runs"]) for values in result["environments"].values())
    result["reference_forecasts_and_cohort_unchanged"] = True
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, default=SOURCE_ROOT)
    parser.add_argument("--root", type=Path, default=ROOT)
    args = parser.parse_args()
    result = run(args.source_root, args.root)
    (args.root / "readout.json").write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    print(json.dumps({env: dict(mean_seed_rmse=values["mean_seed_rmse"],
        pooled_hindsight_trend_weight=values["pooled_moments"]["hindsight_trend_weight"],
        seeds_with_interior_hindsight_weight=values["seeds_with_interior_hindsight_weight"],
        selector_net_excess_squared_error=values["loss_budget"]["net_excess_squared_error"],
        top_positive_excess_fraction=values["loss_budget"]["top_positive_excess_fraction"])
        for env, values in result["environments"].items()}, indent=2))
