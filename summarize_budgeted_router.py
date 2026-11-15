"""Summarize the frozen routing pilot on the server without exporting raw fronts."""

import argparse
import json
from pathlib import Path
from statistics import mean

import numpy as np

import run_budgeted_return_router as pilot
from run_impulse_return_router import PROTOCOL as IMPULSE_PROTOCOL
from summarize_matched_pilot import seed_diagnostics


SEEDS = (26, 27, 28)
PERSISTENT_PROTOCOL = "budgeted_raw_return_router_v1"


def summarize_run(data, condition, seed, protocol=PERSISTENT_PROTOCOL):
    assert protocol in (PERSISTENT_PROTOCOL, IMPULSE_PROTOCOL)
    impulse = protocol == IMPULSE_PROTOCOL
    assert condition in (pilot.CONDITIONS[:3] if impulse else pilot.CONDITIONS)
    trials = condition in pilot.CONDITIONS[:3]
    steps = pilot.MAIN_STEPS if trials else pilot.TOTAL_BUDGET
    archives = pilot.ARCHIVE_MAIN_STEPS if trials else pilot.ARCHIVE_BUDGET_STEPS
    budget = data["budgeted_routing_experiment"]
    expected = {
        "protocol": protocol, "condition": condition, "complete": True,
        "main_environment_steps": steps,
        "trial_environment_steps": 4 * pilot.TRIAL_STEPS if trials else 0,
        "total_training_interactions": pilot.TOTAL_BUDGET,
        "normal_updates": steps - 5000 + 1,
        "fork_updates": 8 * pilot.UPDATES if trials else 0,
        "committed_fork_updates": 4 * pilot.UPDATES if trials else 0,
        "archive_interaction_steps": list(pilot.ARCHIVE_BUDGET_STEPS),
    }
    if impulse:
        expected.update(selection_scope="fork_updates_only", normal_training_gate="default_portfolio")
    for key, value in expected.items():
        assert budget[key] == value, f"Mismatched budget: {key}"
    assert (data["env"], data["seed"], data["total_steps"], data["switch_interval"]) == (
        "mo-halfcheetah-v5", seed, steps, 25000)
    assert data["evaluation_protocol"] == "locked_full_simplex_v1"
    assert data["archive_context"] == "final_encoder"
    weights = pilot.training.full_simplex_grid(2).tolist()
    assert data["evaluation_weights"] == weights
    assert data["episode_seeds_per_mode"] == [
        [100000 + seed * 1000 + mode * 10 + r for r in range(3)] for mode in range(4)]
    assert data["ref_point"] == [-50., -3000.]
    snapshots = data["policy_archive_snapshots"]
    assert [s["step"] for s in snapshots] == list(archives)
    assert [s["interaction_step"] for s in snapshots] == list(pilot.ARCHIVE_BUDGET_STEPS)

    # Recompute within each mode; final actor is not the union of saved actors.
    hv, final_hv, curves = [], [], []
    assert len(data["pareto_per_mode"]) == 4
    for front in data["pareto_per_mode"]:
        assert len(front) == len(archives) * len(weights)
        curve = []
        for index, snap in enumerate(snapshots):
            rows = [p for p in front if p["archive_index"] == index]
            assert [p["weight"] for p in rows] == weights, "Incomplete snapshot grid"
            assert all(p["archive_step"] == snap["step"] and
                       p["archive_interaction_step"] == snap["interaction_step"] for p in rows)
            returns = np.asarray([p["return"] for p in rows])
            assert returns.shape == (21, 2) and np.isfinite(returns).all()
            curve.append({"main_step": snap["step"], "interaction_step": snap["interaction_step"],
                          "hv": pilot.training.compute_hv_nd(returns, data["ref_point"]),
                          "max_speed_return": float(returns[:, 0].max())})
        curves.append(curve)
        hv.append(pilot.training.compute_hv_nd([p["return"] for p in front], data["ref_point"]))
        final_hv.append(curve[-1]["hv"])
    np.testing.assert_allclose(hv, data["hv_per_mode"], rtol=1e-10, atol=1e-6)
    np.testing.assert_allclose(final_hv, data["final_hv_per_mode"], rtol=1e-10, atol=1e-6)
    np.testing.assert_allclose([mean(hv), mean(final_hv)],
                               [data["hv_mean_mode"], data["final_hv_mean_mode"]])
    result = seed_diagnostics(data)
    history = data["ground_truth"]["mode_history"]
    result.update({"budget": expected, "mode_history": history,
                   "training_seconds": data["training_seconds"],
                   "evaluation_seconds": data["evaluation_seconds"],
                   "snapshot_mode_curves": curves,
                   "old_modes": sorted({m for _, m in history} - {history[-1][1]}),
                   "mode_final_to_best_snapshot": [
                       c[-1]["hv"] / max(p["hv"] for p in c) if max(p["hv"] for p in c) else None
                       for c in curves]})
    decisions = []
    if trials:
        routing = data["update_routing"]
        if impulse:
            assert routing["protocol"] == protocol
            assert routing["selection_scope"] == "fork_updates_only"
            assert routing["normal_training_gate"] == "default_portfolio"
            assert "enabled" not in routing
        assert routing["condition"] == condition
        assert routing["decision_steps"] == list(pilot.DECISION_STEPS)
        assert routing["horizons"] == list(pilot.HORIZONS)
        assert (routing["repeats"], routing["updates_per_branch"]) == (pilot.REPEATS, pilot.UPDATES)
        assert routing["extra_environment_steps"] == expected["trial_environment_steps"]
        assert [c["main_step"] for c in routing["cases"]] == list(pilot.DECISION_STEPS)
        enabled = condition != "forced_off"
        for index, case in enumerate(routing["cases"]):
            step = case["main_step"]
            if impulse:
                enabled = True
            assert case["previous_on"] == enabled
            assert case["interaction_step"] == step + (index + 1) * pilot.TRIAL_STEPS
            assert case["extra_environment_steps"] == pilot.TRIAL_STEPS
            assert (case["fork_updates"], case["committed_updates"]) == (2 * pilot.UPDATES, pilot.UPDATES)
            assert case["score_reset_seeds"] == [
                30000000 + seed * 100000 + step * 100 + 20000 + r for r in range(3)]
            assert all(np.asarray(v).shape == (3, 3, 3, 2) for v in case["score_returns"].values())
            scores = pilot.raw_scores(case["score_returns"])
            advantages = np.asarray(scores["mean_scalar_on_minus_off"])
            np.testing.assert_allclose(advantages[:, -1], case["preference_advantages"])
            np.testing.assert_allclose(advantages[:, -1].mean(), case["mean_utility_advantage"])
            active = case["on_candidates"] > 1
            if not active:
                np.testing.assert_array_equal(case["score_returns"]["on"], case["score_returns"]["off"])
            enabled = pilot.choose_branch(advantages[:, -1], condition, enabled, active)
            assert case["chosen"] == ("on" if enabled else "off")
            until = pilot.DECISION_STEPS[index + 1] if index < 3 else steps
            if impulse:
                until = step
            decisions.append({k: case[k] for k in (
                "main_step", "interaction_step", "operator", "on_candidates", "previous_on",
                "chosen", "preference_advantages", "mean_utility_advantage")})
            decisions[-1].update({"mean_advantage_by_horizon": advantages.mean(0).tolist(),
                                  "held_until_main_step": until,
                                  "mode_switches_while_held": [h for h in history if step < h[0] <= until]})
        if not impulse:
            assert routing["enabled"] == enabled
    else:
        assert "update_routing" not in data
    result["decisions"] = decisions
    return result


def paired_comparison(route, controls, label):
    seeds = sorted(route.keys() & controls.keys())
    pair = {"seeds": seeds, "per_seed": {}}
    for seed in seeds:
        a, b = route[seed], controls[seed]
        assert a["mode_history"] == [h for h in b["mode_history"] if h[0] < pilot.MAIN_STEPS], (
            f"Unmatched main schedule prefix: {label}, seed {seed}")
        delta = {metric: a[metric] - b[metric] for metric in ("archive_hv", "final_hv")}
        delta["final_mode_hv"] = [x - y for x, y in zip(a["final_mode_hv"], b["final_mode_hv"])]
        # Keep route's exposure groups when the control later sees additional modes.
        groups = {"route_seen": [m for m, n in enumerate(a["training_steps_per_mode"]) if n],
                  "route_unseen": [m for m, n in enumerate(a["training_steps_per_mode"]) if not n],
                  "route_old": a["old_modes"]}
        delta["final_mode_groups"] = {g: mean(delta["final_mode_hv"][m] for m in modes)
                                     if modes else None for g, modes in groups.items()}
        pair["per_seed"][seed] = delta
    if seeds:
        for metric in ("archive_hv", "final_hv"):
            ds = [pair["per_seed"][s][metric] for s in seeds]
            pair[metric] = {"mean_difference": mean(ds), "wins": sum(d > 0 for d in ds),
                            "relative_mean_difference": mean(ds) / mean(controls[s][metric] for s in seeds)}
    return pair


def summarize(root, protocol=PERSISTENT_PROTOCOL):
    assert protocol in (PERSISTENT_PROTOCOL, IMPULSE_PROTOCOL)
    conditions = pilot.CONDITIONS[:3] if protocol == IMPULSE_PROTOCOL else pilot.CONDITIONS
    output = {"protocol": protocol, "expected": len(conditions) * len(SEEDS), "completed": 0,
              "missing": [], "conditions": {}, "paired_route_minus": {},
              "scope": "Development seeds 26-28. Final-encoder snapshot diagnostics are not "
                       "online learning curves. Forced controls isolate feedback; train-only "
                       "controls match interactions but have longer mode exposure and more updates. "
                       "Routing scores are selection data, not independent outcome labels."}
    for condition in conditions:
        runs = {}
        method = ("bamor_v27_aligned_target_ownpref_actor_fixed_archive_mix"
                  if condition == "train_off" else pilot.forks.METHOD)
        for seed in SEEDS:
            path = root / condition / f"{method}_seed{seed}.json"
            if not path.exists():
                output["missing"].append(f"{condition}/seed{seed}")
                continue
            try:
                runs[seed] = summarize_run(json.loads(path.read_text()), condition, seed, protocol)
            except (AssertionError, KeyError) as exc:
                raise ValueError(f"Invalid result {path}: {exc}") from exc
            output["completed"] += 1
        output["conditions"][condition] = {"seeds": sorted(runs), "per_seed": runs}
        if runs:
            output["conditions"][condition].update({
                f"{metric}_{stat}": function(r[metric] for r in runs.values())
                for metric in ("archive_hv", "final_hv")
                for stat, function in (("mean", mean), ("min", min))})
    route = output["conditions"]["route"]["per_seed"]
    for condition in conditions[1:]:
        controls = output["conditions"][condition]["per_seed"]
        output["paired_route_minus"][condition] = paired_comparison(route, controls, condition)
    output["total_training_interactions"] = sum(
        r["budget"]["total_training_interactions"]
        for c in output["conditions"].values() for r in c["per_seed"].values())
    return output


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path("results/iclr_budgeted_router_20260920"))
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = summarize(args.root)
    args.output.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    print(json.dumps({k: report[k] for k in ("completed", "expected", "missing", "total_training_interactions")}))
