"""Validate saved refresh runs after the tuple/list postprocessing failure."""

import argparse
import json
from pathlib import Path
from statistics import mean

import numpy as np

import run_replay_context_refresh as pilot
from summarize_matched_pilot import seed_diagnostics


SEEDS = (26, 27, 28)
ARCHIVE_STEPS = tuple(range(5000, 50001, 2500))


def metrics(data, seed, expected_method=pilot.METHOD, *,
            env_name="mo-halfcheetah-v5", reward_dim=2, num_modes=4):
    assert (data["method"], data["seed"], data["env"], data["total_steps"], data["switch_interval"]) == (
        expected_method, seed, env_name, 50000, 25000)
    assert data["evaluation_protocol"] == "locked_full_simplex_v1"
    assert data["archive_context"] == "final_encoder"
    weights = pilot.training.full_simplex_grid(reward_dim).tolist()
    assert data["evaluation_weights"] == weights
    assert data["episode_seeds_per_mode"] == [
        [100000 + 1000 * seed + 10 * m + r for r in range(3)] for m in range(num_modes)]
    assert data["ref_point"] == pilot.training.reference_point_for_env(env_name, reward_dim).tolist()
    assert [s["step"] for s in data["policy_archive_snapshots"]] == list(ARCHIVE_STEPS)
    assert len(data["pareto_per_mode"]) == data["ground_truth"]["num_modes"] == num_modes
    assert data["episode_rewards"] and data["episode_rewards"][-1]["step"] <= 50000
    if env_name == "mo-halfcheetah-v5":
        assert data["episode_rewards"][-1]["step"] == 50000
    curves, archive, final = [], [], []
    for front in data["pareto_per_mode"]:
        assert len(front) == len(ARCHIVE_STEPS) * len(weights)
        curve = []
        for index, step in enumerate(ARCHIVE_STEPS):
            rows = [p for p in front if p["archive_index"] == index]
            assert [p["weight"] for p in rows] == weights, "Incomplete snapshot grid"
            assert all(p["archive_step"] == step for p in rows)
            returns = np.asarray([p["return"] for p in rows])
            assert returns.shape == (len(weights), reward_dim) and np.isfinite(returns).all()
            maxima = ({"max_speed_return": float(returns[:, 0].max())} if reward_dim == 2 else
                      {"max_return_per_objective": returns.max(axis=0).tolist()})
            curve.append({"step": step, "hv": pilot.training.compute_hv_nd(returns, data["ref_point"]),
                          **maxima})
        curves.append(curve)
        archive.append(pilot.training.compute_hv_nd([p["return"] for p in front], data["ref_point"]))
        final.append(curve[-1]["hv"])
    np.testing.assert_allclose(archive, data["hv_per_mode"], rtol=1e-10, atol=1e-6)
    np.testing.assert_allclose(final, data["final_hv_per_mode"], rtol=1e-10, atol=1e-6)
    np.testing.assert_allclose([mean(archive), mean(final)],
                               [data["hv_mean_mode"], data["final_hv_mean_mode"]], rtol=1e-10)
    history = data["ground_truth"]["mode_history"]
    best = [max(c, key=lambda p: p["hv"]) for c in curves]
    late = [mean(c[i]["hv"] for c in curves) for i in (-3, -2, -1)]
    return {**seed_diagnostics(data), "mode_history": history,
            "old_modes": sorted({m for _, m in history} - {history[-1][1]}),
            "mode_best_snapshot": best, "snapshot_mode_curves": curves,
            "mode_final_to_best_snapshot": [f / b["hv"] if b["hv"] else None for f, b in zip(final, best)],
            "posthoc_late_snapshot_hv": late, "posthoc_late_mean_hv": mean(late),
            "posthoc_late_mode_mean_hv": [mean(p["hv"] for p in c[-3:]) for c in curves],
            "training_seconds": data["training_seconds"], "evaluation_seconds": data["evaluation_seconds"]}


def validate_trace(trace):
    assert trace["protocol"] == pilot.PROTOCOL
    assert trace["scheduled_steps"] == [c["step"] for c in trace["cases"]] == list(pilot.REFRESH_STEPS)
    assert all(c["transitions_reencoded"] == c["step"] for c in trace["cases"])
    assert trace["transitions_reencoded"] == sum(pilot.REFRESH_STEPS)
    assert trace["extra_environment_steps"] == trace["extra_updates"] == 0
    np.testing.assert_allclose(trace["refresh_seconds"], sum(c["seconds"] for c in trace["cases"]))
    return {k: trace[k] for k in ("protocol", "complete", "transitions_reencoded", "refresh_seconds",
                                  "extra_environment_steps", "extra_updates")}


def summarize(root, reference_root):
    conditions = {c: {"per_seed": {}} for c in ("stored", "refresh")}
    paired, validation = {}, {}
    for seed in SEEDS:
        filename = f"{pilot.METHOD}_seed{seed}.json"
        reference = json.loads((reference_root / filename).read_text())
        data = json.loads((root / filename).read_text())
        trace = json.loads((root / f"seed{seed}_refresh.json").read_text())
        checks = pilot.paired_protocol_check(data, reference)
        trace_summary = validate_trace(trace)
        a, b = metrics(data, seed), metrics(reference, seed)
        a["refresh"] = trace_summary
        a["refresh_fraction_of_training_seconds"] = trace_summary["refresh_seconds"] / a["training_seconds"]
        conditions["stored"]["per_seed"][seed] = b
        conditions["refresh"]["per_seed"][seed] = a
        validation[seed] = {"paired_protocol_checks": checks, "evaluation_recomputed": True,
                            "trace_complete_as_saved": trace["complete"],
                            "runner_metadata_present": "replay_context_refresh" in data}
        paired[seed] = {
            **{k: a[k] - b[k] for k in ("archive_hv", "final_hv")},
            "final_hv_relative_change": a["final_hv"] / b["final_hv"] - 1.,
            "final_mode_hv": [x - y for x, y in zip(a["final_mode_hv"], b["final_mode_hv"])],
            "training_seconds_difference": a["training_seconds"] - b["training_seconds"],
        }
    for c in conditions.values():
        for metric in ("archive_hv", "final_hv", "posthoc_late_mean_hv"):
            for stat, fn in (("mean", mean), ("min", min)):
                c[f"{metric}_{stat}"] = fn(r[metric] for r in c["per_seed"].values())
    a, b = conditions["refresh"], conditions["stored"]
    gates = {
        "final_mean_improved": a["final_hv_mean"] > b["final_hv_mean"],
        "final_minimum_improved": a["final_hv_min"] > b["final_hv_min"],
        "seed28_old_mode2_retention_improved": a["per_seed"][28]["mode_final_to_best_snapshot"][2]
        > b["per_seed"][28]["mode_final_to_best_snapshot"][2],
    }
    return {
        "protocol": pilot.PROTOCOL, "validated_runs": len(paired), "expected": len(SEEDS),
        "sources": {"refresh": str(root), "stored": str(reference_root)},
        "validation": validation, "conditions": conditions, "paired_refresh_minus_stored": paired,
        "final_wins": sum(p["final_hv"] > 0 for p in paired.values()),
        "relative_changes": {k: a[k] / b[k] - 1. for k in (
            "final_hv_mean", "final_hv_min", "archive_hv_mean",
            "posthoc_late_mean_hv_mean", "posthoc_late_mean_hv_min")},
        "development_gates": gates, "advance_to_confirmation": all(gates.values()),
        "new_training_interactions": 150000, "extra_environment_steps": 0, "extra_updates": 0,
        "scope": "Original scheduler jobs exited 1 after saving complete training/evaluation results: "
                 "live tuple mode histories were compared directly with JSON lists. Summary validates "
                 "saved results after fixing that comparison; original files/logs and task states are "
                 "unchanged. All 45 refreshes and four full 19x21 evaluation grids are checked. "
                 "Three development seeds; final-encoder snapshot retention is not an online curve. "
                 "45k/47.5k/50k snapshot averages are posthoc stability diagnostics, not independent "
                 "replicates or a replacement for the frozen final-HV gates. "
                 "Refresh time is measured; historical wall-time differences are not a compute-matched benchmark.",
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path("results/iclr_replay_context_refresh_20260920"))
    parser.add_argument("--reference-root", type=Path, default=pilot.drift.forks.REFERENCE_ROOT)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = summarize(args.root, args.reference_root)
    args.output.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    print(json.dumps({k: report[k] for k in ("validated_runs", "development_gates", "relative_changes")}))
