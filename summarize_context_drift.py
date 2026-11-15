"""Reduce completed replay-context probes without pooling different anchor ages."""

import argparse
import json
import math
from pathlib import Path


SEEDS = (26, 27, 28)
STEPS = (10000, 20000, 30000, 35000, 40000, 42500, 45000, 47500, 50000)
PROTOCOL = "causal_replay_context_drift_v1"
METHOD = "bamor_v27_aligned_target_fixed_archive_mix"
LATENT_METRICS = (
    "latent_current_minus_stored", "next_latent_current_minus_stored",
    "latent_current_minus_anchor", "next_latent_current_minus_anchor",
)
ACTOR_METRICS = ("actor_context_effect", "actor_weight_drift_fixed_anchor_context")
Q_METRICS = (
    "critic_weight_drift_fixed_anchor_inputs", "q_context_effect_fixed_stored_action",
    "q_action_effect_fixed_stored_context", "q_joint_effect",
)


def preference_rms(preferences, metric):
    # All preferences evaluate the same states and action dimensions.
    return math.sqrt(sum(p[metric]["rms"] ** 2 for p in preferences) / len(preferences))


def summarize_run(data):
    assert data["protocol"] == PROTOCOL and data["method"] == METHOD
    assert data["complete"] and data["measurements_complete"]
    assert data["base_reproduction"]["matched"]
    assert data["base_reproduction"]["matched_steps_and_schedule"]
    assert data["base_reproduction"]["max_episode_return_error"] == 0.
    assert data["main_environment_steps"] == data["total_training_interactions"] == 50000
    assert data["extra_environment_steps"] == data["extra_updates"] == 0
    assert data["steps"] == [c["step"] for c in data["cases"]] == list(STEPS)
    assert data["cohort_width"] == 10000 and data["samples_per_cohort"] == 256
    assert data["weights"] == [[0., 1.], [.5, .5], [1., 0.]]
    cohorts = data["cohorts"]
    assert [c["start_transition_index"] for c in cohorts] == list(range(0, 50000, 10000))
    rows = []
    for case in data["cases"]:
        assert [p["start_transition_index"] for p in case["panels"]] == list(
            range(0, case["step"] // 10000 * 10000, 10000))
        for panel in case["panels"]:
            assert panel["anchor_step"] == panel["start_transition_index"] + 10000
            assert panel["samples"] == 256
            prefs = panel["preferences"]
            assert [p["weight"] for p in prefs] == data["weights"]
            rows.append({
                "step": case["step"],
                **{k: panel[k] for k in ("start_transition_index", "anchor_step", "samples")},
                "latent_rms": {k: panel[k]["rms"] for k in LATENT_METRICS},
                "actor_rms_all_preferences": {k: preference_rms(prefs, k) for k in ACTOR_METRICS},
                "preferences": [{
                    "weight": p["weight"],
                    "rms": {k: p[k]["rms"] for k in ACTOR_METRICS + Q_METRICS},
                    "q_delta_mean": {k: p[k]["mean"] for k in Q_METRICS},
                    "q_means": p["q_means"],
                } for p in prefs],
            })
    return {
        "seed": data["seed"], "base_reproduction": data["base_reproduction"],
        "ground_truth_mode_history": data["ground_truth_mode_history"],
        "cohorts": [{k: v for k, v in c.items() if k != "sample_timesteps"} for c in cohorts],
        "rows": rows,
    }


def summarize(root):
    runs = {}
    for seed in SEEDS:
        path = root / f"seed{seed}.json"
        data = json.loads(path.read_text())
        assert data["seed"] == seed
        runs[seed] = {"source": str(path), **summarize_run(data)}
    return {
        "protocol": PROTOCOL, "method": METHOD, "completed": len(runs), "expected": len(SEEDS),
        "total_training_interactions": 150000, "extra_environment_steps": 0, "extra_updates": 0,
        "aggregation": "Equal-sized preference panels use sqrt(mean(rms**2)), not mean(rms). "
                       "No pooling across cohorts, checkpoints or seeds. Repeated panels are correlated. "
                       "Mode counts describe the whole time bin, not mode-specific measurements. "
                       "Actor context sensitivity and anchor-relative weight drift are different contrasts. "
                       "Q changes are soft-critic sensitivity, not measured policy improvement.",
        "per_seed": runs,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path("results/iclr_context_drift_20260920"))
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = summarize(args.root)
    args.output.write_text(json.dumps(report, separators=(",", ":"), allow_nan=False) + "\n")
    print(json.dumps({k: report[k] for k in ("completed", "expected", "total_training_interactions")}))
