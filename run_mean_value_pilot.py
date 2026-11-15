"""Fixed 50/50 actor-value intervention against the observed v27 controls."""

import argparse
import json
from pathlib import Path

import numpy as np

import run_replay_context_confirmation as controls


training = controls.pilot.training
METHOD = "bamor_v28_mean_value_fixed_archive_mix"
BASE = controls.pilot.METHOD
PROTOCOL = "mean_value_actor_pilot_v1"
SEEDS = controls.SEEDS
MAIN_STEPS = 50000
LEARNING_STARTS = 5000
REFERENCE_ROOT = controls.OUTPUT_ROOT / "stored"
OUTPUT_ROOT = Path("results/iclr_mean_value_pilot_20260920")


def paired_protocol_check(result, reference):
    keys = ("env", "seed", "total_steps", "switch_interval", "context_dim",
            "critic_target_rule", "bootstrap_terminal_rule", "critic_candidate_context_rule",
            "v2_context", "evaluation_protocol", "evaluation_weights", "episode_seeds_per_mode",
            "ref_point", "archive_context", "policy_archive_snapshots")
    checks = {key: result[key] == reference[key] for key in keys}
    checks["method_pair"] = result["method"] == METHOD and reference["method"] == BASE
    checks["stored_replay"] = not any(d.get("replay_context_refresh") for d in (result, reference))
    checks["mode_schedule"] = np.array_equal(result["ground_truth"]["mode_history"],
                                             reference["ground_truth"]["mode_history"])
    prefix = lambda d: [e for e in d["episode_rewards"] if e["step"] <= LEARNING_STARTS]
    checks["pre_intervention_returns"] = prefix(result) == prefix(reference)
    for key in ("operator_trainer_configs", "counts", "step_counts", "block_steps", "warmup_sequence"):
        checks[key] = result["adaptive_operator_final"][key] == reference["adaptive_operator_final"][key]
    schedule = lambda d: [(b["operator"], b["end_step"], b["steps"])
                          for b in d["adaptive_operator_final"]["block_history"]]
    checks["operator_schedule"] = schedule(result) == schedule(reference)
    assert all(checks.values()), f"Pilot pairing differs: {checks}"
    return checks


def run(seed, output_root=OUTPUT_ROOT):
    assert seed in SEEDS
    reference_path = REFERENCE_ROOT / f"{BASE}_seed{seed}.json"
    reference = json.loads(reference_path.read_text())
    assert reference["method"] == BASE and reference["seed"] == seed
    assert reference["replay_context_confirmation"]["complete"]
    assert reference["replay_context_confirmation"]["condition"] == "stored"
    output = Path(output_root)
    print(f"MEAN_VALUE_PILOT seed={seed} steps={MAIN_STEPS} mean_fraction=0.5", flush=True)
    result = training.train_method(
        "mo-halfcheetah-v5", METHOD, MAIN_STEPS, 25000, seed, str(output),
        hidden=64, learning_starts=LEARNING_STARTS, eval_freq=100000, device="cpu")
    checks = paired_protocol_check(result, reference)
    assert [s["step"] for s in result["policy_archive_snapshots"]] == list(controls.ARCHIVE_STEPS)
    assert result["actor_update"]["updates"] == MAIN_STEPS - LEARNING_STARTS + 1
    assert result["actor_update"]["single_candidate_updates"] > 0
    assert result["actor_update"]["envelope_updates"] > 0
    assert result["training_interactions"] == MAIN_STEPS
    assert result["evaluation_interactions"] == len(controls.ARCHIVE_STEPS) * 4 * 21 * 3 * 1000
    result["mean_value_pilot"] = {
        "protocol": PROTOCOL, "complete": True, "seeds": list(SEEDS),
        "mean_value_fraction": 0.5, "reference": str(reference_path),
        "paired_protocol_checks": checks, "extra_training_updates": 0,
        "late_snapshot_steps": list(controls.LATE_STEPS),
        "scope": "Observed seeds are development data. Only actor value blending changes. "
                 "Retain stored replay, critic targets, entropy/BC, fixed portfolio and deterministic "
                 "evaluation. Reference results are used for post-training pairing, never selection.",
    }
    (output / f"{METHOD}_seed{seed}.json").write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    print(f"DONE mean_value_pilot seed={seed} training={result['training_interactions']} "
          f"evaluation={result['evaluation_interactions']}", flush=True)
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, choices=SEEDS, required=True)
    parser.add_argument("--output-root", type=Path, default=OUTPUT_ROOT)
    args = parser.parse_args()
    run(args.seed, args.output_root)
