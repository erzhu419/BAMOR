"""Frozen stored/refreshed replay comparison on fresh HalfCheetah seeds."""

import argparse
import json
from pathlib import Path

import run_replay_context_refresh as pilot


PROTOCOL = "replay_context_confirmation_v1"
SEEDS = (29, 30, 31, 32, 33)
CONDITIONS = ("stored", "refresh")
ARCHIVE_STEPS = tuple(range(5000, 50001, 2500))
LATE_STEPS = (45000, 47500, 50000)
OUTPUT_ROOT = Path("results/iclr_replay_context_confirmation_20260920")


def run(condition, seed, output_root=OUTPUT_ROOT):
    assert condition in CONDITIONS and seed in SEEDS
    output = Path(output_root) / condition
    hook = (pilot.ReplayContextRefresh(output / f"seed{seed}_refresh.json")
            if condition == "refresh" else None)
    if hook is not None:
        hook.save()
    print(f"CONFIRMATION condition={condition} seed={seed} steps={pilot.MAIN_STEPS} "
          f"refresh_interval={pilot.INTERVAL if hook is not None else 0}", flush=True)
    result = pilot.training.train_method(
        "mo-halfcheetah-v5", pilot.METHOD, pilot.MAIN_STEPS, 25000, seed, str(output),
        hidden=64, learning_starts=pilot.LEARNING_STARTS, eval_freq=100000, device="cpu",
        training_probe=hook)
    assert [s["step"] for s in result["policy_archive_snapshots"]] == list(ARCHIVE_STEPS)
    assert result["evaluation_protocol"] == "locked_full_simplex_v1"
    assert result["archive_context"] == "final_encoder"
    assert len(result["pareto_per_mode"]) == 4
    assert all(len(front) == len(ARCHIVE_STEPS) * 21 for front in result["pareto_per_mode"])
    if hook is not None:
        assert [c["step"] for c in hook.cases] == list(pilot.REFRESH_STEPS)
        result["replay_context_refresh"] = hook.state_dict(complete=True)
    result["replay_context_confirmation"] = {
        "protocol": PROTOCOL, "condition": condition, "complete": True,
        "seeds": list(SEEDS), "main_environment_steps": pilot.MAIN_STEPS,
        "total_training_interactions": pilot.MAIN_STEPS, "extra_environment_steps": 0,
        "extra_updates": 0, "normal_updates": pilot.MAIN_STEPS - pilot.LEARNING_STARTS + 1,
        "late_snapshot_steps": list(LATE_STEPS),
        "scope": "Frozen five-seed paired confirmation. Only replay latent refreshing differs; "
                 "same fixed portfolio, learner, encoder and evaluation. Pairing is checked after "
                 "both independent jobs finish. Late snapshots use the final encoder, not online "
                 "evaluation. No reference result or evaluation feedback enters training.",
    }
    path = output / f"{pilot.METHOD}_seed{seed}.json"
    path.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    if hook is not None:
        hook.save(complete=True)
    print(f"DONE replay_context_confirmation condition={condition} seed={seed} "
          f"interactions={pilot.MAIN_STEPS}", flush=True)
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--condition", choices=CONDITIONS, required=True)
    parser.add_argument("--seed", type=int, choices=SEEDS, required=True)
    parser.add_argument("--output-root", type=Path, default=OUTPUT_ROOT)
    args = parser.parse_args()
    run(args.condition, args.seed, args.output_root)
