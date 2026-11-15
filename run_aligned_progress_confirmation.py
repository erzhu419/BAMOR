"""Frozen three-arm replication on previously unused training seeds."""

import argparse
import json
from pathlib import Path

import run_aligned_progress_pilot as pilot


PROTOCOL = "aligned_progress_confirmation_v1"
SEEDS = (1001, 1002, 1003, 1004, 1005)
METHODS = {**pilot.METHODS, "legacy": pilot.LEGACY_METHODS["adaptive"]}
OUTPUT_ROOT = Path("results/iclr_aligned_progress_confirmation_20260921")


def metadata(condition):
    return {"protocol": PROTOCOL, "condition": condition, "complete": True,
        "seeds": list(SEEDS), "training_kwargs": dict(pilot.TRAIN_KWARGS),
        "training_steps": 50000, "learning_updates": 45001,
        "extra_training_interactions": 0, "extra_updates": 0,
        "block_steps": 1500, "expected_complete_blocks": 30,
        "late_snapshot_steps": list(pilot.LATE_STEPS), "selection_uses_scores": condition != "blind",
        "critic_candidate_context_rule": "candidate_preference" if condition == "legacy" else "update_preference"}


def run(condition, seed, output_root=OUTPUT_ROOT):
    assert condition in METHODS and seed in SEEDS
    method = METHODS[condition]
    path = Path(output_root) / f"{method}_seed{seed}.json"
    # Recover postprocessing from a saved terminal result without another training run.
    if path.exists():
        result = json.loads(path.read_text())
    else:
        result = pilot.training.train_method("mo-halfcheetah-v5", method, 50000, 25000,
                                             seed, str(output_root), **pilot.TRAIN_KWARGS)
    assert (result["method"], result["seed"], result["total_steps"]) == (method, seed, 50000)
    assert result["training_interactions"] == 50000
    assert [s["step"] for s in result["policy_archive_snapshots"]] == list(pilot.ARCHIVE_STEPS)
    assert result["critic_candidate_context_rule"] == metadata(condition)["critic_candidate_context_rule"]
    state = result["adaptive_operator_final"]
    assert state["score_signal"] == "within_block_normalized_vector_progress"
    assert len(state["block_history"]) == sum(state["counts"].values()) == 30
    result["aligned_progress_confirmation"] = metadata(condition)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    print(f"DONE aligned_progress_confirmation condition={condition} seed={seed} "
          f"training_interactions={result['training_interactions']} "
          f"evaluation_interactions={result['evaluation_interactions']}", flush=True)
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--condition", choices=METHODS, required=True)
    parser.add_argument("--seed", type=int, choices=SEEDS, required=True)
    parser.add_argument("--output-root", type=Path, default=OUTPUT_ROOT)
    args = parser.parse_args()
    run(args.condition, args.seed, args.output_root)
