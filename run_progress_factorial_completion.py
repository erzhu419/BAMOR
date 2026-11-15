"""Complete the missing v25 blind arm without rerunning the three known arms."""

import argparse
import json
from pathlib import Path

import run_aligned_progress_confirmation as previous


PROTOCOL = "progress_factorial_completion_v1"
SEEDS = previous.SEEDS
METHOD = previous.pilot.LEGACY_METHODS["blind"]
OUTPUT_ROOT = Path("results/iclr_progress_factorial_completion_20260921")
REFERENCE_SUMMARY = Path("results/iclr_aligned_progress_confirmation_20260921_summary.json")


def metadata():
    return {**previous.metadata("legacy"), "protocol": PROTOCOL,
            "condition": "legacy_blind", "selection_uses_scores": False}


def run(seed, output_root=OUTPUT_ROOT):
    assert seed in SEEDS
    path = Path(output_root) / f"{METHOD}_seed{seed}.json"
    # A saved terminal result can finish postprocessing without another training run.
    if path.exists():
        result = json.loads(path.read_text())
    else:
        result = previous.pilot.training.train_method(
            "mo-halfcheetah-v5", METHOD, 50000, 25000, seed, str(output_root),
            **previous.pilot.TRAIN_KWARGS)
    assert (result["method"], result["seed"], result["total_steps"]) == (METHOD, seed, 50000)
    assert result["training_interactions"] == 50000
    assert result["evaluation_interactions"] == 4788000
    assert result["environment_initialization_probe_steps"] == 2
    assert result["critic_candidate_context_rule"] == "candidate_preference"
    assert [s["step"] for s in result["policy_archive_snapshots"]] == list(previous.pilot.ARCHIVE_STEPS)
    state = result["adaptive_operator_final"]
    assert state["score_signal"] == "within_block_normalized_vector_progress"
    assert len(state["block_history"]) == sum(state["counts"].values()) == 30
    result["progress_factorial_completion"] = metadata()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    print(f"DONE progress_factorial_completion seed={seed} training_interactions=50000 "
          "evaluation_interactions=4788000", flush=True)
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, choices=SEEDS, required=True)
    parser.add_argument("--output-root", type=Path, default=OUTPUT_ROOT)
    args = parser.parse_args()
    run(args.seed, args.output_root)
