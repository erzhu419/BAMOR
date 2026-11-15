"""End-to-end adaptive/blind progress feedback on the aligned-target backbone."""

import argparse
import json
from pathlib import Path

import train_bamor_mujoco as training


PROTOCOL = "aligned_progress_selector_pilot_v1"
SEEDS = (26, 27, 28)
METHODS = {"adaptive": "bamor_v27_aligned_target_progress_archive_mix",
           "blind": "bamor_v27_aligned_target_progress_fixed_archive_mix"}
LEGACY_METHODS = {"adaptive": "bamor_v25_progress_archive_mix",
                  "blind": "bamor_v25_progress_fixed_archive_mix"}
OUTPUT_ROOT = Path("results/iclr_aligned_progress_20260921")
REFERENCE_ROOT = Path("results/iclr_v25_progress_pilot_20260914/mo_halfcheetah_v5")
ARCHIVE_STEPS = tuple(range(5000, 50001, 2500))
LATE_STEPS = (45000, 47500, 50000)
TRAIN_KWARGS = dict(hidden=64, lr=3e-4, gamma=.99, batch_size=256,
                    learning_starts=5000, eval_freq=100000, device="cpu")


def metadata(condition):
    return {"protocol": PROTOCOL, "condition": condition, "complete": True,
        "seeds": list(SEEDS), "training_kwargs": TRAIN_KWARGS,
        "training_steps": 50000, "learning_updates": 45001,
        "extra_training_interactions": 0, "extra_updates": 0,
        "block_steps": 1500, "expected_complete_blocks": 30,
        "late_snapshot_steps": list(LATE_STEPS), "selection_uses_scores": condition == "adaptive"}


def run(condition, seed, output_root=OUTPUT_ROOT):
    assert condition in METHODS and seed in SEEDS
    method = METHODS[condition]
    output = Path(output_root)
    path = output / f"{method}_seed{seed}.json"
    # A previous pilot retrained after a postprocessing error; reuse its saved terminal result.
    if path.exists():
        result = json.loads(path.read_text())
    else:
        result = training.train_method("mo-halfcheetah-v5", method, 50000, 25000,
                                       seed, str(output), **TRAIN_KWARGS)
    assert (result["method"], result["seed"], result["total_steps"]) == (method, seed, 50000)
    assert result["training_interactions"] == 50000
    assert [s["step"] for s in result["policy_archive_snapshots"]] == list(ARCHIVE_STEPS)
    assert result["critic_candidate_context_rule"] == "update_preference"
    selector = result["adaptive_operator_final"]
    assert selector["score_signal"] == "within_block_normalized_vector_progress"
    assert len(selector["block_history"]) == sum(selector["counts"].values()) == 30
    result["aligned_progress_pilot"] = metadata(condition)
    path.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    print(f"DONE aligned_progress condition={condition} seed={seed} "
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
