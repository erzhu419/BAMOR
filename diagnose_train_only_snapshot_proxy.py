"""Retrospective, zero-interaction test of a causal checkpoint score."""

import argparse
import json
from pathlib import Path
from statistics import mean, median

import numpy as np

from train_bamor_mujoco import compute_hv_nd
import run_scoped_extension as extension


PROTOCOL = "train_only_checkpoint_proxy_v1"
WINDOW = 5000
OUTPUT = Path("results/iclr_train_only_snapshot_proxy_20260926_readout.json")


def training_score(episodes, step, ref_point):
    returns = [row["return"] for row in episodes if step - WINDOW < row["step"] <= step]
    assert returns, f"No completed training episodes in window ending at {step}"
    return compute_hv_nd(np.asarray(returns, dtype=float), ref_point), len(returns)


def single_actor_hv(data, step, ref_point):
    return mean(compute_hv_nd(np.asarray([row["return"] for row in front
                                           if row["archive_step"] == step], dtype=float), ref_point)
                for front in data["pareto_per_mode"])


def summarize(root=extension.ROOT):
    result = {"protocol": PROTOCOL, "score_window_steps": WINDOW,
              "scope": "Retrospective feasibility on inspected seeds; final-encoder snapshot evaluation.",
              "new_training_interactions": 0, "new_evaluation_interactions": 0,
              "environments": {}}
    for env_name, spec in extension.ENVIRONMENTS.items():
        rows = []
        for seed in spec["seeds"]:
            path = (Path(root) / env_name.replace("-", "_") / "legacy_adaptive" /
                    f"{extension.METHODS['adaptive']}_seed{seed}.json")
            data = json.loads(path.read_text())
            assert (data["method"], data["env"], data["seed"], data["total_steps"]) == (
                extension.METHODS["adaptive"], env_name, seed, 50000)
            steps = [snap["step"] for snap in data["policy_archive_snapshots"]]
            assert steps == list(range(5000, 50001, 2500))
            assert data["evaluation_protocol"] == "locked_full_simplex_v1"
            assert len(data["pareto_per_mode"]) == spec["num_modes"]
            ref = np.asarray(data["ref_point"], dtype=float)
            episodes = data["episode_rewards"]
            scores_and_counts = [training_score(episodes, step, ref) for step in steps]
            scores = [score for score, _ in scores_and_counts]
            selected = max(range(len(steps)), key=lambda i: scores[i])
            actor_hvs = [single_actor_hv(data, step, ref) for step in steps]
            best = max(range(len(steps)), key=lambda i: actor_hvs[i])
            assert np.isclose(actor_hvs[-1], data["final_hv_mean_mode"], rtol=1e-10)
            rows.append({"seed": seed, "selected_step": steps[selected],
                         "best_step_hindsight": steps[best],
                         "score_ties_at_max": sum(score == scores[selected] for score in scores),
                         "window_episode_counts": [count for _, count in scores_and_counts],
                         "training_scores": scores,
                         "selected_hv": actor_hvs[selected], "final_hv": actor_hvs[-1],
                         "best_hv_hindsight": actor_hvs[best]})
        deltas = [row["selected_hv"] - row["final_hv"] for row in rows]
        result["environments"][env_name] = {
            "seeds": list(spec["seeds"]), "runs": rows,
            "selected_minus_final_mean": mean(deltas),
            "selected_minus_final_median": median(deltas),
            "selected_minus_final_wins": sum(delta > 0 for delta in deltas),
            "selected_minus_final_losses": sum(delta < 0 for delta in deltas),
            "selected_minus_final_ties": sum(delta == 0 for delta in deltas),
            "selected_minus_final_deltas": deltas,
            "selected_over_hindsight_best_mean": mean(
                row["selected_hv"] / row["best_hv_hindsight"] for row in rows),
        }
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=extension.ROOT)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    args = parser.parse_args()
    readout = summarize(args.root)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(readout, indent=2, allow_nan=False) + "\n")
    print("DONE", PROTOCOL, sum(len(env["runs"]) for env in readout["environments"].values()))
