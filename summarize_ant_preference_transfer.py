"""Evaluate mixed targets using already saved own- and vertex-policy fronts."""

import argparse
import json
from pathlib import Path

import numpy as np

import run_final_policy_fresh as fresh


SOURCE = fresh.ROOT / "mo_ant_v5"
OUTPUT = Path("results/iclr_ant_mixed_critic_20260927/preference_transfer.json")


def summarize(source=SOURCE):
    rows = []
    for seed in fresh.SEEDS:
        data = {}
        for method in ("v25", "capql"):
            folder = source / method
            path = folder / f"{fresh.METHODS[method]}_seed{seed}.json"
            data[method] = json.loads(path.read_text())
            assert (data[method]["seed"], data[method]["env"],
                    data[method]["evaluation_protocol"]) == (
                        seed, "mo-ant-v5", "locked_full_simplex_v1")
        bamor, capql = data["v25"], data["capql"]
        assert bamor["evaluation_weights"] == capql["evaluation_weights"]
        assert bamor["episode_seeds_per_mode"] == capql["episode_seeds_per_mode"]
        weights = np.asarray(bamor["evaluation_weights"], dtype=np.float64)
        vertices = [int(np.flatnonzero(np.all(np.isclose(weights, np.eye(3)[i]), axis=1))[0])
                    for i in range(3)]
        for mode in range(3):
            b_front, c_front = bamor["pareto_per_mode"][mode], capql["pareto_per_mode"][mode]
            assert len(b_front) == len(c_front) == len(weights)
            for wi, weight in enumerate(weights):
                if not 0 < weight[2] < 1:
                    continue
                assert np.allclose(b_front[wi]["weight"], weight)
                assert np.allclose(c_front[wi]["weight"], weight)
                own = float(weight @ np.asarray(b_front[wi]["return"]))
                capql_own = float(weight @ np.asarray(c_front[wi]["return"]))
                vertex = [float(weight @ np.asarray(b_front[index]["return"]))
                          for index in vertices]
                rows.append({"seed": seed, "mode": mode, "weight_index": wi,
                             "weight": weight.tolist(), "bamor_own": own,
                             "bamor_energy_vertex": vertex[2],
                             "bamor_best_vertex": max(vertex),
                             "capql_own": capql_own})

    def aggregate(group):
        own = np.asarray([row["bamor_own"] for row in group])
        energy = np.asarray([row["bamor_energy_vertex"] for row in group])
        best = np.asarray([row["bamor_best_vertex"] for row in group])
        capql = np.asarray([row["capql_own"] for row in group])
        return {"cells": len(group), "bamor_own_minus_capql": float(np.mean(own - capql)),
                "energy_vertex_minus_own": float(np.mean(energy - own)),
                "best_vertex_minus_own": float(np.mean(best - own)),
                "best_vertex_minus_capql": float(np.mean(best - capql)),
                "energy_vertex_beats_own": int(np.sum(energy > own)),
                "best_vertex_beats_own": int(np.sum(best > own)),
                "best_vertex_beats_capql": int(np.sum(best > capql))}

    return {"protocol": "ant_preference_transfer_v1", "source": str(source),
            "scope": "posthoc_reuse_of_inspected_full21_fronts",
            "weights": "all evaluation-grid weights with 0 < energy < 1",
            "mode0": aggregate([row for row in rows if row["mode"] == 0]),
            "per_mode": {str(mode): aggregate([row for row in rows if row["mode"] == mode])
                         for mode in range(3)},
            "per_seed_mode0": {str(seed): aggregate([
                row for row in rows if row["seed"] == seed and row["mode"] == 0])
                for seed in fresh.SEEDS},
            "mode0_per_weight": {str(wi): aggregate([
                row for row in rows if row["mode"] == 0 and row["weight_index"] == wi])
                for wi in sorted({row["weight_index"] for row in rows})},
            "completed_budget": {"training_interactions": 0,
                                 "evaluation_interactions": 0,
                                 "learner_updates": 0}}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=SOURCE)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    args = parser.parse_args()
    result = summarize(args.source)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    print(f"DONE {result['protocol']} {args.output}")
