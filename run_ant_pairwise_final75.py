"""Fresh Ant pilot: pairwise residual x adaptive routing, plus seeded CAPQL."""

import argparse
import json
from pathlib import Path

import torch

import run_aligned_progress_pilot as pilot
import run_ant_ownpref_actor_ablation as actor
import run_final_policy_fresh as fresh
from summarize_capql_archive_extension import paired
import train_bamor_mujoco as training


PROTOCOL = "ant_pairwise_final75_v1"
ROOT = Path("results/iclr_ant_pairwise_final75_20260927")
SEEDS = tuple(range(1301, 1306))
ENV = "mo-ant-v5"
STEPS = 75000
METHODS = {
    "base_adaptive": "bamor_v25_ownpref_both_progress_archive_mix",
    "base_blind": "bamor_v25_ownpref_both_progress_fixed_archive_mix",
    "pairwise_adaptive": "bamor_v25_pairwise_progress_archive_mix",
    "pairwise_blind": "bamor_v25_pairwise_progress_fixed_archive_mix",
    "capql": "capql",
}
METRICS = ("mode0_mixed_energy_utility", "all_mode_direct_utility", "final_hv")


def metrics(raw):
    return {
        "mode0_mixed_energy_utility": actor.mixed_energy_utility(raw),
        "all_mode_direct_utility": fresh.direct_utility(raw)["mean"],
        "final_hv": raw["final_hv_mean_mode"],
    }


def validate(raw, condition, seed, model):
    assert (raw["method"], raw["env"], raw["seed"], raw["total_steps"],
            raw["switch_interval"], raw["evaluation_protocol"]) == (
                METHODS[condition], ENV, seed, STEPS, 25000,
                "locked_full_simplex_v1")
    assert raw["training_interactions"] == STEPS
    assert raw["environment_initialization_probe_steps"] == 2
    assert 0 < raw["evaluation_interactions"] <= 3 * 21 * 3 * 1000
    assert len(raw["evaluation_weights"]) == 21
    assert len(raw["pareto_per_mode"]) == 3
    assert all(len(front) == 21 for front in raw["pareto_per_mode"])
    assert model.exists()
    assert raw["final_hv_mean_mode"] == raw["hv_mean_mode"]
    selector = None
    if condition == "capql":
        assert raw["policy_archive_snapshots"] is None
        assert raw["train_seed"] == seed
    else:
        assert [item["step"] for item in raw["policy_archive_snapshots"]] == [STEPS]
        assert all(point["archive_step"] == STEPS
                   for front in raw["pareto_per_mode"] for point in front)
        selector = raw["adaptive_operator_final"]
        assert selector["score_signal"] == "within_block_normalized_vector_progress"
        assert selector["actor_update_rule"] == "own_preference_with_existing_bc"
        assert selector["critic_candidate_rule"] == "own_preference_only"
        assert all(config["envelope_policy_candidates"] == 1 and
                   config["envelope_target_candidates"] == 1
                   for config in selector["operator_trainer_configs"].values())
        policy = torch.load(model, map_location="cpu", weights_only=True)["policy"]
        assert ("pair_mean_weight" in policy) == condition.startswith("pairwise")
    return {
        "protocol": PROTOCOL, "condition": condition, "seed": seed,
        "complete": True, "metrics": metrics(raw),
        "evaluation": {key: raw[key] for key in (
            "evaluation_weights", "episode_seeds_per_mode", "ref_point")},
        "mode_history": raw["ground_truth"]["mode_history"],
        "operator_configs": selector["operator_trainer_configs"] if selector else None,
        "operator_schedule": [item["operator"] for item in selector["block_history"]]
                             if selector else None,
        "completed_budget": {key: raw[key] for key in (
            "training_interactions", "evaluation_interactions",
            "environment_initialization_probe_steps")},
    }


def run(condition, seed, root=ROOT):
    assert condition in METHODS and seed in SEEDS
    folder = root / condition / f"seed{seed}"
    folder.mkdir(parents=True, exist_ok=True)
    output = folder / "report.json"
    raw = folder / f"{METHODS[condition]}_seed{seed}.json"
    model = folder / f"{METHODS[condition]}_seed{seed}_model.pt"
    if output.exists():
        saved = json.loads(output.read_text())
        if saved.get("complete"):
            assert (saved["protocol"], saved["condition"], saved["seed"]) == (
                PROTOCOL, condition, seed)
            return saved
        if not raw.exists():
            raise RuntimeError(f"Incomplete run at {output}; inspect before retrying")
    else:
        output.write_text(json.dumps({"protocol": PROTOCOL, "condition": condition,
                                      "seed": seed, "complete": False}) + "\n")
    if not raw.exists():
        if condition == "capql":
            training.train_capql(ENV, STEPS, 25000, seed, str(folder),
                                 eval_freq=100000, device="cpu", method="capql")
        else:
            training.train_method(ENV, METHODS[condition], STEPS, 25000, seed,
                                  str(folder), archive_steps=(STEPS,),
                                  **pilot.TRAIN_KWARGS)
    report = validate(json.loads(raw.read_text()), condition, seed, model)
    output.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    print(f"DONE {PROTOCOL} {condition} seed={seed} "
          f"mixed={report['metrics']['mode0_mixed_energy_utility']:.6f}", flush=True)
    return report


def summarize(root=ROOT):
    budget = {key: 0 for key in ("training_interactions", "evaluation_interactions",
                                  "environment_initialization_probe_steps")}
    rows = []
    for seed in SEEDS:
        row = {condition: json.loads((root / condition / f"seed{seed}" /
                                      "report.json").read_text())
               for condition in METHODS}
        assert all((item["protocol"], item["condition"], item["seed"],
                    item["complete"]) == (PROTOCOL, condition, seed, True)
                   for condition, item in row.items())
        reference = row["base_adaptive"]
        assert all(item["evaluation"] == reference["evaluation"] and
                   item["mode_history"] == reference["mode_history"]
                   for item in row.values())
        assert all(row[condition]["operator_configs"] ==
                   reference["operator_configs"] for condition in METHODS
                   if condition != "capql")
        for item in row.values():
            for key in budget:
                budget[key] += item["completed_budget"][key]
        rows.append(row)
    pairs = (
        ("pairwise_adaptive", "base_adaptive"),
        ("pairwise_blind", "base_blind"),
        ("pairwise_adaptive", "pairwise_blind"),
        ("base_adaptive", "base_blind"),
        ("pairwise_adaptive", "capql"),
    )
    comparisons = {
        f"{left}_minus_{right}": {
            metric: paired([row[left]["metrics"][metric] for row in rows],
                           [row[right]["metrics"][metric] for row in rows])
            for metric in METRICS}
        for left, right in pairs
    }
    interaction = [
        row["pairwise_adaptive"]["metrics"][METRICS[0]]
        - row["base_adaptive"]["metrics"][METRICS[0]]
        - row["pairwise_blind"]["metrics"][METRICS[0]]
        + row["base_blind"]["metrics"][METRICS[0]]
        for row in rows]
    return {
        "protocol": PROTOCOL, "seeds": list(SEEDS), "completed_budget": budget,
        "comparisons": comparisons,
        "primary_interaction_per_seed": interaction,
        "runs": [
            {"seed": seed, "metrics": {condition: row[condition]["metrics"]
                                       for condition in METHODS}}
            for seed, row in zip(SEEDS, rows)
        ],
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--condition", choices=METHODS)
    parser.add_argument("--seed", type=int, choices=SEEDS)
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--summarize", action="store_true")
    args = parser.parse_args()
    if args.summarize:
        result = summarize(args.root)
        path = args.root / "readout.json"
        path.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
        print(f"DONE {PROTOCOL} summary {path}")
    else:
        assert args.condition is not None and args.seed is not None
        run(args.condition, args.seed, args.root)
