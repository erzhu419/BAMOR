"""Matched Ant v25 fixed-selector actor-candidate ablation at 75k steps."""

import argparse
import json
from pathlib import Path
from statistics import mean

import numpy as np

import run_aligned_progress_pilot as pilot
import run_ant_selector_ablation as control
import run_final_policy_fresh as fresh
from summarize_capql_archive_extension import paired
import train_bamor_mujoco as training


PROTOCOL = "ant_ownpref_actor_final75_v1"
METHOD = "bamor_v25_ownpref_actor_progress_fixed_archive_mix"
ROOT = Path("results/iclr_ant_ownpref_actor_final75_20260927")
SEEDS = fresh.SEEDS


def mixed_energy_utility(data):
    weights = np.asarray(data["evaluation_weights"], dtype=np.float64)
    selected = [index for index, weight in enumerate(weights) if 0 < weight[2] < 1]
    assert len(selected) == 14 and len(data["pareto_per_mode"]) == 3
    front = data["pareto_per_mode"][0]
    assert len(front) == len(weights)
    return mean(float(weights[index] @ np.asarray(front[index]["return"]))
                for index in selected)


def validate(raw, fixed, seed, model):
    assert (raw["method"], raw["env"], raw["seed"], raw["total_steps"],
            raw["switch_interval"], raw["evaluation_protocol"]) == (
                METHOD, "mo-ant-v5", seed, 75000, 25000, "locked_full_simplex_v1")
    assert model.exists()
    assert raw["training_interactions"] == 75000
    assert raw["environment_initialization_probe_steps"] == 2
    assert 0 < raw["evaluation_interactions"] <= 3 * 21 * 3 * 1000
    assert [item["step"] for item in raw["policy_archive_snapshots"]] == [75000]
    assert all(len(front) == 21 and all(point["archive_step"] == 75000
                                     for point in front)
               for front in raw["pareto_per_mode"])
    assert {key: raw[key] for key in ("evaluation_weights", "episode_seeds_per_mode",
                                     "ref_point")} == {key: fixed[key] for key in (
                                         "evaluation_weights", "episode_seeds_per_mode", "ref_point")}
    assert raw["ground_truth"]["mode_history"] == fixed["ground_truth"]["mode_history"]
    a, b = raw["adaptive_operator_final"], fixed["adaptive_operator_final"]
    assert a["warmup_sequence"] == b["warmup_sequence"]
    assert a["actor_update_rule"] == "own_preference_with_existing_bc"
    for arm, config in a["operator_trainer_configs"].items():
        original = b["operator_trainer_configs"][arm]
        assert config["envelope_policy_candidates"] == 1
        assert {k: v for k, v in config.items() if k != "envelope_policy_candidates"} == {
            k: v for k, v in original.items() if k != "envelope_policy_candidates"}
    schedule = [item["operator"] for item in a["block_history"]]
    fixed_schedule = [item["operator"] for item in b["block_history"]]
    return {"protocol": PROTOCOL, "seed": seed, "complete": True,
            "metrics": {"mode0_mixed_energy_utility": mixed_energy_utility(raw),
                        "all_mode_direct_utility": fresh.direct_utility(raw)["mean"],
                        "final_hv": raw["final_hv_mean_mode"]},
            "operator_schedule_matched": schedule == fixed_schedule,
            "completed_budget": {key: raw[key] for key in (
                "training_interactions", "evaluation_interactions",
                "environment_initialization_probe_steps")}}


def run(seed, root=ROOT):
    assert seed in SEEDS
    folder = root / f"seed{seed}"
    folder.mkdir(parents=True, exist_ok=True)
    output = folder / "report.json"
    raw_path = folder / f"{METHOD}_seed{seed}.json"
    model = folder / f"{METHOD}_seed{seed}_model.pt"
    fixed_path = control.ROOT / f"seed{seed}" / f"{control.METHOD}_seed{seed}.json"
    fixed = json.loads(fixed_path.read_text())
    if output.exists():
        saved = json.loads(output.read_text())
        if saved.get("complete"):
            assert (saved["protocol"], saved["seed"]) == (PROTOCOL, seed)
            return saved
        if not raw_path.exists():
            raise RuntimeError(f"Incomplete run at {output}; inspect before retrying")
    else:
        output.write_text(json.dumps({"protocol": PROTOCOL, "seed": seed,
                                      "complete": False}) + "\n")
    if not raw_path.exists():
        training.train_method("mo-ant-v5", METHOD, 75000, 25000, seed,
                              str(folder), archive_steps=(75000,), **pilot.TRAIN_KWARGS)
    report = validate(json.loads(raw_path.read_text()), fixed, seed, model)
    output.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    print(f"DONE {PROTOCOL} seed={seed} "
          f"mixed={report['metrics']['mode0_mixed_energy_utility']:.6f}", flush=True)
    return report


def summarize(root=ROOT):
    rows = []
    budget = {key: 0 for key in ("training_interactions", "evaluation_interactions",
                                  "environment_initialization_probe_steps")}
    for seed in SEEDS:
        new = json.loads((root / f"seed{seed}" / "report.json").read_text())
        fixed = json.loads((control.ROOT / f"seed{seed}" / f"{control.METHOD}_seed{seed}.json").read_text())
        capql = json.loads((fresh.ROOT / "mo_ant_v5" / "capql" /
                            f"{fresh.METHODS['capql']}_seed{seed}.json").read_text())
        assert (new["complete"], new["protocol"], new["seed"]) == (True, PROTOCOL, seed)
        for key in budget:
            budget[key] += new["completed_budget"][key]
        rows.append({"seed": seed, "new": new,
                     "fixed": {"mode0_mixed_energy_utility": mixed_energy_utility(fixed),
                               "all_mode_direct_utility": fresh.direct_utility(fixed)["mean"],
                               "final_hv": fixed["final_hv_mean_mode"]},
                     "capql": {"mode0_mixed_energy_utility": mixed_energy_utility(capql),
                               "all_mode_direct_utility": fresh.direct_utility(capql)["mean"],
                               "final_hv": capql["final_hv_mean_mode"]}})
    comparisons = {f"new_minus_{baseline}": {metric: paired(
        [row["new"]["metrics"][metric] for row in rows],
        [row[baseline][metric] for row in rows])
        for metric in ("mode0_mixed_energy_utility", "all_mode_direct_utility", "final_hv")}
        for baseline in ("fixed", "capql")}
    return {"protocol": PROTOCOL, "seeds": list(SEEDS), "completed_budget": budget,
            "operator_schedule_matched": {str(row["seed"]): row["new"]["operator_schedule_matched"]
                                          for row in rows},
            "comparisons": comparisons,
            "runs": [{"seed": row["seed"], "metrics": row["new"]["metrics"]} for row in rows]}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--summarize", action="store_true")
    parser.add_argument("--root", type=Path, default=ROOT)
    args = parser.parse_args()
    if args.summarize:
        result = summarize(args.root)
        output = args.root / "readout.json"
        output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
        print(f"DONE {PROTOCOL} summary {output}")
    else:
        assert args.seed is not None
        run(args.seed, args.root)
