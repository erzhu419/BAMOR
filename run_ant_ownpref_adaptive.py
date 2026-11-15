"""Ant v25 own-preference actor with the original adaptive operator selector."""

import argparse
import json
from pathlib import Path

import run_aligned_progress_pilot as pilot
import run_ant_ownpref_actor_ablation as fixed_actor
import run_final_policy_fresh as reference
from summarize_capql_archive_extension import paired
import train_bamor_mujoco as training


PROTOCOL = "ant_ownpref_adaptive_final75_v1"
METHOD = "bamor_v25_ownpref_actor_progress_archive_mix"
ROOT = Path("results/iclr_ant_ownpref_adaptive_final75_20260927")
SEEDS = reference.SEEDS
ENV = "mo-ant-v5"


def validate(raw, original, seed, model):
    assert (raw["method"], raw["env"], raw["seed"], raw["total_steps"],
            raw["switch_interval"], raw["evaluation_protocol"]) == (
                METHOD, ENV, seed, 75000, 25000, "locked_full_simplex_v1")
    assert model.exists()
    assert raw["training_interactions"] == 75000
    assert raw["environment_initialization_probe_steps"] == 2
    assert 0 < raw["evaluation_interactions"] <= 3 * 21 * 3 * 1000
    assert [item["step"] for item in raw["policy_archive_snapshots"]] == [75000]
    assert all(len(front) == 21 and all(point["archive_step"] == 75000
                                     for point in front)
               for front in raw["pareto_per_mode"])
    assert {key: raw[key] for key in ("evaluation_weights", "episode_seeds_per_mode",
                                     "ref_point")} == original["evaluation"]
    assert raw["ground_truth"]["mode_history"] == original["mode_history"]
    selector = raw["adaptive_operator_final"]
    assert selector["score_signal"] == "within_block_normalized_vector_progress"
    assert selector["actor_update_rule"] == "own_preference_with_existing_bc"
    assert selector["warmup_sequence"] == original["warmup_sequence"]
    for arm, config in selector["operator_trainer_configs"].items():
        baseline = original["operator_trainer_configs"][arm]
        assert config["envelope_policy_candidates"] == 1
        assert {key: value for key, value in config.items()
                if key != "envelope_policy_candidates"} == {
                    key: value for key, value in baseline.items()
                    if key != "envelope_policy_candidates"}
    assert len(selector["block_history"]) == len(original["operator_history"])
    return {"protocol": PROTOCOL, "seed": seed, "complete": True,
            "metrics": {"mode0_mixed_energy_utility": fixed_actor.mixed_energy_utility(raw),
                        "all_mode_direct_utility": reference.direct_utility(raw)["mean"],
                        "final_hv": raw["final_hv_mean_mode"]},
            "operator_schedule_matched": [item["operator"] for item in selector["block_history"]]
            == original["operator_history"],
            "completed_budget": {key: raw[key] for key in (
                "training_interactions", "evaluation_interactions",
                "environment_initialization_probe_steps")}}


def run(seed, root=ROOT):
    assert seed in SEEDS
    folder = root / f"seed{seed}"
    folder.mkdir(parents=True, exist_ok=True)
    output = folder / "report.json"
    raw = folder / f"{METHOD}_seed{seed}.json"
    model = folder / f"{METHOD}_seed{seed}_model.pt"
    baseline = json.loads((reference.ROOT / "mo_ant_v5" / "v25" /
                           f"seed{seed}.json").read_text())
    baseline_raw = json.loads((reference.ROOT / "mo_ant_v5" / "v25" /
                               f"{reference.METHODS['v25']}_seed{seed}.json").read_text())
    assert baseline["complete"]
    original = {"evaluation": baseline["evaluation"],
                "mode_history": baseline["mode_history"],
                "warmup_sequence": baseline_raw["adaptive_operator_final"]["warmup_sequence"],
                "operator_trainer_configs": baseline_raw["adaptive_operator_final"]["operator_trainer_configs"],
                "operator_history": [item["operator"] for item in
                                     baseline_raw["adaptive_operator_final"]["block_history"]]}
    if output.exists():
        saved = json.loads(output.read_text())
        if saved.get("complete"):
            assert (saved["protocol"], saved["seed"]) == (PROTOCOL, seed)
            return saved
        if not raw.exists():
            raise RuntimeError(f"Incomplete run at {output}; inspect before retrying")
    else:
        output.write_text(json.dumps({"protocol": PROTOCOL, "seed": seed,
                                      "complete": False}) + "\n")
    if not raw.exists():
        training.train_method(ENV, METHOD, 75000, 25000, seed, str(folder),
                              archive_steps=(75000,), **pilot.TRAIN_KWARGS)
    report = validate(json.loads(raw.read_text()), original, seed, model)
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
        assert (new["protocol"], new["seed"], new["complete"]) == (PROTOCOL, seed, True)
        for key in budget:
            budget[key] += new["completed_budget"][key]
        adaptive = json.loads((reference.ROOT / "mo_ant_v5" / "v25" /
                               f"{reference.METHODS['v25']}_seed{seed}.json").read_text())
        fixed = json.loads((fixed_actor.ROOT / f"seed{seed}" /
                            f"{fixed_actor.METHOD}_seed{seed}.json").read_text())
        capql = json.loads((reference.ROOT / "mo_ant_v5" / "capql" /
                            f"{reference.METHODS['capql']}_seed{seed}.json").read_text())
        rows.append({"seed": seed, "new": new,
                     **{name: {"mode0_mixed_energy_utility": fixed_actor.mixed_energy_utility(raw),
                               "all_mode_direct_utility": reference.direct_utility(raw)["mean"],
                               "final_hv": raw["final_hv_mean_mode"]}
                        for name, raw in (("adaptive", adaptive), ("fixed_actor", fixed),
                                          ("capql", capql))}})
    metrics = ("mode0_mixed_energy_utility", "all_mode_direct_utility", "final_hv")
    comparisons = {f"new_minus_{baseline}": {metric: paired(
        [row["new"]["metrics"][metric] for row in rows],
        [row[baseline][metric] for row in rows]) for metric in metrics}
        for baseline in ("adaptive", "fixed_actor", "capql")}
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
        path = args.root / "readout.json"
        path.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
        print(f"DONE {PROTOCOL} summary {path}")
    else:
        assert args.seed is not None
        run(args.seed, args.root)
