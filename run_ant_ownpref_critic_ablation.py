"""Ant actor-off fixed-selector ablation of cross-preference critic targets."""

import argparse
import json
from pathlib import Path

import run_aligned_progress_pilot as pilot
import run_ant_ownpref_actor_ablation as actor
import run_final_policy_fresh as fresh
from summarize_capql_archive_extension import paired
import train_bamor_mujoco as training


PROTOCOL = "ant_ownpref_critic_final75_v1"
METHOD = "bamor_v25_ownpref_both_progress_fixed_archive_mix"
ROOT = Path("results/iclr_ant_ownpref_critic_final75_20260927")
SEEDS = fresh.SEEDS
ENV = "mo-ant-v5"


def metrics(raw):
    return {"mode0_mixed_energy_utility": actor.mixed_energy_utility(raw),
            "all_mode_direct_utility": fresh.direct_utility(raw)["mean"],
            "final_hv": raw["final_hv_mean_mode"]}


def validate(raw, control, seed, model):
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
                                     "ref_point")} == {key: control[key] for key in (
                                         "evaluation_weights", "episode_seeds_per_mode", "ref_point")}
    assert raw["ground_truth"]["mode_history"] == control["ground_truth"]["mode_history"]
    new = raw["adaptive_operator_final"]
    old = control["adaptive_operator_final"]
    assert new["warmup_sequence"] == old["warmup_sequence"]
    assert new["actor_update_rule"] == old["actor_update_rule"]
    assert new["critic_candidate_rule"] == "own_preference_only"
    assert [item["operator"] for item in new["block_history"]] == [
        item["operator"] for item in old["block_history"]]
    for arm, config in new["operator_trainer_configs"].items():
        original = old["operator_trainer_configs"][arm]
        assert config["envelope_target_candidates"] == 1
        assert {key: value for key, value in config.items()
                if key != "envelope_target_candidates"} == {
                    key: value for key, value in original.items()
                    if key != "envelope_target_candidates"}
    return {"protocol": PROTOCOL, "seed": seed, "complete": True,
            "metrics": metrics(raw),
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
    control = json.loads((actor.ROOT / f"seed{seed}" /
                          f"{actor.METHOD}_seed{seed}.json").read_text())
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
    report = validate(json.loads(raw.read_text()), control, seed, model)
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
        control = json.loads((actor.ROOT / f"seed{seed}" /
                              f"{actor.METHOD}_seed{seed}.json").read_text())
        capql = json.loads((fresh.ROOT / "mo_ant_v5" / "capql" /
                            f"{fresh.METHODS['capql']}_seed{seed}.json").read_text())
        rows.append({"seed": seed, "new": new,
                     "actor_only": metrics(control), "capql": metrics(capql)})
    keys = ("mode0_mixed_energy_utility", "all_mode_direct_utility", "final_hv")
    comparisons = {f"new_minus_{baseline}": {key: paired(
        [row["new"]["metrics"][key] for row in rows],
        [row[baseline][key] for row in rows]) for key in keys}
        for baseline in ("actor_only", "capql")}
    return {"protocol": PROTOCOL, "seeds": list(SEEDS), "completed_budget": budget,
            "comparisons": comparisons,
            "runs": [{"seed": row["seed"], "metrics": row["new"]["metrics"]}
                     for row in rows]}


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
