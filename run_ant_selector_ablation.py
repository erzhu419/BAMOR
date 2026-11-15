"""Matched Ant final-policy diagnostic for v25 adaptive versus fixed selection."""

import argparse
import json
from pathlib import Path
from statistics import mean

import numpy as np

import run_aligned_progress_pilot as pilot
import run_final_policy_fresh as reference
from summarize_capql_archive_extension import paired
import train_bamor_mujoco as training


PROTOCOL = "ant_selector_final75_v1"
ROOT = Path("results/iclr_ant_selector_final75_20260927")
ENV = "mo-ant-v5"
METHOD = "bamor_v25_progress_fixed_archive_mix"
SEEDS = reference.SEEDS


def validate(data, seed, model, adaptive):
    assert (data["method"], data["env"], data["seed"], data["total_steps"],
            data["switch_interval"], data["evaluation_protocol"]) == (
                METHOD, ENV, seed, 75000, 25000, "locked_full_simplex_v1")
    assert model.exists()
    assert data["training_interactions"] == 75000
    assert data["environment_initialization_probe_steps"] == 2
    assert 0 < data["evaluation_interactions"] <= 3 * 21 * 3 * 1000
    assert [item["step"] for item in data["policy_archive_snapshots"]] == [75000]
    assert len(data["pareto_per_mode"]) == 3
    assert all(len(front) == 21 and all(point["archive_step"] == 75000 for point in front)
               for front in data["pareto_per_mode"])
    evaluation = {key: data[key] for key in (
        "evaluation_weights", "episode_seeds_per_mode", "ref_point")}
    assert evaluation == adaptive["evaluation"]
    assert data["ground_truth"]["mode_history"] == adaptive["mode_history"]
    assert np.isclose(data["hv_mean_mode"], data["final_hv_mean_mode"])
    utility = reference.direct_utility(data)
    return {
        "protocol": PROTOCOL, "env": ENV, "method": "v25_fixed", "seed": seed,
        "complete": True,
        "metrics": {"final_hv": data["final_hv_mean_mode"],
                    "mode_final_hv": data["final_hv_per_mode"],
                    "direct_utility": utility,
                    "training_seconds": data["training_seconds"],
                    "evaluation_seconds": data["evaluation_seconds"]},
        "completed_budget": {key: data[key] for key in (
            "training_interactions", "evaluation_interactions",
            "environment_initialization_probe_steps")},
    }


def run(seed, root=ROOT, reference_root=reference.ROOT):
    assert seed in SEEDS
    folder = Path(root) / f"seed{seed}"
    folder.mkdir(parents=True, exist_ok=True)
    output = folder / "report.json"
    raw = folder / f"{METHOD}_seed{seed}.json"
    model = folder / f"{METHOD}_seed{seed}_model.pt"
    adaptive_path = (Path(reference_root) / ENV.replace("-", "_") /
                     "v25" / f"seed{seed}.json")
    adaptive = json.loads(adaptive_path.read_text())
    assert (adaptive["protocol"], adaptive["env"], adaptive["method"],
            adaptive["seed"], adaptive["complete"]) == (
                reference.PROTOCOL, ENV, "v25", seed, True)
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
    report = validate(json.loads(raw.read_text()), seed, model, adaptive)
    output.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    print(f"DONE {PROTOCOL} seed={seed} final_hv={report['metrics']['final_hv']:.6f}",
          flush=True)
    return report


def summarize(root=ROOT, reference_root=reference.ROOT):
    rows = []
    budget = {key: 0 for key in (
        "training_interactions", "evaluation_interactions",
        "environment_initialization_probe_steps")}
    for seed in SEEDS:
        fixed = json.loads((Path(root) / f"seed{seed}" / "report.json").read_text())
        adaptive = json.loads((Path(reference_root) / ENV.replace("-", "_") /
                               "v25" / f"seed{seed}.json").read_text())
        capql = json.loads((Path(reference_root) / ENV.replace("-", "_") /
                            "capql" / f"seed{seed}.json").read_text())
        assert all(row["complete"] for row in (fixed, adaptive, capql))
        for key in budget:
            budget[key] += fixed["completed_budget"][key]
        rows.append({"seed": seed, "v25_fixed": fixed,
                     "v25_adaptive": adaptive, "capql": capql})
    comparisons = {}
    for control in ("v25_adaptive", "capql"):
        comparisons[f"fixed_minus_{control}"] = {
            "final_hv": paired(
                [row["v25_fixed"]["metrics"]["final_hv"] for row in rows],
                [row[control]["metrics"]["final_hv"] for row in rows]),
            "direct_utility": paired(
                [row["v25_fixed"]["metrics"]["direct_utility"]["mean"] for row in rows],
                [row[control]["metrics"]["direct_utility"]["mean"] for row in rows]),
        }
    return {"protocol": PROTOCOL, "scope": "posthoc_ant_mechanism_diagnostic",
            "seeds": list(SEEDS), "completed_budget": budget,
            "comparisons": comparisons, "runs": rows,
            "mode0_direct_utility_fixed_minus_adaptive": mean(
                row["v25_fixed"]["metrics"]["direct_utility"]["per_mode"][0] -
                row["v25_adaptive"]["metrics"]["direct_utility"]["per_mode"][0]
                for row in rows)}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int)
    parser.add_argument("--summarize", action="store_true")
    parser.add_argument("--root", type=Path, default=ROOT)
    args = parser.parse_args()
    if args.summarize:
        output = args.root / "readout.json"
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(summarize(args.root), indent=2, allow_nan=False) + "\n")
        print(f"DONE {PROTOCOL} summary {output}")
    else:
        assert args.seed is not None
        run(args.seed, args.root)
