"""Matched 75k v25/CAPQL budget-sensitivity comparison on paired old seeds."""

import argparse
import json
from pathlib import Path
from statistics import mean

import numpy as np
import torch

import run_aligned_progress_pilot as pilot
import run_scoped_extension as extension
from summarize_capql_archive_extension import paired
import train_bamor_mujoco as training


PROTOCOL = "matched_75k_budget_v1"
ROOT = Path("results/iclr_75k_budget_match_20260926")
CAPQL_50_ROOT = Path("results/iclr_capql_archive_extension_20260926")
TOTAL_STEPS = 75000
STEPS = tuple(range(27500, TOTAL_STEPS + 1, 2500))
METHODS = {"v25": extension.METHODS["adaptive"], "capql": "capql_archive"}
SEEDS = tuple(range(1011, 1016))


def old_folder(env_name, method):
    env_dir = env_name.replace("-", "_")
    if method == "v25":
        return extension.ROOT / env_dir / "legacy_adaptive"
    return CAPQL_50_ROOT / env_dir


def final_utility(data):
    weights = np.asarray(data["evaluation_weights"], dtype=float)
    final_step = data["policy_archive_snapshots"][-1]["step"]
    per_mode = []
    for front in data["pareto_per_mode"]:
        points = [point for point in front if point["archive_step"] == final_step]
        assert len(points) == len(weights)
        assert all(np.allclose(point["weight"], weight, atol=1e-7)
                   for point, weight in zip(points, weights))
        per_mode.append(mean(float(np.dot(weight, point["return"]))
                             for point, weight in zip(points, weights)))
    return {"mean": mean(per_mode), "per_mode": per_mode}


def policy_prefix_equal(old_path, new_path):
    old = torch.load(old_path, map_location="cpu", weights_only=True)["policy"]
    new = torch.load(new_path, map_location="cpu", weights_only=True)["policy_archive"]
    matches = [snapshot["policy"] for snapshot in new if snapshot["step"] == 50000]
    assert len(matches) == 1 and old.keys() == matches[0].keys()
    return all(torch.equal(value, matches[0][key]) for key, value in old.items())


def summarize_run(data):
    return {"archive_hv": data["hv_mean_mode"],
            "final_hv": data["final_hv_mean_mode"],
            "mode_archive_hv": data["hv_per_mode"],
            "mode_final_hv": data["final_hv_per_mode"],
            "final_utility": final_utility(data),
            "training_seconds": data["training_seconds"],
            "evaluation_seconds": data["evaluation_seconds"]}


def validate(data, old, env_name, method, seed, old_model, new_model):
    spec = extension.ENVIRONMENTS[env_name]
    assert (data["method"], data["env"], data["seed"], data["total_steps"],
            data["switch_interval"], data["evaluation_protocol"]) == (
                METHODS[method], env_name, seed, TOTAL_STEPS, 25000,
                "locked_full_simplex_v1")
    assert data["training_interactions"] == TOTAL_STEPS
    assert data["environment_initialization_probe_steps"] == 2
    assert data["evaluation_interactions"] <= spec["num_modes"] * len(STEPS) * 21 * 3 * 1000
    assert len(data["pareto_per_mode"]) == spec["num_modes"]
    assert [snapshot["step"] for snapshot in data["policy_archive_snapshots"]] == list(STEPS)
    assert [snapshot["step"] for snapshot in old["policy_archive_snapshots"]] == list(range(5000, 50001, 2500))
    for key in ("evaluation_weights", "episode_seeds_per_mode", "ref_point"):
        assert data[key] == old[key], key
    assert [tuple(entry) for entry in data["ground_truth"]["mode_history"]
            if entry[0] <= 50000] == [tuple(entry) for entry in old["ground_truth"]["mode_history"]]
    prefix_exact = policy_prefix_equal(old_model, new_model)
    return {"protocol": PROTOCOL, "env": env_name, "method": method,
            "seed": seed, "complete": prefix_exact, "prefix_exact": prefix_exact,
            "validation_issue": None if prefix_exact else "50k_policy_prefix_mismatch",
            "old50": summarize_run(old), "new75": summarize_run(data),
            "completed_budget": {key: data[key] for key in (
                "training_interactions", "evaluation_interactions",
                "environment_initialization_probe_steps")}}


def run(env_name, method, seed, root=ROOT):
    assert env_name in extension.ENVIRONMENTS and method in METHODS and seed in SEEDS
    folder = Path(root) / env_name.replace("-", "_") / method
    folder.mkdir(parents=True, exist_ok=True)
    output = folder / f"seed{seed}.json"
    raw_path = folder / f"{METHODS[method]}_seed{seed}.json"
    model_path = folder / f"{METHODS[method]}_seed{seed}_model.pt"
    old_dir = old_folder(env_name, method)
    old_path = old_dir / f"{METHODS[method]}_seed{seed}.json"
    old_model = old_dir / f"{METHODS[method]}_seed{seed}_model.pt"
    if output.exists():
        saved = json.loads(output.read_text())
        if saved.get("complete"):
            assert (saved["protocol"], saved["env"], saved["method"], saved["seed"]) == (
                PROTOCOL, env_name, method, seed)
            return saved
        if not raw_path.exists():
            raise RuntimeError(f"Incomplete attempt retained at {output}; use a new root for a retry")
    else:
        output.write_text(json.dumps({"protocol": PROTOCOL, "env": env_name,
                                      "method": method, "seed": seed, "complete": False}) + "\n")
    if raw_path.exists():
        data = json.loads(raw_path.read_text())
    elif method == "v25":
        data = training.train_method(env_name, METHODS[method], TOTAL_STEPS, 25000,
                                     seed, str(folder), archive_steps=STEPS,
                                     **pilot.TRAIN_KWARGS)
    else:
        training.train_capql(env_name, TOTAL_STEPS, 25000, seed, str(folder),
                             eval_freq=100000, device="cpu", method=METHODS[method])
        data = json.loads(raw_path.read_text())
    old = json.loads(old_path.read_text())
    report = validate(data, old, env_name, method, seed, old_model, model_path)
    output.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    if report["complete"]:
        print(f"DONE {PROTOCOL} {env_name} {method} seed={seed} "
              f"archive_hv={report['new75']['archive_hv']:.6f}", flush=True)
    else:
        print(f"INVALID {PROTOCOL} {env_name} {method} seed={seed} "
              f"issue={report['validation_issue']}", flush=True)
    return report


def summarize(root=ROOT):
    result = {"protocol": PROTOCOL, "cohort": "inspected_50k_seeds",
              "primary": "v25 minus CAPQL archive HV at 75k",
              "archive_steps": list(STEPS), "environments": {},
              "completed_budget": {"training_interactions": 0,
                                   "evaluation_interactions": 0,
                                   "environment_initialization_probe_steps": 0}}
    for env_name, spec in extension.ENVIRONMENTS.items():
        rows = []
        for seed in SEEDS:
            pair = {method: json.loads((Path(root) / env_name.replace("-", "_") /
                                        method / f"seed{seed}.json").read_text())
                    for method in METHODS}
            assert all((row["protocol"], row["env"], row["method"], row["seed"],
                        row["complete"], row["prefix_exact"]) ==
                       (PROTOCOL, env_name, method, seed, True, True)
                       for method, row in pair.items())
            for row in pair.values():
                for key in result["completed_budget"]:
                    result["completed_budget"][key] += row["completed_budget"][key]
            rows.append({"seed": seed, **pair})
        comparisons = {}
        for metric in ("archive_hv", "final_hv"):
            comparisons[metric] = paired(
                [row["v25"]["new75"][metric] for row in rows],
                [row["capql"]["new75"][metric] for row in rows])
        comparisons["final_utility"] = paired(
            [row["v25"]["new75"]["final_utility"]["mean"] for row in rows],
            [row["capql"]["new75"]["final_utility"]["mean"] for row in rows])
        growth = {method: {metric: paired(
            [row[method]["new75"][metric] for row in rows],
            [row[method]["old50"][metric] for row in rows])
            for metric in ("archive_hv", "final_hv")}
            for method in METHODS}
        for method in METHODS:
            growth[method]["final_utility"] = paired(
                [row[method]["new75"]["final_utility"]["mean"] for row in rows],
                [row[method]["old50"]["final_utility"]["mean"] for row in rows])
        result["environments"][env_name] = {
            "seeds": list(SEEDS), "runs": rows, "paired75": comparisons,
            "within_method_50_to_75": growth,
            "per_mode_archive_mean_delta75": [mean(
                row["v25"]["new75"]["mode_archive_hv"][mode] -
                row["capql"]["new75"]["mode_archive_hv"][mode] for row in rows)
                for mode in range(spec["num_modes"])],
        }
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--env", choices=extension.ENVIRONMENTS)
    parser.add_argument("--method", choices=METHODS)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--summarize", action="store_true")
    args = parser.parse_args()
    if args.summarize:
        output = args.root / "readout.json"
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(summarize(args.root), indent=2, allow_nan=False) + "\n")
        print(f"DONE {PROTOCOL} summary {output}")
    else:
        assert args.env and args.method and args.seed is not None
        run(args.env, args.method, args.seed, args.root)
