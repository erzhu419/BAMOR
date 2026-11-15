"""Fresh 75k final-policy comparison with seeded CAPQL and matched evaluation."""

import argparse
import json
from pathlib import Path
from statistics import mean

import numpy as np

import run_aligned_progress_pilot as pilot
import run_scoped_extension as extension
from summarize_capql_archive_extension import paired
import train_bamor_mujoco as training


PROTOCOL = "final_policy_75k_fresh_v1"
ROOT = Path("results/iclr_75k_final_fresh_20260926")
TOTAL_STEPS = 75000
SEEDS = tuple(range(1201, 1206))
METHODS = {"v25": extension.METHODS["adaptive"], "capql": "capql"}


def direct_utility(data):
    weights = np.asarray(data["evaluation_weights"], dtype=float)
    per_mode = []
    for front in data["pareto_per_mode"]:
        assert len(front) == len(weights)
        assert all(np.allclose(point["weight"], weight, atol=1e-7)
                   for point, weight in zip(front, weights))
        per_mode.append(mean(float(np.dot(weight, point["return"]))
                             for point, weight in zip(front, weights)))
    return {"mean": mean(per_mode), "per_mode": per_mode}


def validate(data, env_name, method, seed, model_path):
    spec = extension.ENVIRONMENTS[env_name]
    training_method = (method if method in training.COMMON_TARGET_METHODS | {"capql_full_simplex"}
                       else METHODS[method])
    assert (data["method"], data["env"], data["seed"], data["total_steps"],
            data["switch_interval"], data["evaluation_protocol"]) == (
                training_method, env_name, seed, TOTAL_STEPS, 25000,
                "locked_full_simplex_v1")
    assert data["training_interactions"] == TOTAL_STEPS
    assert data["environment_initialization_probe_steps"] == 2
    assert 0 < data["evaluation_interactions"] <= spec["num_modes"] * 21 * 3 * 1000
    assert len(data["pareto_per_mode"]) == spec["num_modes"]
    assert len(data["evaluation_weights"]) == 21
    assert all(len(weight) == spec["reward_dim"] for weight in data["evaluation_weights"])
    assert model_path.exists()
    if method == "v25" or method in training.COMMON_TARGET_METHODS:
        assert [item["step"] for item in data["policy_archive_snapshots"]] == [TOTAL_STEPS]
        assert all(point["archive_step"] == TOTAL_STEPS
                   for front in data["pareto_per_mode"] for point in front)
    else:
        assert data["policy_archive_snapshots"] is None
        assert data["train_seed"] == seed
        assert all("archive_step" not in point
                   for front in data["pareto_per_mode"] for point in front)
    assert np.isclose(data["hv_mean_mode"], data["final_hv_mean_mode"])
    return {
        "protocol": PROTOCOL, "env": env_name, "method": method,
        "seed": seed, "complete": True,
        "metrics": {"final_hv": data["final_hv_mean_mode"],
                    "mode_final_hv": data["final_hv_per_mode"],
                    "direct_utility": direct_utility(data),
                    "training_seconds": data["training_seconds"],
                    "evaluation_seconds": data["evaluation_seconds"]},
        "evaluation": {key: data[key] for key in (
            "evaluation_weights", "episode_seeds_per_mode", "ref_point")},
        "mode_history": data["ground_truth"]["mode_history"],
        "completed_budget": {key: data[key] for key in (
            "training_interactions", "evaluation_interactions",
            "environment_initialization_probe_steps")},
    }


def run(env_name, method, seed, root=ROOT):
    assert env_name in extension.ENVIRONMENTS and method in METHODS and seed in SEEDS
    folder = Path(root) / env_name.replace("-", "_") / method
    folder.mkdir(parents=True, exist_ok=True)
    output = folder / f"seed{seed}.json"
    raw = folder / f"{METHODS[method]}_seed{seed}.json"
    model = folder / f"{METHODS[method]}_seed{seed}_model.pt"
    if output.exists():
        saved = json.loads(output.read_text())
        if saved.get("complete"):
            assert (saved["protocol"], saved["env"], saved["method"], saved["seed"]) == (
                PROTOCOL, env_name, method, seed)
            return saved
        if not raw.exists():
            raise RuntimeError(f"Incomplete run at {output}; inspect before retrying")
    else:
        output.write_text(json.dumps({"protocol": PROTOCOL, "env": env_name,
                                      "method": method, "seed": seed,
                                      "complete": False}) + "\n")
    if not raw.exists():
        if method == "v25":
            training.train_method(env_name, METHODS[method], TOTAL_STEPS, 25000,
                                  seed, str(folder), archive_steps=(TOTAL_STEPS,),
                                  **pilot.TRAIN_KWARGS)
        else:
            training.train_capql(env_name, TOTAL_STEPS, 25000, seed, str(folder),
                                 eval_freq=100000, device="cpu", method="capql")
    report = validate(json.loads(raw.read_text()), env_name, method, seed, model)
    output.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    print(f"DONE {PROTOCOL} {env_name} {method} seed={seed} "
          f"final_hv={report['metrics']['final_hv']:.6f}", flush=True)
    return report


def summarize(root=ROOT):
    result = {"protocol": PROTOCOL, "cohort": "fresh_1201_1205",
              "primary": "v25 minus seeded CAPQL final-actor HV at 75k",
              "environments": {}, "completed_budget": {
                  "training_interactions": 0, "evaluation_interactions": 0,
                  "environment_initialization_probe_steps": 0}}
    for env_name, spec in extension.ENVIRONMENTS.items():
        rows = []
        for seed in SEEDS:
            pair = {method: json.loads((Path(root) / env_name.replace("-", "_") /
                                        method / f"seed{seed}.json").read_text())
                    for method in METHODS}
            assert all((row["protocol"], row["env"], row["method"], row["seed"],
                        row["complete"]) == (PROTOCOL, env_name, method, seed, True)
                       for method, row in pair.items())
            assert pair["v25"]["evaluation"] == pair["capql"]["evaluation"]
            assert pair["v25"]["mode_history"] == pair["capql"]["mode_history"]
            for row in pair.values():
                for key in result["completed_budget"]:
                    result["completed_budget"][key] += row["completed_budget"][key]
            rows.append({"seed": seed, **pair})
        comparisons = {}
        for metric in ("final_hv",):
            comparisons[metric] = paired(
                [row["v25"]["metrics"][metric] for row in rows],
                [row["capql"]["metrics"][metric] for row in rows])
        comparisons["direct_utility"] = paired(
            [row["v25"]["metrics"]["direct_utility"]["mean"] for row in rows],
            [row["capql"]["metrics"]["direct_utility"]["mean"] for row in rows])
        result["environments"][env_name] = {
            "seeds": list(SEEDS), "runs": rows, "paired": comparisons,
            "mode_final_hv_mean_delta": [mean(
                row["v25"]["metrics"]["mode_final_hv"][mode] -
                row["capql"]["metrics"]["mode_final_hv"][mode] for row in rows)
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
