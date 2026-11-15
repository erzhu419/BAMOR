"""Print small server-side summaries; leave raw fronts and checkpoints in place."""
import argparse
import json
from pathlib import Path
from statistics import mean


METHODS = ["bamor_v23_extreme_hier_archive_mix",
           "bamor_v23_fixed_selector_archive_mix", "capql_archive"]
ENVS = ["mo-halfcheetah-v5", "mo-ant-v5", "mo-hopper-v5"]


def seed_diagnostics(data):
    truth = data["ground_truth"]
    history = truth["mode_history"]
    mode_steps = [0] * truth["num_modes"]
    for index, (start, mode) in enumerate(history):
        end = history[index + 1][0] if index + 1 < len(history) else data["total_steps"]
        mode_steps[mode] += end - start
    result = {
        "archive_hv": data["hv_mean_mode"],
        "final_hv": data["final_hv_mean_mode"],
        "archive_gain": data["hv_mean_mode"] - data["final_hv_mean_mode"],
        "mode_hv": data["hv_per_mode"],
        "final_mode_hv": data["final_hv_per_mode"],
        "mode_eum": data["eum_per_mode"],
        "training_switches": len(history) - 1,
        "training_steps_per_mode": mode_steps,
    }
    for group, seen in [("seen", True), ("unseen", False)]:
        indices = [mode for mode, steps in enumerate(mode_steps) if (steps > 0) == seen]
        result[f"{group}_mode_hv"] = (
            mean(data["hv_per_mode"][mode] for mode in indices) if indices else None)
    selector = data.get("adaptive_operator_final")
    if selector:
        result["family_steps"] = selector["family_step_counts"]
    return result


def summarize(root, seeds, methods=METHODS, environments=ENVS):
    output = {"protocol": "locked_full_simplex_v1", "completed": 0,
              "expected": len(environments) * len(methods) * len(seeds),
              "reference_method": methods[0], "environments": {}}
    for env in environments:
        records = {}
        missing = []
        for method in methods:
            records[method] = {}
            for seed in seeds:
                path = root / env.replace("-", "_") / f"{method}_seed{seed}.json"
                if not path.exists():
                    missing.append(f"{method}/seed{seed}")
                    continue
                data = json.loads(path.read_text())
                if data.get("evaluation_protocol") != output["protocol"]:
                    raise ValueError(f"Wrong evaluation protocol: {path}")
                steps = [entry["step"] for entry in data["policy_archive_snapshots"]]
                if steps != list(range(5000, 50001, 2500)):
                    raise ValueError(f"Incomplete or mismatched archive: {path}")
                records[method][seed] = data
                output["completed"] += 1
        summary = {"missing": missing, "methods": {}, "paired": {}}
        for method, runs in records.items():
            if not runs:
                continue
            summary["methods"][method] = {
                "seeds": sorted(runs),
                "per_seed": {seed: seed_diagnostics(data) for seed, data in runs.items()},
                "mean_mode_hv": mean(r["hv_mean_mode"] for r in runs.values()),
                "final_mean_mode_hv": mean(r["final_hv_mean_mode"] for r in runs.values()),
                "mode_hv_means": [mean(r["hv_per_mode"][m] for r in runs.values())
                                  for m in range(len(next(iter(runs.values()))["hv_per_mode"]))],
                "training_seconds": mean(r["training_seconds"] for r in runs.values()),
                "evaluation_seconds": mean(r["evaluation_seconds"] for r in runs.values()),
            }
        adaptive = records[methods[0]]
        for comparator in methods[1:]:
            common = sorted(adaptive.keys() & records[comparator].keys())
            if not common:
                continue
            deltas = []
            final_deltas = []
            mode_deltas = []
            exposure_deltas = {"seen": [], "unseen": []}
            for seed in common:
                a, b = adaptive[seed], records[comparator][seed]
                for key in ["evaluation_weights", "episode_seeds_per_mode", "ref_point"]:
                    if a[key] != b[key]:
                        raise ValueError(f"Unmatched {key}: {env}, seed {seed}")
                if a["ground_truth"]["mode_history"] != b["ground_truth"]["mode_history"]:
                    raise ValueError(f"Unmatched training mode schedule: {env}, seed {seed}")
                deltas.append(a["hv_mean_mode"] - b["hv_mean_mode"])
                final_deltas.append(a["final_hv_mean_mode"] - b["final_hv_mean_mode"])
                mode_deltas.append([x - y for x, y in zip(a["hv_per_mode"], b["hv_per_mode"])])
                a_diagnostics, b_diagnostics = seed_diagnostics(a), seed_diagnostics(b)
                for group in exposure_deltas:
                    x, y = a_diagnostics[f"{group}_mode_hv"], b_diagnostics[f"{group}_mode_hv"]
                    exposure_deltas[group].append(x - y if x is not None else None)
            summary["paired"][comparator] = {
                "seeds": common, "adaptive_minus_comparator": deltas,
                "final_actor_differences": final_deltas,
                "mode_hv_differences": mode_deltas,
                "seen_mode_differences": exposure_deltas["seen"],
                "unseen_mode_differences": exposure_deltas["unseen"],
                "mean_difference": mean(deltas), "wins": sum(d > 0 for d in deltas),
            }
        output["environments"][env] = summary
    return output


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path,
                        default=Path("results/iclr_matched_v23_pilot_20260913"))
    parser.add_argument("--seeds", type=int, nargs="+", default=[20, 21, 22])
    parser.add_argument("--methods", nargs="+", default=METHODS,
                        help="First method is the paired-comparison reference.")
    parser.add_argument("--envs", nargs="+", default=ENVS)
    args = parser.parse_args()
    print(json.dumps(summarize(args.root, args.seeds, args.methods, args.envs), indent=2))
