"""Server-side matched CAPQL archive readout for the frozen scoped extension."""

import argparse
import json
from pathlib import Path
from statistics import mean, median

import run_scoped_extension as extension


PROTOCOL = "capql_archive_extension_v1"
ROOT = Path("results/iclr_capql_archive_extension_20260926")
REFERENCE = extension.ROOT
CONDITIONS = ("legacy_adaptive", "legacy_blind", "scoped_adaptive")


def paired(values, controls):
    deltas = [value - control for value, control in zip(values, controls)]
    return {
        "mean_delta": mean(deltas),
        "median_delta": median(deltas),
        "relative_mean_delta": mean(values) / mean(controls) - 1,
        "wins": sum(delta > 0 for delta in deltas),
        "losses": sum(delta < 0 for delta in deltas),
        "ties": sum(delta == 0 for delta in deltas),
        "paired_deltas": deltas,
    }


def summarize(root=ROOT, reference=REFERENCE):
    result = {"protocol": PROTOCOL, "primary": "legacy_adaptive minus capql_archive, "
              "mode-mean archive HV", "environments": {}, "completed_budget": {
                  "training_interactions": 0, "evaluation_interactions": 0,
                  "environment_initialization_probe_steps": 0}}
    for env_name, spec in extension.ENVIRONMENTS.items():
        env_dir = env_name.replace("-", "_")
        rows = []
        for seed in spec["seeds"]:
            capql_path = Path(root) / env_dir / f"capql_archive_seed{seed}.json"
            bamor_path = (Path(reference) / env_dir / "legacy_adaptive" /
                          f"{extension.METHODS['adaptive']}_seed{seed}.json")
            capql = json.loads(capql_path.read_text())
            bamor = json.loads(bamor_path.read_text())
            assert (capql["method"], capql["env"], capql["seed"], capql["total_steps"],
                    capql["switch_interval"], capql["evaluation_protocol"]) == (
                        "capql_archive", env_name, seed, 50000, 25000,
                        "locked_full_simplex_v1")
            assert [entry["step"] for entry in capql["policy_archive_snapshots"]] == [
                entry["step"] for entry in bamor["policy_archive_snapshots"]]
            for key in ("evaluation_weights", "episode_seeds_per_mode", "ref_point"):
                assert capql[key] == bamor[key], (env_name, seed, key)
            assert capql["ground_truth"]["mode_history"] == bamor["ground_truth"]["mode_history"]
            assert len(capql["hv_per_mode"]) == spec["num_modes"]
            assert capql["training_interactions"] == 50000
            assert capql["environment_initialization_probe_steps"] == 2
            assert capql["evaluation_interactions"] <= extension.evaluation_limit(env_name)
            for key in result["completed_budget"]:
                result["completed_budget"][key] += capql[key]
            row = {"seed": seed, "capql_archive_hv": capql["hv_mean_mode"],
                   "capql_final_hv": capql["final_hv_mean_mode"],
                   "capql_mode_hv": capql["hv_per_mode"],
                   "training_seconds": capql["training_seconds"],
                   "evaluation_seconds": capql["evaluation_seconds"]}
            for condition in CONDITIONS:
                path = (Path(reference) / env_dir / condition /
                        f"{extension.METHODS[condition.split('_')[1]]}_seed{seed}.json")
                data = bamor if condition == "legacy_adaptive" else json.loads(path.read_text())
                assert data["evaluation_weights"] == capql["evaluation_weights"]
                assert data["episode_seeds_per_mode"] == capql["episode_seeds_per_mode"]
                assert data["ref_point"] == capql["ref_point"]
                assert data["ground_truth"]["mode_history"] == capql["ground_truth"]["mode_history"]
                row[condition] = {"archive_hv": data["hv_mean_mode"],
                                  "final_hv": data["final_hv_mean_mode"],
                                  "mode_hv": data["hv_per_mode"],
                                  "training_seconds": data["training_seconds"],
                                  "evaluation_seconds": data["evaluation_seconds"]}
            rows.append(row)
        comparison = {}
        for condition in CONDITIONS:
            comparison[condition] = {
                metric: paired([row[condition][metric] for row in rows],
                               [row[f"capql_{metric}"] for row in rows])
                for metric in ("archive_hv", "final_hv")}
            comparison[condition]["per_mode_archive_mean_delta"] = [
                mean(row[condition]["mode_hv"][mode] - row["capql_mode_hv"][mode]
                     for row in rows) for mode in range(spec["num_modes"])]
        result["environments"][env_name] = {
            "seeds": list(spec["seeds"]), "runs": rows, "paired": comparison,
            "capql_archive_mean": mean(row["capql_archive_hv"] for row in rows),
            "capql_final_mean": mean(row["capql_final_hv"] for row in rows)}
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--reference", type=Path, default=REFERENCE)
    parser.add_argument("--output", type=Path, default=ROOT / "readout.json")
    args = parser.parse_args()
    readout = summarize(args.root, args.reference)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(readout, indent=2, allow_nan=False) + "\n")
    print("DONE", PROTOCOL, len(readout["environments"]), "environments", flush=True)
