"""Compare completed impulse routing with fixed branches and the previous pilot."""

import argparse
import json
from pathlib import Path

import summarize_budgeted_router as common


REPRODUCTION_FIELDS = (
    "episode_rewards", "ground_truth", "policy_archive_snapshots", "evaluation_weights",
    "episode_seeds_per_mode", "ref_point", "pareto_per_mode", "hv_per_mode", "final_hv_per_mode",
)


def reproduction_check(current, previous):
    checks = {key: current[key] == previous[key] for key in REPRODUCTION_FIELDS}
    checks["routing_cases"] = current["update_routing"]["cases"] == previous["update_routing"]["cases"]
    assert all(checks.values()), f"Fixed-on reproduction differs: {checks}"
    return checks


def summarize(root, previous_root):
    report = common.summarize(root, common.IMPULSE_PROTOCOL)
    assert report["completed"] == report["expected"] == 9, report["missing"]
    previous_path = previous_root.parent / f"{previous_root.name}_summary.json"
    previous = json.loads(previous_path.read_text())
    assert previous["protocol"] == common.PERSISTENT_PROTOCOL
    assert previous["completed"] == previous["expected"] == 15
    report["previous_summary"] = str(previous_path)
    report["fixed_on_reproduction"] = {}
    report["route_off_equal_logged_trajectory_and_front"] = {}
    for seed in common.SEEDS:
        name = f"{common.pilot.forks.METHOD}_seed{seed}.json"
        current = json.loads((root / "forced_on" / name).read_text())
        old = json.loads((previous_root / "forced_on" / name).read_text())
        report["fixed_on_reproduction"][seed] = reproduction_check(current, old)
        route, off = [json.loads((root / c / name).read_text()) for c in ("route", "forced_off")]
        report["route_off_equal_logged_trajectory_and_front"][seed] = all(
            route[key] == off[key] for key in REPRODUCTION_FIELDS)
    report["cross_pilot_route_minus"] = {}
    report["previous_condition_metrics"] = {}
    for condition in ("route", "train_on", "train_off"):
        old = previous["conditions"][condition]
        runs = {int(k): v for k, v in old["per_seed"].items()}
        report["cross_pilot_route_minus"][condition] = common.paired_comparison(
            report["conditions"]["route"]["per_seed"], runs, f"previous/{condition}")
        report["previous_condition_metrics"][condition] = {
            k: old[k] for k in ("archive_hv_mean", "final_hv_mean", "final_hv_min")}
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path("results/iclr_impulse_router_20260920"))
    parser.add_argument("--previous-root", type=Path, default=Path("results/iclr_budgeted_router_20260920"))
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = summarize(args.root, args.previous_root)
    args.output.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    print(json.dumps({k: report[k] for k in ("completed", "expected", "missing", "total_training_interactions")}))
