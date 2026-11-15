"""Environment-separated contrasts; observed and new cohorts stay explicit."""

import argparse
import json
from pathlib import Path

import run_scoped_extension as experiment
import summarize_scoped_adaptive_confirmation as common


def compact(report):
    report = {**report, "contrasts": {name: {**c, "per_seed": {
        s: {k: v for k, v in row.items() if k not in (
            "terminal_return_differences", "matched_weight_utility_differences")}
        for s, row in c["per_seed"].items()}} for name, c in report["contrasts"].items()}}
    report["per_run"] = {c: {s: {
        "metrics": {k: v for k, v in row["metrics"].items() if k not in (
            "terminal_returns", "snapshot_mode_curves")},
        "targeted_updates": row["routing"]["targeted_updates"],
        "own_selected_updates": row["routing"]["own_selected_updates"],
        "operators": row["selector"]["operators"]}
        for s, row in runs.items()} for c, runs in report["per_run"].items()}
    return report


def summarize(root, reference_root):
    environments, fresh_budget, hc_reports = {}, dict.fromkeys(experiment.base.BUDGET, 0), []
    for env_name, spec in experiment.ENVIRONMENTS.items():
        reports = [json.loads((root / env_name.replace("-", "_") / c / f"seed{s}.json").read_text())
                   for c in experiment.CONDITIONS for s in spec["seeds"]]
        assert all(r["env"] == env_name for r in reports)
        report = common.summarize(reports, seeds=spec["seeds"], protocol=experiment.PROTOCOL,
                                  evaluation_limit=experiment.evaluation_limit(env_name))
        environments[env_name] = compact(report)
        for key in fresh_budget:
            fresh_budget[key] += report["completed_run_budget"][key]
        if env_name == "mo-halfcheetah-v5":
            hc_reports = reports
    old = [json.loads((reference_root / c / f"seed{s}.json").read_text())
           for c in experiment.CONDITIONS for s in experiment.base.SEEDS]
    observed = common.summarize(old)
    pooled = common.summarize(old + hc_reports, seeds=tuple(range(1006, 1026)),
        protocol=experiment.PROTOCOL, source_protocols=(experiment.base.PROTOCOL, experiment.PROTOCOL))
    return dict(protocol=experiment.PROTOCOL, environments=environments,
        halfcheetah_observed_five=compact(observed), halfcheetah_pooled_twenty=compact(pooled),
        new_completed_budget=fresh_budget, reused_completed_budget=observed["completed_run_budget"],
        scope="Primary HalfCheetah inference uses new seeds1011-1025; pooled1006-1025 is descriptive "
              "because expansion followed inspection of the first five. Hopper and Ant use five new seeds "
              "each without tuning. Compare HV within each environment, never pool its raw scale across "
              "environments. All four factorial arms and negative results are retained. Existing presets "
              "include reward caps/scaling, so this is not an unmodified-MuJoCo benchmark. Tests and "
              "interrupted attempts are separate from completed-run budgets.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=experiment.ROOT)
    parser.add_argument("--reference-root", type=Path, default=experiment.base.ROOT)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = summarize(args.root, args.reference_root)
    args.output.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    print(json.dumps({env: r["contrasts"]["update_under_adaptive"]["mean"]
                      for env, r in report["environments"].items()}))
