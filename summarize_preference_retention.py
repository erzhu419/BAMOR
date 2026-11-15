"""Small readout for the frozen two-case preference-retention experiment."""

import argparse
import json
from pathlib import Path

import numpy as np

import run_preference_retention as experiment


def readout(root=experiment.ROOT):
    cases = {}
    for seed in experiment.credit.CASES:
        report = json.loads((Path(root) / f"seed{seed}.json").read_text())
        source, _ = experiment.load_source(seed)
        assert report["complete"] and report["seed"] == seed
        summary = experiment.finalize(report, source)
        assert report["summary"] == summary
        branches = {name: source["branches"][name] for name in ("native", "pulse")}
        branches.update(report["branches"])
        final = {name: branch["measurements"][-1] for name, branch in branches.items()}
        baseline = source["branches"]["native"]["measurements"][0]
        per_branch = {}
        for name, row in final.items():
            entry = dict(hv_mean_mode=row["hv_mean_mode"], hv_per_mode=row["hv_per_mode"],
                         matched_utility_mean=float(np.mean(row["matched_utility_per_mode"])))
            if name in experiment.CONDITIONS:
                audit = branches[name]["measurements"]
                entry["fixed_context_action_mse"] = [float(np.mean(
                    item["retention_audit"]["fixed_context_action_mse_per_template"])) for item in audit]
                entry["operator_mismatches_vs_pulse"] = summary[name]["operator_mismatches_vs_pulse"]
                entry["hv_vs_pulse_pct"] = summary[name]["versus_pulse"][-1]["hv_relative_pct"]
                entry["hv_vs_native_pct"] = summary[name]["versus_native"][-1]["hv_relative_pct"]
            per_branch[name] = entry
        coverage, current = branches["coverage"], branches["current"]
        cases[str(seed)] = dict(background=report["background"], window=report["window"],
            prefix_hv=baseline["hv_mean_mode"],
            native_final_over_prefix_hv=final["native"]["hv_mean_mode"] / baseline["hv_mean_mode"],
            prefix_checks=report["prefix_checks"], completed_budget=report["completed_budget"],
            retention_updates={name: branch["routing"]["retention"]["loss_updates"]
                               for name, branch in (("coverage", coverage), ("current", current))},
            branches=per_branch,
            coverage_vs_current_hv_pct=[row["hv_relative_pct"]
                                        for row in summary["coverage_vs_current"]],
            coverage_vs_current_final_eum_pct=summary["coverage_vs_current"][-1]["front_eum"]["relative_pct"],
            coverage_vs_current_final_mode_hv_delta=summary["coverage_vs_current"][-1]["hv_delta_per_mode"])
    budget = {key: sum(case["completed_budget"][key] for case in cases.values())
              for key in next(iter(cases.values()))["completed_budget"]}
    return dict(protocol="preference_retention_readout_v1", complete=True, cases=cases,
                completed_experiment_budget=budget, readout_training_interactions=0,
                readout_evaluation_interactions=0,
                scope="Outcome-selected Hopper cases. Fixed teacher from the intervention start, "
                      "coefficient 1, seven templates or seven copies of the current preference. "
                      "Identical future operator identities do not imply identical replay, critic or "
                      "encoder states. Fixed-context action MSE is a proxy, not a return metric.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=experiment.ROOT)
    parser.add_argument("--output", type=Path, default=experiment.ROOT / "readout.json")
    args = parser.parse_args()
    result = readout(args.root)
    args.output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    for seed, case in result["cases"].items():
        print(json.dumps(dict(seed=seed, final_hv={name: b["hv_mean_mode"]
            for name, b in case["branches"].items()},
            coverage_vs_current_pct=case["coverage_vs_current_hv_pct"][-1],
            final_action_mse={name: case["branches"][name]["fixed_context_action_mse"][-1]
                              for name in experiment.CONDITIONS})))
    print(json.dumps(dict(budget=result["completed_experiment_budget"],
                          bytes=args.output.stat().st_size)))
