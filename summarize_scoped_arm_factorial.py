"""Casewise factorial effects; no population inference from selected extremes."""

import argparse
import json
from pathlib import Path

import numpy as np

import run_scoped_arm_factorial as experiment


METRICS = ("final_hv", "late_mean_hv", "archive_hv")
CONTRASTS = {
    "envelope_without_bc": ("envelope_only", "legacy_adaptive"),
    "bc_without_envelope": ("bc_only", "legacy_adaptive"),
    "envelope_with_bc": ("scoped_adaptive", "bc_only"),
    "bc_with_envelope": ("scoped_adaptive", "envelope_only"),
    "both_vs_legacy": ("scoped_adaptive", "legacy_adaptive"),
}


def summarize(root=experiment.ROOT, reference_root=experiment.REFERENCE_ROOT):
    cases = {}
    budgets = {name: dict.fromkeys(experiment.base.BUDGET, 0) for name in ("new", "reused")}
    for env_name, seeds in experiment.CASES.items():
        cases[env_name] = {}
        for seed, role in seeds.items():
            reports = experiment.load_references(env_name, seed, reference_root)
            for condition in experiment.TARGETS:
                path = Path(root) / env_name.replace("-", "_") / condition / f"seed{seed}.json"
                report = json.loads(path.read_text())
                assert (report["protocol"], report["env"], report["condition"], report["seed"], report["complete"]) == (
                    experiment.PROTOCOL, env_name, condition, seed, True)
                assert report["routing"]["target_operators"] == list(experiment.TARGETS[condition])
                experiment.validate_pair(report, reports["legacy_adaptive"])
                reports[condition] = report
            for condition, report in reports.items():
                kind = "new" if condition in experiment.TARGETS else "reused"
                for k in budgets[kind]:
                    budgets[kind][k] += report["completed_budget"][k]
            contrasts = {}
            for name, (a_name, b_name) in CONTRASTS.items():
                a, b = reports[a_name], reports[b_name]
                contrasts[name] = {
                    **experiment.base.scoped.matched.paired_difference(
                        a["metrics"], b["metrics"], a["evaluation"]["evaluation_weights"]),
                    **{k: a["metrics"][k] - b["metrics"][k] for k in METRICS},
                    "relative_change_pct": {k: 100 * (a["metrics"][k] / b["metrics"][k] - 1)
                                            if b["metrics"][k] else None for k in METRICS},
                    "different_postwarmup_arms": sum(x != y for x, y in zip(
                        a["selector"]["operators"][10:], b["selector"]["operators"][10:]))}
            interaction = {k: contrasts["envelope_with_bc"][k] - contrasts["envelope_without_bc"][k]
                           for k in METRICS}
            interaction["matched_weight_utility_differences"] = (
                np.asarray(contrasts["envelope_with_bc"]["matched_weight_utility_differences"])
                - np.asarray(contrasts["envelope_without_bc"]["matched_weight_utility_differences"])).tolist()
            cases[env_name][seed] = dict(selected_role=role, contrasts=contrasts, interaction=interaction,
                per_condition={c: dict(metrics={k: v for k, v in r["metrics"].items()
                                                if k != "terminal_returns"},
                    operators=r["selector"]["operators"], routing=r["routing"]) for c, r in reports.items()})
    return dict(protocol=experiment.PROTOCOL, complete=True, cases=cases,
                new_completed_budget=budgets["new"], reused_completed_budget=budgets["reused"],
                scope=experiment.SCOPE + " Report each case without pooled raw HV, bootstrap significance, "
                      "automatic adoption or retuning thresholds. Costs exclude separately logged tests/failed attempts.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=experiment.ROOT)
    parser.add_argument("--reference-root", type=Path, default=experiment.REFERENCE_ROOT)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = summarize(args.root, args.reference_root)
    args.output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    print(json.dumps({"complete": result["complete"], "new_completed_budget": result["new_completed_budget"]}))
