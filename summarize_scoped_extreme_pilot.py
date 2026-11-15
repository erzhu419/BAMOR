"""Frozen full-training development gates; no automatic adaptive deployment."""

import argparse
import json
from pathlib import Path
from statistics import mean

import run_scoped_extreme_pilot as experiment


def summarize(reports):
    assert sorted(r["seed"] for r in reports) == list(experiment.SEEDS)
    for r in reports:
        assert r["protocol"] == experiment.PROTOCOL and r["complete"]
        assert r["controls_passed"] and r["rng_alignment_passed"]
        assert r["completed_budget"] == experiment.NEW_BUDGET
    conditions = {}
    for name in ("factual", "own_matched"):
        rows = [r["branches"][name] for r in reports]
        conditions[name] = {f"{metric}_{stat}": fn(row[metric] for row in rows)
                            for metric in ("final_hv", "late_mean_hv", "archive_hv")
                            for stat, fn in (("mean", mean), ("min", min))}
    a, b = conditions["own_matched"], conditions["factual"]
    paired = {r["seed"]: r["paired_comparisons"]["own_matched_minus_factual"] for r in reports}
    mode_utility = [mean(p["utility_mean_per_mode"][m] for p in paired.values()) for m in range(4)]
    gates = dict(final_mean_improved=a["final_hv_mean"] > b["final_hv_mean"],
                 final_minimum_not_worse=a["final_hv_min"] >= b["final_hv_min"],
                 late_mean_not_worse=a["late_mean_hv_mean"] >= b["late_mean_hv_mean"],
                 archive_mean_not_worse=a["archive_hv_mean"] >= b["archive_hv_mean"],
                 every_mode_mean_matched_utility_not_worse=all(v >= 0 for v in mode_utility))
    return dict(protocol=experiment.PROTOCOL, seeds=list(experiment.SEEDS), conditions=conditions,
                paired=paired, mode_mean_matched_utility_differences=mode_utility,
                development_gates=gates, eligible_to_propose_adaptive_pilot=all(gates.values()),
                new_completed_budget={k: sum(r["completed_budget"][k] for r in reports)
                                      for k in experiment.NEW_BUDGET},
                scope="Observed development seeds only. A gate pass permits proposing an adaptive pilot, "
                      "not algorithm adoption, independent confirmation or20-seed expansion. Old decisions stay unchanged.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=experiment.ROOT)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = summarize([json.loads((args.root / f"seed{s}.json").read_text()) for s in experiment.SEEDS])
    args.output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    print(json.dumps(result["development_gates"]))
