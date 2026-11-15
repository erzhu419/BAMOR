"""Paired update/selector effects without a zero-regression archive veto."""

import argparse
import json
from pathlib import Path
from statistics import mean, median

import numpy as np

import run_scoped_adaptive_confirmation as experiment


METRICS = ("final_hv", "late_mean_hv", "archive_hv")
CONTRASTS = {
    "update_under_blind": ("scoped_blind", "legacy_blind"),
    "update_under_adaptive": ("scoped_adaptive", "legacy_adaptive"),
    "selector_under_legacy": ("legacy_adaptive", "legacy_blind"),
    "selector_under_scoped": ("scoped_adaptive", "scoped_blind"),
}


def summarize(reports, *, seeds=experiment.SEEDS, protocol=experiment.PROTOCOL,
              evaluation_limit=None, source_protocols=None):
    lookup = {(r["condition"], r["seed"]): r for r in reports}
    assert len(lookup) == len(reports) == len(experiment.CONDITIONS) * len(seeds)
    assert set(lookup) == {(c, s) for c in experiment.CONDITIONS for s in seeds}
    num_modes = len(reports[0]["metrics"]["final_mode_hv"])
    for (condition, seed), report in lookup.items():
        assert report["complete"] and report["protocol"] in (source_protocols or (protocol,))
        for key, expected in experiment.BUDGET.items():
            actual = report["completed_budget"][key]
            if key == "evaluation_interactions" and evaluation_limit is not None:
                assert num_modes * 19 * 21 * 3 <= actual <= evaluation_limit
            else:
                assert actual == expected, key
        base = lookup[("legacy_blind", seed)]
        assert report["evaluation"] == base["evaluation"]
        assert report["warmup"] == base["warmup"], "Training differs before the scoped intervention"
        assert report["metrics"]["mode_history"] == base["metrics"]["mode_history"]
    conditions = {c: {f"{k}_{stat}": fn(lookup[c, s]["metrics"][k] for s in seeds)
                     for k in METRICS for stat, fn in (("mean", mean), ("min", min))}
                  for c in experiment.CONDITIONS}
    contrasts = {}
    # Resample whole paired training seeds, never modes, weights, or snapshots.
    indices = np.random.default_rng(0).integers(0, len(seeds), (20000, len(seeds)))
    for name, (candidate, control) in CONTRASTS.items():
        paired = {}
        for seed in seeds:
            a, b = lookup[candidate, seed], lookup[control, seed]
            weights = a["evaluation"]["evaluation_weights"]
            paired[seed] = {
                **experiment.scoped.matched.paired_difference(a["metrics"], b["metrics"], weights),
                **{k: a["metrics"][k] - b["metrics"][k] for k in METRICS},
                "archive_mode_hv": (np.asarray(a["metrics"]["mode_hv"])
                                    - np.asarray(b["metrics"]["mode_hv"])).tolist(),
                "different_postwarmup_arms": sum(x != y for x, y in zip(
                    a["selector"]["operators"][10:], b["selector"]["operators"][10:]))}
        deltas = [p["final_hv"] for p in paired.values()]
        contrasts[name] = dict(per_seed=paired,
            mean={k: mean(p[k] for p in paired.values()) for k in METRICS},
            final_median=median(deltas), final_wins=sum(d > 0 for d in deltas),
            final_losses=sum(d < 0 for d in deltas),
            final_mean_paired_bootstrap_95=np.quantile(np.asarray(deltas)[indices].mean(axis=1), [.025, .975]).tolist(),
            mode_mean_matched_utility=[mean(p["utility_mean_per_mode"][m] for p in paired.values()) for m in range(num_modes)])
    interaction = {s: {k: contrasts["selector_under_scoped"]["per_seed"][s][k]
                        - contrasts["selector_under_legacy"]["per_seed"][s][k] for k in METRICS}
                   for s in seeds}
    return dict(protocol=protocol, seeds=list(seeds), validated_runs=len(reports),
        conditions=conditions, contrasts=contrasts, selector_by_update_interaction=interaction,
        per_run={c: {s: {k: lookup[c, s][k] for k in ("metrics", "selector", "routing")}
                     for s in seeds} for c in experiment.CONDITIONS},
        completed_run_budget={k: sum(r["completed_budget"][k] for r in reports) for k in experiment.BUDGET},
        scope="Primary update contrast: scoped_adaptive minus legacy_adaptive, final-policy HV. "
              "Adaptive evidence separately requires scoped_adaptive versus scoped_blind. Report paired "
              "effects, observed minima and fixed late snapshots; archive and per-mode utility are secondary "
              "tradeoffs, not zero-tolerance vetoes. Paired-seed bootstrap intervals are descriptive, "
              "not equivalence tests or population robustness. Prior frozen failures remain unchanged. "
              "Terminal-run budgets exclude tests and interrupted attempts, which must be counted separately.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=experiment.ROOT)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = summarize([json.loads((args.root / c / f"seed{s}.json").read_text())
                        for c in experiment.CONDITIONS for s in experiment.SEEDS])
    args.output.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    print(json.dumps({k: v["mean"] for k, v in report["contrasts"].items()}))
