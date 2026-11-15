"""Score fresh-reset decisions against cached independent-reset outcomes."""

import argparse
import json
from pathlib import Path

import numpy as np

from diagnose_raw_return_forks import raw_scores
from summarize_update_forks import HORIZONS, WEIGHTS, metrics


def compare(rows):
    # All score prefixes predict the same 512-step outcome; 512 is primary.
    return {str(h): metrics([{**r, "surrogate": r["raw_scores"][hi]} for r in rows], 2)
            for hi, h in enumerate(HORIZONS)}


def summarize(root, cached_root, old_summary):
    prior = json.loads(old_summary.read_text())
    old_rows = {(r["seed"], r["step"], r["weight_index"]): r for r in prior["rows"]}
    rows, nulls, reproduction, costs = [], [], {}, {}
    for seed in (26, 27, 28):
        data = json.loads((root / f"seed{seed}.json").read_text())
        cached = json.loads((cached_root / f"seed{seed}.json").read_text())
        if not data["complete"] or not data["base_reproduction"]["matched"]:
            raise ValueError(f"Incomplete or unmatched main trajectory: seed {seed}")
        assert data["protocol"] == "raw_return_fork_scores_v1" and data["seed"] == seed
        assert data["steps"] == [c["step"] for c in data["cases"]] == list(range(10000, 50001, 5000))
        assert data["horizons"] == HORIZONS and data["primary_horizon"] == 512
        assert data["score_repeats"] == 3 and data["score_seed_offset"] == 20000
        np.testing.assert_array_equal(data["weights"], WEIGHTS)
        originals = {c["step"]: c for c in cached["cases"]}
        reproduction[seed], costs[seed] = data["base_reproduction"], data["cost"]
        assert costs[seed] == {"main_environment_steps": 50000, "extra_environment_steps": 82944,
                               "fork_updates": 2304, "diagnostic_resets": 162}
        for case in data["cases"]:
            original = originals[case["step"]]
            assert case["matched_original_losses"] and case["config"] == original["config"]
            assert case["cached_return_effects"] == original["effects"]
            expected_seeds = [10000000 + seed * 100000 + case["step"] * 100 + 20000 + r
                              for r in range(3)]
            assert case["score_reset_seeds"] == expected_seeds
            assert case["heldout_reset_seeds"] == original["reset_seeds"]
            assert set(expected_seeds).isdisjoint(case["heldout_reset_seeds"])
            assert case["extra_environment_steps"] == 9216
            assert case["diagnostic_resets"] == 18 and case["fork_updates"] == 256
            for values in case["score_returns"].values():
                assert np.shape(values) == (3, 3, 3, 2)
            recomputed = raw_scores(case["score_returns"])
            assert recomputed == case["scores"]
            if case["config"]["envelope_policy_candidates"] == 1:
                assert not np.any(recomputed["mean_scalar_on_minus_off"])
                nulls.append({"seed": seed, "step": case["step"]})
                continue
            for wi in range(3):
                row = old_rows[(seed, case["step"], wi)].copy()
                for effect in ("on_minus_off", "on_minus_base", "off_minus_base"):
                    np.testing.assert_allclose(row[effect], original["effects"][effect][
                        "mean_scalar_by_weight_horizon"][wi], rtol=1e-12, atol=1e-12)
                row["raw_scores"] = recomputed["mean_scalar_on_minus_off"][wi]
                rows.append(row)
    assert len(rows) == len(old_rows) and nulls == prior["identity_controls"]
    per_seed = {str(s): compare([r for r in rows if r["seed"] == s]) for s in (26, 27, 28)}
    macro = {str(h): {field: float(np.mean([v[str(h)][field] for v in per_seed.values()]))
                     for field in ("mean_selected_minus_off", "mean_selected_minus_on",
                                   "mean_selected_minus_base")} for h in HORIZONS}
    return {"protocol": "raw_return_fork_summary_v1", "primary_score_horizon": 512,
            "outcome_horizon": 512, "active_positions": len(rows) // 3,
            "identity_controls": nulls, "base_reproduction": reproduction, "costs": costs,
            "pooled": compare(rows), "per_seed": per_seed, "seed_macro": macro,
            "by_preference": {str(w): compare([r for r in rows if r["weight_index"] == w])
                              for w in range(3)}, "rows": rows,
            "scope": "Strict positive fresh-reset score chooses on, otherwise off. Cached "
                     "outcome resets are disjoint from scoring resets. Comparisons are correlated "
                     "development probes, not independent seeds, deployed routing, or retention."}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path("results/iclr_raw_return_forks_20260920"))
    parser.add_argument("--cached-root", type=Path, default=Path("results/iclr_update_forks_20260919"))
    parser.add_argument("--old-summary", type=Path,
                        default=Path("results/iclr_update_forks_20260919_summary.json"))
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = summarize(args.root, args.cached_root, args.old_summary)
    args.output.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    print(json.dumps({"pooled_primary": report["pooled"]["512"],
                      "per_seed_primary": {s: d["512"] for s, d in report["per_seed"].items()},
                      "costs": report["costs"]}, indent=2))
