"""Compare fork-loss components with cached, paired rollout effects on the server."""

import argparse
import json
from pathlib import Path

import numpy as np

from summarize_update_forks import HORIZONS, WEIGHTS, metrics


COMPONENTS = ("negative_q", "entropy_loss", "bc_loss", "total")
SIGNALS = COMPONENTS + ("deterministic_replay_q", "deterministic_reset_q")
DECISIONS = ("total", "negative_q", "deterministic_replay_q", "deterministic_reset_q")


def group_summary(rows):
    signals = {key: np.array([r["signals"][key] for r in rows]) for key in SIGNALS}
    components = {key: {"mean": float(v.mean()), "min": float(v.min()),
                        "max": float(v.max()), "positive": int((v > 0).sum()),
                        "zero": int((v == 0).sum())} for key, v in signals.items()}
    total = signals["total"]
    flips = {key: int(((total > 0) != ((total - signals[key]) > 0)).sum())
             for key in ("entropy_loss", "bc_loss")}
    decisions = {}
    for key in DECISIONS:
        selected = [{**r, "surrogate": r["signals"][key]} for r in rows]
        decisions[key] = {}
        for hi, horizon in enumerate(HORIZONS):
            m = metrics(selected, hi)
            decisions[key][str(horizon)] = {
                "choose_on": m["surrogate_choose_on"], "ties": m["surrogate_ties"],
                "sign_correct": m["surrogate_sign_correct"],
                "sign_denominator": m["sign_denominator"],
                "selected_minus_off": m["mean_selected_minus_off"],
                "selected_minus_on": m["mean_selected_minus_on"],
                "selected_minus_base": m["mean_selected_minus_base"],
                **m["correlations"]["surrogate"],
            }
    return {"n": len(rows), "components": components, "sign_flips_if_removed": flips,
            "mean_on_minus_off": np.mean([r["on_minus_off"] for r in rows], axis=0).tolist(),
            "decisions": decisions}


def validate_case(case, original):
    assert case["matched_original_losses"]
    assert case["config"] == original["config"]
    assert case["step"] == original["step"] and case["operator"] == original["operator"]
    assert case["cached_return_effects"] == original["effects"]
    assert case["extra_environment_steps"] == 0 and case["diagnostic_resets"] == 3
    measured, saved = case["measurements"], case["on_advantages"]
    for label, values in measured.items():
        np.testing.assert_array_equal(values["total"],
                                      original["own_soft_loss_with_bc_frozen_critic"][label])
        np.testing.assert_allclose(sum(np.array(values[k]) for k in COMPONENTS[:3]),
                                   values["total"], rtol=2e-6, atol=2e-6)
    for key in SIGNALS:
        a, b = ("off", "on") if key in COMPONENTS else ("on", "off")
        delta = np.array(measured[a][key]) - measured[b][key]
        if key == "deterministic_reset_q":
            np.testing.assert_array_equal(delta, saved[key + "_paired"])
            delta = delta.mean(1)
        np.testing.assert_array_equal(delta, saved[key])
    np.testing.assert_array_equal(saved["total"], original["surrogate_on_advantage_by_weight"])
    if case["config"]["envelope_policy_candidates"] == 1:
        assert measured["on"] == measured["off"]
        for effect in case["cached_return_effects"]["on_minus_off"].values():
            assert not np.any(effect), "Identity return control is nonzero"


def summarize(root, cached_root, cached_summary):
    seeds = (26, 27, 28)
    prior = json.loads(cached_summary.read_text())
    old_rows = {(r["seed"], r["step"], r["weight_index"]): r for r in prior["rows"]}
    rows, nulls, reproduction = [], [], {}
    for seed in seeds:
        data = json.loads((root / f"seed{seed}.json").read_text())
        cached = json.loads((cached_root / f"seed{seed}.json").read_text())
        if not data["complete"] or not data["base_reproduction"]["matched"]:
            raise ValueError(f"Incomplete or unmatched reconstruction: seed {seed}")
        assert data["protocol"] == "fork_loss_components_v1" and data["seed"] == seed
        assert data["method"] == cached["method"]
        assert data["steps"] == [c["step"] for c in data["cases"]] == cached["steps"]
        assert data["horizons"] == HORIZONS
        np.testing.assert_array_equal(data["weights"], WEIGHTS)
        reproduction[seed] = data["base_reproduction"]
        originals = {c["step"]: c for c in cached["cases"]}
        for case in data["cases"]:
            validate_case(case, originals[case["step"]])
            if case["config"]["envelope_policy_candidates"] == 1:
                nulls.append({"seed": seed, "step": case["step"]})
                continue
            for wi in range(3):
                row = old_rows[(seed, case["step"], wi)].copy()
                for effect in ("on_minus_off", "on_minus_base", "off_minus_base"):
                    np.testing.assert_allclose(row[effect], case["cached_return_effects"][effect][
                        "mean_scalar_by_weight_horizon"][wi], rtol=1e-12, atol=1e-12)
                row["signals"] = {k: case["on_advantages"][k][wi] for k in SIGNALS}
                rows.append(row)
    assert len(rows) == len(old_rows) and nulls == prior["identity_controls"]
    per_seed = {str(s): group_summary([r for r in rows if r["seed"] == s]) for s in seeds}
    macro = {key: {str(h): {field: float(np.mean([
        per_seed[str(s)]["decisions"][key][str(h)][field] for s in seeds]))
        for field in ("selected_minus_off", "selected_minus_on", "selected_minus_base")}
        for h in HORIZONS} for key in DECISIONS}
    return {"protocol": "fork_loss_components_summary_v1", "seeds": list(seeds),
            "weights": WEIGHTS.tolist(), "horizons": HORIZONS,
            "active_positions": len(rows) // 3, "identity_controls": nulls,
            "base_reproduction": reproduction, "pooled": group_summary(rows),
            "per_seed": per_seed,
            "by_preference": {str(w): group_summary([r for r in rows if r["weight_index"] == w])
                              for w in range(3)},
            "seed_macro": macro, "rows": rows,
            "scope": "Positive signals favor on; zero chooses off but counts as no sign agreement. "
                     "Identity controls are excluded. Raw discounted returns are cached labels, "
                     "not new confirmation. Soft critic continuation differs from deterministic "
                     "finite-horizon rollout continuation. No fitted thresholds or training changes."}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path("results/iclr_fork_loss_components_20260920"))
    parser.add_argument("--cached-root", type=Path, default=Path("results/iclr_update_forks_20260919"))
    parser.add_argument("--cached-summary", type=Path,
                        default=Path("results/iclr_update_forks_20260919_summary.json"))
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = summarize(args.root, args.cached_root, args.cached_summary)
    args.output.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    print(json.dumps({"active_positions": report["active_positions"],
                      "identity_controls": len(report["identity_controls"]),
                      "components": report["pooled"]["components"],
                      "decisions_512": {k: v["512"] for k, v in
                                        report["pooled"]["decisions"].items()}}, indent=2))
