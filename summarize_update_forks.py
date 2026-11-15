"""Summarize live update forks without downloading simulator returns or models."""

import argparse
import json
from pathlib import Path

import numpy as np
from scipy.stats import rankdata


WEIGHTS = np.array([[0., 1.], [.5, .5], [1., 0.]])
HORIZONS = [32, 128, 512]


def correlation(x, y):
    x, y = np.asarray(x), np.asarray(y)
    if len(x) < 3 or np.ptp(x) == 0 or np.ptp(y) == 0:
        return None
    return float(np.corrcoef(x, y)[0, 1])


def metrics(rows, horizon):
    gain = np.array([r["on_minus_off"][horizon] for r in rows])
    on_base = np.array([r["on_minus_base"][horizon] for r in rows])
    off_base = np.array([r["off_minus_base"][horizon] for r in rows])
    signal = np.array([r["surrogate"] for r in rows])
    choose_on = signal > 0
    selected_gain = np.where(choose_on, gain, 0.)
    nonzero = gain != 0
    result = {
        "n": len(rows), "outcome_ties": int((~nonzero).sum()),
        "on_better": int((gain > 0).sum()), "off_better": int((gain < 0).sum()),
        "surrogate_choose_on": int(choose_on.sum()), "surrogate_ties": int((signal == 0).sum()),
        "surrogate_sign_correct": int((signal[nonzero] * gain[nonzero] > 0).sum()),
        "sign_denominator": int(nonzero.sum()),
        "mean_on_minus_off": float(gain.mean()),
        "mean_on_minus_base": float(on_base.mean()),
        "mean_off_minus_base": float(off_base.mean()),
        "mean_selected_minus_off": float(selected_gain.mean()),
        "mean_selected_minus_on": float((selected_gain - gain).mean()),
        "mean_selected_minus_base": float((off_base + selected_gain).mean()),
        "on_all_repeats_better": sum(all(v > 0 for v in r["paired_on_minus_off"][horizon])
                                     for r in rows),
        "on_all_repeats_worse": sum(all(v < 0 for v in r["paired_on_minus_off"][horizon])
                                    for r in rows),
        "correlations": {},
    }
    for field in ("surrogate", "gradient_alignment_advantage", "own_min_other_cosine"):
        pairs = [(r[field], g) for r, g in zip(rows, gain) if r[field] is not None]
        x, y = zip(*pairs) if pairs else ([], [])
        result["correlations"][field] = {
            "n": len(pairs), "pearson": correlation(x, y),
            "spearman": correlation(rankdata(x), rankdata(y))}
    return result


def grouped(rows):
    return {str(horizon): {
        "all_preferences": metrics(rows, hi),
        "by_preference": {str(wi): metrics([r for r in rows if r["weight_index"] == wi], hi)
                          for wi in range(3)},
    } for hi, horizon in enumerate(HORIZONS)}


def summarize(root, seeds=(26, 27, 28)):
    records, nulls, reproduction = [], [], {}
    for seed in seeds:
        data = json.loads((root / f"seed{seed}.json").read_text())
        if not data["complete"] or not data["base_reproduction"]["matched"]:
            raise ValueError(f"Incomplete or unmatched main trajectory: seed {seed}")
        assert data["protocol"] == "live_paired_update_forks_v1"
        assert data["seed"] == seed
        assert data["steps"] == [c["step"] for c in data["cases"]] == list(range(10000, 50001, 5000))
        assert data["horizons"] == HORIZONS
        np.testing.assert_array_equal(data["weights"], WEIGHTS)
        reproduction[seed] = data["base_reproduction"]
        for case in data["cases"]:
            assert case["updates_per_branch"] == 128
            arrays = {name: np.asarray(case["returns"][name]) for name in ("base", "on", "off")}
            for arr in arrays.values():
                assert arr.shape == (3, 3, 3, 2) and np.isfinite(arr).all()
            effects = {}
            for name, a, b in [("on_minus_off", "on", "off"),
                               ("on_minus_base", "on", "base"),
                               ("off_minus_base", "off", "base")]:
                effects[name] = np.einsum("wrhd,wd->wrh", arrays[a] - arrays[b], WEIGHTS)
                np.testing.assert_allclose(effects[name], case["effects"][name][
                    "scalar_by_weight_repeat_horizon"], rtol=1e-12, atol=1e-12)
            losses = case["own_soft_loss_with_bc_frozen_critic"]
            surrogate = np.array(losses["off"]) - losses["on"]
            np.testing.assert_array_equal(surrogate, case["surrogate_on_advantage_by_weight"])
            if case["config"]["envelope_policy_candidates"] == 1:
                np.testing.assert_array_equal(arrays["on"], arrays["off"])
                np.testing.assert_array_equal(surrogate, np.zeros(3))
                nulls.append({"seed": seed, "step": case["step"]})
                continue
            signals = case["signals_before"]
            for wi in range(3):
                a = signals["mean_search_to_each_own_cosine"][wi]
                b = signals["mean_own_to_each_own_cosine"][wi]
                own = [v for i, v in enumerate(signals["own_preference_cosines"][wi])
                       if i != wi and v is not None]
                row = {"seed": seed, "step": case["step"], "operator": case["operator"],
                       "weight_index": wi, "surrogate": float(surrogate[wi]),
                       "gradient_alignment_advantage": a - b if a is not None and b is not None else None,
                       "own_min_other_cosine": min(own) if own else None,
                       "paired_on_minus_off": effects["on_minus_off"][wi].T.tolist()}
                row.update({k: v[wi].mean(0).tolist() for k, v in effects.items()})
                records.append(row)
    per_seed = {seed: grouped([r for r in records if r["seed"] == seed]) for seed in seeds}
    gain_fields = ("mean_on_minus_off", "mean_on_minus_base", "mean_off_minus_base",
                   "mean_selected_minus_off", "mean_selected_minus_on", "mean_selected_minus_base")
    macro = {str(h): {k: float(np.mean([per_seed[s][str(h)]["all_preferences"][k] for s in seeds]))
                     for k in gain_fields} for h in HORIZONS}
    return {"protocol": "live_paired_update_forks_v1", "seeds": list(seeds),
            "weights": WEIGHTS.tolist(), "horizons": HORIZONS,
            "active_cases": len(records) // 3, "identity_controls": nulls,
            "base_reproduction": reproduction, "per_seed": per_seed,
            "pooled_active": grouped(records), "seed_macro_means": macro, "rows": records,
            "interpretation": "Strict positive surrogate chooses on; otherwise off. "
                              "Identity controls excluded from discrimination. Correlated development "
                              "probes, not independent seeds or a deployed selector. Gradient "
                              "alignment difference is descriptive, not a fitted routing rule."}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path("results/iclr_update_forks_20260919"))
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = summarize(args.root)
    args.output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    print(json.dumps({"active_cases": result["active_cases"],
                      "identity_controls": len(result["identity_controls"]),
                      "pooled_512": result["pooled_active"]["512"],
                      "per_seed_512": {s: d["512"]["all_preferences"]
                                       for s, d in result["per_seed"].items()},
                      "seed_macro_512": result["seed_macro_means"]["512"]}, indent=2))
