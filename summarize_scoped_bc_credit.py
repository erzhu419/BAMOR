"""Compact single-block readout, with posthoc preference-envelope diagnostics."""

import argparse
import json
from pathlib import Path

import numpy as np

import run_scoped_bc_credit as experiment


PROTOCOL = "scoped_bc_credit_readout_v1"


def relative_pct(candidate, control):
    return 100 * (candidate / control - 1) if control else None


def preference_decomposition(candidate, control, weights):
    a, b, w = map(np.asarray, (candidate, control, weights))
    matched = [(r * w).sum(axis=-1) for r in (a, b)]
    envelope = [(r @ w.T).max(axis=1) for r in (a, b)]
    result = {}
    for name, (x, y) in (("matched_utility", matched), ("front_eum", envelope)):
        delta = x - y
        result[name] = dict(native_mean=float(y.mean()), pulse_mean=float(x.mean()),
            relative_pct=relative_pct(float(x.mean()), float(y.mean())),
            delta_per_mode=delta.mean(axis=1).tolist(),
            wins_per_mode=(delta > 0).sum(axis=1).tolist(),
            losses_per_mode=(delta < 0).sum(axis=1).tolist(),
            ties_per_mode=(delta == 0).sum(axis=1).tolist())
    return result


def block_comparison(native, pulse):
    assert native["operator"] == pulse["operator"] == experiment.B
    assert native["half_counts"] == pulse["half_counts"] == [750, 750]
    np.testing.assert_array_equal(native["scale_before_block"], pulse["scale_before_block"])
    a, b = np.asarray(pulse["half_mean_rewards"]), np.asarray(native["half_mean_rewards"])
    return dict(native=native, pulse=pulse, half_mean_reward_delta=(a - b).tolist(),
                mean_reward_native=b.mean(axis=0).tolist(), mean_reward_pulse=a.mean(axis=0).tolist(),
                mean_reward_delta=(a - b).mean(axis=0).tolist())


def read_case(report, seed):
    background, cut = experiment.CASES[seed]
    assert (report["protocol"], report["seed"], report["env"], report["complete"], report["background"]) == (
        experiment.PROTOCOL, seed, experiment.ENV, True, background)
    assert report["window"] == [cut, cut + experiment.BLOCK]
    checks = report["control_reproduction"]
    assert all(check["passed"] for check in checks.values())
    assert checks[str(experiment.END)]["final_returns_matched"]
    summary = experiment.summarize(report)
    assert summary == report["summary"], "Saved summary differs from the completed branch data"
    native = report["branches"]["native"]["measurements"]
    pulse = report["branches"]["pulse"]["measurements"]
    contrasts = []
    for frozen, a, b in zip(summary["contrasts"], pulse, native[1:]):
        for row in (a, b):
            assert row["encoder"] == "contemporaneous" and row["policy"] == "current_only"
            hv = [experiment.training.compute_hv_nd(front, np.asarray(row["ref_point"]))
                  for front in row["returns"]]
            np.testing.assert_allclose(hv, row["hv_per_mode"], rtol=1e-12, atol=1e-6)
        contrasts.append({**frozen, "native_hv": b["hv_mean_mode"], "pulse_hv": a["hv_mean_mode"],
            **preference_decomposition(a["returns"], b["returns"], a["evaluation_weights"])})
    first = [next(block for block in row["feedback"] if block["end_step"] == report["window"][1])
             for row in (native[1], pulse[0])]
    return dict(seed=seed, background=background, window=report["window"],
        control_reproduction_passed=True, completed_budget=report["completed_budget"],
        mode_history=report["branches"]["native"]["mode_history"],
        block_feedback=block_comparison(*first),
        block_progress_delta=summary["block_progress_delta"],
        bc_score_delta_at_block_end=summary["bc_score_delta_at_block_end"],
        subsequent_operator_mismatches=summary["subsequent_operator_mismatches"],
        first_subsequent_operator_mismatch=summary["first_subsequent_operator_mismatch"],
        contrasts=contrasts)


def summarize(root=experiment.ROOT):
    cases = {str(seed): read_case(json.loads((Path(root) / f"seed{seed}.json").read_text()), seed)
             for seed in experiment.CASES}
    budget = {key: sum(case["completed_budget"][key] for case in cases.values())
              for key in next(iter(cases.values()))["completed_budget"]}
    return dict(protocol=PROTOCOL, complete=True, cases=cases, completed_experiment_budget=budget,
        new_readout_interactions=0, new_readout_updates=0,
        scope="Two outcome-selected development cases, not independent confirmation. HV and matched "
              "utility retain their frozen definitions. Front EUM and preference win/loss counts are "
              "posthoc diagnostics on the same grid: each target preference may select the best return "
              "among all sampled conditioning preferences. This is not native one-query execution. "
              "Counts are correlated grid points, not independent trials. Paired score differences "
              "are diagnostic contrasts, not counterfactual labels available online. No causal split "
              "among subsequent actor, critic, encoder and replay changes is identified.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=experiment.ROOT)
    parser.add_argument("--output", type=Path, default=experiment.ROOT / "readout.json")
    args = parser.parse_args()
    result = summarize(args.root)
    args.output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    print(json.dumps(dict(complete=True, budget=result["completed_experiment_budget"],
                         readout_bytes=args.output.stat().st_size)))
    for seed, case in result["cases"].items():
        print(json.dumps(dict(seed=seed, progress_delta=case["block_progress_delta"],
            operator_mismatches=case["subsequent_operator_mismatches"],
            trajectory=[dict(step=row["step"], hv_pct=row["hv_relative_pct"],
                             matched_pct=row["matched_utility"]["relative_pct"],
                             eum_pct=row["front_eum"]["relative_pct"]) for row in case["contrasts"]])))
