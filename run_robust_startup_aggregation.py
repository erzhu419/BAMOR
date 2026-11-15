"""Compare fixed temporal block medians with the frozen startup mean detector."""

import argparse
import copy
import json
from pathlib import Path
import time

import numpy as np
import torch

import collect_startup_confirmation as acquisition
import run_startup_confirmation as confirmation
from collect_matched_scale_normals import ROOT as FRESH, normal_data as fresh_data
from headwise_change_evidence import CONFIG
from run_calibration_expansion import verify_windows
from run_headwise_change_probe import (
    CASES, ENVIRONMENTS, SEEDS, SOURCE, fold_roles, normal_data,
    summarize_heads, case_summary, screen, write)
from run_sensitivity_guarded_selection import GuardedEvidence, GUARDED


PROTOCOL = "fixed_temporal_block_median_startup_development_v1"
ROOT = Path("results/iclr_robust_startup_aggregation_20261001")
MEAN = "mean_window"
PRIMARY = "block_median_5x20"
ARMS = (MEAN, PRIMARY)
AGGREGATION = dict(window=100, block_size=20, blocks=5,
    empty_block="omit_from_median", head_eligibility="unchanged_predictor_support")
ZERO_WORK = ("additional_interactions", "initialization_probes", "validation_scale_optimizer_steps",
             "policy_optimizer_steps", "critic_optimizer_steps", "encoder_optimizer_steps")


def aggregate_windows(evidence, errors, supported):
    assert evidence.config["window"] == AGGREGATION["window"]
    mean = evidence._windows(errors, supported)
    robust = copy.deepcopy(mean)
    for window in robust:
        start, end = window["start_step"] - 1, window["end_step"]
        for h, name in enumerate(evidence.head_names):
            values = errors[start:end, h].reshape(5, 20)
            masks = supported[start:end, h].reshape(5, 20)
            blocks = [float(v[m].mean()) if m.any() else None for v, m in zip(values, masks)]
            head = window["heads"][name]
            head.update(block_means=blocks, block_supported_steps=masks.sum(1).tolist())
            if head["eligible"]:
                head["statistic"] = float(np.median([v for v in blocks if v is not None]))
        eligible = [h["statistic"] for h in window["heads"].values() if h["eligible"]]
        window["statistic"] = max(eligible) if eligible else None
    return {MEAN: mean, PRIMARY: robust}


def paired_windows(evidence, inputs, targets):
    support = evidence.distances(inputs) <= evidence.radii
    errors = evidence.raw_losses(inputs, targets) / evidence.scales(inputs, GUARDED)
    return aggregate_windows(evidence, errors, support), support


def mark_alarms(windows, threshold):
    for window in windows:
        window["alarm"] = bool(window["eligible"] and window["statistic"] > threshold)
        for head in window["heads"].values():
            head["alarm"] = bool(head["eligible"] and head["statistic"] > threshold)


def calibrate(evidence, inputs, targets):
    arms, _ = paired_windows(evidence, inputs, targets)
    return {arm: dict(error_threshold=float(np.quantile(
        [w["statistic"] for w in windows if w["eligible"]], CONFIG["quantile"])),
        support_radii=dict(zip(evidence.head_names, evidence.radii.tolist())),
        eligible_windows=sum(w["eligible"] for w in windows), total_windows=len(windows), windows=windows)
        for arm, windows in arms.items()}


def verify_coverage(first, second):
    assert len(first) == len(second)
    for a, b in zip(first, second):
        assert all(a[k] == b[k] for k in ("start_step", "end_step", "eligible", "supported_steps"))
        assert a["heads"].keys() == b["heads"].keys()
        for name in a["heads"]:
            assert all(a["heads"][name][k] == b["heads"][name][k]
                       for k in ("eligible", "supported_steps", "domain"))


def validate(row, env, seed):
    assert (row["protocol"], row["env"], row["roles"], row["config"], row["complete"], row["aggregation"]) == (
        PROTOCOL, env, fold_roles(seed), CONFIG, True, AGGREGATION)
    assert all(row[k] == 0 for k in ZERO_WORK)
    assert row["predictor_optimizer_steps"] == 2 * CONFIG["updates"]
    assert row["residual_scale_optimizer_steps"] == (row["reward_dim"] + 1) * CONFIG["scale_updates"]
    assert row["mean_control_reproduced"] and row["thresholds_frozen_before_evaluation_read"]
    assert row["scale_fit_sources"] == row["roles"]["fit_seeds"] + [row["roles"]["scale_seed"]]
    assert set(row["arms"]) == set(ARMS)
    a, b = (row["arms"][arm] for arm in ARMS)
    assert a["calibration"]["support_radii"] == b["calibration"]["support_radii"]
    verify_coverage(a["calibration"]["windows"], b["calibration"]["windows"])
    for case in CASES:
        verify_coverage(a["cases"][case]["windows"], b["cases"][case]["windows"])


def run(env, seed, source=SOURCE, root=ROOT, parent=confirmation.ROOT, fresh=FRESH):
    output = root / env.replace("-", "_") / f"test_seed{seed}" / "readout.json"
    if output.exists():
        saved = json.loads(output.read_text())
        validate(saved, env, seed)
        return saved
    prior = json.loads((parent / env.replace("-", "_") / f"test_seed{seed}" / "readout.json").read_text())
    assert (prior["protocol"], prior["env"], prior["complete"], prior["config"], prior["roles"]) == (
        confirmation.PROTOCOL, env, True, CONFIG, fold_roles(seed))
    started, roles = time.monotonic(), prior["roles"]
    report = dict(protocol=PROTOCOL, env=env, roles=roles, config=CONFIG, complete=False,
        aggregation=AGGREGATION, development_only=True, reconstruction_only=True,
        evaluation_scope="reuse_of_seen_confirmation_trajectories", evidence_integrated=False,
        frozen_selection=prior["frozen_selection"], retained_method="v25", arms={},
        predictor_optimizer_steps=0, residual_scale_optimizer_steps=0, affine_fits=0,
        reused_acquisition_interactions_counted_in_parent=250190, **dict.fromkeys(ZERO_WORK, 0))
    write(output, report)
    fit_rows = [normal_data(source, env, s, "expanded_normal") for s in roles["fit_seeds"]]
    fit_x, fit_y = (np.concatenate([r[i] for r in fit_rows]) for i in (0, 1))
    scale_rows = [fresh_data(fresh, env, s) for s in roles["fit_seeds"]]
    scale_rows.append(normal_data(source, env, roles["scale_seed"], "expanded_normal"))
    scale_x, scale_y = (np.concatenate([r[i] for r in scale_rows]) for i in (0, 1))
    evidence = GuardedEvidence(fit_x.shape[1], fit_y.shape[1], fit_rows[0][2]["reward_dim"])
    evidence.weights = {GUARDED: prior["frozen_selection"]["global_weight"]}
    work = evidence.fit(fit_x, fit_y, len(fit_rows[0][0]), 80000000 + 10 * seed)
    assert work == prior["predictor_work"]
    report.update(predictor_optimizer_steps=work["optimizer_steps"], predictor_work=work, affine_fits=2)
    write(output, report)
    work = evidence.fit_scale_losses(scale_x, evidence.raw_losses(scale_x, scale_y), 81000000 + 10 * seed)
    assert work["optimizer_steps"] == prior["scale_work"]["optimizer_steps"]
    np.testing.assert_allclose(work["global_scales"], prior["scale_work"]["global_scales"], rtol=1e-6)
    report.update(residual_scale_optimizer_steps=work["optimizer_steps"], scale_work=work,
        reward_dim=evidence.reward_dim, scale_fit_sources=prior["scale_fit_sources"])
    old = prior["arms"][confirmation.PRIMARY]
    evidence.radii = np.array([old["calibration"]["support_radii"][name] for name in evidence.head_names])
    startup, _ = acquisition.data(parent / "collection", env, seed, "startup")
    x, y = (np.concatenate([r[i] for r in startup.values()]) for i in (0, 1))
    calibration = calibrate(evidence, x, y)
    verify_windows(calibration[MEAN]["windows"], old["calibration"]["windows"])
    np.testing.assert_allclose(calibration[MEAN]["error_threshold"], old["calibration"]["error_threshold"], rtol=1e-6)
    report["arms"] = {arm: dict(calibration=calibration[arm], cases={}) for arm in ARMS}
    report["thresholds_frozen_before_evaluation_read"] = True
    write(output, report)
    cases, metadata = acquisition.data(parent / "collection", env, seed, "evaluation")
    for case, (x, y) in cases.items():
        windows, support = paired_windows(evidence, x, y)
        for arm in ARMS:
            mark_alarms(windows[arm], calibration[arm]["error_threshold"])
            evaluation = dict(windows=windows[arm], supported=support.any(1), head_supported=support)
            summary = case_summary(evaluation)
            summary["heads"] = summarize_heads(evidence, evaluation)
            if case in ("sampler_change", "actor_change"):
                summary["intervention_active"] = metadata["cases"][case]["intervention_active"]
            post = [w for w in windows[arm] if w["eligible"] and w["start_step"] > 500]
            summary["peak_post_score"] = max((w["statistic"] for w in post), default=None)
            if arm == MEAN:
                verify_windows(windows[arm], old["cases"][case]["windows"])
                assert [w["alarm"] for w in windows[arm]] == [w["alarm"] for w in old["cases"][case]["windows"]]
            report["arms"][arm]["cases"][case] = summary
        write(output, report)
    report.update(complete=True, mean_control_reproduced=True, seconds=time.monotonic() - started)
    validate(report, env, seed)
    write(output, report)
    print(f"DONE {PROTOCOL} {env} seed={seed} interactions=0 "
          f"thresholds={ {arm: calibration[arm]['error_threshold'] for arm in ARMS} }", flush=True)
    return report


def paired_readout(runs):
    result = dict(normal_alarms={arm: 0 for arm in ARMS}, normal_eligible_windows=0,
        mode_alarms={arm: 0 for arm in ARMS}, mode_eligible_windows=0,
        normal_resolved=[], normal_added=[], mode_lost=[], mode_added=[], timely_lost=[], timely_added=[])
    for row in runs:
        seed = row["roles"]["test_seed"]
        for case in CASES:
            a, b = (row["arms"][arm]["cases"][case] for arm in ARMS)
            mode = case.startswith("mode")
            if mode and a["timely_alarm"] != b["timely_alarm"]:
                result["timely_lost" if a["timely_alarm"] else "timely_added"].append(dict(seed=seed, case=case))
            for first, second in zip(a["windows"], b["windows"]):
                if first["end_step"] <= 500 and case != "stationary":
                    continue
                category = "mode" if mode else "normal"
                result[f"{category}_eligible_windows"] += first["eligible"]
                for arm, w in zip(ARMS, (first, second)):
                    result[f"{category}_alarms"][arm] += w["alarm"]
                if first["alarm"] != second["alarm"]:
                    change = "lost" if mode else "resolved"
                    result[f"{category}_{change if first['alarm'] else 'added'}"].append(
                        dict(seed=seed, case=case, start_step=first["start_step"], end_step=first["end_step"]))
    return result


def passes_paired_screen(environments):
    passed = all(e["arms"][PRIMARY]["development_screen_passed"] for e in environments.values())
    no_lost = all(not e["paired"]["timely_lost"] for e in environments.values())
    improved = any(e["paired"]["normal_alarms"][PRIMARY] < e["paired"]["normal_alarms"][MEAN]
                   for e in environments.values())
    nonworse = all(e["paired"]["normal_alarms"][PRIMARY] <= e["paired"]["normal_alarms"][MEAN]
                   for e in environments.values())
    return passed and no_lost and improved and nonworse


def summarize(root=ROOT):
    report = dict(protocol=PROTOCOL, complete=True, development_only=True, aggregation=AGGREGATION,
        retained_method="v25", evidence_integrated=False, environments={},
        predictor_optimizer_steps=0, residual_scale_optimizer_steps=0, affine_fits=0, **dict.fromkeys(ZERO_WORK, 0))
    for env in ENVIRONMENTS:
        runs = [json.loads((root / env.replace("-", "_") / f"test_seed{s}" / "readout.json").read_text()) for s in SEEDS]
        for seed, row in zip(SEEDS, runs):
            validate(row, env, seed)
            for key in ("predictor_optimizer_steps", "residual_scale_optimizer_steps", "affine_fits"):
                report[key] += row[key]
        report["environments"][env] = dict(arms={arm: screen(runs, arm) for arm in ARMS},
            paired=paired_readout(runs), runs=runs)
    report["decision"] = "consider_fresh_trajectory_robust_confirmation" if passes_paired_screen(report["environments"]) else "hold_conditional_evidence_integration"
    return report


def compact_report(report):
    result = copy.deepcopy(report)
    for env in result["environments"].values():
        for row in env["runs"]:
            for arm in row["arms"].values():
                arm["calibration"].pop("windows")
                for case in arm["cases"].values():
                    case.pop("windows")
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--env", choices=ENVIRONMENTS)
    parser.add_argument("--test-seed", type=int, choices=SEEDS)
    parser.add_argument("--summarize", action="store_true")
    args = parser.parse_args()
    torch.set_num_threads(2)
    if args.summarize:
        report = summarize()
        write(ROOT / "readout.json", report)
        write(ROOT / "compact_readout.json", compact_report(report))
        print(json.dumps(dict(decision=report["decision"], environments={env: dict(arms=e["arms"], paired=e["paired"])
            for env, e in report["environments"].items()})))
    else:
        run(args.env, args.test_seed)
