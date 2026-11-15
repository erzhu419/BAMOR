"""Confirm the frozen 5x20 statistic on new startup and evaluation trajectories."""

import argparse
import json
from pathlib import Path
import time

import numpy as np
import torch

import collect_startup_confirmation as acquisition
import run_robust_startup_aggregation as development
from collect_matched_scale_normals import ROOT as FRESH, normal_data as fresh_data
from headwise_change_evidence import CONFIG
from run_headwise_change_probe import (
    CASES, ENVIRONMENTS, SEEDS, SOURCE, fold_roles, normal_data,
    summarize_heads, case_summary, screen, write)
from run_sensitivity_guarded_selection import GuardedEvidence, GUARDED


PROTOCOL = "frozen_5x20_new_trajectory_confirmation_v1"
ROOT = Path("results/iclr_robust_startup_confirmation_20261001")
ARMS, PRIMARY, MEAN = development.ARMS, development.PRIMARY, development.MEAN
ZERO_WORK = ("validation_scale_optimizer_steps", "policy_optimizer_steps",
             "critic_optimizer_steps", "encoder_optimizer_steps")


def seeds(seed, group, index=0):
    if group == "startup":
        return dict(environment_seed=270000000 + 10 * seed + index,
            reset_seed=800000000 + 50000 * seed + 10000 * index,
            weight_seed=280000000 + 10 * seed + index)
    assert group == "evaluation"
    return dict(environment_seed=290000000 + 10 * seed,
        reset_seed=1000000000 + 50000 * seed, weight_seed=300000000 + 10 * seed)


def validate(row, env, seed):
    assert (row["protocol"], row["env"], row["roles"], row["config"], row["complete"], row["aggregation"]) == (
        PROTOCOL, env, fold_roles(seed), CONFIG, True, development.AGGREGATION)
    assert row["predictor_optimizer_steps"] == 2 * CONFIG["updates"]
    assert row["residual_scale_optimizer_steps"] == (row["reward_dim"] + 1) * CONFIG["scale_updates"]
    assert row["scale_fit_sources"] == row["roles"]["fit_seeds"] + [row["roles"]["scale_seed"]]
    assert all(row[k] == 0 for k in ZERO_WORK)
    assert row["thresholds_frozen_before_evaluation_collection"]
    assert row["frozen_model_work_reproduced"] and set(row["arms"]) == set(ARMS)
    for group in ("startup", "evaluation"):
        acquisition.validate(row["acquisition"][group], env, seed, group, seed_fn=seeds)
    assert row["additional_interactions"] == sum(r["additional_interactions"] for r in row["acquisition"].values())
    assert row["initialization_probes"] == sum(r["initialization_probes"] for r in row["acquisition"].values())
    a, b = (row["arms"][arm] for arm in ARMS)
    assert a["calibration"]["support_radii"] == b["calibration"]["support_radii"]
    development.verify_coverage(a["calibration"]["windows"], b["calibration"]["windows"])
    for case in CASES:
        development.verify_coverage(a["cases"][case]["windows"], b["cases"][case]["windows"])


def run(env, seed, source=SOURCE, root=ROOT, parent=development.ROOT, fresh=FRESH):
    output = root / env.replace("-", "_") / f"test_seed{seed}" / "readout.json"
    if output.exists():
        saved = json.loads(output.read_text())
        if not saved["complete"]:
            raise RuntimeError("Incomplete confirmation retained; inspect acquisition ledgers before retrying")
        validate(saved, env, seed)
        return saved
    prior = json.loads((parent / env.replace("-", "_") / f"test_seed{seed}" / "readout.json").read_text())
    development.validate(prior, env, seed)
    started, roles = time.monotonic(), prior["roles"]
    report = dict(protocol=PROTOCOL, env=env, roles=roles, config=CONFIG, complete=False,
        aggregation=development.AGGREGATION, confirmation_scope="new_trajectories_same_frozen_policy_sources",
        retained_method="v25", evidence_integrated=False, frozen_selection=prior["frozen_selection"],
        predictor_optimizer_steps=0, residual_scale_optimizer_steps=0, affine_fits=0,
        additional_interactions=0, initialization_probes=0, acquisition={}, arms={}, **dict.fromkeys(ZERO_WORK, 0))
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
    report.update(predictor_work=work, predictor_optimizer_steps=work["optimizer_steps"], affine_fits=2)
    write(output, report)
    work = evidence.fit_scale_losses(scale_x, evidence.raw_losses(scale_x, scale_y), 81000000 + 10 * seed)
    assert work["optimizer_steps"] == prior["scale_work"]["optimizer_steps"]
    np.testing.assert_allclose(work["global_scales"], prior["scale_work"]["global_scales"], rtol=1e-6)
    report.update(scale_work=work, residual_scale_optimizer_steps=work["optimizer_steps"], reward_dim=evidence.reward_dim,
        scale_fit_sources=prior["scale_fit_sources"], frozen_model_work_reproduced=True)
    radii = prior["arms"][MEAN]["calibration"]["support_radii"]
    evidence.radii = np.array([radii[name] for name in evidence.head_names])
    collect_root = root / "collection"
    for group in ("startup", "evaluation"):
        # Record partial spending even if collection or its physical audit fails.
        try:
            acquisition.run_group(env, seed, group, collect_root, seed_fn=seeds)
        finally:
            ledger = acquisition.folder(collect_root, env, seed, group) / "readout.json"
            if ledger.exists():
                report["acquisition"][group] = json.loads(ledger.read_text())
                report.update(additional_interactions=sum(r["additional_interactions"] for r in report["acquisition"].values()),
                    initialization_probes=sum(r["initialization_probes"] for r in report["acquisition"].values()))
                write(output, report)
        rows, metadata = acquisition.data(collect_root, env, seed, group, seed_fn=seeds)
        report["acquisition"][group] = metadata
        if group == "startup":
            x, y = (np.concatenate([r[i] for r in rows.values()]) for i in (0, 1))
            calibration = development.calibrate(evidence, x, y)
            report["arms"] = {arm: dict(calibration=calibration[arm], cases={}) for arm in ARMS}
            report["thresholds_frozen_before_evaluation_collection"] = True
            write(output, report)
        else:
            for case, (x, y) in rows.items():
                windows, support = development.paired_windows(evidence, x, y)
                for arm in ARMS:
                    development.mark_alarms(windows[arm], report["arms"][arm]["calibration"]["error_threshold"])
                    evaluation = dict(windows=windows[arm], supported=support.any(1), head_supported=support)
                    summary = case_summary(evaluation)
                    summary["heads"] = summarize_heads(evidence, evaluation)
                    if case in ("sampler_change", "actor_change"):
                        summary["intervention_active"] = metadata["cases"][case]["intervention_active"]
                    post = [w for w in windows[arm] if w["eligible"] and w["start_step"] > 500]
                    summary["peak_post_score"] = max((w["statistic"] for w in post), default=None)
                    report["arms"][arm]["cases"][case] = summary
                write(output, report)
    report.update(complete=True, seconds=time.monotonic() - started)
    validate(report, env, seed)
    write(output, report)
    print(f"DONE {PROTOCOL} {env} seed={seed} new_interactions={report['additional_interactions']}", flush=True)
    return report


def summarize(root=ROOT):
    report = dict(protocol=PROTOCOL, complete=True, aggregation=development.AGGREGATION,
        confirmation_scope="new_trajectories_same_frozen_policy_sources", retained_method="v25",
        evidence_integrated=False, environments={}, prior_acquisition_interactions=250190,
        additional_interactions=0, initialization_probes=0, predictor_optimizer_steps=0,
        residual_scale_optimizer_steps=0, affine_fits=0, **dict.fromkeys(ZERO_WORK, 0))
    for env in ENVIRONMENTS:
        runs = [json.loads((root / env.replace("-", "_") / f"test_seed{s}" / "readout.json").read_text()) for s in SEEDS]
        for seed, row in zip(SEEDS, runs):
            validate(row, env, seed)
            for key in ("additional_interactions", "initialization_probes", "predictor_optimizer_steps", "residual_scale_optimizer_steps", "affine_fits"):
                report[key] += row[key]
        report["environments"][env] = dict(arms={arm: screen(runs, arm) for arm in ARMS},
            paired=development.paired_readout(runs), runs=runs)
    report["cumulative_acquisition_interactions"] = report["prior_acquisition_interactions"] + report["additional_interactions"]
    report["decision"] = "consider_budgeted_robust_detector_integration" if development.passes_paired_screen(
        report["environments"]) else "hold_conditional_evidence_integration"
    return report


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
        write(ROOT / "compact_readout.json", development.compact_report(report))
        print(json.dumps(dict(decision=report["decision"], environments={env: dict(arms=e["arms"], paired=e["paired"])
            for env, e in report["environments"].items()}, additional_interactions=report["additional_interactions"])))
    else:
        run(args.env, args.test_seed)
