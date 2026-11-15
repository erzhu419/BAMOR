"""Confirm frozen startup calibration using newly acquired trajectories."""

import argparse
import copy
import json
from pathlib import Path
import time

import numpy as np
import torch

import collect_startup_confirmation as acquisition
from collect_matched_scale_normals import ROOT as FRESH, normal_data as fresh_data
from headwise_change_evidence import CONFIG
from run_cross_policy_calibration import calibrate_fixed_support
from run_calibration_expansion import verify_windows
from run_deployment_startup_calibration import ROOT as PREVIOUS, PRIMARY as OLD_STARTUP, validate as validate_previous
from run_headwise_change_probe import (
    CASES, ENVIRONMENTS, SEEDS, SOURCE, fold_roles, normal_data,
    summarize_heads, case_summary, screen, write)
from run_sensitivity_guarded_selection import GuardedEvidence, GUARDED


PROTOCOL = "frozen_startup_new_trajectory_confirmation_v1"
ROOT = Path("results/iclr_startup_confirmation_20261001")
PRIMARY = "fresh_deployment_startup_5k"
CONTROLS = ("original_calibration", "pooled_two_policy", "frozen_old_startup")
ARMS = (*CONTROLS, PRIMARY)


def validate(row, env, seed):
    assert (row["protocol"], row["env"], row["roles"], row["complete"], row["config"]) == (
        PROTOCOL, env, fold_roles(seed), True, CONFIG)
    assert row["predictor_optimizer_steps"] == 2 * CONFIG["updates"]
    assert row["residual_scale_optimizer_steps"] == (row["reward_dim"] + 1) * CONFIG["scale_updates"]
    assert row["scale_fit_sources"] == row["roles"]["fit_seeds"] + [row["roles"]["scale_seed"]]
    assert row["shared_evaluation_scores_verified"]
    assert set(row["arms"]) == set(ARMS)
    for group in ("startup", "evaluation"):
        acquisition.validate(row["acquisition"][group], env, seed, group)
    assert row["additional_interactions"] == sum(r["additional_interactions"] for r in row["acquisition"].values())
    assert row["initialization_probes"] == sum(r["initialization_probes"] for r in row["acquisition"].values())
    assert all(row[key] == 0 for key in ("validation_scale_optimizer_steps", "policy_optimizer_steps",
        "critic_optimizer_steps", "encoder_optimizer_steps"))
    radii = row["arms"][PRIMARY]["calibration"]["support_radii"]
    assert all(a["calibration"]["support_radii"] == radii for a in row["arms"].values())


def run(env, seed, source=SOURCE, root=ROOT, previous=PREVIOUS, fresh=FRESH):
    roles = fold_roles(seed)
    output = root / env.replace("-", "_") / f"test_seed{seed}" / "readout.json"
    if output.exists():
        saved = json.loads(output.read_text())
        if not saved["complete"]:
            raise RuntimeError("Incomplete confirmation retained; inspect acquisition ledger before retrying")
        validate(saved, env, seed)
        return saved
    prior = json.loads((previous / env.replace("-", "_") / f"test_seed{seed}" / "readout.json").read_text())
    validate_previous(prior, env, seed)
    started = time.monotonic()
    report = dict(protocol=PROTOCOL, env=env, roles=roles, config=CONFIG, complete=False,
        confirmation_scope="new_trajectories_same_frozen_policy_sources", primary_calibration=PRIMARY,
        frozen_selection=prior["frozen_selection"], arms={}, acquisition={}, additional_interactions=0,
        initialization_probes=0, validation_scale_optimizer_steps=0, policy_optimizer_steps=0,
        critic_optimizer_steps=0, encoder_optimizer_steps=0, predictor_optimizer_steps=0,
        residual_scale_optimizer_steps=0)
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
    report.update(predictor_work=work, predictor_optimizer_steps=work["optimizer_steps"])
    write(output, report)
    work = evidence.fit_scale_losses(scale_x, evidence.raw_losses(scale_x, scale_y), 81000000 + 10 * seed)
    assert work["optimizer_steps"] == prior["scale_work"]["optimizer_steps"]
    np.testing.assert_allclose(work["global_scales"], prior["scale_work"]["global_scales"], rtol=1e-6)
    report.update(scale_work=work, residual_scale_optimizer_steps=work["optimizer_steps"],
        reward_dim=evidence.reward_dim, scale_fit_sources=prior["scale_fit_sources"], fit_steps=len(fit_x), scale_fit_steps=len(scale_x))
    evidence.thresholds = {}
    evidence.radii = np.array([prior["arms"][OLD_STARTUP]["calibration"]["support_radii"][name] for name in evidence.head_names])
    for arm in CONTROLS:
        old = prior["arms"][OLD_STARTUP if arm == "frozen_old_startup" else arm]
        report["arms"][arm] = dict(calibration=copy.deepcopy(old["calibration"]), cases={})
    collect_root = root / "collection"
    acquisition.run_group(env, seed, "startup", collect_root)
    startup, meta = acquisition.data(collect_root, env, seed, "startup")
    report["acquisition"]["startup"] = meta
    report.update(additional_interactions=meta["additional_interactions"], initialization_probes=meta["initialization_probes"])
    x, y = (np.concatenate([rows[i] for rows in startup.values()]) for i in (0, 1))
    report["arms"][PRIMARY] = dict(calibration=calibrate_fixed_support(evidence, x, y), cases={})
    report["threshold_frozen_before_evaluation_collection"] = True
    write(output, report)
    acquisition.run_group(env, seed, "evaluation", collect_root)
    evaluation_cases, meta = acquisition.data(collect_root, env, seed, "evaluation")
    report["acquisition"]["evaluation"] = meta
    report.update(additional_interactions=sum(m["additional_interactions"] for m in report["acquisition"].values()),
        initialization_probes=sum(m["initialization_probes"] for m in report["acquisition"].values()))
    for arm in ARMS:
        evidence.thresholds[GUARDED] = report["arms"][arm]["calibration"]["error_threshold"]
        for case in CASES:
            evaluation = evidence.evaluate(*evaluation_cases[case], GUARDED)
            summary = case_summary(evaluation)
            summary["heads"] = summarize_heads(evidence, evaluation)
            if case in ("sampler_change", "actor_change"):
                summary["intervention_active"] = meta["cases"][case]["intervention_active"]
            if arm != CONTROLS[0]:
                verify_windows(summary["windows"], report["arms"][CONTROLS[0]]["cases"][case]["windows"])
            report["arms"][arm]["cases"][case] = summary
        write(output, report)
    report.update(complete=True, shared_evaluation_scores_verified=True, seconds=time.monotonic() - started)
    validate(report, env, seed)
    write(output, report)
    print(f"DONE {env} seed={seed} new_interactions={report['additional_interactions']} "
        f"startup_threshold={report['arms'][PRIMARY]['calibration']['error_threshold']}", flush=True)
    return report


def summarize(root=ROOT):
    report = dict(protocol=PROTOCOL, config=CONFIG, complete=True,
        confirmation_scope="new_trajectories_same_frozen_policy_sources", environments={},
        additional_interactions=0, initialization_probes=0, predictor_optimizer_steps=0,
        residual_scale_optimizer_steps=0, validation_scale_optimizer_steps=0,
        policy_optimizer_steps=0, critic_optimizer_steps=0, encoder_optimizer_steps=0,
        prior_acquisition_interactions=140100, retained_method="v25", evidence_integrated=False)
    for env in ENVIRONMENTS:
        runs = [json.loads((root / env.replace("-", "_") / f"test_seed{seed}" / "readout.json").read_text()) for seed in SEEDS]
        for seed, row in zip(SEEDS, runs):
            validate(row, env, seed)
            assert row["threshold_frozen_before_evaluation_collection"]
            for key in ("additional_interactions", "initialization_probes", "predictor_optimizer_steps", "residual_scale_optimizer_steps"):
                report[key] += row[key]
        report["environments"][env] = dict(arms={arm: screen(runs, arm) for arm in ARMS}, runs=runs)
    report["cumulative_acquisition_interactions"] = report["prior_acquisition_interactions"] + report["additional_interactions"]
    passed = all(e["arms"][PRIMARY]["development_screen_passed"] for e in report["environments"].values())
    report["decision"] = "consider_budgeted_detector_integration" if passed else "hold_conditional_evidence_integration"
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--env", choices=ENVIRONMENTS)
    parser.add_argument("--test-seed", type=int, choices=SEEDS)
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--summarize", action="store_true")
    args = parser.parse_args()
    torch.set_num_threads(2)
    if args.summarize:
        report = summarize(args.root)
        write(args.root / "readout.json", report)
        print(json.dumps(dict(environments={env: e["arms"] for env, e in report["environments"].items()},
            decision=report["decision"], new_interactions=report["additional_interactions"])))
    else:
        run(args.env, args.test_seed, root=args.root)
