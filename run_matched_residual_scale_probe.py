"""Scale fitting from the deployed predictor's errors on fresh normal inputs."""

import argparse
import copy
import json
from pathlib import Path
import time

import numpy as np
import torch

from affine_residual_change_evidence import AffineResidualHeadwiseEvidence
from collect_matched_scale_normals import ROOT as FRESH, normal_data as fresh_data
from headwise_change_evidence import ARMS, CONFIG
from run_affine_residual_change_probe import ROOT as CONTROL, PROTOCOL as CONTROL_PROTOCOL
from run_cross_source_scale_probe import validate_run
from run_headwise_change_probe import (
    CASES, ENVIRONMENTS, SEEDS, fold_roles, summarize_heads, case_summary,
    SOURCE, normal_data, source_folder, source_readout, screen, write)


PROTOCOL = "matched_deployed_residual_scale_development_v1"
ROOT = Path("results/iclr_matched_residual_scale_20261001")


def run(env, test_seed, source=SOURCE, root=ROOT, control=CONTROL, fresh=FRESH):
    roles = fold_roles(test_seed)
    output = root / env.replace("-", "_") / f"test_seed{test_seed}" / "readout.json"
    if output.exists():
        saved = json.loads(output.read_text())
        validate_run(saved, env, test_seed, PROTOCOL)
        return saved
    old = json.loads((control / env.replace("-", "_") / f"test_seed{test_seed}" / "readout.json").read_text())
    validate_run(old, env, test_seed, CONTROL_PROTOCOL)
    fresh_rows = [fresh_data(fresh, env, seed) for seed in roles["fit_seeds"]]
    original = source_readout(source, env, test_seed)
    started = time.monotonic()
    report = dict(protocol=PROTOCOL, env=env, roles=roles, config=CONFIG, complete=False,
        development_only=True, additional_interactions=0, initialization_probes=0,
        predictor_optimizer_steps=0, residual_scale_optimizer_steps=0,
        policy_optimizer_steps=0, critic_optimizer_steps=0, encoder_optimizer_steps=0, arms={},
        affine_residual_control=dict(protocol=CONTROL_PROTOCOL, reused=True, arms=old["arms"]))
    write(output, report)
    fit_rows = [normal_data(source, env, seed, "expanded_normal") for seed in roles["fit_seeds"]]
    fit_x, fit_y = (np.concatenate([row[index] for row in fit_rows]) for index in (0, 1))
    scale_x, scale_y, scale_meta = normal_data(source, env, roles["scale_seed"], "expanded_normal")
    cal_x, cal_y, cal_meta = normal_data(source, env, roles["calibration_seed"], "expanded_normal")
    evidence = AffineResidualHeadwiseEvidence(fit_x.shape[1], fit_y.shape[1], cal_meta["reward_dim"])
    predictor_work = evidence.fit(fit_x, fit_y, len(fit_rows[0][0]), 80000000 + 10 * test_seed)
    assert predictor_work == old["predictor_work"], "Deployed mean prediction must reproduce"
    inputs = np.concatenate([row[0] for row in fresh_rows] + [scale_x])
    targets = np.concatenate([row[1] for row in fresh_rows] + [scale_y])
    losses = evidence.raw_losses(inputs, targets)
    scale_work = evidence.fit_scale_losses(inputs, losses, 81000000 + 10 * test_seed)
    report.update(fit_sources=[row[2] for row in fit_rows], scale_source=scale_meta,
        calibration_source=cal_meta, fit_steps=len(fit_x), scale_steps=len(scale_x), calibration_steps=len(cal_x),
        scale_fit_steps=len(inputs), scale_fit_sources=roles["fit_seeds"] + [roles["scale_seed"]],
        fresh_fit_sources=[dict(seed=row[2]["seed"], steps=len(row[0]), source_step=row[2]["source_step"])
                           for row in fresh_rows], predictor_work=predictor_work, scale_work=scale_work,
        predictor_optimizer_steps=predictor_work["optimizer_steps"],
        residual_scale_optimizer_steps=scale_work["optimizer_steps"],
        residual_blocks=[dict(seed=seed, fresh=index < 2, steps=5000,
            normalized_head_mse=dict(zip(evidence.head_names, losses[index * 5000:(index + 1) * 5000].mean(0).tolist())))
            for index, seed in enumerate(roles["fit_seeds"] + [roles["scale_seed"]])])
    calibration = evidence.calibrate(cal_x, cal_y)
    for arm in ARMS:
        row = dict(calibration=calibration[arm], cases={})
        report["arms"][arm] = row
        assert calibration[arm]["support_radii"] == old["arms"][arm]["calibration"]["support_radii"]
        with np.load(source_folder(source, env, test_seed) / "transitions.npz") as data:
            for case in CASES:
                evaluation = evidence.evaluate(data[f"{case}_inputs"], data[f"{case}_targets"], arm)
                summary = case_summary(evaluation)
                summary["heads"] = summarize_heads(evidence, evaluation)
                for new_window, old_window in zip(summary["windows"], old["arms"][arm]["cases"][case]["windows"]):
                    assert {h: v["supported_steps"] for h, v in new_window["heads"].items()} == {
                        h: v["supported_steps"] for h, v in old_window["heads"].items()}
                if case in ("actor_change", "sampler_change"):
                    summary["intervention_active"] = original["cases"][case]["intervention_active"]
                row["cases"][case] = summary
        write(output, report)
        print(f"ARM {env} seed={test_seed} {arm} prefix_alarms={row['cases']['stationary']['pre_alarm_windows']}", flush=True)
    report.update(complete=True, seconds=time.monotonic() - started)
    write(output, report)
    print(f"DONE {PROTOCOL} {env} seed={test_seed} offline_interactions=0", flush=True)
    return report


def summarize(root=ROOT, fresh=FRESH):
    report = dict(protocol=PROTOCOL, config=CONFIG, complete=True, development_only=True,
        additional_interactions=0, collection_interactions=0, initialization_probes=0,
        predictor_optimizer_steps=0, residual_scale_optimizer_steps=0,
        policy_optimizer_steps=0, critic_optimizer_steps=0, encoder_optimizer_steps=0,
        environments={}, retained_method="v25", evidence_integrated=False)
    for env in ENVIRONMENTS:
        # Count every physical collection once, not once per dependent fold using it.
        collections = [fresh_data(fresh, env, seed)[2] for seed in SEEDS]
        for row in collections:
            assert all(row[k] == 0 for k in ("policy_optimizer_steps", "critic_optimizer_steps", "encoder_optimizer_steps"))
            for key in ("additional_interactions", "collection_interactions", "initialization_probes"):
                report[key] += row[key]
        runs = [json.loads((root / env.replace("-", "_") / f"test_seed{seed}" / "readout.json").read_text()) for seed in SEEDS]
        for seed, row in zip(SEEDS, runs):
            validate_run(row, env, seed, PROTOCOL)
            assert row["scale_fit_steps"] == 15000
            assert row["scale_fit_sources"] == row["roles"]["fit_seeds"] + [row["roles"]["scale_seed"]]
            assert [r["seed"] for r in row["fresh_fit_sources"]] == row["roles"]["fit_seeds"]
            assert all(row[k] == 0 for k in ("additional_interactions", "initialization_probes",
                                           "policy_optimizer_steps", "critic_optimizer_steps", "encoder_optimizer_steps"))
            for key in ("predictor_optimizer_steps", "residual_scale_optimizer_steps"):
                report[key] += row[key]
        report["environments"][env] = dict(arms={arm: screen(runs, arm) for arm in ARMS},
            affine_residual_control={arm: screen([r["affine_residual_control"] for r in runs], arm) for arm in ARMS},
            collections=[{k:r[k] for k in ("seed", "collection_interactions", "initialization_probes", "additional_interactions")}
                         for r in collections], runs=runs)
    passed = all(row["arms"]["conditional_scale"]["development_screen_passed"] for row in report["environments"].values())
    report["decision"] = "consider_fresh_seed_evidence_probe" if passed else "hold_conditional_evidence_integration"
    return report


def compact_report(report):
    compact = copy.deepcopy(report)
    for environment in compact["environments"].values():
        for row in environment["runs"]:
            for collection in (row["arms"], row["affine_residual_control"]["arms"]):
                for arm in collection.values():
                    arm["calibration"].pop("windows")
                    for case in arm["cases"].values():
                        case.pop("windows")
    return compact


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--env", choices=ENVIRONMENTS)
    parser.add_argument("--test-seed", type=int, choices=SEEDS)
    parser.add_argument("--source", type=Path, default=SOURCE)
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--control", type=Path, default=CONTROL)
    parser.add_argument("--fresh", type=Path, default=FRESH)
    parser.add_argument("--summarize", action="store_true")
    args = parser.parse_args()
    torch.set_num_threads(2)
    if args.summarize:
        report = summarize(args.root, args.fresh)
        write(args.root / "readout.json", report)
        write(args.root / "compact_readout.json", compact_report(report))
        print(json.dumps({env: row["arms"] for env, row in report["environments"].items()}))
    else:
        run(args.env, args.test_seed, args.source, args.root, args.control, args.fresh)
