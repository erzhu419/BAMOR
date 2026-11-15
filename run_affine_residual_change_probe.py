"""Four-role affine-residual development, paired with saved plain-MLP controls."""

import argparse
import copy
import json
from pathlib import Path
import time

import numpy as np
import torch

from affine_residual_change_evidence import AffineResidualHeadwiseEvidence
from headwise_change_evidence import CONFIG, ARMS
from run_headwise_change_probe import (
    CASES, CHANGE_AT, ENVIRONMENTS, SEEDS, fold_roles, summarize_heads, case_summary,
    SOURCE, normal_data, source_folder, source_readout, screen, write,
    ROOT as CONTROL, PROTOCOL as CONTROL_PROTOCOL)


PROTOCOL = "four_role_affine_residual_change_development_v1"
ROOT = Path("results/iclr_affine_residual_change_20261001")


def validate_control(row, env, seed):
    assert (row["protocol"], row["env"], row["roles"], row["complete"], row["config"]) == (
        CONTROL_PROTOCOL, env, fold_roles(seed), True, CONFIG)
    assert row["predictor_optimizer_steps"] == 2 * CONFIG["updates"]
    assert row["residual_scale_optimizer_steps"] == (row["scale_source"]["reward_dim"] + 1) * CONFIG["scale_updates"]
    assert (row["fit_steps"], row["scale_steps"], row["calibration_steps"]) == (10000, 5000, 5000)


def loss_summary(evidence, inputs, targets):
    affine, corrected = evidence.affine_losses(inputs, targets), evidence.raw_losses(inputs, targets)
    return dict(steps=len(inputs), affine_head_mse=dict(zip(evidence.head_names, affine.mean(0).tolist())),
                residual_head_mse=dict(zip(evidence.head_names, corrected.mean(0).tolist())))


def run(env, test_seed, source=SOURCE, root=ROOT, control=CONTROL):
    roles = fold_roles(test_seed)
    output = root / env.replace("-", "_") / f"test_seed{test_seed}" / "readout.json"
    if output.exists():
        saved = json.loads(output.read_text())
        assert (saved["protocol"], saved["env"], saved["roles"], saved["complete"], saved["config"]) == (
            PROTOCOL, env, roles, True, CONFIG)
        return saved
    baseline = json.loads((control / env.replace("-", "_") / f"test_seed{test_seed}" / "readout.json").read_text())
    validate_control(baseline, env, test_seed)
    original = source_readout(source, env, test_seed)
    started = time.monotonic()
    report = dict(protocol=PROTOCOL, env=env, roles=roles, config=CONFIG,
        complete=False, development_only=True, additional_interactions=0, initialization_probes=0,
        predictor_optimizer_steps=0, residual_scale_optimizer_steps=0,
        policy_optimizer_steps=0, critic_optimizer_steps=0, encoder_optimizer_steps=0,
        arms={}, plain_mlp_control=dict(protocol=CONTROL_PROTOCOL, reused=True,
            arms=baseline["arms"], global_scales=baseline["scale_work"]["global_scales"]), normal_losses={})
    write(output, report)
    fit_rows = [normal_data(source, env, seed, "expanded_normal") for seed in roles["fit_seeds"]]
    fit_x, fit_y = (np.concatenate([row[index] for row in fit_rows]) for index in (0, 1))
    scale_x, scale_y, scale_meta = normal_data(source, env, roles["scale_seed"], "expanded_normal")
    cal_x, cal_y, cal_meta = normal_data(source, env, roles["calibration_seed"], "expanded_normal")
    evidence = AffineResidualHeadwiseEvidence(fit_x.shape[1], fit_y.shape[1], cal_meta["reward_dim"])
    predictor_work = evidence.fit(fit_x, fit_y, len(fit_rows[0][0]), 80000000 + 10 * test_seed)
    scale_work = evidence.fit_scales(scale_x, scale_y, 81000000 + 10 * test_seed)
    assert predictor_work["action_heads"] == baseline["predictor_work"]["action_heads"]
    report.update(fit_sources=[row[2] for row in fit_rows], scale_source=scale_meta,
        calibration_source=cal_meta, fit_steps=len(fit_x), scale_steps=len(scale_x), calibration_steps=len(cal_x),
        predictor_work=predictor_work, scale_work=scale_work,
        predictor_optimizer_steps=predictor_work["optimizer_steps"],
        residual_scale_optimizer_steps=scale_work["optimizer_steps"])
    for name, inputs, targets in (("fit", fit_x, fit_y), ("scale", scale_x, scale_y), ("calibration", cal_x, cal_y)):
        report["normal_losses"][name] = loss_summary(evidence, inputs, targets)
    calibration = evidence.calibrate(cal_x, cal_y)
    for arm in ARMS:
        row = dict(calibration=calibration[arm], cases={})
        report["arms"][arm] = row
        assert calibration[arm]["support_radii"] == baseline["arms"][arm]["calibration"]["support_radii"]
        with np.load(source_folder(source, env, test_seed) / "transitions.npz") as data:
            for case in CASES:
                inputs, targets = data[f"{case}_inputs"], data[f"{case}_targets"]
                evaluation = evidence.evaluate(inputs, targets, arm)
                summary = case_summary(evaluation)
                summary["heads"] = summarize_heads(evidence, evaluation)
                for new, old in zip(summary["windows"], baseline["arms"][arm]["cases"][case]["windows"]):
                    assert {h: v["supported_steps"] for h, v in new["heads"].items()} == {
                        h: v["supported_steps"] for h, v in old["heads"].items()}
                if case in ("actor_change", "sampler_change"):
                    summary["intervention_active"] = original["cases"][case]["intervention_active"]
                row["cases"][case] = summary
                if arm == ARMS[0] and case in CASES[:3]:
                    start = 0 if case == "stationary" else CHANGE_AT
                    report["normal_losses"][f"test_{case}"] = loss_summary(evidence, inputs[start:], targets[start:])
        write(output, report)
        print(f"ARM {env} seed={test_seed} {arm} prefix_alarms={row['cases']['stationary']['pre_alarm_windows']}", flush=True)
    report.update(complete=True, seconds=time.monotonic() - started)
    write(output, report)
    print(f"DONE {PROTOCOL} {env} seed={test_seed} interactions=0", flush=True)
    return report


def summarize(root=ROOT):
    report = dict(protocol=PROTOCOL, config=CONFIG, complete=True, development_only=True,
        additional_interactions=0, initialization_probes=0, predictor_optimizer_steps=0,
        residual_scale_optimizer_steps=0, policy_optimizer_steps=0, critic_optimizer_steps=0,
        encoder_optimizer_steps=0, environments={}, retained_method="v25", evidence_integrated=False)
    for env in ENVIRONMENTS:
        runs = [json.loads((root / env.replace("-", "_") / f"test_seed{seed}" / "readout.json").read_text()) for seed in SEEDS]
        for seed, row in zip(SEEDS, runs):
            assert (row["protocol"], row["env"], row["roles"], row["complete"], row["config"]) == (
                PROTOCOL, env, fold_roles(seed), True, CONFIG)
            assert row["predictor_optimizer_steps"] == 2 * CONFIG["updates"]
            assert row["residual_scale_optimizer_steps"] == (row["scale_source"]["reward_dim"] + 1) * CONFIG["scale_updates"]
            assert all(row[k] == 0 for k in ("additional_interactions", "initialization_probes",
                                           "policy_optimizer_steps", "critic_optimizer_steps", "encoder_optimizer_steps"))
            for key in ("predictor_optimizer_steps", "residual_scale_optimizer_steps"):
                report[key] += row[key]
        controls = [row["plain_mlp_control"] for row in runs]
        report["environments"][env] = dict(arms={arm: screen(runs, arm) for arm in ARMS},
            plain_mlp_control={arm: screen(controls, arm) for arm in ARMS}, runs=runs)
    passed = all(row["arms"]["conditional_scale"]["development_screen_passed"] for row in report["environments"].values())
    report["decision"] = "consider_fresh_seed_evidence_probe" if passed else "hold_conditional_evidence_integration"
    return report


def compact_report(report):
    compact = copy.deepcopy(report)
    for environment in compact["environments"].values():
        for row in environment["runs"]:
            for collection in (row["arms"], row["plain_mlp_control"]["arms"]):
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
    parser.add_argument("--summarize", action="store_true")
    args = parser.parse_args()
    torch.set_num_threads(2)
    if args.summarize:
        report = summarize(args.root)
        write(args.root / "readout.json", report)
        write(args.root / "compact_readout.json", compact_report(report))
        print(json.dumps({env: row["arms"] for env, row in report["environments"].items()}))
    else:
        run(args.env, args.test_seed, args.source, args.root, args.control)
