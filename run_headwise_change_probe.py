"""Four-role, headwise change-evidence development from stored transitions."""

import argparse
import json
from pathlib import Path
import time

import numpy as np
import torch

from headwise_change_evidence import HeadwiseChangeEvidence, CONFIG, ARMS
from run_conditional_change_probe import CASES, CHANGE_AT, ENVIRONMENTS, SEEDS, case_summary
from run_episode_blocked_change_probe import SOURCE, normal_data, source_folder, source_readout, screen, write


PROTOCOL = "four_role_headwise_change_development_v1"
ROOT = Path("results/iclr_headwise_change_20260930")


def fold_roles(test_seed):
    index = SEEDS.index(test_seed)
    calibration_seed = SEEDS[(index + 1) % len(SEEDS)]
    scale_seed = SEEDS[(index + 2) % len(SEEDS)]
    fit_seeds = [seed for seed in SEEDS if seed not in (test_seed, calibration_seed, scale_seed)]
    return dict(fit_seeds=fit_seeds, scale_seed=scale_seed,
                calibration_seed=calibration_seed, test_seed=test_seed)


def summarize_heads(evidence, evaluation):
    result = {}
    for index, name in enumerate(evidence.head_names):
        windows = [window["heads"][name] for window in evaluation["windows"]
                   if window["start_step"] > CHANGE_AT]
        result[name] = dict(domain=evidence.domains[index],
            post_supported_fraction=float(evaluation["head_supported"][CHANGE_AT:, index].mean()),
            post_eligible_windows=sum(row["eligible"] for row in windows),
            post_alarm_windows=sum(row["alarm"] for row in windows))
    return result


def run(env, test_seed, source=SOURCE, root=ROOT):
    roles = fold_roles(test_seed)
    output = root / env.replace("-", "_") / f"test_seed{test_seed}" / "readout.json"
    if output.exists():
        saved = json.loads(output.read_text())
        assert (saved["protocol"], saved["env"], saved["roles"], saved["complete"], saved["config"]) == (
            PROTOCOL, env, roles, True, CONFIG)
        return saved
    original = source_readout(source, env, test_seed)
    started = time.monotonic()
    report = dict(protocol=PROTOCOL, env=env, roles=roles, config=CONFIG,
        complete=False, development_only=True, additional_interactions=0, initialization_probes=0,
        predictor_optimizer_steps=0, residual_scale_optimizer_steps=0,
        policy_optimizer_steps=0, critic_optimizer_steps=0, encoder_optimizer_steps=0, arms={})
    write(output, report)
    fit_rows = [normal_data(source, env, seed, "expanded_normal") for seed in roles["fit_seeds"]]
    fit_x, fit_y = (np.concatenate([row[index] for row in fit_rows]) for index in (0, 1))
    scale_x, scale_y, scale_meta = normal_data(source, env, roles["scale_seed"], "expanded_normal")
    cal_x, cal_y, cal_meta = normal_data(source, env, roles["calibration_seed"], "expanded_normal")
    reward_dim = cal_meta["reward_dim"]
    assert all(row[2]["reward_dim"] == reward_dim for row in fit_rows) and scale_meta["reward_dim"] == reward_dim
    evidence = HeadwiseChangeEvidence(fit_x.shape[1], fit_y.shape[1], reward_dim)
    predictor_work = evidence.fit(fit_x, fit_y, len(fit_rows[0][0]), 80000000 + 10 * test_seed)
    scale_work = evidence.fit_scales(scale_x, scale_y, 81000000 + 10 * test_seed)
    report.update(fit_sources=[row[2] for row in fit_rows], scale_source=scale_meta,
        calibration_source=cal_meta, fit_steps=len(fit_x), scale_steps=len(scale_x), calibration_steps=len(cal_x),
        predictor_work=predictor_work, scale_work=scale_work,
        predictor_optimizer_steps=predictor_work["optimizer_steps"],
        residual_scale_optimizer_steps=scale_work["optimizer_steps"])
    calibration = evidence.calibrate(cal_x, cal_y)
    for arm in ARMS:
        row = dict(calibration=calibration[arm], cases={})
        report["arms"][arm] = row
        with np.load(source_folder(source, env, test_seed) / "transitions.npz") as data:
            for case in CASES:
                evaluation = evidence.evaluate(data[f"{case}_inputs"], data[f"{case}_targets"], arm)
                summary = case_summary(evaluation)
                summary["heads"] = summarize_heads(evidence, evaluation)
                if case in ("actor_change", "sampler_change"):
                    summary["intervention_active"] = original["cases"][case]["intervention_active"]
                row["cases"][case] = summary
        write(output, report)
        print(f"ARM {env} test_seed={test_seed} arm={arm} "
              f"actor_alarm={row['cases']['actor_change']['any_post_alarm']} "
              f"mode2_available={row['cases']['mode2']['available']}", flush=True)
    report.update(complete=True, seconds=time.monotonic() - started)
    write(output, report)
    print(f"DONE {PROTOCOL} {env} test_seed={test_seed} interactions=0", flush=True)
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
            assert (row["fit_steps"], row["scale_steps"], row["calibration_steps"]) == (10000, 5000, 5000)
            assert all(row[key] == 0 for key in ("additional_interactions", "initialization_probes",
                                                "policy_optimizer_steps", "critic_optimizer_steps", "encoder_optimizer_steps"))
            for key in ("predictor_optimizer_steps", "residual_scale_optimizer_steps"):
                report[key] += row[key]
        report["environments"][env] = dict(arms={arm: screen(runs, arm) for arm in ARMS}, runs=runs)
    passed = all(row["arms"]["conditional_scale"]["development_screen_passed"] for row in report["environments"].values())
    report["decision"] = "consider_fresh_seed_evidence_probe" if passed else "hold_conditional_evidence_integration"
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--env", choices=ENVIRONMENTS)
    parser.add_argument("--test-seed", type=int, choices=SEEDS)
    parser.add_argument("--source", type=Path, default=SOURCE)
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--summarize", action="store_true")
    args = parser.parse_args()
    torch.set_num_threads(2)
    if args.summarize:
        report = summarize(args.root)
        write(args.root / "readout.json", report)
        compact = json.loads(json.dumps(report))
        for environment in compact["environments"].values():
            for row in environment["runs"]:
                for arm in row["arms"].values():
                    arm["calibration"].pop("windows")
                    for case in arm["cases"].values():
                        case.pop("windows")
        write(args.root / "compact_readout.json", compact)
        print(json.dumps({env: row["arms"] for env, row in report["environments"].items()}))
    else:
        run(args.env, args.test_seed, args.source, args.root)
