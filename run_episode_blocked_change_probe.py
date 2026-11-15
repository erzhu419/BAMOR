"""Source-blocked development screen using existing normal trajectories only."""

import argparse
import json
from pathlib import Path
import time

import numpy as np
import torch

from episode_blocked_change_evidence import EpisodeBlockedChangeEvidence, ENSEMBLE_SIZE, ENSEMBLE_CONFIG
from run_conditional_change_probe import (CASES, CHANGE_AT, ENVIRONMENTS, SEEDS,
                                          PROTOCOL as SOURCE_PROTOCOL, case_summary)


PROTOCOL = "source_blocked_change_development_v1"
SOURCE = Path("results/iclr_conditional_change_20260930")
ROOT = Path("results/iclr_episode_blocked_change_20260930")
ARMS = ("fixed_only", "expanded_normal")


def write(path, report):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")


def fold_roles(test_seed):
    index = SEEDS.index(test_seed)
    calibration_seed = SEEDS[(index + 1) % len(SEEDS)]
    fit_seeds = [seed for seed in SEEDS if seed not in (test_seed, calibration_seed)]
    return dict(fit_seeds=fit_seeds, calibration_seed=calibration_seed, test_seed=test_seed)


def source_folder(source, env, seed):
    return source / env.replace("-", "_") / f"seed{seed}"


def source_readout(source, env, seed):
    original = json.loads((source_folder(source, env, seed) / "readout.json").read_text())
    assert (original["protocol"], original["env"], original["seed"], original["complete"]) == (
        SOURCE_PROTOCOL, env, seed, True)
    assert original["paired_prefixes_verified"] == len(CASES) - 1
    # Only branches collected with the environment unchanged may enter development fits.
    assert all(original["cases"][case]["ground_truth_mode_history_audit_only"] == [[0, 0]]
               for case in CASES[:3])
    return original


def normal_data(source, env, seed, arm):
    source_readout(source, env, seed)
    with np.load(source_folder(source, env, seed) / "transitions.npz") as data:
        inputs = [data["prefix_inputs"], data["stationary_inputs"]]
        targets = [data["prefix_targets"], data["stationary_targets"]]
        if arm == "expanded_normal":
            for case in ("sampler_change", "actor_change"):
                # Their shared initial500 steps are already in stationary_inputs.
                inputs.append(data[f"{case}_inputs"][CHANGE_AT:])
                targets.append(data[f"{case}_targets"][CHANGE_AT:])
        completed = sum(int(data[f"{case}_terminals"].any(1).sum()) for case in ("prefix", "stationary"))
        metadata = dict(seed=seed, steps=sum(map(len, inputs)),
            fixed_rollout_completed_episodes=completed, reward_dim=data["prefix_weights"].shape[1])
        return np.concatenate(inputs), np.concatenate(targets), metadata


def run(env, test_seed, source=SOURCE, root=ROOT):
    roles = fold_roles(test_seed)
    output = root / env.replace("-", "_") / f"test_seed{test_seed}" / "readout.json"
    if output.exists():
        saved = json.loads(output.read_text())
        assert (saved["protocol"], saved["env"], saved["roles"], saved["complete"]) == (PROTOCOL, env, roles, True)
        assert saved["predictor_config"] == ENSEMBLE_CONFIG and saved["ensemble_size"] == ENSEMBLE_SIZE
        return saved
    original = source_readout(source, env, test_seed)
    report = dict(protocol=PROTOCOL, env=env, roles=roles, complete=False,
        predictor_config=ENSEMBLE_CONFIG, ensemble_size=ENSEMBLE_SIZE,
        additional_interactions=0, initialization_probes=0, predictor_optimizer_steps=0,
        policy_optimizer_steps=0, critic_optimizer_steps=0, encoder_optimizer_steps=0,
        development_only=True, arms={})
    write(output, report)
    started = time.monotonic()
    for arm in ARMS:
        fit_rows = [normal_data(source, env, seed, arm) for seed in roles["fit_seeds"]]
        fit_x = np.concatenate([row[0] for row in fit_rows])
        fit_y = np.concatenate([row[1] for row in fit_rows])
        cal_x, cal_y, calibration_source = normal_data(source, env, roles["calibration_seed"], arm)
        reward_dim = calibration_source["reward_dim"]
        assert all(row[2]["reward_dim"] == reward_dim for row in fit_rows)
        evidence = EpisodeBlockedChangeEvidence(fit_x.shape[1], fit_y.shape[1], reward_dim)
        work = evidence.fit(fit_x, fit_y, 70000000 + 10 * test_seed)
        report["predictor_optimizer_steps"] += work["optimizer_steps"]
        row = dict(fit_sources=[item[2] for item in fit_rows], calibration_source=calibration_source,
            fit_steps=len(fit_x), calibration_steps=len(cal_x), predictor_work=work,
            calibration=evidence.calibrate(cal_x, cal_y), cases={})
        report["arms"][arm] = row
        write(output, report)
        with np.load(source_folder(source, env, test_seed) / "transitions.npz") as data:
            for case in CASES:
                inputs, targets = data[f"{case}_inputs"], data[f"{case}_targets"]
                evaluation = evidence.evaluate(inputs, targets)
                parts = evidence.components(inputs, targets)
                summary = case_summary(evaluation)
                supported = evaluation["supported"][CHANGE_AT:]
                summary["post_raw_error_mean"] = float(parts["raw_errors"][CHANGE_AT:][supported].mean()) if supported.any() else None
                summary["post_disagreement_mean"] = float(parts["disagreement"][CHANGE_AT:][supported].mean()) if supported.any() else None
                if case in ("actor_change", "sampler_change"):
                    summary["intervention_active"] = original["cases"][case]["intervention_active"]
                row["cases"][case] = summary
        write(output, report)
        print(f"ARM {env} test_seed={test_seed} arm={arm} fit={len(fit_x)} cal={len(cal_x)} "
              f"actor_available={row['cases']['actor_change']['available']}", flush=True)
    report.update(complete=True, seconds=time.monotonic() - started)
    write(output, report)
    print(f"DONE {PROTOCOL} {env} test_seed={test_seed} interactions=0", flush=True)
    return report


def screen(runs, arm):
    cases = {case: dict(available=sum(row["arms"][arm]["cases"][case]["available"] for row in runs),
        post_alarm=sum(row["arms"][arm]["cases"][case]["any_post_alarm"] for row in runs),
        timely_alarm=sum(row["arms"][arm]["cases"][case]["timely_alarm"] for row in runs)) for case in CASES}
    prefix = dict(available=sum(row["arms"][arm]["cases"]["stationary"]["pre_available"] for row in runs),
        alarm=sum(row["arms"][arm]["cases"]["stationary"]["pre_alarm_windows"] > 0 for row in runs))
    passed = (prefix["available"] >= 4 and prefix["alarm"] <= 1
        and all(cases[case]["available"] >= 4 and cases[case]["post_alarm"] <= 1 for case in CASES[:3])
        and all(cases[case]["timely_alarm"] >= 4 for case in CASES[3:])
        and all(row["arms"][arm]["cases"][case]["intervention_active"] for row in runs
                for case in ("sampler_change", "actor_change")))
    return dict(shared_prefix=prefix, cases=cases, development_screen_passed=passed)


def summarize(root=ROOT):
    report = dict(protocol=PROTOCOL, complete=True, development_only=True,
        additional_interactions=0, initialization_probes=0, predictor_optimizer_steps=0,
        policy_optimizer_steps=0, critic_optimizer_steps=0, encoder_optimizer_steps=0,
        predictor_config=ENSEMBLE_CONFIG, ensemble_size=ENSEMBLE_SIZE, environments={},
        retained_method="v25", evidence_integrated=False)
    for env in ENVIRONMENTS:
        runs = [json.loads((root / env.replace("-", "_") / f"test_seed{seed}" / "readout.json").read_text()) for seed in SEEDS]
        for seed, row in zip(SEEDS, runs):
            assert (row["protocol"], row["env"], row["roles"], row["complete"]) == (PROTOCOL, env, fold_roles(seed), True)
            assert row["predictor_config"] == ENSEMBLE_CONFIG and row["ensemble_size"] == ENSEMBLE_SIZE
            assert row["predictor_optimizer_steps"] == len(ARMS) * ENSEMBLE_SIZE * ENSEMBLE_CONFIG["updates"]
            assert all(row[key] == 0 for key in ("additional_interactions", "initialization_probes",
                                                "policy_optimizer_steps", "critic_optimizer_steps", "encoder_optimizer_steps"))
            report["predictor_optimizer_steps"] += row["predictor_optimizer_steps"]
        report["environments"][env] = dict(arms={arm: screen(runs, arm) for arm in ARMS}, runs=runs)
    passed = all(row["arms"]["expanded_normal"]["development_screen_passed"] for row in report["environments"].values())
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
            for run_row in environment["runs"]:
                for arm in run_row["arms"].values():
                    arm["calibration"].pop("windows")
                    for case in arm["cases"].values():
                        case.pop("windows")
        write(args.root / "compact_readout.json", compact)
        print(json.dumps({env: row["arms"] for env, row in report["environments"].items()}))
    else:
        run(args.env, args.test_seed, args.source, args.root)
