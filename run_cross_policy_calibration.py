"""Compare calibration sources with a frozen sensitivity-guarded predictor."""

import argparse
import copy
import json
from pathlib import Path
import time

import numpy as np
import torch

from collect_matched_scale_normals import ROOT as FRESH, normal_data as fresh_data
from headwise_change_evidence import CONFIG
from run_calibration_expansion import verify_windows
from run_cross_source_scale_probe import validate_run
from run_headwise_change_probe import (
    CASES, ENVIRONMENTS, SEEDS, SOURCE, fold_roles, normal_data, source_folder,
    source_readout, summarize_heads, case_summary, screen, write)
from run_sensitivity_guarded_selection import (
    GuardedEvidence, GUARDED, ROOT as PREVIOUS, PROTOCOL as PREVIOUS_PROTOCOL)


PROTOCOL = "fixed_model_cross_policy_calibration_v1"
ROOT = Path("results/iclr_cross_policy_calibration_20261001")
CALIBRATIONS = ("original_calibration", "fresh_same_policy", "fresh_training_policy", "pooled_two_policy")
PRIMARY = "pooled_two_policy"
ZERO_WORK = ("additional_interactions", "initialization_probes", "validation_scale_optimizer_steps",
             "policy_optimizer_steps", "critic_optimizer_steps", "encoder_optimizer_steps")


def balanced_windows(same, training, window):
    assert len(same[0]) == len(same[1]) == len(training[0]) == len(training[1]) == 5000
    count = 5000 // window
    assert 5000 % window == 0 and count % 2 == 0
    indices = (np.arange(0, count, 2), np.arange(1, count, 2))
    pooled = tuple(np.concatenate([row[i].reshape(count, window, -1)[index].reshape(-1, row[i].shape[1])
        for row, index in zip((same, training), indices)]) for i in (0, 1))
    sources = [dict(seed=row[2]["seed"], role=role, steps=2500, window_indices=index.tolist(),
                    dataset="fresh_normal")
        for row, index, role in zip((same, training), indices, ("calibration_policy", "training_scale_policy"))]
    return (*pooled, sources)


def calibrate_fixed_support(evidence, inputs, targets):
    support = evidence.distances(inputs) <= evidence.radii
    windows = evidence._windows(evidence.raw_losses(inputs, targets) / evidence.scales(inputs, GUARDED), support)
    eligible = [w["statistic"] for w in windows if w["eligible"]]
    assert len(eligible) >= 2
    evidence.thresholds[GUARDED] = float(np.quantile(eligible, evidence.config["quantile"]))
    return dict(error_threshold=evidence.thresholds[GUARDED],
        support_radii=dict(zip(evidence.head_names, evidence.radii.tolist())),
        support_fixed_from_original=True, eligible_windows=len(eligible), total_windows=len(windows), windows=windows)


def validate(row, env, seed):
    assert (row["protocol"], row["env"], row["roles"], row["complete"], row["config"]) == (
        PROTOCOL, env, fold_roles(seed), True, CONFIG)
    assert row["original_calibration_reproduced"] and row["evaluation_scores_unchanged"]
    assert (row["fit_steps"], row["scale_fit_steps"]) == (10000, 15000)
    assert row["scale_fit_sources"] == row["roles"]["fit_seeds"] + [row["roles"]["scale_seed"]]
    assert row["predictor_optimizer_steps"] == 2 * CONFIG["updates"]
    assert row["residual_scale_optimizer_steps"] == (row["reward_dim"] + 1) * CONFIG["scale_updates"]
    assert all(row[key] == 0 for key in ZERO_WORK)
    assert set(row["arms"]) == set(CALIBRATIONS)
    radii = row["arms"]["original_calibration"]["calibration"]["support_radii"]
    for arm in row["arms"].values():
        assert arm["calibration_steps"] == 5000
        assert sum(s["steps"] for s in arm["sources"]) == 5000
        assert arm["calibration"]["total_windows"] == 5000 // CONFIG["window"]
        assert arm["calibration"]["support_radii"] == radii
        assert all(s["seed"] != seed for s in arm["sources"])


def run(env, test_seed, source=SOURCE, root=ROOT, previous=PREVIOUS, fresh=FRESH):
    roles = fold_roles(test_seed)
    output = root / env.replace("-", "_") / f"test_seed{test_seed}" / "readout.json"
    if output.exists():
        saved = json.loads(output.read_text())
        validate(saved, env, test_seed)
        return saved
    prior = json.loads((previous / env.replace("-", "_") / f"test_seed{test_seed}" / "readout.json").read_text())
    validate_run(prior, env, test_seed, PREVIOUS_PROTOCOL)
    assert prior["scale_fit_steps"] == 15000
    started = time.monotonic()
    report = dict(protocol=PROTOCOL, env=env, roles=roles, config=CONFIG, complete=False,
        development_only=True, reconstruction_only=True, arms={},
        predictor_optimizer_steps=0, residual_scale_optimizer_steps=0,
        frozen_selection=dict(protocol=PREVIOUS_PROTOCOL, arm=GUARDED,
            global_weight=prior["selections"][GUARDED]["global_weight"]), primary_calibration=PRIMARY)
    report.update({key: 0 for key in ZERO_WORK})
    write(output, report)
    fit_rows = [normal_data(source, env, s, "expanded_normal") for s in roles["fit_seeds"]]
    fit_x, fit_y = (np.concatenate([row[i] for row in fit_rows]) for i in (0, 1))
    scale_rows = [fresh_data(fresh, env, s) for s in roles["fit_seeds"]]
    scale_rows.append(normal_data(source, env, roles["scale_seed"], "expanded_normal"))
    scale_x, scale_y = (np.concatenate([row[i] for row in scale_rows]) for i in (0, 1))
    evidence = GuardedEvidence(fit_x.shape[1], fit_y.shape[1], fit_rows[0][2]["reward_dim"])
    evidence.weights = {arm: selection["global_weight"] for arm, selection in prior["selections"].items()}
    work = evidence.fit(fit_x, fit_y, len(fit_rows[0][0]), 80000000 + 10 * test_seed)
    assert work == prior["predictor_work"]
    report.update(predictor_optimizer_steps=work["optimizer_steps"], predictor_work=work)
    write(output, report)
    work = evidence.fit_scale_losses(scale_x, evidence.raw_losses(scale_x, scale_y), 81000000 + 10 * test_seed)
    assert work["optimizer_steps"] == prior["scale_work"]["optimizer_steps"]
    np.testing.assert_allclose(work["global_scales"], prior["scale_work"]["global_scales"], rtol=1e-6)
    report.update(residual_scale_optimizer_steps=work["optimizer_steps"], scale_work=work,
        reward_dim=evidence.reward_dim, fit_steps=len(fit_x), scale_fit_steps=len(scale_x),
        scale_fit_sources=prior["scale_fit_sources"])
    write(output, report)
    old = prior["arms"][GUARDED]
    evidence.thresholds = {}
    evidence.radii = np.array([old["calibration"]["support_radii"][name] for name in evidence.head_names])
    original = normal_data(source, env, roles["calibration_seed"], "expanded_normal")
    same = fresh_data(fresh, env, roles["calibration_seed"])
    training = fresh_data(fresh, env, roles["scale_seed"])
    pooled_x, pooled_y, pooled_sources = balanced_windows(same, training, CONFIG["window"])
    sources = [dict(seed=row[2]["seed"], steps=5000, dataset=dataset, role=role)
        for row, dataset, role in zip((original, same, training),
            ("original_normal", "fresh_normal", "fresh_normal"),
            ("calibration_policy", "calibration_policy", "training_scale_policy"))]
    calibrations = [(*row[:2], [meta]) for row, meta in zip((original, same, training), sources)]
    calibrations.append((pooled_x, pooled_y, pooled_sources))
    original_cases = source_readout(source, env, test_seed)
    with np.load(source_folder(source, env, test_seed) / "transitions.npz") as data:
        for arm, (x, y, provenance) in zip(CALIBRATIONS, calibrations):
            assert len(x) == len(y) == 5000
            calibration = calibrate_fixed_support(evidence, x, y)
            if arm == "original_calibration":
                verify_windows(calibration["windows"], old["calibration"]["windows"])
                np.testing.assert_allclose(calibration["error_threshold"], old["calibration"]["error_threshold"],
                    rtol=1e-5, atol=1e-8)
            row = dict(calibration=calibration, calibration_steps=len(x), sources=provenance, cases={})
            report["arms"][arm] = row
            for case in CASES:
                evaluation = evidence.evaluate(data[f"{case}_inputs"], data[f"{case}_targets"], GUARDED)
                summary = case_summary(evaluation)
                summary["heads"] = summarize_heads(evidence, evaluation)
                verify_windows(summary["windows"], old["cases"][case]["windows"])
                if arm == "original_calibration":
                    assert [w["alarm"] for w in summary["windows"]] == [w["alarm"] for w in old["cases"][case]["windows"]]
                if case in ("sampler_change", "actor_change"):
                    summary["intervention_active"] = original_cases["cases"][case]["intervention_active"]
                row["cases"][case] = summary
            write(output, report)
            print(f"ARM {env} seed={test_seed} {arm} threshold={calibration['error_threshold']}", flush=True)
    report.update(complete=True, original_calibration_reproduced=True, evaluation_scores_unchanged=True,
        seconds=time.monotonic() - started)
    validate(report, env, test_seed)
    write(output, report)
    print(f"DONE {env} seed={test_seed} added_interactions=0", flush=True)
    return report


def summarize(root=ROOT):
    report = dict(protocol=PROTOCOL, config=CONFIG, complete=True, development_only=True,
        reconstruction_only=True, environments={}, retained_method="v25", evidence_integrated=False,
        primary_calibration=PRIMARY, predictor_optimizer_steps=0, residual_scale_optimizer_steps=0,
        reused_acquisition_interactions_counted_in_parent=140100,
        reused_acquisition_ledgers=dict(original_change_probe=90070, fresh_normal_collection=50030))
    report.update({key: 0 for key in ZERO_WORK})
    for env in ENVIRONMENTS:
        runs = [json.loads((root / env.replace("-", "_") / f"test_seed{seed}" / "readout.json").read_text()) for seed in SEEDS]
        for seed, row in zip(SEEDS, runs):
            validate(row, env, seed)
            for key in ("predictor_optimizer_steps", "residual_scale_optimizer_steps"):
                report[key] += row[key]
        report["environments"][env] = dict(arms={arm: screen(runs, arm) for arm in CALIBRATIONS}, runs=runs)
    passed = all(e["arms"][PRIMARY]["development_screen_passed"] for e in report["environments"].values())
    report["decision"] = "consider_fresh_seed_evidence_probe" if passed else "hold_conditional_evidence_integration"
    return report


def compact_report(report):
    compact = copy.deepcopy(report)
    for environment in compact["environments"].values():
        for row in environment["runs"]:
            for arm in row["arms"].values():
                arm["calibration"].pop("windows")
                for case in arm["cases"].values():
                    case.pop("windows")
                    case.pop("heads")
    return compact


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
        output = args.root / "compact_readout.json"
        write(output, compact_report(report))
        print(json.dumps(dict(environments={env: e["arms"] for env, e in report["environments"].items()},
            decision=report["decision"], compact_readout_bytes=output.stat().st_size)))
    else:
        run(args.env, args.test_seed, root=args.root)
