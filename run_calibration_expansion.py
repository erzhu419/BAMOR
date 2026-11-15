"""Expand held-source calibration while preserving prediction, scales and support."""

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
from run_cross_source_scale_probe import validate_run
from run_headwise_change_probe import (
    CASES, ENVIRONMENTS, SEEDS, SOURCE, fold_roles, normal_data, source_folder,
    source_readout, summarize_heads, case_summary, screen, write)
from run_matched_residual_scale_probe import ROOT as CONTROL, PROTOCOL as CONTROL_PROTOCOL


PROTOCOL = "fixed_support_calibration_expansion_v1"
ROOT = Path("results/iclr_calibration_expansion_20261001")


def verify_windows(reconstructed, saved):
    assert len(reconstructed) == len(saved)
    for new, old in zip(reconstructed, saved):
        for key in ("start_step", "end_step", "eligible", "supported_steps"):
            assert new[key] == old[key], f"Original window changed {key}"
        assert new["heads"].keys() == old["heads"].keys()
        for name in new["heads"]:
            for key in ("eligible", "supported_steps", "domain"):
                assert new["heads"][name][key] == old["heads"][name][key]
            if old["heads"][name]["eligible"]:
                np.testing.assert_allclose(new["heads"][name]["statistic"],
                    old["heads"][name]["statistic"], rtol=1e-5, atol=1e-8)
        if old["eligible"]:
            np.testing.assert_allclose(new["statistic"], old["statistic"], rtol=1e-5, atol=1e-8)


def calibrate_fixed_support(evidence, inputs, targets, original_steps):
    supported = evidence.distances(inputs) <= evidence.radii
    raw = evidence.raw_losses(inputs, targets)
    report = {}
    for arm in ARMS:
        windows = evidence._windows(raw / evidence.scales(inputs, arm), supported)
        eligible = [window["statistic"] for window in windows if window["eligible"]]
        assert len(eligible) >= 2
        threshold = float(np.quantile(eligible, evidence.config["quantile"]))
        evidence.thresholds[arm] = threshold
        report[arm] = dict(error_threshold=threshold,
            support_radii=dict(zip(evidence.head_names, evidence.radii.tolist())),
            support_fixed_from_original=True, eligible_windows=len(eligible), total_windows=len(windows),
            fresh_eligible_windows=sum(w["eligible"] for w in windows if w["start_step"] > original_steps),
            windows=windows)
    return report


def run(env, test_seed, source=SOURCE, root=ROOT, control=CONTROL, fresh=FRESH):
    roles = fold_roles(test_seed)
    output = root / env.replace("-", "_") / f"test_seed{test_seed}" / "readout.json"
    if output.exists():
        saved = json.loads(output.read_text())
        assert (saved["protocol"], saved["env"], saved["roles"], saved["complete"], saved["config"]) == (
            PROTOCOL, env, roles, True, CONFIG)
        return saved
    previous = json.loads((control / env.replace("-", "_") / f"test_seed{test_seed}" / "readout.json").read_text())
    validate_run(previous, env, test_seed, CONTROL_PROTOCOL)
    assert previous["scale_fit_steps"] == 15000
    assert previous["scale_fit_sources"] == roles["fit_seeds"] + [roles["scale_seed"]]
    started = time.monotonic()
    report = dict(protocol=PROTOCOL, env=env, roles=roles, config=CONFIG, complete=False,
        development_only=True, additional_interactions=0, initialization_probes=0,
        predictor_optimizer_steps=0, residual_scale_optimizer_steps=0,
        policy_optimizer_steps=0, critic_optimizer_steps=0, encoder_optimizer_steps=0,
        reconstruction_only=True, arms={},
        matched_scale_control=dict(protocol=CONTROL_PROTOCOL, reused=True, arms=previous["arms"]))
    write(output, report)
    fit_rows = [normal_data(source, env, seed, "expanded_normal") for seed in roles["fit_seeds"]]
    fit_x, fit_y = (np.concatenate([row[index] for row in fit_rows]) for index in (0, 1))
    scale_rows = [fresh_data(fresh, env, seed) for seed in roles["fit_seeds"]]
    scale_rows.append(normal_data(source, env, roles["scale_seed"], "expanded_normal"))
    scale_x, scale_y = (np.concatenate([row[index] for row in scale_rows]) for index in (0, 1))
    cal_x, cal_y, cal_meta = normal_data(source, env, roles["calibration_seed"], "expanded_normal")
    fresh_x, fresh_y, fresh_meta = fresh_data(fresh, env, roles["calibration_seed"])
    assert (len(fit_x), len(scale_x), len(cal_x), len(fresh_x)) == (10000, 15000, 5000, 5000)
    evidence = AffineResidualHeadwiseEvidence(fit_x.shape[1], fit_y.shape[1], cal_meta["reward_dim"])
    predictor_work = evidence.fit(fit_x, fit_y, len(fit_rows[0][0]), 80000000 + 10 * test_seed)
    report.update(predictor_optimizer_steps=predictor_work["optimizer_steps"], predictor_work=predictor_work)
    write(output, report)
    assert predictor_work == previous["predictor_work"]
    scale_work = evidence.fit_scale_losses(scale_x, evidence.raw_losses(scale_x, scale_y), 81000000 + 10 * test_seed)
    report.update(residual_scale_optimizer_steps=scale_work["optimizer_steps"], scale_work=scale_work)
    write(output, report)
    assert scale_work["optimizer_steps"] == previous["scale_work"]["optimizer_steps"]
    np.testing.assert_allclose(scale_work["global_scales"], previous["scale_work"]["global_scales"], rtol=1e-6)
    reconstructed = evidence.calibrate(cal_x, cal_y)
    for arm in ARMS:
        original = previous["arms"][arm]["calibration"]
        assert reconstructed[arm]["support_radii"] == original["support_radii"]
        np.testing.assert_allclose(reconstructed[arm]["error_threshold"], original["error_threshold"], rtol=1e-5, atol=1e-8)
        verify_windows(reconstructed[arm]["windows"], original["windows"])
    expanded_x, expanded_y = np.concatenate([cal_x, fresh_x]), np.concatenate([cal_y, fresh_y])
    calibration = calibrate_fixed_support(evidence, expanded_x, expanded_y, len(cal_x))
    report.update(fit_steps=len(fit_x), scale_fit_steps=len(scale_x), calibration_steps=len(expanded_x),
        original_calibration_steps=len(cal_x), fresh_calibration_steps=len(fresh_x),
        scale_fit_sources=previous["scale_fit_sources"], calibration_source=cal_meta,
        fresh_calibration_source=dict(seed=fresh_meta["seed"], steps=len(fresh_x), source_step=fresh_meta["source_step"]),
        original_calibration_reproduced=True, evaluation_scores_unchanged=False)
    original_cases = source_readout(source, env, test_seed)
    for arm in ARMS:
        assert calibration[arm]["support_radii"] == previous["arms"][arm]["calibration"]["support_radii"]
        verify_windows(calibration[arm]["windows"][:len(cal_x) // CONFIG["window"]], reconstructed[arm]["windows"])
        row = dict(calibration=calibration[arm], cases={})
        report["arms"][arm] = row
        with np.load(source_folder(source, env, test_seed) / "transitions.npz") as data:
            for case in CASES:
                evaluation = evidence.evaluate(data[f"{case}_inputs"], data[f"{case}_targets"], arm)
                summary = case_summary(evaluation)
                summary["heads"] = summarize_heads(evidence, evaluation)
                old = previous["arms"][arm]["cases"][case]
                verify_windows(summary["windows"], old["windows"])
                for key in ("available", "pre_available", "post_eligible_windows"):
                    assert summary[key] == old[key]
                if case in ("actor_change", "sampler_change"):
                    summary["intervention_active"] = original_cases["cases"][case]["intervention_active"]
                row["cases"][case] = summary
        write(output, report)
        print(f"ARM {env} seed={test_seed} {arm} "
              f"threshold={previous['arms'][arm]['calibration']['error_threshold']} -> {calibration[arm]['error_threshold']}", flush=True)
    report.update(complete=True, evaluation_scores_unchanged=True, seconds=time.monotonic() - started)
    write(output, report)
    return report


def summarize(root=ROOT):
    report = dict(protocol=PROTOCOL, config=CONFIG, complete=True, development_only=True,
        additional_interactions=0, initialization_probes=0, predictor_optimizer_steps=0,
        residual_scale_optimizer_steps=0, policy_optimizer_steps=0, critic_optimizer_steps=0,
        encoder_optimizer_steps=0, reconstruction_only=True, environments={}, retained_method="v25",
        evidence_integrated=False, reused_acquisition_interactions_counted_in_parent=50030)
    for env in ENVIRONMENTS:
        runs = [json.loads((root / env.replace("-", "_") / f"test_seed{seed}" / "readout.json").read_text()) for seed in SEEDS]
        for seed, row in zip(SEEDS, runs):
            assert (row["protocol"], row["env"], row["roles"], row["complete"], row["config"]) == (
                PROTOCOL, env, fold_roles(seed), True, CONFIG)
            assert row["original_calibration_reproduced"] and row["evaluation_scores_unchanged"]
            assert (row["fit_steps"], row["scale_fit_steps"], row["calibration_steps"],
                    row["original_calibration_steps"], row["fresh_calibration_steps"]) == (10000, 15000, 10000, 5000, 5000)
            assert row["scale_fit_sources"] == row["roles"]["fit_seeds"] + [row["roles"]["scale_seed"]]
            assert row["fresh_calibration_source"]["seed"] == row["roles"]["calibration_seed"]
            assert all(row[key] == 0 for key in ("additional_interactions", "initialization_probes",
                "policy_optimizer_steps", "critic_optimizer_steps", "encoder_optimizer_steps"))
            for key in ("predictor_optimizer_steps", "residual_scale_optimizer_steps"):
                report[key] += row[key]
        report["environments"][env] = dict(arms={arm: screen(runs, arm) for arm in ARMS},
            matched_scale_control={arm: screen([row["matched_scale_control"] for row in runs], arm) for arm in ARMS}, runs=runs)
    passed = all(e["arms"]["conditional_scale"]["development_screen_passed"] for e in report["environments"].values())
    report["decision"] = "consider_fresh_seed_evidence_probe" if passed else "hold_conditional_evidence_integration"
    return report


def compact_report(report):
    compact = copy.deepcopy(report)
    for environment in compact["environments"].values():
        for row in environment["runs"]:
            for collection in (row["arms"], row["matched_scale_control"]["arms"]):
                for arm in collection.values():
                    arm["calibration"].pop("windows")
                    for case in arm["cases"].values():
                        case.pop("windows")
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
                              compact_readout_bytes=output.stat().st_size)))
    else:
        run(args.env, args.test_seed, root=args.root)
