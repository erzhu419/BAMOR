"""Calibrate on independent unchanged startup data from the deployment policy."""

import argparse
import copy
import json
from pathlib import Path
import time

import numpy as np
import torch

from collect_matched_scale_normals import (
    ROOT as FRESH, COLLECTION_STEPS, normal_data as fresh_data, seeds as collection_seeds)
from headwise_change_evidence import CONFIG
from run_cross_policy_calibration import (
    ROOT as PREVIOUS, calibrate_fixed_support, validate as validate_previous, ZERO_WORK)
from run_calibration_expansion import verify_windows
from run_headwise_change_probe import (
    CASES, ENVIRONMENTS, SEEDS, SOURCE, fold_roles, normal_data, source_folder,
    source_readout, summarize_heads, case_summary, screen, write)
from run_sensitivity_guarded_selection import GuardedEvidence, GUARDED


PROTOCOL = "independent_deployment_policy_startup_calibration_v1"
ROOT = Path("results/iclr_deployment_startup_calibration_20261001")
CONTROLS = ("original_calibration", "pooled_two_policy")
PRIMARY = "deployment_startup_5k"
ARMS = (*CONTROLS, PRIMARY)


def startup_provenance(metadata, seed):
    assert metadata["seed"] == seed and metadata["source_step"] == 75000
    evaluation_rng = dict(environment_seed=seed + 100,
        reset_seed=50000000 + 1000 * (seed + 100), weight_seed=60000000 + seed + 100)
    rngs = {}
    for index, case in enumerate(COLLECTION_STEPS):
        rng = metadata["cases"][case]["seeds"]
        assert rng == collection_seeds(seed, index)
        assert all(rng[key] != evaluation_rng[key] for key in evaluation_rng)
        rngs[case] = rng
    return dict(seed=seed, source_step=metadata["source_step"], case_seeds=rngs,
        evaluation_seeds=evaluation_rng, separate_reset_streams=True,
        calibration_transitions=metadata["collection_interactions"],
        initialization_probes=metadata["initialization_probes"],
        required_interactions=metadata["additional_interactions"],
        reused=True, already_charged_in_fresh_collection=True, new_interactions=0)


def validate(row, env, seed):
    assert (row["protocol"], row["env"], row["roles"], row["complete"], row["config"]) == (
        PROTOCOL, env, fold_roles(seed), True, CONFIG)
    assert row["external_controls_reproduced"] and row["evaluation_scores_unchanged"]
    assert (row["fit_steps"], row["scale_fit_steps"]) == (10000, 15000)
    assert row["scale_fit_sources"] == row["roles"]["fit_seeds"] + [row["roles"]["scale_seed"]]
    assert seed not in row["scale_fit_sources"]
    assert row["predictor_optimizer_steps"] == 2 * CONFIG["updates"]
    assert row["residual_scale_optimizer_steps"] == (row["reward_dim"] + 1) * CONFIG["scale_updates"]
    assert all(row[key] == 0 for key in ZERO_WORK)
    startup = row["startup_acquisition"]
    assert startup["seed"] == seed and startup["separate_reset_streams"]
    assert (startup["calibration_transitions"], startup["initialization_probes"],
            startup["required_interactions"], startup["new_interactions"]) == (5000, 3, 5003, 0)
    assert set(row["arms"]) == set(ARMS)
    radii = row["arms"][CONTROLS[0]]["calibration"]["support_radii"]
    for arm, result in row["arms"].items():
        assert result["calibration_steps"] == 5000
        assert result["calibration"]["total_windows"] == 5000 // CONFIG["window"]
        assert result["calibration"]["support_radii"] == radii
        if arm == PRIMARY:
            assert result["sources"] == [dict(seed=seed, steps=5000, dataset="fresh_normal", role="deployment_startup_policy")]
        else:
            assert all(s["seed"] != seed for s in result["sources"])


def run(env, test_seed, source=SOURCE, root=ROOT, previous=PREVIOUS, fresh=FRESH):
    roles = fold_roles(test_seed)
    output = root / env.replace("-", "_") / f"test_seed{test_seed}" / "readout.json"
    if output.exists():
        saved = json.loads(output.read_text())
        validate(saved, env, test_seed)
        return saved
    prior = json.loads((previous / env.replace("-", "_") / f"test_seed{test_seed}" / "readout.json").read_text())
    validate_previous(prior, env, test_seed)
    started = time.monotonic()
    report = dict(protocol=PROTOCOL, env=env, roles=roles, config=CONFIG, complete=False,
        development_only=True, reconstruction_only=True, arms={}, frozen_selection=prior["frozen_selection"],
        primary_calibration=PRIMARY, predictor_optimizer_steps=0, residual_scale_optimizer_steps=0,
        calibration_source_assumption="known_unchanged_deployment_policy_startup_stream")
    report.update({key: 0 for key in ZERO_WORK})
    write(output, report)
    fit_rows = [normal_data(source, env, s, "expanded_normal") for s in roles["fit_seeds"]]
    fit_x, fit_y = (np.concatenate([row[i] for row in fit_rows]) for i in (0, 1))
    scale_rows = [fresh_data(fresh, env, s) for s in roles["fit_seeds"]]
    scale_rows.append(normal_data(source, env, roles["scale_seed"], "expanded_normal"))
    scale_x, scale_y = (np.concatenate([row[i] for row in scale_rows]) for i in (0, 1))
    evidence = GuardedEvidence(fit_x.shape[1], fit_y.shape[1], fit_rows[0][2]["reward_dim"])
    evidence.weights = {GUARDED: prior["frozen_selection"]["global_weight"]}
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
    original = prior["arms"][CONTROLS[0]]
    evidence.radii = np.array([original["calibration"]["support_radii"][name] for name in evidence.head_names])
    evidence.thresholds = {}
    startup_x, startup_y, startup_meta = fresh_data(fresh, env, test_seed)
    report["startup_acquisition"] = startup_provenance(startup_meta, test_seed)
    calibration = calibrate_fixed_support(evidence, startup_x, startup_y)
    report["arms"] = {arm: copy.deepcopy(prior["arms"][arm]) for arm in CONTROLS}
    report["arms"][PRIMARY] = dict(calibration=calibration, calibration_steps=len(startup_x),
        sources=[dict(seed=test_seed, steps=len(startup_x), dataset="fresh_normal", role="deployment_startup_policy")], cases={})
    write(output, report)
    # Evaluation starts only after the startup threshold is fixed.
    original_cases = source_readout(source, env, test_seed)
    with np.load(source_folder(source, env, test_seed) / "transitions.npz") as data:
        for arm in ARMS:
            evidence.thresholds[GUARDED] = report["arms"][arm]["calibration"]["error_threshold"]
            for case in CASES:
                evaluation = evidence.evaluate(data[f"{case}_inputs"], data[f"{case}_targets"], GUARDED)
                summary = case_summary(evaluation)
                summary["heads"] = summarize_heads(evidence, evaluation)
                saved = prior["arms"][arm if arm in CONTROLS else CONTROLS[0]]["cases"][case]
                verify_windows(summary["windows"], saved["windows"])
                if arm in CONTROLS:
                    assert [w["alarm"] for w in summary["windows"]] == [w["alarm"] for w in saved["windows"]]
                if case in ("sampler_change", "actor_change"):
                    summary["intervention_active"] = original_cases["cases"][case]["intervention_active"]
                report["arms"][arm]["cases"][case] = summary
            write(output, report)
            print(f"ARM {env} seed={test_seed} {arm} threshold={evidence.thresholds[GUARDED]}", flush=True)
    report.update(complete=True, external_controls_reproduced=True, evaluation_scores_unchanged=True,
        seconds=time.monotonic() - started)
    validate(report, env, test_seed)
    write(output, report)
    print(f"DONE {env} seed={test_seed} added_interactions=0 startup_required_interactions=5003", flush=True)
    return report


def summarize(root=ROOT):
    report = dict(protocol=PROTOCOL, config=CONFIG, complete=True, development_only=True,
        reconstruction_only=True, environments={}, primary_calibration=PRIMARY,
        retained_method="v25", evidence_integrated=False, predictor_optimizer_steps=0,
        residual_scale_optimizer_steps=0, deployment_startup_required_interactions=0,
        reused_acquisition_interactions_counted_in_parent=140100,
        reused_acquisition_ledgers=dict(original_change_probe=90070, fresh_normal_collection=50030))
    report.update({key: 0 for key in ZERO_WORK})
    for env in ENVIRONMENTS:
        runs = [json.loads((root / env.replace("-", "_") / f"test_seed{seed}" / "readout.json").read_text()) for seed in SEEDS]
        for seed, row in zip(SEEDS, runs):
            validate(row, env, seed)
            for key in ("predictor_optimizer_steps", "residual_scale_optimizer_steps"):
                report[key] += row[key]
            report["deployment_startup_required_interactions"] += row["startup_acquisition"]["required_interactions"]
        report["environments"][env] = dict(arms={arm: screen(runs, arm) for arm in ARMS}, runs=runs)
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
            decision=report["decision"], startup_required_interactions=report["deployment_startup_required_interactions"],
            compact_readout_bytes=output.stat().st_size)))
    else:
        run(args.env, args.test_seed, root=args.root)
