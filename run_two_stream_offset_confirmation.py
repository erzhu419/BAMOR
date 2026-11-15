"""Confirm the fixed mean-offset rule on fresh fit, calibration and evaluation streams."""

import argparse
import copy
import json
from pathlib import Path
import time

import numpy as np
import torch

import run_two_stream_mean_offset as development


PROTOCOL = "frozen_two_stream_mean_offset_fresh_confirmation_v1"
ROOT = Path("results/iclr_two_stream_offset_confirmation_20261001")
PARENT = development.PARENT
shared = development.shared
acquisition = PARENT.acquisition
ANCHOR, CONTROL, PRIMARY = development.ARMS
ARMS = development.ARMS
GROUPS = ("offset", "startup", "evaluation")
ZERO_WORK = PARENT.ZERO_WORK + ("offset_optimizer_steps",)


def seeds(seed, group, index=0):
    if group == "offset":
        return acquisition.seeds(seed, group, index)
    if group == "startup":
        return dict(environment_seed=330000000 + 10 * seed + index,
            reset_seed=1400000000 + 50000 * seed + 10000 * index,
            weight_seed=340000000 + 10 * seed + index)
    assert group == "evaluation"
    return dict(environment_seed=350000000 + 10 * seed,
        reset_seed=1600000000 + 50000 * seed, weight_seed=360000000 + 10 * seed)


def validate(row, env, seed):
    assert (row["protocol"], row["env"], row["roles"], row["config"], row["complete"]) == (
        PROTOCOL, env, PARENT.fold_roles(seed), PARENT.CONFIG, True)
    assert all(row[k] == 0 for k in ZERO_WORK)
    assert row["predictor_optimizer_steps"] == 2 * PARENT.CONFIG["updates"]
    assert row["residual_scale_optimizer_steps"] == (row["reward_dim"] + 1) * PARENT.CONFIG["scale_updates"]
    assert row["scale_fit_sources"] == row["roles"]["fit_seeds"] + [row["roles"]["scale_seed"]]
    assert row["offset_fit"]["fit_steps"] == 3000 and row["offset_fit"]["fit_calls"] == row["offset_fit_calls"] == 1
    assert row["calibration_steps"] == 5000 and row["used_startup_steps"] == 8000
    assert row["offset_fit_case"] == "stationary" and row["calibration_cases"] == list(acquisition.STARTUP_STEPS)
    assert row["thresholds_and_offset_frozen_before_evaluation_collection"] and row["frozen_anchor_calibration_retained"]
    for group in GROUPS:
        metadata = row["acquisition"][group]
        acquisition.validate(metadata, env, seed, group, seed_fn=seeds)
        assert all(metadata[k] == row["policy_source"][k] for k in ("source", "source_methods", "source_step"))
    assert row["additional_interactions"] == sum(m["additional_interactions"] for m in row["acquisition"].values())
    assert row["initialization_probes"] == sum(m["initialization_probes"] for m in row["acquisition"].values())
    assert set(row["arms"]) == set(ARMS)
    a, b = (row["arms"][arm]["calibration"] for arm in (CONTROL, PRIMARY))
    assert a["support_radii"] == b["support_radii"] == row["arms"][ANCHOR]["calibration"]["support_radii"]
    PARENT.development.verify_coverage(a["windows"], b["windows"])
    for case in PARENT.CASES:
        for control in (ANCHOR, CONTROL):
            PARENT.development.verify_coverage(row["arms"][control]["cases"][case]["windows"],
                row["arms"][PRIMARY]["cases"][case]["windows"])
        shared.verify_windows(row["arms"][CONTROL]["cases"][case]["windows"],
            row["arms"][ANCHOR]["cases"][case]["windows"])


def run(env, seed, root=ROOT, parent=development.ROOT, source=PARENT.SOURCE, fresh=PARENT.FRESH):
    output = root / env.replace("-", "_") / f"test_seed{seed}" / "readout.json"
    if output.exists():
        saved = json.loads(output.read_text())
        if not saved["complete"]:
            raise RuntimeError("Incomplete offset confirmation retained; inspect spent work before retrying")
        validate(saved, env, seed)
        return saved
    prior = json.loads((parent / env.replace("-", "_") / f"test_seed{seed}" / "readout.json").read_text())
    development.validate(prior, env, seed)
    started, roles = time.monotonic(), prior["roles"]
    report = dict(protocol=PROTOCOL, env=env, roles=roles, config=PARENT.CONFIG, complete=False,
        confirmation_scope="new_trajectories_same_frozen_policy_sources", retained_method="v25",
        evidence_integrated=False, frozen_selection=prior["frozen_selection"], used_startup_steps=8000,
        offset_fit_case="stationary", calibration_cases=list(acquisition.STARTUP_STEPS),
        policy_source={k: prior["startup_sources"]["calibration"][k] for k in ("source", "source_methods", "source_step")},
        predictor_optimizer_steps=0, residual_scale_optimizer_steps=0, affine_fits=0, offset_fit_calls=0,
        additional_interactions=0, initialization_probes=0, acquisition={}, arms={}, **dict.fromkeys(ZERO_WORK, 0))
    PARENT.write(output, report)
    fit = [PARENT.normal_data(source, env, s, "expanded_normal") for s in roles["fit_seeds"]]
    fit_x, fit_y = (np.concatenate([r[i] for r in fit]) for i in (0, 1))
    scale = [PARENT.fresh_data(fresh, env, s) for s in roles["fit_seeds"]]
    scale.append(PARENT.normal_data(source, env, roles["scale_seed"], "expanded_normal"))
    scale_x, scale_y = (np.concatenate([r[i] for r in scale]) for i in (0, 1))
    evidence = shared.MeanOffsetEvidence(fit_x.shape[1], fit_y.shape[1], fit[0][2]["reward_dim"])
    evidence.weights = {PARENT.GUARDED: prior["frozen_selection"]["global_weight"]}
    work = evidence.fit(fit_x, fit_y, len(fit[0][0]), 80000000 + 10 * seed)
    assert work == prior["predictor_work"]
    report.update(predictor_work=work, predictor_optimizer_steps=work["optimizer_steps"], affine_fits=2)
    PARENT.write(output, report)
    work = evidence.fit_scale_losses(scale_x, evidence.raw_losses(scale_x, scale_y), 81000000 + 10 * seed)
    assert work["optimizer_steps"] == prior["scale_work"]["optimizer_steps"]
    np.testing.assert_allclose(work["global_scales"], prior["scale_work"]["global_scales"], rtol=1e-6)
    report.update(scale_work=work, residual_scale_optimizer_steps=work["optimizer_steps"],
        reward_dim=evidence.reward_dim, scale_fit_sources=prior["scale_fit_sources"])
    anchor = prior["arms"][ANCHOR]["calibration"]
    evidence.radii = np.array([anchor["support_radii"][n] for n in evidence.head_names])
    PARENT.write(output, report)
    collection = root / "collection"
    for group in GROUPS:
        # Persist physical spending even if collection or its audit is interrupted.
        try:
            acquisition.run_group(env, seed, group, collection, seed_fn=seeds)
        finally:
            ledger = acquisition.folder(collection, env, seed, group) / "readout.json"
            if ledger.exists():
                report["acquisition"][group] = json.loads(ledger.read_text())
                report.update(additional_interactions=sum(m["additional_interactions"] for m in report["acquisition"].values()),
                    initialization_probes=sum(m["initialization_probes"] for m in report["acquisition"].values()))
                PARENT.write(output, report)
        rows, metadata = acquisition.data(collection, env, seed, group, seed_fn=seeds)
        assert all(metadata[k] == report["policy_source"][k] for k in report["policy_source"])
        if group == "offset":
            report.update(offset_fit=evidence.fit_offset(*rows["stationary"]), offset_fit_calls=1)
        elif group == "startup":
            x, y = (np.concatenate([r[i] for r in rows.values()]) for i in (0, 1))
            calibration = shared.calibrate(evidence, x, y, arms=ARMS)
            calibration[ANCHOR] = copy.deepcopy(anchor)
            report.update(calibration_steps=len(x), thresholds_and_offset_frozen_before_evaluation_collection=True,
                frozen_anchor_calibration_retained=True,
                arms={a: dict(calibration=calibration[a], cases={}) for a in ARMS})
        else:
            for case, (x, y) in rows.items():
                paired, support = shared.windows(evidence, x, y, arms=ARMS)
                for arm in ARMS:
                    PARENT.development.mark_alarms(paired[arm], calibration[arm]["error_threshold"])
                    summary = PARENT.case_summary(dict(windows=paired[arm], supported=support.any(1)))
                    summary["heads"] = PARENT.summarize_heads(evidence, dict(windows=paired[arm], head_supported=support))
                    if case in ("sampler_change", "actor_change"):
                        summary["intervention_active"] = metadata["cases"][case]["intervention_active"]
                    report["arms"][arm]["cases"][case] = summary
        PARENT.write(output, report)
    report.update(complete=True, seconds=time.monotonic() - started)
    validate(report, env, seed)
    PARENT.write(output, report)
    print(f"DONE {PROTOCOL} {env} seed={seed} new_interactions={report['additional_interactions']}", flush=True)
    return report


def summarize(root=ROOT):
    report = dict(protocol=PROTOCOL, complete=True, retained_method="v25", evidence_integrated=False,
        confirmation_scope="new_trajectories_same_frozen_policy_sources", prior_acquisition_interactions=360280,
        environments={}, additional_interactions=0, initialization_probes=0, predictor_optimizer_steps=0,
        residual_scale_optimizer_steps=0, affine_fits=0, offset_fit_calls=0, used_startup_steps=0, **dict.fromkeys(ZERO_WORK, 0))
    for env in PARENT.ENVIRONMENTS:
        runs = [json.loads((root / env.replace("-", "_") / f"test_seed{s}" / "readout.json").read_text()) for s in PARENT.SEEDS]
        for seed, row in zip(PARENT.SEEDS, runs):
            validate(row, env, seed)
            for key in ("additional_interactions", "initialization_probes", "predictor_optimizer_steps",
                        "residual_scale_optimizer_steps", "affine_fits", "offset_fit_calls", "used_startup_steps"):
                report[key] += row[key]
        report["environments"][env] = dict(arms={a: PARENT.screen(runs, a) for a in ARMS},
            paired={a: shared.paired_readout(runs, a, primary=PRIMARY) for a in (CONTROL, ANCHOR)}, runs=runs)
    report["cumulative_acquisition_interactions"] = report["prior_acquisition_interactions"] + report["additional_interactions"]
    report["decision"] = "consider_budgeted_mean_offset_detector_integration" if shared.passes_screen(
        report["environments"], control=CONTROL, primary=PRIMARY) else "hold_conditional_evidence_integration"
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--env", choices=PARENT.ENVIRONMENTS)
    parser.add_argument("--test-seed", type=int, choices=PARENT.SEEDS)
    parser.add_argument("--summarize", action="store_true")
    args = parser.parse_args()
    torch.set_num_threads(2)
    if args.summarize:
        report = summarize()
        PARENT.write(ROOT / "readout.json", report)
        PARENT.write(ROOT / "compact_readout.json", PARENT.development.compact_report(report))
        print(json.dumps({k: v for k, v in report.items() if k != "environments"}))
    else:
        run(args.env, args.test_seed)
