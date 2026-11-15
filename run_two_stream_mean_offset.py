"""Isolate mean correction using old offset fitting and full new startup calibration."""

import argparse
import copy
import json
from pathlib import Path
import time

import numpy as np
import torch

import run_startup_mean_offset as shared
from run_startup_confirmation import ROOT as EARLIER


PROTOCOL = "two_stream_full_startup_mean_offset_development_v1"
ROOT = Path("results/iclr_two_stream_mean_offset_20261001")
PARENT = shared.parent_probe
ANCHOR = shared.ANCHOR
CONTROL = "full_startup_mean"
PRIMARY = "two_stream_mean_offset"
ARMS = (ANCHOR, CONTROL, PRIMARY)


def verify_streams(old, new, env, seed):
    PARENT.acquisition.validate(old, env, seed, "startup")
    PARENT.acquisition.validate(new, env, seed, "startup", seed_fn=PARENT.seeds)
    assert all(old[k] == new[k] for k in ("env", "seed", "source", "source_methods", "source_step"))
    fit_seeds = old["cases"]["stationary"]["seeds"]
    assert all(all(fit_seeds[k] != case["seeds"][k] for k in fit_seeds) for case in new["cases"].values())


def offset_data(earlier, env, seed):
    folder = PARENT.acquisition.folder(earlier / "collection", env, seed, "startup")
    metadata = json.loads((folder / "readout.json").read_text())
    PARENT.acquisition.validate(metadata, env, seed, "startup")
    with np.load(folder / "transitions.npz") as data:
        x, y = data["stationary_inputs"].copy(), data["stationary_targets"].copy()
    assert len(x) == len(y) == 3000
    return x, y, metadata


def validate(row, env, seed):
    assert (row["protocol"], row["env"], row["roles"], row["config"], row["complete"]) == (
        PROTOCOL, env, PARENT.fold_roles(seed), PARENT.CONFIG, True)
    assert all(row[k] == 0 for k in shared.ZERO_WORK)
    assert row["predictor_optimizer_steps"] == 2 * PARENT.CONFIG["updates"]
    assert row["residual_scale_optimizer_steps"] == (row["reward_dim"] + 1) * PARENT.CONFIG["scale_updates"]
    assert row["scale_fit_sources"] == row["roles"]["fit_seeds"] + [row["roles"]["scale_seed"]]
    assert row["offset_fit"]["fit_steps"] == 3000 and row["offset_fit"]["fit_calls"] == row["offset_fit_calls"] == 1
    assert row["calibration_steps"] == 5000 and row["used_startup_steps"] == 8000
    assert row["offset_fit_case"] == "stationary" and row["calibration_cases"] == list(PARENT.acquisition.STARTUP_STEPS)
    assert row["thresholds_and_offset_frozen_before_evaluation_read"] and row["original_anchor_reproduced"]
    verify_streams(row["startup_sources"]["offset"], row["startup_sources"]["calibration"], env, seed)
    assert set(row["arms"]) == set(ARMS)
    a, b = (row["arms"][arm]["calibration"] for arm in (CONTROL, PRIMARY))
    assert a["support_radii"] == b["support_radii"] == row["arms"][ANCHOR]["calibration"]["support_radii"]
    PARENT.development.verify_coverage(a["windows"], b["windows"])
    shared.verify_windows(a["windows"], row["arms"][ANCHOR]["calibration"]["windows"])
    assert a["error_threshold"] == row["arms"][ANCHOR]["calibration"]["error_threshold"]
    for case in PARENT.CASES:
        for control in (ANCHOR, CONTROL):
            PARENT.development.verify_coverage(row["arms"][control]["cases"][case]["windows"],
                row["arms"][PRIMARY]["cases"][case]["windows"])
        assert row["arms"][CONTROL]["cases"][case] == row["arms"][ANCHOR]["cases"][case]


def run(env, seed, root=ROOT, parent=PARENT.ROOT, earlier=EARLIER, source=PARENT.SOURCE, fresh=PARENT.FRESH):
    output = root / env.replace("-", "_") / f"test_seed{seed}" / "readout.json"
    if output.exists():
        saved = json.loads(output.read_text())
        if not saved["complete"]:
            raise RuntimeError("Incomplete two-stream development retained; inspect work before retrying")
        validate(saved, env, seed)
        return saved
    prior = json.loads((parent / env.replace("-", "_") / f"test_seed{seed}" / "readout.json").read_text())
    PARENT.validate(prior, env, seed)
    offset_x, offset_y, old_metadata = offset_data(earlier, env, seed)
    startup, new_metadata = PARENT.acquisition.data(parent / "collection", env, seed, "startup", seed_fn=PARENT.seeds)
    verify_streams(old_metadata, new_metadata, env, seed)
    roles, started = prior["roles"], time.monotonic()
    report = dict(protocol=PROTOCOL, env=env, roles=roles, config=PARENT.CONFIG, complete=False,
        development_only=True, evaluation_scope="seen_confirmation_trajectories", retained_method="v25",
        evidence_integrated=False, reconstruction_only=True, frozen_selection=prior["frozen_selection"],
        startup_sources=dict(offset=old_metadata, calibration=new_metadata), used_startup_steps=8000,
        offset_fit_case="stationary", calibration_cases=list(PARENT.acquisition.STARTUP_STEPS),
        reused_acquisition_interactions_counted_in_parent=360280, predictor_optimizer_steps=0,
        residual_scale_optimizer_steps=0, affine_fits=0, offset_fit_calls=0, arms={}, **dict.fromkeys(shared.ZERO_WORK, 0))
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
    anchor = prior["arms"][PARENT.MEAN]
    radii = anchor["calibration"]["support_radii"]
    evidence.radii = np.array([radii[n] for n in evidence.head_names])
    report.update(offset_fit=evidence.fit_offset(offset_x, offset_y), offset_fit_calls=1)
    x, y = (np.concatenate([r[i] for r in startup.values()]) for i in (0, 1))
    calibration = shared.calibrate(evidence, x, y, arms=ARMS)
    shared.verify_windows(calibration[CONTROL]["windows"], anchor["calibration"]["windows"])
    np.testing.assert_allclose(calibration[CONTROL]["error_threshold"], anchor["calibration"]["error_threshold"], rtol=1e-6)
    # The matched control and original anchor use exactly the same complete calibration.
    calibration[CONTROL]["error_threshold"] = anchor["calibration"]["error_threshold"]
    calibration[ANCHOR] = copy.deepcopy(anchor["calibration"])
    report.update(calibration_steps=len(x), thresholds_and_offset_frozen_before_evaluation_read=True,
        arms={a: dict(calibration=calibration[a], cases={}) for a in ARMS})
    PARENT.write(output, report)
    evaluation, metadata = PARENT.acquisition.data(parent / "collection", env, seed, "evaluation", seed_fn=PARENT.seeds)
    for case, (x, y) in evaluation.items():
        paired, support = shared.windows(evidence, x, y, arms=ARMS)
        for arm in ARMS:
            PARENT.development.mark_alarms(paired[arm], calibration[arm]["error_threshold"])
            summary = PARENT.case_summary(dict(windows=paired[arm], supported=support.any(1)))
            summary["heads"] = PARENT.summarize_heads(evidence, dict(windows=paired[arm], head_supported=support))
            if case in ("sampler_change", "actor_change"):
                summary["intervention_active"] = metadata["cases"][case]["intervention_active"]
            if arm in (ANCHOR, CONTROL):
                old = anchor["cases"][case]["windows"]
                shared.verify_windows(paired[arm], old)
                assert [w["alarm"] for w in paired[arm]] == [w["alarm"] for w in old]
            report["arms"][arm]["cases"][case] = summary
        PARENT.write(output, report)
    report.update(complete=True, original_anchor_reproduced=True, seconds=time.monotonic() - started)
    validate(report, env, seed)
    PARENT.write(output, report)
    print(f"DONE {PROTOCOL} {env} seed={seed} interactions=0 "
        f"thresholds={ {a: calibration[a]['error_threshold'] for a in ARMS} }", flush=True)
    return report


def summarize(root=ROOT):
    report = dict(protocol=PROTOCOL, complete=True, development_only=True, retained_method="v25",
        evidence_integrated=False, environments={}, predictor_optimizer_steps=0, residual_scale_optimizer_steps=0,
        affine_fits=0, offset_fit_calls=0, used_startup_steps=0, **dict.fromkeys(shared.ZERO_WORK, 0))
    for env in PARENT.ENVIRONMENTS:
        runs = [json.loads((root / env.replace("-", "_") / f"test_seed{s}" / "readout.json").read_text()) for s in PARENT.SEEDS]
        for seed, row in zip(PARENT.SEEDS, runs):
            validate(row, env, seed)
            for key in ("predictor_optimizer_steps", "residual_scale_optimizer_steps", "affine_fits", "offset_fit_calls", "used_startup_steps"):
                report[key] += row[key]
        report["environments"][env] = dict(arms={a: PARENT.screen(runs, a) for a in ARMS},
            paired={a: shared.paired_readout(runs, a, primary=PRIMARY) for a in (CONTROL, ANCHOR)}, runs=runs)
    report["decision"] = "consider_fresh_two_stream_mean_offset_confirmation" if shared.passes_screen(
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
