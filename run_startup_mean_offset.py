"""Develop a disjoint normal-startup mean offset on already collected streams."""

import argparse
import copy
import json
from pathlib import Path
import time

import numpy as np
import torch

import run_robust_startup_confirmation as parent_probe
from run_calibration_expansion import verify_windows
from startup_mean_offset import MeanOffsetEvidence


PROTOCOL = "disjoint_normal_startup_mean_offset_development_v1"
ROOT = Path("results/iclr_startup_mean_offset_20261001")
ANCHOR = "original_startup_mean"
CONTROL = "split_startup_mean"
PRIMARY = "startup_mean_offset"
ARMS = (ANCHOR, CONTROL, PRIMARY)
CALIBRATION_CASES = ("sampler_change", "actor_change")
ZERO_WORK = ("additional_interactions", "initialization_probes", "validation_scale_optimizer_steps",
             "offset_optimizer_steps", "policy_optimizer_steps", "critic_optimizer_steps", "encoder_optimizer_steps")


def windows(evidence, inputs, targets, arms=ARMS):
    support = evidence.distances(inputs) <= evidence.radii
    scales = evidence.scales(inputs, parent_probe.GUARDED)
    original = evidence._windows(evidence.raw_losses(inputs, targets) / scales, support)
    corrected = evidence._windows(evidence.corrected_losses(inputs, targets) / scales, support)
    parent_probe.development.verify_coverage(original, corrected)
    anchor, control, primary = arms
    return {anchor: copy.deepcopy(original), control: original, primary: corrected}, support


def calibrate(evidence, inputs, targets, arms=ARMS):
    paired, _ = windows(evidence, inputs, targets, arms)
    return {arm: dict(error_threshold=float(np.quantile(
        [w["statistic"] for w in paired[arm] if w["eligible"]], parent_probe.CONFIG["quantile"])),
        support_radii=dict(zip(evidence.head_names, evidence.radii.tolist())),
        eligible_windows=sum(w["eligible"] for w in paired[arm]), total_windows=len(paired[arm]),
        windows=paired[arm]) for arm in arms[1:]}


def validate(row, env, seed):
    assert (row["protocol"], row["env"], row["roles"], row["config"], row["complete"]) == (
        PROTOCOL, env, parent_probe.fold_roles(seed), parent_probe.CONFIG, True)
    assert all(row[k] == 0 for k in ZERO_WORK)
    assert row["predictor_optimizer_steps"] == 2 * parent_probe.CONFIG["updates"]
    assert row["residual_scale_optimizer_steps"] == (row["reward_dim"] + 1) * parent_probe.CONFIG["scale_updates"]
    assert row["offset_fit"]["fit_steps"] == 3000 and row["offset_fit"]["fit_calls"] == row["offset_fit_calls"] == 1
    assert row["offset_fit_case"] == "stationary" and row["calibration_cases"] == list(CALIBRATION_CASES)
    assert row["calibration_steps"] == 2000 and row["startup_steps"] == 5000
    assert row["thresholds_and_offset_frozen_before_evaluation_read"] and row["original_anchor_reproduced"]
    assert row["scale_fit_sources"] == row["roles"]["fit_seeds"] + [row["roles"]["scale_seed"]]
    assert set(row["arms"]) == set(ARMS)
    a, b = (row["arms"][arm]["calibration"] for arm in (CONTROL, PRIMARY))
    assert a["support_radii"] == b["support_radii"] == row["arms"][ANCHOR]["calibration"]["support_radii"]
    parent_probe.development.verify_coverage(a["windows"], b["windows"])
    for case in parent_probe.CASES:
        for control in (ANCHOR, CONTROL):
            parent_probe.development.verify_coverage(row["arms"][control]["cases"][case]["windows"],
                row["arms"][PRIMARY]["cases"][case]["windows"])


def run(env, seed, root=ROOT, parent=parent_probe.ROOT, source=parent_probe.SOURCE, fresh=parent_probe.FRESH):
    output = root / env.replace("-", "_") / f"test_seed{seed}" / "readout.json"
    if output.exists():
        saved = json.loads(output.read_text())
        if not saved["complete"]:
            raise RuntimeError("Incomplete mean-offset development retained; inspect work before retrying")
        validate(saved, env, seed)
        return saved
    prior = json.loads((parent / env.replace("-", "_") / f"test_seed{seed}" / "readout.json").read_text())
    parent_probe.validate(prior, env, seed)
    started, roles = time.monotonic(), prior["roles"]
    report = dict(protocol=PROTOCOL, env=env, roles=roles, config=parent_probe.CONFIG, complete=False,
        development_only=True, evaluation_scope="seen_confirmation_trajectories", retained_method="v25",
        evidence_integrated=False, frozen_selection=prior["frozen_selection"], reconstruction_only=True,
        reused_acquisition_interactions_counted_in_parent=360280, predictor_optimizer_steps=0,
        residual_scale_optimizer_steps=0, affine_fits=0, offset_fit_calls=0, arms={}, **dict.fromkeys(ZERO_WORK, 0))
    parent_probe.write(output, report)
    fit = [parent_probe.normal_data(source, env, s, "expanded_normal") for s in roles["fit_seeds"]]
    fit_x, fit_y = (np.concatenate([r[i] for r in fit]) for i in (0, 1))
    scale = [parent_probe.fresh_data(fresh, env, s) for s in roles["fit_seeds"]]
    scale.append(parent_probe.normal_data(source, env, roles["scale_seed"], "expanded_normal"))
    scale_x, scale_y = (np.concatenate([r[i] for r in scale]) for i in (0, 1))
    evidence = MeanOffsetEvidence(fit_x.shape[1], fit_y.shape[1], fit[0][2]["reward_dim"])
    evidence.weights = {parent_probe.GUARDED: prior["frozen_selection"]["global_weight"]}
    work = evidence.fit(fit_x, fit_y, len(fit[0][0]), 80000000 + 10 * seed)
    assert work == prior["predictor_work"]
    report.update(predictor_work=work, predictor_optimizer_steps=work["optimizer_steps"], affine_fits=2)
    parent_probe.write(output, report)
    work = evidence.fit_scale_losses(scale_x, evidence.raw_losses(scale_x, scale_y), 81000000 + 10 * seed)
    assert work["optimizer_steps"] == prior["scale_work"]["optimizer_steps"]
    np.testing.assert_allclose(work["global_scales"], prior["scale_work"]["global_scales"], rtol=1e-6)
    report.update(scale_work=work, residual_scale_optimizer_steps=work["optimizer_steps"],
        reward_dim=evidence.reward_dim, scale_fit_sources=prior["scale_fit_sources"])
    anchor = prior["arms"][parent_probe.MEAN]
    radii = anchor["calibration"]["support_radii"]
    evidence.radii = np.array([radii[n] for n in evidence.head_names])
    startup, _ = parent_probe.acquisition.data(parent / "collection", env, seed, "startup", seed_fn=parent_probe.seeds)
    x, y = (np.concatenate([r[i] for r in startup.values()]) for i in (0, 1))
    support = evidence.distances(x) <= evidence.radii
    original = evidence._windows(evidence.raw_losses(x, y) / evidence.scales(x, parent_probe.GUARDED), support)
    verify_windows(original, anchor["calibration"]["windows"])
    np.testing.assert_allclose(np.quantile([w["statistic"] for w in original if w["eligible"]],
        parent_probe.CONFIG["quantile"]), anchor["calibration"]["error_threshold"], rtol=1e-6)
    report.update(offset_fit=evidence.fit_offset(*startup["stationary"]), offset_fit_case="stationary",
        offset_fit_calls=1, startup_steps=len(x), calibration_cases=list(CALIBRATION_CASES))
    x, y = (np.concatenate([startup[c][i] for c in CALIBRATION_CASES]) for i in (0, 1))
    calibration = calibrate(evidence, x, y)
    calibration[ANCHOR] = copy.deepcopy(anchor["calibration"])
    report.update(calibration_steps=len(x), thresholds_and_offset_frozen_before_evaluation_read=True,
        arms={a: dict(calibration=calibration[a], cases={}) for a in ARMS})
    parent_probe.write(output, report)
    evaluation, metadata = parent_probe.acquisition.data(parent / "collection", env, seed, "evaluation", seed_fn=parent_probe.seeds)
    for case, (x, y) in evaluation.items():
        paired, support = windows(evidence, x, y)
        for arm in ARMS:
            parent_probe.development.mark_alarms(paired[arm], calibration[arm]["error_threshold"])
            summary = parent_probe.case_summary(dict(windows=paired[arm], supported=support.any(1), head_supported=support))
            summary["heads"] = parent_probe.summarize_heads(evidence, dict(windows=paired[arm], head_supported=support))
            if case in CALIBRATION_CASES:
                summary["intervention_active"] = metadata["cases"][case]["intervention_active"]
            if arm == ANCHOR:
                old = anchor["cases"][case]["windows"]
                verify_windows(paired[arm], old)
                assert [w["alarm"] for w in paired[arm]] == [w["alarm"] for w in old]
            report["arms"][arm]["cases"][case] = summary
        parent_probe.write(output, report)
    report.update(complete=True, original_anchor_reproduced=True, seconds=time.monotonic() - started)
    validate(report, env, seed)
    parent_probe.write(output, report)
    print(f"DONE {PROTOCOL} {env} seed={seed} interactions=0 "
        f"thresholds={ {a: calibration[a]['error_threshold'] for a in ARMS} }", flush=True)
    return report


def paired_readout(runs, control, primary=PRIMARY):
    result = dict(normal_alarms={a: 0 for a in (control, primary)}, normal_eligible_windows=0,
        mode_alarms={a: 0 for a in (control, primary)}, mode_eligible_windows=0,
        normal_resolved=[], normal_added=[], mode_lost=[], mode_added=[], timely_lost=[], timely_added=[])
    for row in runs:
        seed = row["roles"]["test_seed"]
        for case in parent_probe.CASES:
            a, b = (row["arms"][arm]["cases"][case] for arm in (control, primary))
            mode = case.startswith("mode")
            if mode and a["timely_alarm"] != b["timely_alarm"]:
                result["timely_lost" if a["timely_alarm"] else "timely_added"].append(dict(seed=seed, case=case))
            for first, second in zip(a["windows"], b["windows"]):
                if first["end_step"] <= 500 and case != "stationary":
                    continue
                category = "mode" if mode else "normal"
                result[f"{category}_eligible_windows"] += first["eligible"]
                for arm, w in zip((control, primary), (first, second)):
                    result[f"{category}_alarms"][arm] += w["alarm"]
                if first["alarm"] != second["alarm"]:
                    change = ("lost" if mode else "resolved") if first["alarm"] else "added"
                    result[f"{category}_{change}"].append(dict(seed=seed, case=case,
                        start_step=first["start_step"], end_step=first["end_step"]))
    return result


def passes_screen(environments, control=CONTROL, primary=PRIMARY):
    complete = all(e["arms"][primary]["development_screen_passed"] for e in environments.values())
    safe = all(not p["timely_lost"] and p["normal_alarms"][primary] <= p["normal_alarms"][baseline]
        for e in environments.values() for baseline, p in e["paired"].items())
    improves = any(e["paired"][control]["normal_alarms"][primary] < e["paired"][control]["normal_alarms"][control]
        for e in environments.values())
    return complete and safe and improves


def summarize(root=ROOT):
    report = dict(protocol=PROTOCOL, complete=True, development_only=True, retained_method="v25",
        evidence_integrated=False, environments={}, predictor_optimizer_steps=0,
        residual_scale_optimizer_steps=0, affine_fits=0, offset_fit_calls=0, **dict.fromkeys(ZERO_WORK, 0))
    for env in parent_probe.ENVIRONMENTS:
        runs = [json.loads((root / env.replace("-", "_") / f"test_seed{s}" / "readout.json").read_text()) for s in parent_probe.SEEDS]
        for seed, row in zip(parent_probe.SEEDS, runs):
            validate(row, env, seed)
            for key in ("predictor_optimizer_steps", "residual_scale_optimizer_steps", "affine_fits", "offset_fit_calls"):
                report[key] += row[key]
        report["environments"][env] = dict(arms={a: parent_probe.screen(runs, a) for a in ARMS},
            paired={a: paired_readout(runs, a) for a in (CONTROL, ANCHOR)}, runs=runs)
    report["decision"] = "consider_fresh_mean_offset_confirmation" if passes_screen(report["environments"]) else "hold_conditional_evidence_integration"
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--env", choices=parent_probe.ENVIRONMENTS)
    parser.add_argument("--test-seed", type=int, choices=parent_probe.SEEDS)
    parser.add_argument("--summarize", action="store_true")
    args = parser.parse_args()
    torch.set_num_threads(2)
    if args.summarize:
        report = summarize()
        parent_probe.write(ROOT / "readout.json", report)
        parent_probe.write(ROOT / "compact_readout.json", parent_probe.development.compact_report(report))
        print(json.dumps({k: v for k, v in report.items() if k != "environments"}))
    else:
        run(args.env, args.test_seed)
