"""Decompose the two Hopper stationary alarms without changing the detector."""

import argparse
import json
from pathlib import Path
import time

import numpy as np
import torch

import collect_startup_confirmation as acquisition
import run_startup_confirmation as confirmation
from collect_matched_scale_normals import ROOT as FRESH, normal_data as fresh_data
from headwise_change_evidence import CONFIG, action_basis
from run_calibration_expansion import verify_windows
from run_headwise_change_probe import SOURCE, fold_roles, normal_data, write
from run_sensitivity_guarded_selection import GuardedEvidence, GUARDED


PROTOCOL = "startup_hopper_stationary_alarm_decomposition_v1"
ENV = "mo-hopper-v5"
SEEDS = (2102, 2103)
ROOT = confirmation.ROOT / "false_alarm_diagnostic"


def tail(windows, quantile):
    ordered = sorted((w for w in windows if w["eligible"]), key=lambda w: w["statistic"])
    rank = (len(ordered) - 1) * quantile
    lower, upper = int(np.floor(rank)), int(np.ceil(rank))
    fraction = rank - lower
    threshold = (1 - fraction) * ordered[lower]["statistic"] + fraction * ordered[upper]["statistic"]
    return dict(eligible_windows=len(ordered), quantile=quantile, rank=rank,
        lower_index=lower, upper_index=upper, upper_weight=fraction,
        threshold=float(threshold), maximum=ordered[-1]["statistic"],
        top_windows=[dict(start_step=w["start_step"], end_step=w["end_step"],
            statistic=w["statistic"], dominant_head=winner(w)) for w in ordered[-2:]])


def winner(window):
    eligible = {name: row["statistic"] for name, row in window["heads"].items() if row["eligible"]}
    return max(eligible, key=eligible.get) if eligible else None


def head_decomposition(raw, affine, conditional, guarded, global_scale, supported, eligible):
    result = dict(eligible=eligible, supported_steps=int(supported.sum()))
    if not eligible:
        return result
    raw, affine, conditional, guarded = (a[supported] for a in (raw, affine, conditional, guarded))
    raw_mean = float(raw.mean())
    conditional_score, guarded_score = float((raw / conditional).mean()), float((raw / guarded).mean())
    global_score = float(raw.mean() / global_scale)
    result.update(raw_mean_loss=raw_mean, affine_only_mean_loss=float(affine.mean()),
        global_scale=float(global_scale), global_score=global_score,
        conditional_score=conditional_score, guarded_score=guarded_score,
        effective_conditional_scale=raw_mean / conditional_score if conditional_score > 0 else None,
        amplification_vs_global=conditional_score / global_score if global_score > 0 else None,
        conditional_scale_min_median_max=np.quantile(conditional, [0, .5, 1]).tolist(),
        normalized_top1_share=float(np.sort(raw / guarded)[-1:].sum() / (raw / guarded).sum()) if guarded_score > 0 else 0.,
        normalized_top5_share=float(np.sort(raw / guarded)[-5:].sum() / (raw / guarded).sum()) if guarded_score > 0 else 0.)
    return result


def scores(evidence, inputs, targets):
    raw = evidence.raw_losses(inputs, targets)
    affine = evidence.affine_losses(inputs, targets)
    conditional = evidence.scales(inputs, "conditional_scale")
    guarded = evidence.scales(inputs, GUARDED)
    support = evidence.distances(inputs) <= evidence.radii
    windows = evidence._windows(raw / guarded, support)
    for w in windows:
        w["alarm"] = bool(w["eligible"] and w["statistic"] > evidence.thresholds[GUARDED])
    return dict(raw=raw, affine=affine, conditional=conditional, guarded=guarded,
        support=support, windows=windows)


def window_detail(evidence, inputs, targets, terminals, arrays, window):
    start, end = window["start_step"] - 1, window["end_step"]
    segment = slice(start, end)
    heads = {}
    for h, name in enumerate(evidence.head_names):
        heads[name] = head_decomposition(*(arrays[key][segment, h] for key in
            ("raw", "affine", "conditional", "guarded")), evidence.global_scales[h],
            arrays["support"][segment, h], window["heads"][name]["eligible"])
    dominant = winner(window)
    result = dict(start_step=start + 1, end_step=end, statistic=window["statistic"],
        alarm=window["alarm"], dominant_head=dominant, heads=heads,
        terminated_steps=(np.flatnonzero(terminals[segment, 0]) + start + 1).tolist(),
        truncated_steps=(np.flatnonzero(terminals[segment, 1]) + start + 1).tolist(), top_steps=[])
    if dominant is None:
        return result
    h = evidence.head_names.index(dominant)
    indices = np.flatnonzero(arrays["support"][segment, h]) + start
    ordered = indices[np.argsort(arrays["raw"][indices, h] / arrays["guarded"][indices, h])[-5:][::-1]]
    prediction, affine_prediction = evidence.reward_predictor.predict(inputs), evidence.reward_predictor.affine_predict(inputs)
    actions = action_basis(inputs[:, evidence.obs_dim:]) @ evidence.action_coefficients
    for i in ordered:
        row = dict(step=int(i + 1), raw_loss=float(arrays["raw"][i, h]),
            conditional_scale=float(arrays["conditional"][i, h]),
            guarded_normalized_loss=float(arrays["raw"][i, h] / arrays["guarded"][i, h]),
            terminated=bool(terminals[i, 0]), truncated=bool(terminals[i, 1]),
            first_after_reset=bool(i > 0 and terminals[i - 1].any()))
        if h < evidence.reward_dim:
            predicted = actions[i, h] if evidence.action_heads[h] else prediction[i, h]
            affine_predicted = actions[i, h] if evidence.action_heads[h] else affine_prediction[i, h]
            row.update(target=float(targets[i, h]), prediction=float(predicted),
                affine_prediction=float(affine_predicted), target_normalization_scale=float(evidence.y_scale[h]))
        result["top_steps"].append(row)
    return result


def selected_windows(windows, group, case):
    eligible = [w for w in windows if w["eligible"] and (group == "startup" or w["start_step"] > 500)]
    if not eligible:
        return []
    selected = [max(eligible, key=lambda w: w["statistic"])]
    if group == "evaluation":
        alarms = [w for w in eligible if w["alarm"]]
        selected += alarms[:1] if case.startswith("mode") else alarms
    return sorted({w["start_step"]: w for w in selected}.values(), key=lambda w: w["start_step"])


def run(seed, root=ROOT, parent=confirmation.ROOT, source=SOURCE, fresh=FRESH):
    output = root / f"seed{seed}" / "readout.json"
    if output.exists():
        saved = json.loads(output.read_text())
        assert saved["protocol"] == PROTOCOL and saved["seed"] == seed and saved["complete"]
        return saved
    prior = json.loads((parent / ENV.replace("-", "_") / f"test_seed{seed}" / "readout.json").read_text())
    assert (prior["protocol"], prior["complete"], prior["config"], prior["roles"]) == (
        confirmation.PROTOCOL, True, CONFIG, fold_roles(seed))
    roles, calibration = prior["roles"], prior["arms"][confirmation.PRIMARY]["calibration"]
    started = time.monotonic()
    report = dict(protocol=PROTOCOL, env=ENV, seed=seed, complete=False, roles=roles,
        retained_method="v25", decision="hold_conditional_evidence_integration", reconstruction_only=True,
        frozen_selection=prior["frozen_selection"], frozen_threshold=calibration["error_threshold"],
        additional_interactions=0, initialization_probes=0, policy_optimizer_steps=0,
        critic_optimizer_steps=0, encoder_optimizer_steps=0, validation_scale_optimizer_steps=0,
        predictor_optimizer_steps=0, residual_scale_optimizer_steps=0, affine_fits=0, groups={})
    write(output, report)
    fit_rows = [normal_data(source, ENV, s, "expanded_normal") for s in roles["fit_seeds"]]
    fit_x, fit_y = (np.concatenate([r[i] for r in fit_rows]) for i in (0, 1))
    scale_rows = [fresh_data(fresh, ENV, s) for s in roles["fit_seeds"]]
    scale_rows.append(normal_data(source, ENV, roles["scale_seed"], "expanded_normal"))
    scale_x, scale_y = (np.concatenate([r[i] for r in scale_rows]) for i in (0, 1))
    evidence = GuardedEvidence(fit_x.shape[1], fit_y.shape[1], fit_rows[0][2]["reward_dim"])
    evidence.weights = {GUARDED: prior["frozen_selection"]["global_weight"]}
    work = evidence.fit(fit_x, fit_y, len(fit_rows[0][0]), 80000000 + 10 * seed)
    assert work == prior["predictor_work"]
    report.update(predictor_optimizer_steps=work["optimizer_steps"], affine_fits=2)
    write(output, report)
    work = evidence.fit_scale_losses(scale_x, evidence.raw_losses(scale_x, scale_y), 81000000 + 10 * seed)
    assert work["optimizer_steps"] == prior["scale_work"]["optimizer_steps"]
    np.testing.assert_allclose(work["global_scales"], prior["scale_work"]["global_scales"], rtol=1e-6)
    report.update(residual_scale_optimizer_steps=work["optimizer_steps"],
        global_scales=dict(zip(evidence.head_names, work["global_scales"])), support_domains=evidence.domains)
    evidence.radii = np.array([calibration["support_radii"][name] for name in evidence.head_names])
    evidence.thresholds = {GUARDED: calibration["error_threshold"]}
    for group in ("startup", "evaluation"):
        rows, _ = acquisition.data(parent / "collection", ENV, seed, group)
        group_result, combined_windows, offset = {}, [], 0
        with np.load(acquisition.folder(parent / "collection", ENV, seed, group) / "transitions.npz") as stored:
            for case, (x, y) in rows.items():
                arrays = scores(evidence, x, y)
                if group == "evaluation":
                    old = prior["arms"][confirmation.PRIMARY]["cases"][case]["windows"]
                    verify_windows(arrays["windows"], old)
                    assert [w["alarm"] for w in arrays["windows"]] == [w["alarm"] for w in old]
                group_result[case] = dict(eligible_windows=sum(w["eligible"] for w in arrays["windows"]),
                    head_reference={}, selected_windows=[window_detail(evidence, x, y, stored[f"{case}_terminals"], arrays, w)
                        for w in selected_windows(arrays["windows"], group, case)])
                for h, name in enumerate(evidence.head_names):
                    decomposed = [head_decomposition(*(arrays[k][w["start_step"] - 1:w["end_step"], h]
                        for k in ("raw", "affine", "conditional", "guarded")), evidence.global_scales[h],
                        arrays["support"][w["start_step"] - 1:w["end_step"], h], True)
                        for w in arrays["windows"] if w["heads"][name]["eligible"]]
                    group_result[case]["head_reference"][name] = {key: np.quantile(
                        [r[key] for r in decomposed], [.5, .99, 1]).tolist()
                        for key in ("raw_mean_loss", "affine_only_mean_loss", "global_score", "conditional_score", "guarded_score")} if decomposed else {}
                for w in arrays["windows"]:
                    combined_windows.append({**w, "start_step": w["start_step"] + offset, "end_step": w["end_step"] + offset})
                offset += len(x)
        report["groups"][group] = group_result
        if group == "startup":
            verify_windows(combined_windows, calibration["windows"])
            report["calibration_tail"] = tail(combined_windows, CONFIG["quantile"])
            np.testing.assert_allclose(report["calibration_tail"]["threshold"], evidence.thresholds[GUARDED], rtol=1e-6)
        write(output, report)
    report.update(complete=True, frozen_scores_and_alarms_reproduced=True, seconds=time.monotonic() - started)
    write(output, report)
    print(f"DONE {PROTOCOL} seed={seed} interactions=0 seconds={report['seconds']:.1f}", flush=True)
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--test-seed", type=int, choices=SEEDS, required=True)
    args = parser.parse_args()
    torch.set_num_threads(2)
    run(args.test_seed)
