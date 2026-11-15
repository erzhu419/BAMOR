"""Decompose frozen HalfCheetah normal alarms and mode1 misses on stored data."""

import argparse
import json
from pathlib import Path
import time

import numpy as np
import torch

import run_robust_startup_confirmation as confirmation
from diagnose_startup_false_alarms import head_decomposition, scores
from headwise_change_evidence import action_basis
from run_calibration_expansion import verify_windows


PROTOCOL = "frozen_halfcheetah_residual_failure_decomposition_v1"
ENV = "mo-halfcheetah-v5"
SEEDS = (2103, 2104)
ROOT = confirmation.ROOT / "residual_failure_diagnostic"
QUANTILES = (0., .01, .5, .99, 1.)


def moments(residual):
    residual = np.asarray(residual, np.float64)
    mean, variance = residual.mean(0), residual.var(0)
    bias = float(np.mean(mean ** 2))
    spread = float(variance.mean())
    return dict(normalized_residual_mean=mean.tolist(),
        normalized_residual_variance=variance.tolist(), mean_squared_bias=bias,
        within_window_variance=spread, normalized_mse=bias + spread,
        bias_fraction=bias / (bias + spread) if bias + spread > 0 else 0.)


def residual_arrays(evidence, inputs, targets):
    prediction = np.concatenate([evidence.reward_predictor.predict(inputs),
        evidence.dynamics_predictor.predict(inputs)], axis=1)
    affine = np.concatenate([evidence.reward_predictor.affine_predict(inputs),
        evidence.dynamics_predictor.affine_predict(inputs)], axis=1)
    action = action_basis(inputs[:, evidence.obs_dim:]) @ evidence.action_coefficients
    for head in np.flatnonzero(evidence.action_heads):
        prediction[:, head] = affine[:, head] = action[:, head]
    return dict(signed=(prediction.astype(np.float64) - targets) / evidence.y_scale,
        correction=(prediction.astype(np.float64) - affine) / evidence.y_scale,
        prediction=prediction, targets=targets)


def details(evidence, arrays, window, robust):
    segment = slice(window["start_step"] - 1, window["end_step"])
    heads = {}
    for h, name in enumerate(evidence.head_names):
        support = arrays["support"][segment, h]
        row = head_decomposition(*(arrays[k][segment, h] for k in
            ("raw", "affine", "conditional", "guarded")), evidence.global_scales[h],
            support, window["heads"][name]["eligible"])
        if row["eligible"]:
            outputs = slice(h, h + 1) if h < evidence.reward_dim else slice(evidence.reward_dim, None)
            row.update(moments(arrays["signed"][segment, outputs][support]))
            np.testing.assert_allclose(row["normalized_mse"], row["raw_mean_loss"], rtol=2e-5, atol=1e-10)
            row.update(correction_mse=float(np.mean(arrays["correction"][segment, outputs][support] ** 2)),
                target_mean=arrays["targets"][segment, outputs][support].mean(0).tolist(),
                target_std=arrays["targets"][segment, outputs][support].std(0).tolist(),
                prediction_mean=arrays["prediction"][segment, outputs][support].mean(0).tolist(),
                prediction_std=arrays["prediction"][segment, outputs][support].std(0).tolist(),
                robust_score=robust["heads"][name]["statistic"],
                block_means=robust["heads"][name]["block_means"])
        heads[name] = row
    return dict(start_step=window["start_step"], end_step=window["end_step"],
        eligible=window["eligible"], mean_score=window["statistic"], mean_alarm=window["alarm"],
        robust_score=robust["statistic"], robust_alarm=robust["alarm"], heads=heads)


def reference(windows, names):
    result = {}
    for name in names:
        rows = [w["heads"][name] for w in windows if w["heads"][name]["eligible"]]
        result[name] = dict(eligible_windows=len(rows), quantiles=list(QUANTILES))
        if rows:
            for key in ("raw_mean_loss", "affine_only_mean_loss", "guarded_score", "global_score",
                        "correction_mse", "bias_fraction", "normalized_residual_mean"):
                result[name][key] = np.quantile([r[key] for r in rows], QUANTILES, axis=0).tolist()
    return result


def compare_reference(windows, startup):
    # These ranks describe seen trajectories, not a newly calibrated detector.
    for window in windows:
        for name, row in window["heads"].items():
            baseline = [w["heads"][name] for w in startup if w["heads"][name]["eligible"]]
            if not row["eligible"] or not baseline:
                continue
            row["startup_empirical_cdf"] = {key: float(np.mean([
                r[key] <= row[key] for r in baseline])) for key in
                ("raw_mean_loss", "affine_only_mean_loss", "guarded_score", "global_score")}
            median = float(np.median([r["raw_mean_loss"] for r in baseline]))
            row["raw_loss_to_startup_median"] = row["raw_mean_loss"] / median if median > 0 else None
    return windows


def run(seed, root=ROOT, parent=confirmation.ROOT, source=confirmation.SOURCE, fresh=confirmation.FRESH):
    output = root / f"seed{seed}" / "readout.json"
    if output.exists():
        saved = json.loads(output.read_text())
        assert (saved["protocol"], saved["seed"], saved["complete"]) == (PROTOCOL, seed, True)
        return saved
    prior = json.loads((parent / ENV.replace("-", "_") / f"test_seed{seed}" / "readout.json").read_text())
    confirmation.validate(prior, ENV, seed)
    roles, started = prior["roles"], time.monotonic()
    report = dict(protocol=PROTOCOL, env=ENV, seed=seed, complete=False, roles=roles,
        config=confirmation.CONFIG, retained_method="v25", evidence_integrated=False,
        decision="hold_conditional_evidence_integration", reconstruction_only=True,
        evaluation_scope="diagnostic_on_seen_confirmation_trajectories",
        additional_interactions=0, initialization_probes=0, predictor_optimizer_steps=0,
        residual_scale_optimizer_steps=0, validation_scale_optimizer_steps=0, affine_fits=0,
        policy_optimizer_steps=0, critic_optimizer_steps=0, encoder_optimizer_steps=0,
        frozen_selection=prior["frozen_selection"],
        frozen_thresholds={a: prior["arms"][a]["calibration"]["error_threshold"] for a in confirmation.ARMS})
    confirmation.write(output, report)
    fit = [confirmation.normal_data(source, ENV, s, "expanded_normal") for s in roles["fit_seeds"]]
    fit_x, fit_y = (np.concatenate([r[i] for r in fit]) for i in (0, 1))
    scale = [confirmation.fresh_data(fresh, ENV, s) for s in roles["fit_seeds"]]
    scale.append(confirmation.normal_data(source, ENV, roles["scale_seed"], "expanded_normal"))
    scale_x, scale_y = (np.concatenate([r[i] for r in scale]) for i in (0, 1))
    evidence = confirmation.GuardedEvidence(fit_x.shape[1], fit_y.shape[1], fit[0][2]["reward_dim"])
    evidence.weights = {confirmation.GUARDED: prior["frozen_selection"]["global_weight"]}
    work = evidence.fit(fit_x, fit_y, len(fit[0][0]), 80000000 + 10 * seed)
    assert work == prior["predictor_work"]
    report.update(predictor_optimizer_steps=work["optimizer_steps"], affine_fits=2)
    confirmation.write(output, report)
    work = evidence.fit_scale_losses(scale_x, evidence.raw_losses(scale_x, scale_y), 81000000 + 10 * seed)
    assert work["optimizer_steps"] == prior["scale_work"]["optimizer_steps"]
    np.testing.assert_allclose(work["global_scales"], prior["scale_work"]["global_scales"], rtol=1e-6)
    report.update(residual_scale_optimizer_steps=work["optimizer_steps"],
        global_scales=dict(zip(evidence.head_names, work["global_scales"])), action_heads=evidence.action_heads.tolist())
    calibration = prior["arms"][confirmation.MEAN]["calibration"]
    evidence.radii = np.array([calibration["support_radii"][n] for n in evidence.head_names])
    evidence.thresholds = {confirmation.GUARDED: calibration["error_threshold"]}
    for group in ("startup", "evaluation"):
        rows, _ = confirmation.acquisition.data(parent / "collection", ENV, seed, group, seed_fn=confirmation.seeds)
        all_windows, group_result, combined, offset = [], {}, {a: [] for a in confirmation.ARMS}, 0
        for case, (x, y) in rows.items():
            arrays = scores(evidence, x, y)
            arrays.update(residual_arrays(evidence, x, y))
            paired = confirmation.development.aggregate_windows(evidence, arrays["raw"] / arrays["guarded"], arrays["support"])
            for arm in confirmation.ARMS:
                windows = paired[arm]
                confirmation.development.mark_alarms(windows, report["frozen_thresholds"][arm])
                if group == "evaluation":
                    old = prior["arms"][arm]["cases"][case]["windows"]
                    verify_windows(windows, old)
                    assert [w["alarm"] for w in windows] == [w["alarm"] for w in old]
                combined[arm].extend([{**w, "start_step": w["start_step"] + offset,
                    "end_step": w["end_step"] + offset} for w in windows])
            case_windows = [details(evidence, arrays, a, b) for a, b in zip(
                paired[confirmation.MEAN], paired[confirmation.PRIMARY])]
            if group == "startup":
                all_windows.extend([{**w, "case": case, "start_step": w["start_step"] + offset,
                    "end_step": w["end_step"] + offset} for w in case_windows])
            else:
                selected = case_windows if case == "stationary" else [w for w in case_windows if w["start_step"] > 500]
                group_result[case] = dict(windows=compare_reference(selected, report["startup_windows"]),
                    outcomes={a: {k: prior["arms"][a]["cases"][case][k] for k in
                        ("available", "any_post_alarm", "timely_alarm")} for a in confirmation.ARMS})
            offset += len(x)
        if group == "startup":
            for arm in confirmation.ARMS:
                old = prior["arms"][arm]["calibration"]["windows"]
                verify_windows(combined[arm], old)
                threshold = np.quantile([w["statistic"] for w in combined[arm] if w["eligible"]], confirmation.CONFIG["quantile"])
                np.testing.assert_allclose(threshold, report["frozen_thresholds"][arm], rtol=1e-6)
            report.update(startup_windows=all_windows, startup_reference=reference(all_windows, evidence.head_names))
        else:
            report["cases"] = group_result
        confirmation.write(output, report)
    report.update(complete=True, frozen_scores_and_alarms_reproduced=True, seconds=time.monotonic() - started)
    confirmation.write(output, report)
    print(f"DONE {PROTOCOL} seed={seed} interactions=0 seconds={report['seconds']:.1f}", flush=True)
    return report


def summarize(root=ROOT):
    result = dict(protocol=PROTOCOL, complete=True, retained_method="v25", evidence_integrated=False,
        decision="hold_conditional_evidence_integration", additional_interactions=0,
        initialization_probes=0, predictor_optimizer_steps=0, residual_scale_optimizer_steps=0,
        affine_fits=0, policy_optimizer_steps=0, critic_optimizer_steps=0, encoder_optimizer_steps=0, folds=[])
    for seed in SEEDS:
        row = json.loads((root / f"seed{seed}" / "readout.json").read_text())
        assert (row["protocol"], row["seed"], row["complete"], row["frozen_scores_and_alarms_reproduced"]) == (
            PROTOCOL, seed, True, True)
        for key in ("additional_interactions", "initialization_probes", "policy_optimizer_steps",
                    "critic_optimizer_steps", "encoder_optimizer_steps"):
            assert row[key] == 0
        for key in ("predictor_optimizer_steps", "residual_scale_optimizer_steps", "affine_fits"):
            result[key] += row[key]
        fold = {k: row[k] for k in ("seed", "roles", "frozen_selection", "frozen_thresholds",
            "global_scales", "action_heads", "startup_reference", "seconds")}
        fold["startup_case_reference"] = {case: reference(
            [w for w in row["startup_windows"] if w["case"] == case], row["startup_reference"])
            for case in confirmation.acquisition.STARTUP_STEPS}
        fold["cases"] = {}
        for case, data in row["cases"].items():
            windows = []
            for window in data["windows"]:
                copied = {k: v for k, v in window.items() if k != "heads"}
                copied["heads"] = {}
                for name, head in window["heads"].items():
                    trimmed = {k: v for k, v in head.items() if k not in (
                        "normalized_residual_mean", "normalized_residual_variance", "target_mean",
                        "target_std", "prediction_mean", "prediction_std")}
                    if head["eligible"]:
                        means = np.asarray(head["normalized_residual_mean"])
                        component = int(np.argmax(np.abs(means)))
                        trimmed.update(largest_bias_component=component,
                            largest_signed_normalized_bias=float(means[component]))
                        if name != "dynamics":
                            trimmed.update({k: head[k] for k in ("normalized_residual_mean",
                                "target_mean", "target_std", "prediction_mean", "prediction_std")})
                    copied["heads"][name] = trimmed
                windows.append(copied)
            fold["cases"][case] = dict(outcomes=data["outcomes"], windows=windows)
        result["folds"].append(fold)
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--test-seed", type=int, choices=SEEDS)
    parser.add_argument("--summarize", action="store_true")
    args = parser.parse_args()
    torch.set_num_threads(2)
    if args.summarize:
        report = summarize()
        confirmation.write(ROOT / "compact_readout.json", report)
        print(json.dumps({k: v for k, v in report.items() if k != "folds"}))
    else:
        run(args.test_seed)
