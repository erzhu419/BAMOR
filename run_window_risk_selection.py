"""Cross-source joint-window risk selection with separate training-side calibration."""

import argparse
import copy
import json
from pathlib import Path
import time

import numpy as np
import torch

from collect_matched_scale_normals import ROOT as FRESH, normal_data as fresh_data
from affine_residual_change_evidence import AffineResidualHeadwiseEvidence
from headwise_change_evidence import ARMS, CONFIG
from run_calibration_expansion import verify_windows
from run_cross_source_scale_probe import validate_run
from run_headwise_change_probe import (
    CASES, ENVIRONMENTS, SEEDS, SOURCE, fold_roles, normal_data, source_folder,
    source_readout, summarize_heads, case_summary, screen, write)
from run_matched_residual_scale_probe import ROOT as CONTROL, PROTOCOL as CONTROL_PROTOCOL
from run_residual_scale_shrinkage import GLOBAL_WEIGHTS, select_weight
from run_source_held_scale_shrinkage import aggregate_selection as aggregate_nll


PROTOCOL = "cross_source_joint_window_risk_selection_v1"
ROOT = Path("results/iclr_window_risk_selection_20261001")
RISK = "window_risk_selected"
NLL = "matched_nll_selected"
ALL_ARMS = (*ARMS, RISK, NLL)


class WindowRiskEvidence(AffineResidualHeadwiseEvidence):
    def scales(self, inputs, arm):
        if arm not in (RISK, NLL):
            return super().scales(inputs, arm)
        weight = self.weights[arm]
        return weight * self.global_scales + (1 - weight) * super().scales(inputs, "conditional_scale")

    def calibrate(self, inputs, targets):
        report = super().calibrate(inputs, targets)
        supported = self.distances(inputs) <= self.radii
        losses = self.raw_losses(inputs, targets)
        for arm in (RISK, NLL):
            windows = self._windows(losses / self.scales(inputs, arm), supported)
            eligible = [w["statistic"] for w in windows if w["eligible"]]
            assert len(eligible) >= 2
            self.thresholds[arm] = float(np.quantile(eligible, self.config["quantile"]))
            report[arm] = dict(error_threshold=self.thresholds[arm],
                support_radii=dict(zip(self.head_names, self.radii.tolist())),
                eligible_windows=len(eligible), total_windows=len(windows), windows=windows)
        return report


def window_selection(evidence, validation, calibration):
    val_x, val_y, _ = validation
    cal_x, cal_y, cal_meta = calibration
    radii = np.quantile(evidence.distances(cal_x), evidence.config["quantile"], axis=0)
    cal_support = evidence.distances(cal_x) <= radii
    val_support = evidence.distances(val_x) <= radii
    cal_losses, val_losses = evidence.raw_losses(cal_x, cal_y), evidence.raw_losses(val_x, val_y)
    cal_conditional = evidence.scales(cal_x, "conditional_scale")
    val_conditional = evidence.scales(val_x, "conditional_scale")
    report = select_weight(val_losses, val_conditional, evidence.global_scales)
    for row in report["candidates"]:
        weight = row["global_weight"]
        cal_scale = weight * evidence.global_scales + (1 - weight) * cal_conditional
        val_scale = weight * evidence.global_scales + (1 - weight) * val_conditional
        cal_windows = evidence._windows(cal_losses / cal_scale, cal_support)
        val_windows = evidence._windows(val_losses / val_scale, val_support)
        cal_scores = [w["statistic"] for w in cal_windows if w["eligible"]]
        val_scores = [w["statistic"] for w in val_windows if w["eligible"]]
        assert len(cal_scores) >= 2 and len(val_scores) > 0, "No supported calibration/validation windows"
        threshold = float(np.quantile(cal_scores, evidence.config["quantile"]))
        exceedances = sum(score > threshold for score in val_scores)
        row.update(error_threshold=threshold, calibration_eligible_windows=len(cal_scores),
            validation_eligible_windows=len(val_scores), validation_total_windows=len(val_windows),
            joint_window_exceedances=exceedances, window_risk=exceedances / len(val_scores))
    report.update(inner_calibration_seed=cal_meta["seed"], inner_calibration_steps=len(cal_x),
        support_radii=dict(zip(evidence.head_names, radii.tolist())),
        validation_head_supported_fraction=val_support.mean(0).tolist(),
        candidate_independent_support=True)
    return report


def source_validation_folds(evidence, rows, calibration, seed):
    assert len(rows) == 3 and len({row[2]["seed"] for row in rows}) == 3
    losses = [evidence.raw_losses(row[0], row[1]) for row in rows]
    for held in range(len(rows)):
        training = [index for index in range(len(rows)) if index != held]
        train_x = np.concatenate([rows[index][0] for index in training])
        train_losses = np.concatenate([losses[index] for index in training])
        work = evidence.fit_scale_losses(train_x, train_losses, seed)
        yield dict(held_source=rows[held][2]["seed"],
            train_sources=[rows[index][2]["seed"] for index in training],
            train_steps=len(train_x), validation_steps=len(rows[held][0]), seed=seed,
            scale_work=work, selection=window_selection(evidence, rows[held], calibration))


def aggregate_selection(folds):
    nll = aggregate_nll(folds)
    candidates = copy.deepcopy(nll["candidates"])
    for index, row in enumerate(candidates):
        scores = [fold["selection"]["candidates"][index] for fold in folds]
        row.update(window_risk=float(np.mean([s["window_risk"] for s in scores])),
            joint_window_exceedances=sum(s["joint_window_exceedances"] for s in scores),
            validation_eligible_windows=sum(s["validation_eligible_windows"] for s in scores),
            validation_total_windows=sum(s["validation_total_windows"] for s in scores))
    chosen = min(candidates, key=lambda row: (row["window_risk"], row["normal_nll"], -row["global_weight"]))
    risk = dict(nll, global_weight=chosen["global_weight"], normal_nll=chosen["normal_nll"],
        window_risk=chosen["window_risk"], candidates=candidates,
        objective="mean_source_joint_window_exceedance_rate", tie_break="nll_then_larger_global_weight")
    return {RISK: risk, NLL: nll}


def run(env, test_seed, source=SOURCE, root=ROOT, control=CONTROL, fresh=FRESH):
    roles = fold_roles(test_seed)
    output = root / env.replace("-", "_") / f"test_seed{test_seed}" / "readout.json"
    if output.exists():
        saved = json.loads(output.read_text())
        validate_run(saved, env, test_seed, PROTOCOL)
        assert saved["validation_scale_optimizer_steps"] == 3 * saved["residual_scale_optimizer_steps"]
        return saved
    parent = json.loads((control / env.replace("-", "_") / f"test_seed{test_seed}" / "readout.json").read_text())
    validate_run(parent, env, test_seed, CONTROL_PROTOCOL)
    assert parent["scale_fit_steps"] == 15000
    assert parent["scale_fit_sources"] == roles["fit_seeds"] + [roles["scale_seed"]]
    started = time.monotonic()
    report = dict(protocol=PROTOCOL, env=env, roles=roles, config=CONFIG, complete=False,
        development_only=True, additional_interactions=0, initialization_probes=0,
        predictor_optimizer_steps=0, residual_scale_optimizer_steps=0,
        validation_scale_optimizer_steps=0, total_residual_scale_optimizer_steps=0,
        policy_optimizer_steps=0, critic_optimizer_steps=0, encoder_optimizer_steps=0,
        reconstruction_only=True, arms={}, source_validation_folds=[], control_scores_reproduced=False,
        matched_scale_control=dict(protocol=CONTROL_PROTOCOL, reused=True, arms=parent["arms"]))
    write(output, report)
    fit_rows = [normal_data(source, env, seed, "expanded_normal") for seed in roles["fit_seeds"]]
    fit_x, fit_y = (np.concatenate([row[index] for row in fit_rows]) for index in (0, 1))
    scale_rows = [fresh_data(fresh, env, seed) for seed in roles["fit_seeds"]]
    scale_rows.append(normal_data(source, env, roles["scale_seed"], "expanded_normal"))
    assert len(fit_x) == 10000 and all(len(row[0]) == 5000 for row in scale_rows)
    inner_calibration = fresh_data(fresh, env, roles["scale_seed"])
    assert len(inner_calibration[0]) == 5000
    report["inner_calibration_source"] = inner_calibration[2]
    evidence = WindowRiskEvidence(fit_x.shape[1], fit_y.shape[1], fit_rows[0][2]["reward_dim"])
    predictor_work = evidence.fit(fit_x, fit_y, len(fit_rows[0][0]), 80000000 + 10 * test_seed)
    report.update(predictor_optimizer_steps=predictor_work["optimizer_steps"], predictor_work=predictor_work)
    write(output, report)
    assert predictor_work == parent["predictor_work"]
    scale_seed = 81000000 + 10 * test_seed
    for fold in source_validation_folds(evidence, scale_rows, inner_calibration, scale_seed):
        report["source_validation_folds"].append(fold)
        report["validation_scale_optimizer_steps"] += fold["scale_work"]["optimizer_steps"]
        report["total_residual_scale_optimizer_steps"] = report["validation_scale_optimizer_steps"]
        write(output, report)
    selection = aggregate_selection(report["source_validation_folds"])
    evidence.weights = {arm: row["global_weight"] for arm, row in selection.items()}
    report["selections"] = selection
    write(output, report)
    # Refit the original deployed scale on all sources after selecting the coefficient.
    scale_x, scale_y = (np.concatenate([row[index] for row in scale_rows]) for index in (0, 1))
    scale_work = evidence.fit_scale_losses(scale_x, evidence.raw_losses(scale_x, scale_y), scale_seed)
    report.update(residual_scale_optimizer_steps=scale_work["optimizer_steps"], scale_work=scale_work,
        total_residual_scale_optimizer_steps=report["validation_scale_optimizer_steps"] + scale_work["optimizer_steps"],
        fit_steps=len(fit_x), scale_steps=len(scale_rows[-1][0]), scale_source=scale_rows[-1][2],
        scale_fit_steps=len(scale_x), scale_fit_sources=parent["scale_fit_sources"])
    write(output, report)
    assert scale_work["optimizer_steps"] == parent["scale_work"]["optimizer_steps"]
    np.testing.assert_allclose(scale_work["global_scales"], parent["scale_work"]["global_scales"], rtol=1e-6)
    cal_x, cal_y, cal_meta = normal_data(source, env, roles["calibration_seed"], "expanded_normal")
    assert len(cal_x) == 5000
    calibration = evidence.calibrate(cal_x, cal_y)
    report.update(calibration_steps=len(cal_x), calibration_source=cal_meta)
    for arm in ARMS:
        old = parent["arms"][arm]["calibration"]
        assert calibration[arm]["support_radii"] == old["support_radii"]
        np.testing.assert_allclose(calibration[arm]["error_threshold"], old["error_threshold"], rtol=1e-5, atol=1e-8)
        verify_windows(calibration[arm]["windows"], old["windows"])
    original = source_readout(source, env, test_seed)
    for arm in ALL_ARMS:
        row = dict(calibration=calibration[arm], cases={})
        report["arms"][arm] = row
        assert calibration[arm]["support_radii"] == parent["arms"]["conditional_scale"]["calibration"]["support_radii"]
        with np.load(source_folder(source, env, test_seed) / "transitions.npz") as data:
            for case in CASES:
                evaluation = evidence.evaluate(data[f"{case}_inputs"], data[f"{case}_targets"], arm)
                summary = case_summary(evaluation)
                summary["heads"] = summarize_heads(evidence, evaluation)
                old = parent["arms"][arm if arm in ARMS else "conditional_scale"]["cases"][case]
                if arm in ARMS:
                    verify_windows(summary["windows"], old["windows"])
                    assert [w["alarm"] for w in summary["windows"]] == [w["alarm"] for w in old["windows"]]
                for key in ("available", "pre_available", "post_eligible_windows"):
                    assert summary[key] == old[key]
                if case in ("actor_change", "sampler_change"):
                    summary["intervention_active"] = original["cases"][case]["intervention_active"]
                row["cases"][case] = summary
        write(output, report)
    report.update(complete=True, control_scores_reproduced=True, seconds=time.monotonic() - started)
    write(output, report)
    print(f"DONE {env} seed={test_seed} weights={evidence.weights} "
          f"scale_updates={report['total_residual_scale_optimizer_steps']} added_interactions=0", flush=True)
    return report


def summarize(root=ROOT):
    report = dict(protocol=PROTOCOL, config=CONFIG, complete=True, development_only=True,
        additional_interactions=0, initialization_probes=0, predictor_optimizer_steps=0,
        residual_scale_optimizer_steps=0, validation_scale_optimizer_steps=0,
        total_residual_scale_optimizer_steps=0, policy_optimizer_steps=0, critic_optimizer_steps=0,
        encoder_optimizer_steps=0, reconstruction_only=True, environments={}, retained_method="v25",
        evidence_integrated=False, reused_acquisition_interactions_counted_in_parent=50030)
    for env in ENVIRONMENTS:
        runs = [json.loads((root / env.replace("-", "_") / f"test_seed{seed}" / "readout.json").read_text()) for seed in SEEDS]
        for seed, row in zip(SEEDS, runs):
            validate_run(row, env, seed, PROTOCOL)
            assert row["control_scores_reproduced"] and row["scale_fit_steps"] == 15000
            sources = row["roles"]["fit_seeds"] + [row["roles"]["scale_seed"]]
            assert row["scale_fit_sources"] == sources
            assert row["inner_calibration_source"]["seed"] == row["roles"]["scale_seed"]
            for selected in row["selections"].values():
                assert selected["validation_steps"] == 15000
                assert selected["validation_sources"] == sources
                assert selected["global_weight"] in GLOBAL_WEIGHTS
            assert [fold["held_source"] for fold in row["source_validation_folds"]] == sources
            for fold in row["source_validation_folds"]:
                assert fold["train_sources"] == [s for s in sources if s != fold["held_source"]]
                assert (fold["train_steps"], fold["validation_steps"]) == (10000, 5000)
                assert fold["selection"]["inner_calibration_seed"] == row["roles"]["scale_seed"]
                assert fold["selection"]["inner_calibration_steps"] == 5000
                counts = [c["validation_eligible_windows"] for c in fold["selection"]["candidates"]]
                assert len(set(counts)) == 1
            assert row["validation_scale_optimizer_steps"] == sum(f["scale_work"]["optimizer_steps"] for f in row["source_validation_folds"]) == 3 * row["residual_scale_optimizer_steps"]
            assert row["total_residual_scale_optimizer_steps"] == row["validation_scale_optimizer_steps"] + row["residual_scale_optimizer_steps"]
            assert all(row[key] == 0 for key in ("additional_interactions", "initialization_probes",
                "policy_optimizer_steps", "critic_optimizer_steps", "encoder_optimizer_steps"))
            for key in ("predictor_optimizer_steps", "residual_scale_optimizer_steps",
                        "validation_scale_optimizer_steps", "total_residual_scale_optimizer_steps"):
                report[key] += row[key]
        report["environments"][env] = dict(arms={arm: screen(runs, arm) for arm in ALL_ARMS},
            global_weights={arm: {str(row["roles"]["test_seed"]): row["selections"][arm]["global_weight"]
                for row in runs} for arm in (RISK, NLL)}, runs=runs)
    passed = all(e["arms"][RISK]["development_screen_passed"] for e in report["environments"].values())
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
        print(json.dumps(dict(environments={env: dict(arms=e["arms"], global_weights=e["global_weights"])
            for env, e in report["environments"].items()}, decision=report["decision"],
            compact_readout_bytes=output.stat().st_size)))
    else:
        run(args.env, args.test_seed, root=args.root)

