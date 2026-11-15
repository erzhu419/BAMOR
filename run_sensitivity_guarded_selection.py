"""Select normal-window risk subject to held training-source change sensitivity."""

import argparse
import copy
import json
from pathlib import Path
import time

import numpy as np
import torch

from collect_matched_scale_normals import ROOT as FRESH, normal_data as fresh_data
from headwise_change_evidence import ARMS, CONFIG
from run_calibration_expansion import verify_windows
from run_cross_source_scale_probe import validate_run
from run_headwise_change_probe import (
    CASES, ENVIRONMENTS, SEEDS, SOURCE, fold_roles, normal_data, source_folder,
    source_readout, summarize_heads, case_summary, screen, write)
from run_matched_residual_scale_probe import ROOT as CONTROL, PROTOCOL as CONTROL_PROTOCOL
from run_residual_scale_shrinkage import GLOBAL_WEIGHTS
from run_window_risk_selection import (
    WindowRiskEvidence, window_selection, aggregate_selection as unguarded_selection,
    ROOT as PREVIOUS, PROTOCOL as PREVIOUS_PROTOCOL, RISK, NLL)


PROTOCOL = "training_change_sensitivity_guarded_selection_v1"
ROOT = Path("results/iclr_sensitivity_guarded_selection_20261001")
GUARDED = "sensitivity_guarded_selected"
ALL_ARMS = (*ARMS, RISK, NLL, GUARDED)
MODE_CASES = ("mode1", "mode2", "mode3")


class GuardedEvidence(WindowRiskEvidence):
    def scales(self, inputs, arm):
        if arm != GUARDED:
            return super().scales(inputs, arm)
        weight = self.weights[arm]
        return weight * self.global_scales + (1 - weight) * super().scales(inputs, "conditional_scale")

    def calibrate(self, inputs, targets):
        report = super().calibrate(inputs, targets)
        supported = self.distances(inputs) <= self.radii
        windows = self._windows(self.raw_losses(inputs, targets) / self.scales(inputs, GUARDED), supported)
        eligible = [w["statistic"] for w in windows if w["eligible"]]
        assert len(eligible) >= 2
        self.thresholds[GUARDED] = float(np.quantile(eligible, self.config["quantile"]))
        report[GUARDED] = dict(error_threshold=self.thresholds[GUARDED],
            support_radii=dict(zip(self.head_names, self.radii.tolist())),
            eligible_windows=len(eligible), total_windows=len(windows), windows=windows)
        return report


def training_changes(source, env, seed):
    audit = source_readout(source, env, seed)
    assert all(audit["cases"][case]["ground_truth_mode_history_audit_only"] ==
        [[0, 0], [500, int(case[-1])]] for case in MODE_CASES)
    with np.load(source_folder(source, env, seed) / "transitions.npz") as data:
        rows = {case: (data[f"{case}_inputs"].copy(), data[f"{case}_targets"].copy()) for case in MODE_CASES}
    assert all(len(x) == len(y) == 1000 for x, y in rows.values())
    return rows


def add_change_sensitivity(evidence, probes, selection):
    radii = np.array([selection["support_radii"][name] for name in evidence.head_names])
    for candidate in selection["candidates"]:
        candidate["change_probes"] = {}
    for case, (inputs, targets) in probes.items():
        raw = evidence.raw_losses(inputs, targets)
        conditional = evidence.scales(inputs, "conditional_scale")
        support = evidence.distances(inputs) <= radii
        for candidate in selection["candidates"]:
            weight = candidate["global_weight"]
            scales = weight * evidence.global_scales + (1 - weight) * conditional
            windows = evidence._windows(raw / scales, support)
            for window in windows:
                window["alarm"] = bool(window["eligible"] and window["statistic"] > candidate["error_threshold"])
            summary = case_summary(dict(windows=windows, supported=support.any(1)))
            summary.pop("windows")
            candidate["change_probes"][case] = summary
    selection["change_probe_steps"] = sum(len(x) for x, _ in probes.values())
    selection["change_probe_cases"] = list(probes)
    return selection


def source_validation_folds(evidence, rows, calibration, changes, seed):
    assert len(rows) == 3 and len({row[2]["seed"] for row in rows}) == 3
    losses = [evidence.raw_losses(row[0], row[1]) for row in rows]
    for held in range(len(rows)):
        training = [i for i in range(len(rows)) if i != held]
        train_x = np.concatenate([rows[i][0] for i in training])
        train_losses = np.concatenate([losses[i] for i in training])
        work = evidence.fit_scale_losses(train_x, train_losses, seed)
        source_seed = rows[held][2]["seed"]
        selection = window_selection(evidence, rows[held], calibration)
        add_change_sensitivity(evidence, changes[source_seed], selection)
        yield dict(held_source=source_seed, train_sources=[rows[i][2]["seed"] for i in training],
            train_steps=len(train_x), validation_steps=len(rows[held][0]), seed=seed,
            scale_work=work, selection=selection)


def aggregate_selection(folds):
    selections = unguarded_selection(folds)
    candidates = copy.deepcopy(selections[RISK]["candidates"])
    for index, candidate in enumerate(candidates):
        protected, lost = [], []
        for fold in folds:
            baseline = fold["selection"]["candidates"][-1]
            probe = fold["selection"]["candidates"][index]
            assert baseline["global_weight"] == 0.
            for case in MODE_CASES:
                old, new = baseline["change_probes"][case], probe["change_probes"][case]
                for key in ("available", "post_eligible_windows"):
                    assert old[key] == new[key], "Scale choice must not change probe coverage"
                if old["timely_alarm"]:
                    reference = dict(source_seed=fold["held_source"], case=case)
                    protected.append(reference)
                    if not new["timely_alarm"]:
                        lost.append(reference)
        candidate.update(protected_probes=protected, lost_protected_probes=lost,
                         sensitivity_feasible=not lost)
    assert candidates[-1]["sensitivity_feasible"]
    feasible = [c for c in candidates if c["sensitivity_feasible"]]
    chosen = min(feasible, key=lambda c: (c["window_risk"], c["normal_nll"], -c["global_weight"]))
    selections[GUARDED] = dict(selections[RISK], global_weight=chosen["global_weight"],
        normal_nll=chosen["normal_nll"], window_risk=chosen["window_risk"], candidates=candidates,
        protected_probes=chosen["protected_probes"], lost_protected_probes=chosen["lost_protected_probes"],
        constraint="preserve_each_conditional_scale_training_probe_timely_detection",
        constraint_baseline_global_weight=0., training_intervention_labels_used=True)
    return selections


def run(env, test_seed, source=SOURCE, root=ROOT, control=CONTROL, fresh=FRESH, previous=PREVIOUS):
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
    prior = json.loads((previous / env.replace("-", "_") / f"test_seed{test_seed}" / "readout.json").read_text())
    validate_run(prior, env, test_seed, PREVIOUS_PROTOCOL)
    started = time.monotonic()
    report = dict(protocol=PROTOCOL, env=env, roles=roles, config=CONFIG, complete=False,
        development_only=True, additional_interactions=0, initialization_probes=0,
        predictor_optimizer_steps=0, residual_scale_optimizer_steps=0,
        validation_scale_optimizer_steps=0, total_residual_scale_optimizer_steps=0,
        policy_optimizer_steps=0, critic_optimizer_steps=0, encoder_optimizer_steps=0,
        reconstruction_only=True, arms={}, source_validation_folds=[], control_scores_reproduced=False,
        prior_selection_controls_reproduced=False, training_intervention_labels_used=True,
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
    changes = {row[2]["seed"]: training_changes(source, env, row[2]["seed"]) for row in scale_rows}
    report["change_probe_sources"] = list(changes)
    report["training_change_probe_steps"] = sum(len(x) for rows in changes.values() for x, _ in rows.values())
    evidence = GuardedEvidence(fit_x.shape[1], fit_y.shape[1], fit_rows[0][2]["reward_dim"])
    predictor_work = evidence.fit(fit_x, fit_y, len(fit_rows[0][0]), 80000000 + 10 * test_seed)
    report.update(predictor_optimizer_steps=predictor_work["optimizer_steps"], predictor_work=predictor_work)
    write(output, report)
    assert predictor_work == parent["predictor_work"]
    scale_seed = 81000000 + 10 * test_seed
    for fold in source_validation_folds(evidence, scale_rows, inner_calibration, changes, scale_seed):
        report["source_validation_folds"].append(fold)
        report["validation_scale_optimizer_steps"] += fold["scale_work"]["optimizer_steps"]
        report["total_residual_scale_optimizer_steps"] = report["validation_scale_optimizer_steps"]
        write(output, report)
    selection = aggregate_selection(report["source_validation_folds"])
    for arm in (RISK, NLL):
        assert selection[arm] == prior["selections"][arm]
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
    for arm in (RISK, NLL):
        assert report["arms"][arm] == prior["arms"][arm]
    report.update(complete=True, control_scores_reproduced=True, prior_selection_controls_reproduced=True,
        seconds=time.monotonic() - started)
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
        evidence_integrated=False, reused_acquisition_interactions_counted_in_parent=140100,
        reused_acquisition_ledgers=dict(original_change_probe=90070, fresh_normal_collection=50030))
    for env in ENVIRONMENTS:
        runs = [json.loads((root / env.replace("-", "_") / f"test_seed{seed}" / "readout.json").read_text()) for seed in SEEDS]
        for seed, row in zip(SEEDS, runs):
            validate_run(row, env, seed, PROTOCOL)
            assert row["control_scores_reproduced"] and row["prior_selection_controls_reproduced"]
            assert row["scale_fit_steps"] == 15000
            sources = row["roles"]["fit_seeds"] + [row["roles"]["scale_seed"]]
            assert row["scale_fit_sources"] == row["change_probe_sources"] == sources
            assert row["training_change_probe_steps"] == 9000
            assert row["selections"][GUARDED]["lost_protected_probes"] == []
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
                for row in runs} for arm in (RISK, NLL, GUARDED)}, runs=runs)
    passed = all(e["arms"][GUARDED]["development_screen_passed"] for e in report["environments"].values())
    report["decision"] = "consider_fresh_seed_evidence_probe" if passed else "hold_conditional_evidence_integration"
    return report


def compact_report(report):
    compact = copy.deepcopy(report)
    for environment in compact["environments"].values():
        for row in environment["runs"]:
            row["matched_scale_control"].pop("arms")
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
        print(json.dumps(dict(environments={env: dict(arms=e["arms"], global_weights=e["global_weights"])
            for env, e in report["environments"].items()}, decision=report["decision"],
            compact_readout_bytes=output.stat().st_size)))
    else:
        run(args.env, args.test_seed, root=args.root)


