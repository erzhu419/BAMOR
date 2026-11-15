"""Select residual-scale shrinkage on held training-side normal trajectories."""

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
from run_calibration_expansion import verify_windows
from run_cross_source_scale_probe import validate_run
from run_headwise_change_probe import (
    CASES, ENVIRONMENTS, SEEDS, SOURCE, fold_roles, normal_data, source_folder,
    source_readout, summarize_heads, case_summary, screen, write)
from run_matched_residual_scale_probe import ROOT as CONTROL, PROTOCOL as CONTROL_PROTOCOL


PROTOCOL = "held_normal_residual_scale_shrinkage_v1"
ROOT = Path("results/iclr_residual_scale_shrinkage_20261001")
SELECTED = "selected_shrinkage"
ALL_ARMS = (*ARMS, SELECTED)
GLOBAL_WEIGHTS = (1., .75, .5, .25, 0.)


def select_weight(losses, conditional, global_scales):
    losses = np.asarray(losses, np.float64)
    conditional = np.asarray(conditional, np.float64)
    global_scales = np.asarray(global_scales, np.float64)
    assert losses.shape == conditional.shape and losses.shape[1] == len(global_scales)
    assert np.all(losses >= 0) and np.all(conditional > 0) and np.all(global_scales > 0)
    candidates = []
    for weight in GLOBAL_WEIGHTS:
        scales = weight * global_scales + (1 - weight) * conditional
        # Equal weight per transition and head; the log reference is constant in weight.
        per_head = (losses / scales + np.log(scales / global_scales)).mean(0)
        candidates.append(dict(global_weight=weight, normal_nll=float(per_head.mean()),
                               head_normal_nll=per_head.tolist()))
    selected = min(candidates, key=lambda row: row["normal_nll"])
    return dict(global_weight=selected["global_weight"], normal_nll=selected["normal_nll"],
                candidates=candidates, validation_steps=len(losses), shared_across_heads=True,
                objective="mean_head_transition_residual_nll", tie_break="larger_global_weight")


class ShrunkResidualEvidence(AffineResidualHeadwiseEvidence):
    def select_shrinkage(self, inputs, targets):
        report = select_weight(self.raw_losses(inputs, targets),
            super().scales(inputs, "conditional_scale"), self.global_scales)
        self.global_weight = report["global_weight"]
        return report

    def scales(self, inputs, arm):
        if arm != SELECTED:
            return super().scales(inputs, arm)
        conditional = super().scales(inputs, "conditional_scale")
        return self.global_weight * self.global_scales + (1 - self.global_weight) * conditional

    def calibrate(self, inputs, targets):
        report = super().calibrate(inputs, targets)
        supported = self.distances(inputs) <= self.radii
        windows = self._windows(self.raw_losses(inputs, targets) / self.scales(inputs, SELECTED), supported)
        eligible = [w["statistic"] for w in windows if w["eligible"]]
        assert len(eligible) >= 2
        self.thresholds[SELECTED] = float(np.quantile(eligible, self.config["quantile"]))
        report[SELECTED] = dict(error_threshold=self.thresholds[SELECTED],
            support_radii=dict(zip(self.head_names, self.radii.tolist())),
            eligible_windows=len(eligible), total_windows=len(windows), windows=windows)
        return report


def run(env, test_seed, source=SOURCE, root=ROOT, control=CONTROL, fresh=FRESH):
    roles = fold_roles(test_seed)
    output = root / env.replace("-", "_") / f"test_seed{test_seed}" / "readout.json"
    if output.exists():
        saved = json.loads(output.read_text())
        validate_run(saved, env, test_seed, PROTOCOL)
        return saved
    parent = json.loads((control / env.replace("-", "_") / f"test_seed{test_seed}" / "readout.json").read_text())
    validate_run(parent, env, test_seed, CONTROL_PROTOCOL)
    assert parent["scale_fit_steps"] == 15000
    assert parent["scale_fit_sources"] == roles["fit_seeds"] + [roles["scale_seed"]]
    started = time.monotonic()
    report = dict(protocol=PROTOCOL, env=env, roles=roles, config=CONFIG, complete=False,
        development_only=True, additional_interactions=0, initialization_probes=0,
        predictor_optimizer_steps=0, residual_scale_optimizer_steps=0,
        policy_optimizer_steps=0, critic_optimizer_steps=0, encoder_optimizer_steps=0,
        reconstruction_only=True, arms={}, control_scores_reproduced=False,
        matched_scale_control=dict(protocol=CONTROL_PROTOCOL, reused=True, arms=parent["arms"]))
    write(output, report)
    fit_rows = [normal_data(source, env, seed, "expanded_normal") for seed in roles["fit_seeds"]]
    fit_x, fit_y = (np.concatenate([r[index] for r in fit_rows]) for index in (0, 1))
    scale_rows = [fresh_data(fresh, env, seed) for seed in roles["fit_seeds"]]
    scale_rows.append(normal_data(source, env, roles["scale_seed"], "expanded_normal"))
    scale_x, scale_y = (np.concatenate([r[index] for r in scale_rows]) for index in (0, 1))
    validation_x, validation_y, validation_meta = fresh_data(fresh, env, roles["scale_seed"])
    assert (len(fit_x), len(scale_x), len(validation_x)) == (10000, 15000, 5000)
    evidence = ShrunkResidualEvidence(fit_x.shape[1], fit_y.shape[1], fit_rows[0][2]["reward_dim"])
    predictor_work = evidence.fit(fit_x, fit_y, len(fit_rows[0][0]), 80000000 + 10 * test_seed)
    report.update(predictor_optimizer_steps=predictor_work["optimizer_steps"], predictor_work=predictor_work)
    write(output, report)
    assert predictor_work == parent["predictor_work"]
    scale_work = evidence.fit_scale_losses(scale_x, evidence.raw_losses(scale_x, scale_y), 81000000 + 10 * test_seed)
    report.update(residual_scale_optimizer_steps=scale_work["optimizer_steps"], scale_work=scale_work)
    write(output, report)
    assert scale_work["optimizer_steps"] == parent["scale_work"]["optimizer_steps"]
    np.testing.assert_allclose(scale_work["global_scales"], parent["scale_work"]["global_scales"], rtol=1e-6)
    selection = evidence.select_shrinkage(validation_x, validation_y)
    selection.update(source_seed=roles["scale_seed"], source_step=validation_meta["source_step"],
                     trajectory_held_out_from_mean_and_scale=True, policy_source_held_out_from_scale=False)
    report.update(selection=selection, fit_steps=len(fit_x), scale_steps=len(scale_rows[-1][0]),
        scale_source=scale_rows[-1][2], scale_fit_steps=len(scale_x),
        scale_fit_sources=parent["scale_fit_sources"], validation_source_seed=roles["scale_seed"])
    write(output, report)
    # Calibration and test targets are accessed only after selecting the weight.
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
    print(f"DONE {env} seed={test_seed} global_weight={selection['global_weight']} added_interactions=0", flush=True)
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
            validate_run(row, env, seed, PROTOCOL)
            assert row["control_scores_reproduced"]
            assert (row["fit_steps"], row["scale_fit_steps"], row["calibration_steps"], row["selection"]["validation_steps"]) == (10000, 15000, 5000, 5000)
            assert row["scale_fit_sources"] == row["roles"]["fit_seeds"] + [row["roles"]["scale_seed"]]
            assert row["selection"]["source_seed"] == row["roles"]["scale_seed"] == row["validation_source_seed"]
            assert row["selection"]["global_weight"] in GLOBAL_WEIGHTS
            assert all(row[key] == 0 for key in ("additional_interactions", "initialization_probes",
                "policy_optimizer_steps", "critic_optimizer_steps", "encoder_optimizer_steps"))
            for key in ("predictor_optimizer_steps", "residual_scale_optimizer_steps"):
                report[key] += row[key]
        report["environments"][env] = dict(arms={arm: screen(runs, arm) for arm in ALL_ARMS},
            global_weights={str(row["roles"]["test_seed"]): row["selection"]["global_weight"] for row in runs}, runs=runs)
    passed = all(e["arms"][SELECTED]["development_screen_passed"] for e in report["environments"].values())
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
