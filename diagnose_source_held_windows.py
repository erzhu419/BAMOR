"""Read-only paired-window algebra for whole-source scale selection."""

import argparse
from collections import Counter
import copy
import json
from pathlib import Path
import time

import numpy as np

from diagnose_matched_scale_transfer import pair_window, GROUPS, group_name, ZERO_WORK
from run_cross_source_scale_probe import validate_run
from run_headwise_change_probe import CHANGE_AT, ENVIRONMENTS, SEEDS, fold_roles, write
from run_residual_scale_shrinkage import SELECTED
from run_source_held_scale_shrinkage import ROOT as INPUT, PROTOCOL as INPUT_PROTOCOL


PROTOCOL = "stored_source_held_window_diagnostic_v1"
ROOT = INPUT / "window_diagnostic"
WORK = {**ZERO_WORK, "validation_scale_optimizer_steps": 0,
        "total_residual_scale_optimizer_steps": 0}
NORMAL_CASES = ("stationary", "sampler_change", "actor_change")
MODE_CASES = ("mode1", "mode2", "mode3")


def paired_window(global_row, old, new, global_scales, thresholds):
    row = pair_window(dict(global_scale=global_row, conditional_scale=old),
        dict(global_scale=global_row, conditional_scale=new), global_scales, global_scales, thresholds)
    eligible = [name for name, head in row["heads"].items() if head["eligible"]]
    row["old_dominant_head"] = max(eligible, key=lambda name: row["heads"][name]["old_score"]) if eligible else None
    row["winner_changed"] = row["old_dominant_head"] != row["dominant_head"]
    row["score_only_alarm"] = bool(eligible and row["new_joint_score"] > thresholds[0])
    row["threshold_only_alarm"] = bool(eligible and row["old_joint_score"] > thresholds[1])
    row["sufficient_change"] = None
    if row["new_alarm"] and not row["old_alarm"]:
        score, threshold = row["score_only_alarm"], row["threshold_only_alarm"]
        row["sufficient_change"] = ("both_sufficient" if score and threshold else
            "score_sufficient" if score else "threshold_sufficient" if threshold else "joint_required")
    for side, window, threshold in (("old", old, thresholds[0]), ("new", new, thresholds[1])):
        if "alarm" in window:
            assert row[f"{side}_alarm"] == bool(eligible and row[f"{side}_joint_score"] > threshold)
        if eligible:
            np.testing.assert_allclose(row[f"{side}_joint_score"],
                max(head[f"{side}_score"] for head in row["heads"].values() if head["eligible"]))
    return row


def paired_series(global_rows, old, new, global_scales, thresholds):
    assert len(global_rows) == len(old) == len(new)
    return [paired_window(g, a, b, global_scales, thresholds) for g, a, b in zip(global_rows, old, new)]


def describe(rows):
    result = dict(windows=len(rows), eligible_windows=sum(row["eligible"] for row in rows),
        old_alarm_windows=sum(row["old_alarm"] for row in rows), new_alarm_windows=sum(row["new_alarm"] for row in rows),
        winner_changes=sum(row["winner_changed"] for row in rows),
        sufficient_changes=dict(Counter(row["sufficient_change"] for row in rows if row["sufficient_change"])),
        new_dominant_heads=dict(Counter(row["dominant_head"] for row in rows if row["dominant_head"])),
        head_transitions=dict(Counter(f"{row['old_dominant_head']} -> {row['dominant_head']}"
            for row in rows if row["winner_changed"])))
    for field in ("scale_factor", "threshold_factor", "raw_error_factor"):
        values = [row["heads"][row["dominant_head"]].get(field) for row in rows if row["dominant_head"]]
        values = [value for value in values if value is not None]
        result[field] = dict(median=float(np.median(values)), minimum=float(min(values)),
                             maximum=float(max(values))) if values else None
    return result


def calibration_tail(rows, side, threshold, quantile):
    key = f"{side}_joint_score"
    ordered = sorted((row for row in rows if row["eligible"]), key=lambda row: row[key])
    np.testing.assert_allclose(np.quantile([row[key] for row in ordered], quantile), threshold)
    position = (len(ordered) - 1) * quantile
    lower, upper = int(np.floor(position)), int(np.ceil(position))
    operands = []
    for index, weight in ((lower, 1. - position + lower), (upper, position - lower)):
        if weight:
            row = ordered[index]
            winner = row["old_dominant_head"] if side == "old" else row["dominant_head"]
            operands.append(dict(weight=float(weight), end_step=row["end_step"],
                joint_score=row[key], dominant_head=winner, old_joint_score=row["old_joint_score"],
                new_joint_score=row["new_joint_score"], decomposition=row["heads"][winner]))
    return dict(threshold=threshold, operands=operands)


def analyze_fold(source):
    old, new = source["arms"]["conditional_scale"], source["arms"][SELECTED]
    assert old == source["trajectory_control"]["arms"][SELECTED]
    assert source["trajectory_control"]["selection"]["global_weight"] == 0
    assert old["calibration"]["support_radii"] == new["calibration"]["support_radii"]
    scales = source["scale_work"]["global_scales"]
    thresholds = (old["calibration"]["error_threshold"], new["calibration"]["error_threshold"])
    global_arm = source["arms"]["global_scale"]
    calibration = paired_series(global_arm["calibration"]["windows"],
        old["calibration"]["windows"], new["calibration"]["windows"], scales, thresholds)
    normal, modes = [], []
    prefixes = {}
    for case in (*NORMAL_CASES, *MODE_CASES):
        paired = paired_series(global_arm["cases"][case]["windows"],
            old["cases"][case]["windows"], new["cases"][case]["windows"], scales, thresholds)
        prefixes[case] = [row for row in paired if row["end_step"] <= CHANGE_AT]
        if case == "stationary":
            normal.extend(dict(row, case="shared_prefix") for row in prefixes[case])
        target = normal if case in NORMAL_CASES else modes
        target.extend(dict(row, case=case) for row in paired if row["start_step"] > CHANGE_AT)
    assert all(prefix == prefixes["stationary"] for prefix in prefixes.values()), "Shared prefixes must reproduce"
    selection = copy.deepcopy(source["selection"])
    original_nll = next(row["normal_nll"] for row in selection["candidates"] if row["global_weight"] == 0)
    selection["nll_gain_over_zero_shrinkage"] = original_nll - selection["normal_nll"]
    return dict(selection=selection,
        calibration=dict(summary=describe(calibration),
            old_tail=calibration_tail(calibration, "old", thresholds[0], source["config"]["quantile"]),
            new_tail=calibration_tail(calibration, "new", thresholds[1], source["config"]["quantile"]),
            windows=calibration),
        normal_groups={name: dict(summary=describe([row for row in normal if group_name(row) == name]),
            windows=[row for row in normal if group_name(row) == name]) for name in GROUPS},
        mode_groups={name: dict(summary=describe([row for row in modes if group_name(row) == name]),
            windows=[row for row in modes if group_name(row) == name]) for name in GROUPS})


def run(env, test_seed, root=ROOT, inputs=INPUT):
    output = root / env.replace("-", "_") / f"test_seed{test_seed}" / "readout.json"
    if output.exists():
        saved = json.loads(output.read_text())
        assert (saved["protocol"], saved["env"], saved["roles"], saved["complete"]) == (
            PROTOCOL, env, fold_roles(test_seed), True)
        return saved
    source = json.loads((inputs / env.replace("-", "_") / f"test_seed{test_seed}" / "readout.json").read_text())
    validate_run(source, env, test_seed, INPUT_PROTOCOL)
    assert source["control_scores_reproduced"]
    started = time.monotonic()
    report = dict(protocol=PROTOCOL, input_protocol=INPUT_PROTOCOL, env=env, roles=source["roles"],
        complete=False, development_only=True, retained_method="v25", evidence_integrated=False, **WORK)
    write(output, report)
    report.update(analyze_fold(source), complete=True, seconds=time.monotonic() - started)
    write(output, report)
    return report


def summarize(root=ROOT):
    report = dict(protocol=PROTOCOL, complete=True, development_only=True, retained_method="v25",
        evidence_integrated=False, decision="hold_conditional_evidence_integration", environments={}, **WORK)
    for env in ENVIRONMENTS:
        runs = [json.loads((root / env.replace("-", "_") / f"test_seed{seed}" / "readout.json").read_text()) for seed in SEEDS]
        for seed, row in zip(SEEDS, runs):
            assert (row["protocol"], row["env"], row["roles"], row["complete"]) == (
                PROTOCOL, env, fold_roles(seed), True)
            assert all(row[key] == 0 for key in WORK)
        report["environments"][env] = dict(runs=runs, **{kind: {name: describe(
            [window for run in runs for window in run[kind][name]["windows"]]) for name in GROUPS}
            for kind in ("normal_groups", "mode_groups")})
    return report


def compact_report(report):
    compact = copy.deepcopy(report)
    for environment in compact["environments"].values():
        for run in environment["runs"]:
            run["calibration"].pop("windows")
            for kind in ("normal_groups", "mode_groups"):
                for name, group in run[kind].items():
                    if name not in ("new_only_alarm", "resolved_alarm"):
                        group.pop("windows")
    return compact


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--inputs", type=Path, default=INPUT)
    args = parser.parse_args()
    for env in ENVIRONMENTS:
        for seed in SEEDS:
            run(env, seed, args.root, args.inputs)
    report = summarize(args.root)
    write(args.root / "readout.json", report)
    output = args.root / "compact_readout.json"
    write(output, compact_report(report))
    print(json.dumps(dict(environments={env: {kind: e[kind] for kind in ("normal_groups", "mode_groups")}
        for env, e in report["environments"].items()}, compact_readout_bytes=output.stat().st_size, **WORK)))
