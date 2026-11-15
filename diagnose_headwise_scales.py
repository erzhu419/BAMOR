"""Decompose paired stored speed-head scores; no trajectories or model work."""

import argparse
import json
from pathlib import Path

import numpy as np

from diagnose_episode_blocked_calibration import write


ROOT = Path("results/iclr_headwise_change_20260930")
PROTOCOL = "stored_headwise_speed_scale_v1"
INPUT_PROTOCOL = "four_role_headwise_change_development_v1"
ENV = "mo-halfcheetah-v5"
SEEDS = tuple(range(2101, 2106))
CASES = ("stationary", "sampler_change", "actor_change", "mode1", "mode2", "mode3")
ARMS = ("global_scale", "conditional_scale")
HEAD = "reward_0"


def pair_window(global_window, conditional_window, mean_scale, thresholds):
    for key in ("start_step", "end_step", "eligible", "supported_steps"):
        assert global_window[key] == conditional_window[key], f"Different paired {key}"
    for head, left in global_window["heads"].items():
        right = conditional_window["heads"][head]
        for key in ("eligible", "supported_steps", "domain"):
            assert left[key] == right[key], f"Different paired head {key}"
    left, right = global_window["heads"][HEAD], conditional_window["heads"][HEAD]
    result = dict(start_step=global_window["start_step"], end_step=global_window["end_step"],
        eligible=left["eligible"], supported_steps=left["supported_steps"],
        global_joint_alarm=bool(global_window.get("alarm", False)),
        conditional_joint_alarm=bool(conditional_window.get("alarm", False)),
        conditional_alarm_heads=[name for name, row in conditional_window["heads"].items() if row.get("alarm", False)],
        global_score=left["statistic"], conditional_score=right["statistic"],
        raw_loss_mean=None, effective_conditional_scale=None, scale_amplification=None,
        threshold_factor=thresholds[0] / thresholds[1], global_ratio=None, conditional_ratio=None)
    if not left["eligible"]:
        assert left["statistic"] is None and right["statistic"] is None
        return result
    global_score, conditional_score = left["statistic"], right["statistic"]
    assert global_score >= 0 and conditional_score >= 0 and mean_scale > 0
    result.update(raw_loss_mean=mean_scale * global_score,
                  global_ratio=global_score / thresholds[0], conditional_ratio=conditional_score / thresholds[1])
    if global_score == 0:
        assert conditional_score == 0
        return result
    assert conditional_score > 0
    amplification = conditional_score / global_score
    result.update(effective_conditional_scale=mean_scale / amplification, scale_amplification=amplification)
    np.testing.assert_allclose(result["conditional_ratio"],
        result["global_ratio"] * amplification * result["threshold_factor"], rtol=1e-10)
    return result


def pair_windows(left, right, mean_scale, thresholds):
    assert len(left) == len(right)
    return [pair_window(a, b, mean_scale, thresholds) for a, b in zip(left, right)]


def distribution(windows):
    rows = [row for row in windows if row["eligible"]]
    result = dict(total_windows=len(windows), eligible_windows=len(rows))
    for field in ("raw_loss_mean", "effective_conditional_scale", "scale_amplification", "global_ratio", "conditional_ratio"):
        values = [row[field] for row in rows if row[field] is not None]
        result[field] = dict(count=len(values), median=float(np.median(values)),
                             maximum=float(max(values))) if values else None
    return result


def tail(arm, paired, quantile):
    original = arm["calibration"]["windows"]
    eligible = [row for row in original if row["eligible"]]
    values = np.sort([row["statistic"] for row in eligible])
    threshold = arm["calibration"]["error_threshold"]
    np.testing.assert_allclose(np.quantile(values, quantile), threshold, rtol=1e-6)
    position = (len(values) - 1) * quantile
    lower, upper = int(np.floor(position)), int(np.ceil(position))
    operands = []
    for index, weight in ((lower, 1. - position + lower), (upper, position - lower)):
        if weight:
            matches = []
            for window, decomposition in zip(original, paired):
                if window["eligible"] and window["statistic"] == values[index]:
                    winners = [name for name, row in window["heads"].items()
                               if row["eligible"] and row["statistic"] == window["statistic"]]
                    matches.append(dict(decomposition, joint_winners=winners))
            operands.append(dict(weight=float(weight), joint_score=float(values[index]), windows=matches))
    return dict(threshold=threshold, quantile_operands=operands)


def fold(row):
    assert (row["protocol"], row["env"], row["complete"]) == (INPUT_PROTOCOL, ENV, True)
    global_arm, conditional_arm = (row["arms"][name] for name in ARMS)
    assert global_arm["calibration"]["support_radii"] == conditional_arm["calibration"]["support_radii"]
    mean_scale = row["scale_work"]["global_scales"][0]
    thresholds = tuple(row["calibration"]["error_threshold"] for row in (global_arm, conditional_arm))
    assert all(value > 0 for value in thresholds)
    cal = pair_windows(global_arm["calibration"]["windows"], conditional_arm["calibration"]["windows"], mean_scale, thresholds)
    cases = {}
    normal = []
    for case in CASES:
        paired = pair_windows(global_arm["cases"][case]["windows"], conditional_arm["cases"][case]["windows"], mean_scale, thresholds)
        post = [window for window in paired if window["start_step"] > 500]
        cases[case] = dict(distribution=distribution(post), windows=post,
            outcomes={name: {key: row["arms"][name]["cases"][case][key] for key in
                ("available", "any_post_alarm", "first_post_alarm_delay", "timely_alarm")} for name in ARMS})
        if case in CASES[:3]:
            normal.extend(dict(window, case=case, phase="post") for window in post)
        if case == "stationary":
            prefix = [window for window in paired if window["end_step"] <= 500]
            normal.extend(dict(window, case="shared_prefix", phase="pre") for window in prefix)
    groups = {}
    for name, predicate in (
        ("shared_alarm", lambda window: window["global_joint_alarm"] and window["conditional_joint_alarm"]),
        ("conditional_only_alarm", lambda window: not window["global_joint_alarm"] and window["conditional_joint_alarm"]),
        ("global_only_alarm", lambda window: window["global_joint_alarm"] and not window["conditional_joint_alarm"]),
        ("neither_alarm", lambda window: not window["global_joint_alarm"] and not window["conditional_joint_alarm"])):
        windows = [window for window in normal if predicate(window)]
        groups[name] = dict(distribution=distribution(windows), windows=windows)
    return dict(roles=row["roles"], global_scale=mean_scale, thresholds=dict(zip(ARMS, thresholds)),
        calibration_distribution=distribution(cal), calibration_windows=cal,
        calibration_tails={name: tail(row["arms"][name], cal, row["config"]["quantile"]) for name in ARMS},
        normal_groups=groups, cases=cases)


def run(root=ROOT):
    rows = []
    for seed in SEEDS:
        original = json.loads((root / ENV.replace("-", "_") / f"test_seed{seed}" / "readout.json").read_text())
        assert original["roles"]["test_seed"] == seed
        rows.append(fold(original))
    group_counts = {name: sum(len(row["normal_groups"][name]["windows"]) for row in rows)
                    for name in rows[0]["normal_groups"]}
    alarms = [dict(window, seed=row["roles"]["test_seed"])
        for row in rows for name in ("shared_alarm", "conditional_only_alarm")
        for window in row["normal_groups"][name]["windows"]]
    return dict(protocol=PROTOCOL, complete=True, env=ENV, folds=rows,
        normal_group_counts=group_counts, conditional_normal_alarm_windows=len(alarms),
        conditional_alarm_heads=sorted({head for row in alarms for head in row["conditional_alarm_heads"]}),
        additional_interactions=0, initialization_probes=0, predictor_optimizer_steps=0,
        residual_scale_optimizer_steps=0, policy_optimizer_steps=0, critic_optimizer_steps=0,
        encoder_optimizer_steps=0, retained_method="v25", evidence_integrated=False,
        development_only=True, decision="hold_conditional_evidence_integration")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT)
    args = parser.parse_args()
    report = run(args.root)
    output = args.root / "scale_diagnostic"
    write(output / "readout.json", report)
    compact = json.loads(json.dumps(report))
    for row in compact["folds"]:
        row.pop("calibration_windows")
        row["normal_groups"]["neither_alarm"].pop("windows")
        for case in row["cases"].values():
            case.pop("windows")
    path = output / "compact_readout.json"
    write(path, compact)
    print(json.dumps(dict(protocol=PROTOCOL, folds=len(report["folds"]),
        normal_group_counts=report["normal_group_counts"],
        conditional_alarm_heads=report["conditional_alarm_heads"], compact_json_bytes=path.stat().st_size,
        predictor_optimizer_steps=0, residual_scale_optimizer_steps=0, additional_interactions=0)))
