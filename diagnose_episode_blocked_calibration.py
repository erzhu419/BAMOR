"""Attribute stored calibration tails and test failures without predictor work."""

import argparse
import json
from pathlib import Path

import numpy as np


ROOT = Path("results/iclr_episode_blocked_change_20260930")
SOURCE = Path("results/iclr_conditional_change_20260930")
PROTOCOL = "stored_blocked_calibration_tail_v1"
INPUT_PROTOCOL = "source_blocked_change_development_v1"
ENVIRONMENTS = ("mo-halfcheetah-v5", "mo-hopper-v5")
SEEDS = tuple(range(2101, 2106))
ARMS = ("fixed_only", "expanded_normal")
CASES = ("stationary", "sampler_change", "actor_change", "mode1", "mode2", "mode3")
SEGMENTS = (("prefix", 0, 3000, 0), ("stationary", 3000, 4000, 0),
            ("sampler_change", 4000, 4500, 500), ("actor_change", 4500, 5000, 500))
CHANGE_AT, MAXIMUM_DELAY, MINIMUM_WINDOWS = 500, 300, 3


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def terminal_context(source, env, seed):
    folder = source / env.replace("-", "_") / f"seed{seed}"
    original = json.loads((folder / "readout.json").read_text())
    result = {}
    with np.load(folder / "transitions.npz") as data:
        for segment, _, _, _ in SEGMENTS:
            terminal = data[f"{segment}_terminals"].any(1)
            indices = np.arange(len(terminal))
            starts = np.r_[0, (np.flatnonzero(terminal) + 1)]
            episode = np.searchsorted(starts, indices, side="right") - 1
            age = indices - starts[episode]
            family = "prefix" if segment == "prefix" else "branch"
            reset_seed = (20000000 if family == "prefix" else 50000000) + 1000 * original["probe_seed"]
            result[segment] = dict(terminal=terminal, episode=episode, age=age,
                                   reset_seed=reset_seed, family=family)
    return result


def provenance(window, context):
    for segment, offset, end, suffix_start in SEGMENTS:
        if offset < window["start_step"] <= window["end_step"] <= end:
            start = window["start_step"] - offset + suffix_start
            stop = window["end_step"] - offset + suffix_start
            info = context[segment]
            episode = info["episode"][start - 1:stop]
            age = info["age"][start - 1:stop]
            return dict(segment=segment, calibration_end_step=window["end_step"],
                source_start_step=start, source_end_step=stop, statistic=window["statistic"],
                supported_steps=window["supported_steps"], reset_family=info["family"],
                episode_ids=np.unique(episode).tolist(),
                reset_seeds=(info["reset_seed"] + np.unique(episode)).tolist(),
                all_window_steps_early_fraction=float((age < 100).mean()),
                all_window_steps_reset_events=int(info["terminal"][start - 1:stop].sum()))
    raise ValueError("Calibration window crosses frozen source segments")


def calibration_tail(arm, quantile, context):
    calibration = arm["calibration"]
    windows = calibration["windows"]
    assert windows[-1]["end_step"] == arm["calibration_steps"]
    eligible = [window for window in windows if window["eligible"]]
    values = np.asarray([window["statistic"] for window in eligible])
    threshold = calibration["error_threshold"]
    np.testing.assert_allclose(np.quantile(values, quantile), threshold, rtol=1e-6, atol=1e-8)
    ordered = np.sort(values)
    position = (len(values) - 1) * quantile
    lower, upper = int(np.floor(position)), int(np.ceil(position))
    weight = float(position - lower)
    operands = []
    for index, interpolation_weight in ((lower, 1. - weight), (upper, weight)):
        if interpolation_weight:
            value = float(ordered[index])
            operands.append(dict(value=value, weight=interpolation_weight,
                windows=[provenance(window, context) for window in eligible if window["statistic"] == value]))
    segments = {}
    for segment, offset, end, _ in SEGMENTS:
        selected = [window for window in windows if offset < window["start_step"] <= window["end_step"] <= end]
        supported = [window["statistic"] for window in selected if window["eligible"]]
        if selected:
            segments[segment] = dict(windows=len(selected), eligible=len(supported),
                maximum=float(max(supported)) if supported else None,
                mean=float(np.mean(supported)) if supported else None)
    return dict(threshold=threshold, quantile=quantile, eligible_windows=len(eligible), segments=segments,
                quantile_operands=operands)


def test_phase(windows, threshold):
    eligible = [window for window in windows if window["eligible"]]
    alarms = [window for window in windows if window["alarm"]]
    early = [window for window in eligible if window["end_step"] <= CHANGE_AT + MAXIMUM_DELAY]
    delay = alarms[0]["end_step"] - CHANGE_AT if alarms else None
    available = len(eligible) >= MINIMUM_WINDOWS
    outcome = ("coverage_unavailable" if not available else "no_alarm" if not alarms
               else "timely_alarm" if delay <= MAXIMUM_DELAY else "late_alarm")
    return dict(outcome=outcome, available=available, eligible_windows=len(eligible),
        early_eligible_windows=len(early), alarm_windows=len(alarms), first_alarm_delay=delay,
        maximum_ratio=max(window["statistic"] / threshold for window in eligible) if eligible else None,
        early_maximum_ratio=max(window["statistic"] / threshold for window in early) if early else None,
        windows=[dict(end_step=window["end_step"], supported_steps=window["supported_steps"],
            eligible=window["eligible"], alarm=window["alarm"], statistic=window["statistic"],
            ratio=window["statistic"] / threshold if window["eligible"] else None) for window in windows])


def audit_run(row, context):
    assert row["protocol"] == INPUT_PROTOCOL and row["complete"]
    result = dict(env=row["env"], roles=row["roles"], arms={})
    for name in ARMS:
        arm = row["arms"][name]
        tail = calibration_tail(arm, row["predictor_config"]["quantile"], context)
        threshold = tail["threshold"]
        cases = {}
        for case in CASES:
            old = arm["cases"][case]
            post = [window for window in old["windows"] if window["start_step"] > CHANGE_AT]
            phase = test_phase(post, threshold)
            assert phase["available"] == old["available"]
            assert bool(phase["alarm_windows"]) == old["any_post_alarm"]
            assert phase["first_alarm_delay"] == old["first_post_alarm_delay"]
            assert (phase["outcome"] == "timely_alarm") == old["timely_alarm"]
            cases[case] = phase
        prefix = [window for window in arm["cases"]["stationary"]["windows"] if window["end_step"] <= CHANGE_AT]
        supported = [window for window in prefix if window["eligible"]]
        result["arms"][name] = dict(calibration=tail, cases=cases,
            shared_prefix=dict(eligible_windows=len(supported), alarm_windows=sum(window["alarm"] for window in prefix),
                maximum_ratio=max(window["statistic"] / threshold for window in supported) if supported else None))
    return result


def threshold_obstruction(rows, arm, positive="mode1", negative="stationary"):
    existing_negative, timely, unpenalized, dominated = [], [], [], []
    quantiles = [row["arms"][arm]["calibration"]["quantile"] for row in rows]
    assert len(set(quantiles)) == 1
    for row in rows:
        seed = row["roles"]["test_seed"]
        cases = row["arms"][arm]["cases"]
        null, changed = cases[negative], cases[positive]
        if null["alarm_windows"]:
            existing_negative.append(seed)
        if changed["outcome"] == "timely_alarm":
            timely.append(seed)
        elif changed["available"] and changed["early_maximum_ratio"] is not None:
            null_peak, changed_peak = null["maximum_ratio"], changed["early_maximum_ratio"]
            if null_peak is not None and null_peak >= changed_peak:
                dominated.append(dict(seed=seed, negative_maximum_ratio=null_peak,
                    positive_early_maximum_ratio=changed_peak, already_negative_alarm=bool(null["alarm_windows"])))
                if null["alarm_windows"]:
                    unpenalized.append(seed)
            else:
                unpenalized.append(seed)
    # Lowering the score quantile preserves existing alarms; each new dominated
    # source rescued by the300-step deadline adds a distinct negative alarm source.
    remaining_negative_slots = max(0, 1 - len(existing_negative))
    lower_q_upper_bound = min(len(rows), len(timely) + len(unpenalized) + remaining_negative_slots)
    lower_q_blocked = len(existing_negative) > 1 or lower_q_upper_bound < 4
    higher_q_blocked = len(timely) < 4
    return dict(positive=positive, negative=negative, baseline_quantile=quantiles[0],
        fixed_scores_and_support=True, minimum_timely_sources=4, maximum_negative_alarm_sources=1,
        existing_negative_sources=existing_negative, existing_timely_sources=timely,
        dominated_missed_sources=dominated, unpenalized_rescuable_sources=unpenalized,
        lower_quantile_timely_upper_bound=lower_q_upper_bound,
        lower_quantile_blocked=lower_q_blocked, higher_or_equal_quantile_blocked=higher_q_blocked,
        no_shared_score_quantile_can_pass=bool(lower_q_blocked and higher_q_blocked))


def run(root=ROOT, source=SOURCE):
    report = dict(protocol=PROTOCOL, complete=True, additional_interactions=0,
        initialization_probes=0, predictor_optimizer_steps=0, policy_optimizer_steps=0,
        critic_optimizer_steps=0, encoder_optimizer_steps=0, retained_method="v25",
        decision="hold_conditional_evidence_integration", development_only=True, environments={})
    for env in ENVIRONMENTS:
        rows = []
        for seed in SEEDS:
            original = json.loads((root / env.replace("-", "_") / f"test_seed{seed}" / "readout.json").read_text())
            assert original["env"] == env and original["roles"]["test_seed"] == seed
            context = terminal_context(source, env, original["roles"]["calibration_seed"])
            rows.append(audit_run(original, context))
        report["environments"][env] = dict(runs=rows,
            score_quantile_obstructions={arm: threshold_obstruction(rows, arm) for arm in ARMS})
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--source", type=Path, default=SOURCE)
    args = parser.parse_args()
    report = run(args.root, args.source)
    folder = args.root / "calibration_tail_diagnostic"
    write(folder / "readout.json", report)
    compact = json.loads(json.dumps(report))
    for environment in compact["environments"].values():
        for row in environment["runs"]:
            for arm in row["arms"].values():
                for case in arm["cases"].values():
                    case.pop("windows")
    output = folder / "compact_readout.json"
    write(output, compact)
    print(json.dumps(dict(protocol=PROTOCOL,
        completed_folds=sum(len(data["runs"]) for data in report["environments"].values()),
        predictor_optimizer_steps=0, additional_interactions=0, compact_json_bytes=output.stat().st_size)))
