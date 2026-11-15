"""Read-only decomposition and input geometry for matched-scale false alarms."""

import argparse
from collections import Counter
import copy
import json
from pathlib import Path
import time

import numpy as np
import torch

from collect_matched_scale_normals import ROOT as FRESH, normal_data as fresh_data
from diagnose_scale_support import nearest
from run_cross_source_scale_probe import validate_run
from run_headwise_change_probe import (
    CHANGE_AT, ENVIRONMENTS, SEEDS, SOURCE, normal_data, source_folder, write)
from run_matched_residual_scale_probe import (
    ROOT as INPUT, PROTOCOL as INPUT_PROTOCOL, CONTROL, CONTROL_PROTOCOL)


PROTOCOL = "stored_matched_scale_transfer_v1"
ROOT = INPUT / "transfer_diagnostic"
GROUPS = ("new_only_alarm", "shared_alarm", "resolved_alarm", "neither_alarm")
ZERO_WORK = dict(additional_interactions=0, initialization_probes=0,
    predictor_optimizer_steps=0, residual_scale_optimizer_steps=0,
    policy_optimizer_steps=0, critic_optimizer_steps=0, encoder_optimizer_steps=0)


def pair_window(old, new, old_scales, new_scales, thresholds):
    windows = [row[arm] for row in (old, new) for arm in ("global_scale", "conditional_scale")]
    first = windows[0]
    for window in windows[1:]:
        for key in ("start_step", "end_step", "eligible", "supported_steps"):
            assert window[key] == first[key], f"Paired window changed {key}"
        assert window["heads"].keys() == first["heads"].keys()
    heads = {}
    for index, name in enumerate(first["heads"]):
        values = [window["heads"][name] for window in windows]
        for value in values[1:]:
            for key in ("eligible", "supported_steps", "domain"):
                assert value[key] == values[0][key], f"Paired head changed {key}"
        head = {key: values[0][key] for key in ("eligible", "supported_steps", "domain")}
        if head["eligible"]:
            raw_old, raw_new = values[0]["statistic"] * old_scales[index], values[2]["statistic"] * new_scales[index]
            np.testing.assert_allclose(raw_new, raw_old, rtol=1e-5, atol=1e-10)
            score_old, score_new = values[1]["statistic"], values[3]["statistic"]
            head.update(raw_loss_mean=float(raw_new), old_score=score_old, new_score=score_new,
                old_ratio=score_old / thresholds[0], new_ratio=score_new / thresholds[1],
                old_effective_scale=None, new_effective_scale=None, scale_factor=None,
                raw_error_factor=None, threshold_factor=thresholds[0] / thresholds[1])
            if raw_old > 0 and raw_new > 0:
                assert score_old > 0 and score_new > 0
                head.update(old_effective_scale=raw_old / score_old,
                    new_effective_scale=raw_new / score_new,
                    scale_factor=(raw_old / score_old) / (raw_new / score_new),
                    raw_error_factor=raw_new / raw_old)
                np.testing.assert_allclose(head["new_ratio"], head["old_ratio"] *
                    head["raw_error_factor"] * head["scale_factor"] * head["threshold_factor"], rtol=1e-10)
            else:
                assert raw_old == raw_new == score_old == score_new == 0
        heads[name] = head
    old_cond, new_cond = windows[1], windows[3]
    eligible = [name for name, head in heads.items() if head["eligible"]]
    dominant = max(eligible, key=lambda name: heads[name]["new_score"]) if eligible else None
    return dict(start_step=first["start_step"], end_step=first["end_step"],
        eligible=first["eligible"], old_joint_score=old_cond["statistic"],
        new_joint_score=new_cond["statistic"], old_threshold=thresholds[0], new_threshold=thresholds[1],
        old_alarm=bool(old_cond.get("alarm", False)), new_alarm=bool(new_cond.get("alarm", False)),
        dominant_head=dominant, heads=heads)


def pair_series(old, new, old_scales, new_scales, thresholds):
    count = len(old["global_scale"])
    assert all(len(rows) == count for arm in (old, new) for rows in arm.values())
    return [pair_window({arm: rows[i] for arm, rows in old.items()},
                        {arm: rows[i] for arm, rows in new.items()},
                        old_scales, new_scales, thresholds) for i in range(count)]


def group_name(row):
    return ("shared_alarm" if row["old_alarm"] else "new_only_alarm") if row["new_alarm"] else (
        "resolved_alarm" if row["old_alarm"] else "neither_alarm")


def describe(rows):
    selected = [row["heads"][row["dominant_head"]] for row in rows if row["dominant_head"]]
    result = dict(windows=len(rows), eligible_windows=len(selected),
        dominant_heads=dict(Counter(row["dominant_head"] for row in rows if row["dominant_head"])),
        sufficient_changes=dict(Counter()))
    changes = Counter()
    for row in rows:
        if row["new_alarm"] and not row["old_alarm"]:
            scale = row["new_joint_score"] > row["old_threshold"]
            threshold = row["old_joint_score"] > row["new_threshold"]
            changes["both_sufficient" if scale and threshold else "scale_sufficient" if scale else
                    "threshold_sufficient" if threshold else "joint_change_required"] += 1
    result["sufficient_changes"] = dict(changes)
    for field in ("scale_factor", "threshold_factor", "raw_error_factor"):
        values = [head[field] for head in selected if head.get(field) is not None]
        result[field] = dict(median=float(np.median(values)), maximum=float(max(values))) if values else None
    geometry = {}
    for name in ("old_scale", "new_scale", "old_scale_calibrated", "new_scale_calibrated", "calibration"):
        values = [head["geometry"][name]["supported_fraction"] for head in selected if "geometry" in head]
        geometry[name] = dict(windows_with_80pct_coverage=sum(value >= .8 for value in values),
            mean_supported_fraction=float(np.mean(values)) if values else None)
    result["geometry"] = geometry
    return result


def calibration_tail(windows, threshold, quantile):
    ordered = sorted((row for row in windows if row["eligible"]), key=lambda row: row["new_joint_score"])
    np.testing.assert_allclose(np.quantile([row["new_joint_score"] for row in ordered], quantile), threshold)
    position = (len(ordered) - 1) * quantile
    lower, upper = int(np.floor(position)), int(np.ceil(position))
    operands = []
    for index, weight in ((lower, 1. - position + lower), (upper, position - lower)):
        if weight:
            row = ordered[index]
            operands.append(dict(weight=float(weight), end_step=row["end_step"],
                joint_score=row["new_joint_score"], dominant_head=row["dominant_head"],
                decomposition=row["heads"][row["dominant_head"]]))
    return dict(threshold=threshold, operands=operands)


def build_inputs(row, source=SOURCE, fresh=FRESH):
    env, roles = row["env"], row["roles"]
    fit = np.concatenate([normal_data(source, env, seed, "expanded_normal")[0] for seed in roles["fit_seeds"]])
    scale = normal_data(source, env, roles["scale_seed"], "expanded_normal")[0]
    cal, targets, meta = normal_data(source, env, roles["calibration_seed"], "expanded_normal")
    fresh_rows = [fresh_data(fresh, env, seed)[0] for seed in roles["fit_seeds"]]
    new_scale = np.concatenate([*fresh_rows, scale])
    assert (len(fit), len(scale), len(cal), len(new_scale)) == (
        row["fit_steps"], row["scale_steps"], row["calibration_steps"], row["scale_fit_steps"])
    mean, deviation = fit.mean(0), np.maximum(fit.std(0), row["config"]["scale_floor"])
    references = {name: (inputs - mean) / deviation for name, inputs in
        (("predictor", fit), ("old_scale", scale), ("new_scale", new_scale), ("calibration", cal))}
    with np.load(source_folder(source, env, roles["test_seed"]) / "transitions.npz") as data:
        tests = {case: (data[f"{case}_inputs"] - mean) / deviation
                 for case in ("stationary", "sampler_change", "actor_change")}
    return references, tests, targets.shape[1] - meta["reward_dim"]


def attach_geometry(calibration, normal, references, queries, domains, obs_dim, radii, config):
    for domain in dict.fromkeys(domains.values()):
        features = {name: inputs[:, obs_dim:] if domain == "action" else inputs
                    for name, inputs in references.items()}
        test = queries[:, obs_dim:] if domain == "action" else queries
        cal_distance = {name: nearest(features["calibration"], inputs,
            exclude_window=config["window"] if name == "calibration" else None)
            for name, inputs in features.items()}
        distance = {name: nearest(test, inputs) for name, inputs in features.items()}
        head_names = [name for name, value in domains.items() if value == domain]
        radius = radii[head_names[0]]
        assert all(radii[name] == radius for name in head_names)
        np.testing.assert_allclose(np.quantile(cal_distance["predictor"], config["quantile"]), radius, rtol=1e-6, atol=1e-8)
        limits = {name: float(np.quantile(nearest(features[name], features[name],
            exclude_window=config["window"]), config["quantile"])) for name in ("old_scale", "new_scale")}
        limits["calibration"] = float(np.quantile(cal_distance["calibration"], config["quantile"]))
        limits.update({f"{name}_calibrated": float(np.quantile(cal_distance[name], config["quantile"]))
                       for name in ("old_scale", "new_scale")})
        for rows, distances in ((calibration, cal_distance), (normal, distance)):
            for index, row in enumerate(rows):
                start, end = index * config["window"], (index + 1) * config["window"]
                mask = distances["predictor"][start:end] <= radius
                for name in head_names:
                    head = row["heads"][name]
                    assert int(mask.sum()) == head["supported_steps"], "Predictor geometry must reproduce"
                    head["geometry"] = {}
                    if not mask.any():
                        continue
                    for key, limit in limits.items():
                        values = distances[key.removesuffix("_calibrated")][start:end][mask]
                        head["geometry"][key] = dict(supported_fraction=float((values <= limit).mean()),
                            mean_distance_ratio=float(values.mean() / limit) if limit > 0 else None)
        for name in head_names:
            references_for_head = {**limits, "predictor": radius}
            domains[name] = dict(domain=domain, radii=references_for_head)


def analyze_fold(row, old, source=SOURCE, fresh=FRESH):
    assert row["roles"] == old["roles"] and row["predictor_work"] == old["predictor_work"]
    old_scales, new_scales = old["scale_work"]["global_scales"], row["scale_work"]["global_scales"]
    thresholds = tuple(run["arms"]["conditional_scale"]["calibration"]["error_threshold"] for run in (old, row))
    series = lambda run: {arm: data["calibration"]["windows"] for arm, data in run["arms"].items()}
    calibration = pair_series(series(old), series(row), old_scales, new_scales, thresholds)
    normal = []
    for case in ("stationary", "sampler_change", "actor_change"):
        series = lambda run: {arm: data["cases"][case]["windows"] for arm, data in run["arms"].items()}
        paired = pair_series(series(old), series(row), old_scales, new_scales, thresholds)
        normal.extend(dict(window, case="shared_prefix" if window["end_step"] <= CHANGE_AT else case)
                      for window in paired if case == "stationary" or window["start_step"] > CHANGE_AT)
    references, tests, obs_dim = build_inputs(row, source, fresh)
    queries = np.concatenate([tests["stationary"], tests["sampler_change"][CHANGE_AT:], tests["actor_change"][CHANGE_AT:]])
    assert len(queries) == len(normal) * row["config"]["window"]
    domains = dict(zip(normal[0]["heads"], row["predictor_work"]["support_domains"]))
    assert row["arms"]["conditional_scale"]["calibration"]["support_radii"] == old["arms"]["conditional_scale"]["calibration"]["support_radii"]
    attach_geometry(calibration, normal, references, queries, domains, obs_dim,
        row["arms"]["conditional_scale"]["calibration"]["support_radii"], row["config"])
    return dict(roles=row["roles"], geometry_references=domains,
        calibration=dict(summary=describe(calibration), tail=calibration_tail(calibration, thresholds[1], row["config"]["quantile"]),
                         windows=calibration),
        normal_groups={name: dict(summary=describe([w for w in normal if group_name(w) == name]),
            windows=[w for w in normal if group_name(w) == name]) for name in GROUPS})


def run(env, test_seed, root=ROOT, inputs=INPUT, control=CONTROL, source=SOURCE, fresh=FRESH):
    output = root / env.replace("-", "_") / f"test_seed{test_seed}" / "readout.json"
    if output.exists():
        saved = json.loads(output.read_text())
        assert (saved["protocol"], saved["env"], saved["roles"]["test_seed"], saved["complete"]) == (PROTOCOL, env, test_seed, True)
        return saved
    folder = Path(env.replace("-", "_")) / f"test_seed{test_seed}" / "readout.json"
    row, old = (json.loads((base / folder).read_text()) for base in (inputs, control))
    validate_run(row, env, test_seed, INPUT_PROTOCOL)
    validate_run(old, env, test_seed, CONTROL_PROTOCOL)
    started = time.monotonic()
    report = dict(protocol=PROTOCOL, env=env, roles=row["roles"], complete=False,
        development_only=True, retained_method="v25", evidence_integrated=False, **ZERO_WORK)
    write(output, report)
    report.update(analyze_fold(row, old, source, fresh), complete=True, seconds=time.monotonic() - started)
    write(output, report)
    print(json.dumps(dict(env=env, test_seed=test_seed,
        groups={name: data["summary"] for name, data in report["normal_groups"].items()}, **ZERO_WORK)), flush=True)
    return report


def summarize(root=ROOT):
    report = dict(protocol=PROTOCOL, complete=True, development_only=True, retained_method="v25",
        evidence_integrated=False, decision="hold_conditional_evidence_integration", environments={}, **ZERO_WORK)
    for env in ENVIRONMENTS:
        runs = [json.loads((root / env.replace("-", "_") / f"test_seed{seed}" / "readout.json").read_text()) for seed in SEEDS]
        for seed, run in zip(SEEDS, runs):
            assert (run["protocol"], run["env"], run["roles"]["test_seed"], run["complete"]) == (PROTOCOL, env, seed, True)
            assert all(run[key] == 0 for key in ZERO_WORK)
        report["environments"][env] = dict(runs=runs, normal_groups={name: describe(
            [window for run in runs for window in run["normal_groups"][name]["windows"]]) for name in GROUPS})
    return report


def compact_report(report):
    compact = copy.deepcopy(report)
    for environment in compact["environments"].values():
        for run in environment["runs"]:
            run["calibration"].pop("windows")
            for name, group in run["normal_groups"].items():
                if name not in ("new_only_alarm", "resolved_alarm"):
                    group.pop("windows")
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
        print(json.dumps(dict(environments={env: e["normal_groups"] for env, e in report["environments"].items()},
                              compact_readout_bytes=output.stat().st_size)))
    else:
        run(args.env, args.test_seed, args.root)
