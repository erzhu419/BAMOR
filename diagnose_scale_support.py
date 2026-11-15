"""Stored input geometry for mean-predictor and residual-scale support."""

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from diagnose_episode_blocked_calibration import write
from diagnose_headwise_scales import ENV, HEAD, ROOT, SEEDS, CASES, PROTOCOL as SCORE_PROTOCOL
from run_episode_blocked_change_probe import SOURCE, normal_data, source_folder


PROTOCOL = "stored_speed_scale_input_support_v1"


@torch.no_grad()
def nearest(inputs, reference, exclude_window=None):
    inputs, reference = torch.as_tensor(inputs, dtype=torch.float32), torch.as_tensor(reference, dtype=torch.float32)
    if exclude_window:
        assert len(inputs) == len(reference) and len(reference) > exclude_window
        reference_blocks = torch.arange(len(reference)) // exclude_window
    output = []
    for start in range(0, len(inputs), 128):
        chunk = inputs[start:start + 128]
        distances = torch.cdist(chunk, reference)
        if exclude_window:
            query_blocks = torch.arange(start, start + len(chunk)) // exclude_window
            distances.masked_fill_(query_blocks[:, None] == reference_blocks[None, :], float("inf"))
        output.append(distances.min(1).values)
    return torch.cat(output).numpy()


def geometry_windows(fit_distances, scale_distances, radii, original, decomposed):
    assert len(fit_distances) == len(scale_distances) == original[-1]["end_step"]
    result = []
    for window in original:
        start, end = window["start_step"] - 1, window["end_step"]
        fit, scale = fit_distances[start:end], scale_distances[start:end]
        supported = fit <= radii["predictor"]
        saved = window["heads"][HEAD]
        assert int(supported.sum()) == saved["supported_steps"], "Predictor support did not reproduce"
        row = dict(decomposed[end], predictor_supported_fraction=float(supported.mean()),
            scale_calibration_supported_fraction=float((scale <= radii["scale_calibration"]).mean()),
            scale_leave_window_supported_fraction=float((scale <= radii["scale_leave_window"]).mean()))
        for name, values, radius in (("predictor", fit, radii["predictor"]),
                                     ("scale_calibration", scale, radii["scale_calibration"]),
                                     ("scale_leave_window", scale, radii["scale_leave_window"])):
            row[f"{name}_distance_mean"] = float(values.mean())
            row[f"{name}_distance_ratio"] = float(values.mean() / radius) if radius > 0 else None
        for name in ("scale_calibration", "scale_leave_window"):
            row[f"{name}_supported_on_predictor_steps"] = (
                float((scale[supported] <= radii[name]).mean()) if supported.any() else None)
        result.append(row)
    return result


def summary(rows):
    result = dict(windows=len(rows), speed_eligible_windows=sum(row["eligible"] for row in rows))
    for name in ("scale_calibration", "scale_leave_window"):
        values = [row[f"{name}_supported_on_predictor_steps"] for row in rows
                  if row["eligible"] and row[f"{name}_supported_on_predictor_steps"] is not None]
        result[name] = dict(eligible_windows_with_80pct_support=sum(value >= .8 for value in values),
            mean_supported_fraction=float(np.mean(values)) if values else None,
            minimum_supported_fraction=float(min(values)) if values else None)
    return result


def analyze_fold(raw, scores, source=SOURCE):
    roles = raw["roles"]
    assert roles == scores["roles"] and raw["env"] == ENV and raw["complete"]
    assert raw["predictor_work"]["support_domains"][0] == "state_action"
    fit = np.concatenate([normal_data(source, ENV, seed, "expanded_normal")[0] for seed in roles["fit_seeds"]])
    scale = normal_data(source, ENV, roles["scale_seed"], "expanded_normal")[0]
    calibration = normal_data(source, ENV, roles["calibration_seed"], "expanded_normal")[0]
    mean, deviation = fit.mean(0), np.maximum(fit.std(0), raw["config"]["scale_floor"])
    fit, scale, calibration = ((inputs - mean) / deviation for inputs in (fit, scale, calibration))
    cal_fit, cal_scale = nearest(calibration, fit), nearest(calibration, scale)
    self_scale = nearest(scale, scale, exclude_window=raw["config"]["window"])
    quantile = raw["config"]["quantile"]
    radius = raw["arms"]["conditional_scale"]["calibration"]["support_radii"][HEAD]
    np.testing.assert_allclose(np.quantile(cal_fit, quantile), radius, rtol=1e-6, atol=1e-8)
    radii = dict(predictor=radius, scale_calibration=float(np.quantile(cal_scale, quantile)),
                 scale_leave_window=float(np.quantile(self_scale, quantile)))
    cal_map = {row["end_step"]: row for row in scores["calibration_windows"]}
    cal_windows = geometry_windows(cal_fit, cal_scale, radii,
        raw["arms"]["conditional_scale"]["calibration"]["windows"], cal_map)
    groups = {name: [] for name in scores["normal_groups"]}
    normal_map = {}
    for name, data in scores["normal_groups"].items():
        for row in data["windows"]:
            normal_map[(row["case"], row["end_step"])] = (name, row)
    cases = {}
    folder = source_folder(source, ENV, roles["test_seed"])
    with np.load(folder / "transitions.npz") as data:
        for case in CASES:
            inputs = (data[f"{case}_inputs"] - mean) / deviation
            fit_distances, scale_distances = nearest(inputs, fit), nearest(inputs, scale)
            original = raw["arms"]["conditional_scale"]["cases"][case]["windows"]
            mapping = {row["end_step"]: row for row in scores["cases"][case]["windows"]}
            for window in original[:5]:
                mapping[window["end_step"]] = normal_map[("shared_prefix", window["end_step"])][1]
            rows = geometry_windows(fit_distances, scale_distances, radii, original, mapping)
            post = [row for row in rows if row["start_step"] > 500]
            cases[case] = dict(summary=summary(post), windows=post,
                               outcomes=scores["cases"][case]["outcomes"])
            if case in CASES[:3]:
                for row in post:
                    name, _ = normal_map[(case, row["end_step"])]
                    groups[name].append(dict(row, case=case, phase="post"))
            if case == "stationary":
                for row in rows[:5]:
                    name, _ = normal_map[("shared_prefix", row["end_step"])]
                    groups[name].append(dict(row, case="shared_prefix", phase="pre"))
    return dict(roles=roles, radii=radii, calibration=dict(summary=summary(cal_windows), windows=cal_windows),
                normal_groups={name: dict(summary=summary(rows), windows=rows) for name, rows in groups.items()}, cases=cases)


def run(root=ROOT, source=SOURCE):
    previous = json.loads((root / "scale_diagnostic" / "readout.json").read_text())
    assert previous["protocol"] == SCORE_PROTOCOL and previous["complete"]
    rows = []
    for seed, score in zip(SEEDS, previous["folds"]):
        assert score["roles"]["test_seed"] == seed
        raw = json.loads((root / ENV.replace("-", "_") / f"test_seed{seed}" / "readout.json").read_text())
        rows.append(analyze_fold(raw, score, source))
        print(f"FOLD seed={seed} normal_windows="
              f"{sum(group['summary']['windows'] for group in rows[-1]['normal_groups'].values())}", flush=True)
    groups = {name: summary([window for row in rows for window in row["normal_groups"][name]["windows"]])
              for name in rows[0]["normal_groups"]}
    return dict(protocol=PROTOCOL, complete=True, env=ENV, folds=rows, normal_groups=groups,
        additional_interactions=0, initialization_probes=0, predictor_optimizer_steps=0,
        residual_scale_optimizer_steps=0, policy_optimizer_steps=0, critic_optimizer_steps=0,
        encoder_optimizer_steps=0, retained_method="v25", evidence_integrated=False,
        development_only=True, decision="hold_conditional_evidence_integration")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--source", type=Path, default=SOURCE)
    args = parser.parse_args()
    torch.set_num_threads(2)
    report = run(args.root, args.source)
    folder = args.root / "scale_support_diagnostic"
    write(folder / "readout.json", report)
    compact = json.loads(json.dumps(report))
    for row in compact["folds"]:
        row["calibration"].pop("windows")
        row["normal_groups"]["neither_alarm"].pop("windows")
    output = folder / "compact_readout.json"
    write(output, compact)
    print(json.dumps(dict(protocol=PROTOCOL, folds=len(report["folds"]), normal_groups=report["normal_groups"],
        additional_interactions=0, predictor_optimizer_steps=0, residual_scale_optimizer_steps=0,
        compact_json_bytes=output.stat().st_size)), flush=True)
