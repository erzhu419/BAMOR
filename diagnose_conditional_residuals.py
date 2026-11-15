"""Reconstruct the frozen probe predictor and diagnose stored residuals only."""

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from conditional_change_evidence import ConditionalChangeEvidence, CONFIG


SOURCE = Path("results/iclr_conditional_change_20260930")
ROOT = SOURCE / "residual_diagnostic"
PROTOCOL = "stored_conditional_residual_v1"
ENVS = ("mo-halfcheetah-v5", "mo-hopper-v5")
SEEDS = tuple(range(2101, 2106))
CASES = ("stationary", "sampler_change", "actor_change", "mode1", "mode2", "mode3")
FIT_STEPS, CHANGE_AT, NEIGHBORS = 2000, 500, 10


def write(path, report):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")


def episode_age(terminals):
    ages, episodes, age, episode = [], [], 0, 0
    for terminal in terminals:
        ages.append(age)
        episodes.append(episode)
        age += 1
        if np.any(terminal):
            age, episode = 0, episode + 1
    return np.asarray(ages), np.asarray(episodes)


def losses(evidence, inputs, targets):
    residual = (evidence.predict(inputs) - targets) / evidence.y_scale
    reward = np.mean(residual[:, :evidence.reward_dim] ** 2, axis=1)
    dynamics = np.mean(residual[:, evidence.reward_dim:] ** 2, axis=1)
    return np.column_stack([np.maximum(reward, dynamics), reward, dynamics])


def neighbors(queries, reference, reference_losses, k=NEIGHBORS,
              query_blocks=None, reference_blocks=None):
    estimates, distances = [], []
    reference = torch.as_tensor(reference)
    for start in range(0, len(queries), 128):
        distance = torch.cdist(torch.as_tensor(queries[start:start + 128]), reference)
        if query_blocks is not None:
            same_block = query_blocks[start:start + 128, None] == reference_blocks[None, :]
            distance[torch.as_tensor(same_block)] = torch.inf
        nearest = distance.topk(k, largest=False)
        estimates.append(reference_losses[nearest.indices.numpy()].mean(axis=1))
        distances.append(nearest.values[:, -1].numpy())
    return np.concatenate(estimates), np.concatenate(distances)


def gaps(observed, estimated, baseline, baseline_estimated):
    observed_gap = observed.mean(0) - baseline.mean(0)
    mix = estimated.mean(0) - baseline_estimated.mean(0)
    remaining = (observed - estimated).mean(0) - (baseline - baseline_estimated).mean(0)
    return dict(observed_gap=observed_gap.astype(float).tolist(),
                input_mix_gap=mix.astype(float).tolist(),
                remaining_gap=remaining.astype(float).tolist())


def describe_phase(rows):
    matched = [row for row in rows if row["original_eligible"] and row["matched_eligible"]]
    alarms = [row for row in rows if row["original_alarm"]]
    result = dict(windows=len(rows), original_eligible=sum(row["original_eligible"] for row in rows),
        original_alarms=len(alarms), matched_eligible=len(matched),
        matched_alarm_windows=sum(row["matched_eligible"] for row in alarms),
        fit_supported_fraction=sum(row["fit_supported_steps"] for row in rows) / (100 * len(rows)),
        matched_fraction=sum(row["matched_steps"] for row in rows) / (100 * len(rows)),
        early_episode_fraction=float(np.mean([row["early_episode_fraction"] for row in rows])),
        loss_means=(np.mean([row["loss_means"] for row in rows if row["loss_means"] is not None], axis=0).tolist()
                    if any(row["loss_means"] is not None for row in rows) else None))
    for key in ("observed_gap", "input_mix_gap", "remaining_gap"):
        result[key] = np.mean([row["gaps"][key] for row in matched], axis=0).tolist() if matched else None
    result["alarm_details"] = [{key: row[key] for key in
        ("end_step", "matched_steps", "loss_means", "gaps", "early_episode_fraction")}
        for row in alarms]
    return result


def run(env, seed, source=SOURCE, root=ROOT):
    output = root / env.replace("-", "_") / f"seed{seed}" / "readout.json"
    if output.exists():
        saved = json.loads(output.read_text())
        assert (saved["protocol"], saved["env"], saved["seed"], saved["complete"]) == (PROTOCOL, env, seed, True)
        return saved
    folder = source / env.replace("-", "_") / f"seed{seed}"
    original = json.loads((folder / "readout.json").read_text())
    assert (original["protocol"], original["env"], original["seed"], original["complete"]) == (
        "conditional_change_evidence_v1", env, seed, True)
    assert original["predictor_config"] == CONFIG and original["probe_seed"] == seed + 100
    report = dict(protocol=PROTOCOL, env=env, seed=seed, complete=False,
        additional_interactions=0, initialization_probes=0, policy_optimizer_steps=0,
        critic_optimizer_steps=0, encoder_optimizer_steps=0, predictor_reconstruction_steps=0)
    write(output, report)
    with np.load(folder / "transitions.npz") as data:
        reward_dim = data["prefix_weights"].shape[1]
        evidence = ConditionalChangeEvidence(data["prefix_inputs"].shape[1],
                                              data["prefix_targets"].shape[1], reward_dim)
        evidence.fit(data["prefix_inputs"][:FIT_STEPS], data["prefix_targets"][:FIT_STEPS],
                     40000000 + original["probe_seed"])
        report["predictor_reconstruction_steps"] = evidence.optimizer_steps
        write(output, report)
        cal_x, cal_y = data["prefix_inputs"][FIT_STEPS:], data["prefix_targets"][FIT_STEPS:]
        calibration = evidence.calibrate(cal_x, cal_y)
        for key in ("support_radius", "error_threshold"):
            np.testing.assert_allclose(calibration[key], original["calibration"][key], rtol=1e-5, atol=1e-7)
        for reproduced, stored in zip(calibration["windows"], original["calibration"]["windows"]):
            assert reproduced["eligible"] == stored["eligible"]
            assert reproduced["supported_steps"] == stored["supported_steps"]
            if stored["eligible"]:
                np.testing.assert_allclose(reproduced["statistic"], stored["statistic"], rtol=1e-5, atol=1e-7)
        cal_distance = evidence.distances(cal_x)
        cal_support = cal_distance <= evidence.support_radius
        cal_losses = losses(evidence, cal_x, cal_y).astype(np.float64)
        normalized = (cal_x - evidence.x_mean) / evidence.x_scale
        blocks = np.arange(len(cal_x)) // CONFIG["window"]
        cal_estimate, cal_neighbor_distance = neighbors(normalized, normalized[cal_support],
            cal_losses[cal_support], query_blocks=blocks, reference_blocks=blocks[cal_support])
        radius = float(np.quantile(cal_neighbor_distance[cal_support], CONFIG["quantile"]))
        cal_match = cal_support & (cal_neighbor_distance <= radius)
        ages, episodes = episode_age(data["prefix_terminals"])
        report["calibration"] = dict(original_threshold=evidence.threshold, original_radius=evidence.support_radius,
            matching_radius=radius, reference_steps=int(cal_support.sum()), matched_steps=int(cal_match.sum()),
            episode_segments=int(np.unique(episodes[FIT_STEPS:]).size),
            early_episode_fraction=float((ages[FIT_STEPS:] < 100).mean()),
            loss_means=cal_losses[cal_match].mean(0).tolist(),
            local_reference_means=cal_estimate[cal_match].mean(0).tolist())
        report["phases"], maximum_error_delta = {}, 0.
        for case in CASES:
            inputs, targets = data[f"{case}_inputs"], data[f"{case}_targets"]
            observed = losses(evidence, inputs, targets).astype(np.float64)
            np.testing.assert_allclose(observed[:, 0], data[f"{case}_errors"], rtol=1e-5, atol=1e-7)
            maximum_error_delta = max(maximum_error_delta, float(np.abs(observed[:, 0] - data[f"{case}_errors"]).max()))
            distance = evidence.distances(inputs)
            np.testing.assert_allclose(distance, data[f"{case}_distances"], rtol=1e-5, atol=1e-7)
            support = distance <= evidence.support_radius
            np.testing.assert_array_equal(support, data[f"{case}_supported"])
            estimate, neighbor_distance = neighbors((inputs - evidence.x_mean) / evidence.x_scale,
                normalized[cal_support], cal_losses[cal_support])
            match = support & (neighbor_distance <= radius)
            age, _ = episode_age(data[f"{case}_terminals"])
            windows = []
            for index, old in enumerate(original["cases"][case]["windows"]):
                start, end = index * CONFIG["window"], (index + 1) * CONFIG["window"]
                selected, supported = match[start:end], support[start:end]
                means = observed[start:end][supported].mean(0).tolist() if supported.any() else None
                if old["eligible"]:
                    np.testing.assert_allclose(means[0], old["statistic"], rtol=1e-5, atol=1e-7)
                assert bool(old["eligible"] and means[0] > evidence.threshold) == old["alarm"]
                matched_eligible = bool(selected.mean() >= CONFIG["minimum_supported_fraction"])
                windows.append(dict(end_step=end, original_eligible=old["eligible"], original_alarm=old["alarm"],
                    fit_supported_steps=int(supported.sum()), matched_steps=int(selected.sum()),
                    matched_eligible=matched_eligible, loss_means=means,
                    early_episode_fraction=float((age[start:end] < 100).mean()),
                    gaps=(gaps(observed[start:end][selected], estimate[start:end][selected],
                               cal_losses[cal_match], cal_estimate[cal_match]) if matched_eligible else None)))
            if case == "stationary":
                report["phases"]["shared_prefix"] = describe_phase(windows[:CHANGE_AT // CONFIG["window"]])
            report["phases"][case] = describe_phase(windows[CHANGE_AT // CONFIG["window"]:])
        report.update(complete=True, original_reproduced=True, maximum_error_delta=maximum_error_delta,
                      decision="hold_conditional_evidence_integration", retained_method="v25")
        write(output, report)
    print(f"DONE {PROTOCOL} {env} seed={seed} reproduced=True interactions=0", flush=True)
    return report


def summarize(root=ROOT):
    report = dict(protocol=PROTOCOL, complete=True, additional_interactions=0,
        initialization_probes=0, policy_optimizer_steps=0, critic_optimizer_steps=0,
        encoder_optimizer_steps=0, predictor_reconstruction_steps=0,
        neighbors=NEIGHBORS, matching_quantile=CONFIG["quantile"],
        loss_order=["max_reward_dynamics", "reward", "dynamics"],
        decision="hold_conditional_evidence_integration", retained_method="v25", environments={})
    for env in ENVS:
        runs = [json.loads((root / env.replace("-", "_") / f"seed{seed}" / "readout.json").read_text()) for seed in SEEDS]
        for seed, row in zip(SEEDS, runs):
            assert (row["protocol"], row["env"], row["seed"], row["complete"], row["original_reproduced"]) == (
                PROTOCOL, env, seed, True, True)
            assert all(row[key] == 0 for key in ("additional_interactions", "initialization_probes",
                                                "policy_optimizer_steps", "critic_optimizer_steps", "encoder_optimizer_steps"))
            assert row["predictor_reconstruction_steps"] == CONFIG["updates"]
            report["predictor_reconstruction_steps"] += row["predictor_reconstruction_steps"]
        phases = {}
        for phase in ("shared_prefix", *CASES):
            rows = [row["phases"][phase] for row in runs]
            result = {key: sum(row[key] for row in rows) for key in
                ("windows", "original_eligible", "original_alarms", "matched_eligible", "matched_alarm_windows")}
            result.update({key: float(np.mean([row[key] for row in rows])) for key in
                           ("fit_supported_fraction", "matched_fraction", "early_episode_fraction")})
            for key in ("observed_gap", "input_mix_gap", "remaining_gap"):
                values = [row[key] for row in rows if row[key] is not None]
                result[key] = np.mean(values, axis=0).tolist() if values else None
            result["runs_with_gaps"] = sum(row["observed_gap"] is not None for row in rows)
            phases[phase] = result
        report["environments"][env] = dict(phases=phases, runs=runs)
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--env", choices=ENVS)
    parser.add_argument("--seed", type=int, choices=SEEDS)
    parser.add_argument("--source", type=Path, default=SOURCE)
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--summarize", action="store_true")
    args = parser.parse_args()
    torch.set_num_threads(2)
    if args.summarize:
        report = summarize(args.root)
        write(args.root / "compact_readout.json", report)
        print(json.dumps({env: row["phases"] for env, row in report["environments"].items()}))
    else:
        run(args.env, args.seed, args.source, args.root)
