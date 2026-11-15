"""Budgeted, mode-blind post-training actor calibration on saved archives."""

import argparse
import json
from pathlib import Path
from statistics import mean, median

import numpy as np
import torch

import run_scoped_extension as extension
import train_bamor_mujoco as training


PROTOCOL = "archive_actor_probe_v1"
ROOT = Path("results/iclr_archive_actor_probe_20260926")
PILOT_SEEDS = tuple(range(1011, 1016))
STEPS = (10000, 25000, 40000, 50000)
PROBE_STEPS = 10000
PROBE_DWELL = 5000
METHOD = extension.METHODS["adaptive"]


def weights_for_dim(dim):
    if dim == 2:
        return np.asarray([[1., 0.], [.5, .5], [0., 1.]], dtype=np.float32)
    assert dim == 3
    return np.asarray([[1., 0., 0.], [0., 1., 0.], [0., 0., 1.],
                       [.2, .4, .4], [.4, .2, .4], [.4, .4, .2]], dtype=np.float32)


class ActorProbe:
    def __init__(self, policy, reward_dim):
        self.policy = policy
        self.reward_dim = reward_dim

    def select_action(self, state, context, deterministic=True):
        assert deterministic
        return self.policy.get_action(state, context, deterministic=True)


def load_models(env_name, seed, source=extension.ROOT):
    folder = Path(source) / env_name.replace("-", "_") / "legacy_adaptive"
    raw = json.loads((folder / f"{METHOD}_seed{seed}.json").read_text())
    ckpt = torch.load(folder / f"{METHOD}_seed{seed}_model.pt",
                      map_location="cpu", weights_only=True)
    env = training.make_piecewise_mo_env(env_name, switch_interval=25000, seed=seed + 1000)
    try:
        obs_dim, act_dim, reward_dim = env.observation_space.shape[0], env.action_space.shape[0], env.reward_dim
        context_dim = training.context_dim_for_method(
            METHOD, obs_dim, act_dim, reward_dim, env.num_modes)
        policy = training.PolicyNet(obs_dim, act_dim, context_dim, hidden=64)
        encoder = training.LatentTransitionEncoder(
            obs_dim, act_dim, reward_dim, latent_dim=training.LATENT_CONTEXT_DIM, hidden=64)
    finally:
        env.close()
    encoder.load_state_dict(ckpt["encoder"])
    encoder.eval()
    archive = {int(snap["step"]): snap["policy"] for snap in ckpt["policy_archive"]}
    assert all(step in archive for step in STEPS)
    assert raw["evaluation_protocol"] == "locked_full_simplex_v1"
    return raw, policy, encoder, archive


def cached_return(front, step, weight):
    matches = [point["return"] for point in front
               if point["archive_step"] == step and np.allclose(point["weight"], weight, atol=1e-7)]
    assert len(matches) == 1, (step, weight)
    return np.asarray(matches[0], dtype=float)


def reproduce_final_cell(env_name, seed, raw, policy, encoder, archive):
    weight = weights_for_dim(raw["ground_truth"]["reward_dim"])[0]
    env = training.make_piecewise_mo_env(env_name, switch_interval=25000, seed=seed + 1000)
    try:
        env._lock_mode = True
        env._apply_mode(0)
        policy.load_state_dict(archive[50000])
        policy.eval()
        actual = training.evaluate_pareto_v2(
            env, ActorProbe(policy, env.reward_dim), METHOD, encoder, [weight], num_episodes=3,
            device="cpu", episode_seeds=raw["episode_seeds_per_mode"][0])[0]["return"]
        expected = cached_return(raw["pareto_per_mode"][0], 50000, weight)
        error = float(np.max(np.abs(np.asarray(actual) - expected)))
        assert error <= 1e-6, error
        return {"max_return_error": error, "evaluation_interactions": env._total_steps,
                "initialization_probe_steps": 1}
    finally:
        env.close()


def probe_cell(env_name, seed, index, weight, policy, encoder):
    env = training.make_piecewise_mo_env(
        env_name, switch_interval=PROBE_DWELL,
        seed=4000000 + 1000 * seed + index)
    try:
        total_scalar = 0.0
        episode = 0
        while env._total_steps < PROBE_STEPS:
            reset_seed = 5000000 + 10000 * seed + 100 * index + episode
            point = training.evaluate_pareto_v2(
                env, ActorProbe(policy, env.reward_dim), METHOD, encoder, [weight], num_episodes=1,
                device="cpu", episode_seeds=[reset_seed])[0]
            total_scalar += float(np.dot(weight, point["return"]))
            episode += 1
        return {"score_per_step": total_scalar / env._total_steps,
                "interactions": env._total_steps, "episodes": episode,
                "mode_history_audit_only": env.get_ground_truth()["mode_history"]}
    finally:
        env.close()


def utility(values, weights):
    return mean(float(np.dot(weight, value)) for weight, value in zip(weights, values))


def run(env_name, seed, root=ROOT, source=extension.ROOT, baseline_root=ROOT.parent / "iclr_capql_archive_extension_20260926"):
    assert env_name in extension.ENVIRONMENTS and seed in PILOT_SEEDS
    torch.set_num_threads(2)
    raw, policy, encoder, archive = load_models(env_name, seed, source)
    baseline = json.loads((Path(baseline_root) / env_name.replace("-", "_") /
                           f"capql_archive_seed{seed}.json").read_text())
    for key in ("evaluation_weights", "episode_seeds_per_mode", "ref_point"):
        assert baseline[key] == raw[key]
    assert baseline["ground_truth"]["mode_history"] == raw["ground_truth"]["mode_history"]
    weights = weights_for_dim(raw["ground_truth"]["reward_dim"])
    assert all(any(np.allclose(weight, existing, atol=1e-7)
                   for existing in raw["evaluation_weights"]) for weight in weights)
    report = {"protocol": PROTOCOL, "env": env_name, "seed": seed,
              "candidate_steps": list(STEPS), "probe_weights": weights.tolist(),
              "new_training_interactions": 0, "new_learner_updates": 0,
              "probe_evaluation_interactions": 0, "initialization_probe_steps": 0,
              "cells": [], "test": {}}
    for index, weight in enumerate(weights):
        cells = []
        for step in STEPS:
            policy.load_state_dict(archive[step])
            policy.eval()
            cell = probe_cell(env_name, seed, index, weight, policy, encoder)
            cell["actor_step"] = step
            cells.append(cell)
            report["probe_evaluation_interactions"] += cell["interactions"]
            report["initialization_probe_steps"] += 1
        selected = max(range(len(STEPS)), key=lambda i: cells[i]["score_per_step"])
        report["cells"].append({"weight": weight.tolist(), "selected_step": STEPS[selected],
                                "probes": cells})
    ref = np.asarray(raw["ref_point"], dtype=float)
    selected_returns, final_returns, capql_returns, oracle_returns = [], [], [], []
    for mode, front in enumerate(raw["pareto_per_mode"]):
        selected_returns.append([])
        final_returns.append([])
        capql_returns.append([])
        oracle_returns.append([])
        for index, weight in enumerate(weights):
            selected_step = report["cells"][index]["selected_step"]
            selected_returns[-1].append(cached_return(front, selected_step, weight))
            final_returns[-1].append(cached_return(front, 50000, weight))
            capql_returns[-1].append(cached_return(baseline["pareto_per_mode"][mode], 50000, weight))
    for index, weight in enumerate(weights):
        best_step = max(STEPS, key=lambda step: mean(float(np.dot(weight, cached_return(
            front, step, weight))) for front in raw["pareto_per_mode"]))
        for mode, front in enumerate(raw["pareto_per_mode"]):
            oracle_returns[mode].append(cached_return(front, best_step, weight))
    for name, fronts in (("selected", selected_returns), ("final", final_returns),
                         ("capql_final", capql_returns), ("hindsight_best", oracle_returns)):
        report["test"][name] = {
            "mean_same_preference_utility": mean(utility(front, weights) for front in fronts),
            "subset_mode_mean_hv": mean(training.compute_hv_nd(np.asarray(front), ref)
                                        for front in fronts)}
    output = Path(root) / env_name.replace("-", "_") / f"seed{seed}.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    print(f"DONE {PROTOCOL} {env_name} seed={seed} probe={report['probe_evaluation_interactions']} "
          f"selected_utility={report['test']['selected']['mean_same_preference_utility']:.6f}", flush=True)
    return report


def summarize(root=ROOT):
    result = {"protocol": PROTOCOL, "cohort": "inspected_seed_development_pilot",
              "new_training_interactions": 0, "new_learner_updates": 0,
              "probe_evaluation_interactions": 0, "initialization_probe_steps": 0,
              "environments": {}}
    for env_name in extension.ENVIRONMENTS:
        rows = [json.loads((Path(root) / env_name.replace("-", "_") /
                            f"seed{seed}.json").read_text()) for seed in PILOT_SEEDS]
        assert all((row["protocol"], row["env"], row["seed"]) ==
                   (PROTOCOL, env_name, seed) for row, seed in zip(rows, PILOT_SEEDS))
        for row in rows:
            result["probe_evaluation_interactions"] += row["probe_evaluation_interactions"]
            result["initialization_probe_steps"] += row["initialization_probe_steps"]
        deltas = [row["test"]["selected"]["mean_same_preference_utility"] -
                  row["test"]["final"]["mean_same_preference_utility"] for row in rows]
        compact_rows = [{"seed": row["seed"],
                         "selected_steps": [cell["selected_step"] for cell in row["cells"]],
                         "probe_evaluation_interactions": row["probe_evaluation_interactions"],
                         "test": row["test"]} for row in rows]
        result["environments"][env_name] = {
            "seeds": list(PILOT_SEEDS), "probe_weights": rows[0]["probe_weights"],
            "runs": compact_rows,
            "selected_minus_final_mean_utility": mean(deltas),
            "selected_minus_final_median_utility": median(deltas),
            "selected_minus_final_wins": sum(delta > 0 for delta in deltas),
            "selected_minus_final_losses": sum(delta < 0 for delta in deltas),
            "selected_minus_final_ties": sum(delta == 0 for delta in deltas),
            "selected_minus_final_deltas": deltas,
            "mean_test_utility": {name: mean(row["test"][name]["mean_same_preference_utility"]
                                      for row in rows) for name in rows[0]["test"]},
            "mean_subset_hv": {name: mean(row["test"][name]["subset_mode_mean_hv"]
                                    for row in rows) for name in rows[0]["test"]},
        }
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--env", choices=extension.ENVIRONMENTS)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--summarize", action="store_true")
    parser.add_argument("--root", type=Path, default=ROOT)
    args = parser.parse_args()
    if args.summarize:
        readout = summarize(args.root)
        path = args.root / "readout.json"
        path.write_text(json.dumps(readout, indent=2, allow_nan=False) + "\n")
        print("DONE", PROTOCOL, "summary")
    elif args.smoke:
        assert args.env and args.seed is not None
        raw, policy, encoder, archive = load_models(args.env, args.seed)
        print(reproduce_final_cell(args.env, args.seed, raw, policy, encoder, archive))
    else:
        assert args.env and args.seed is not None
        run(args.env, args.seed, args.root)
