"""Frozen-network stochastic/mean-action comparison on paired encoder snapshots."""

import argparse
import json
from pathlib import Path

import numpy as np
import torch

import diagnose_snapshot_encoders as snapshots


training = snapshots.training
METHOD = snapshots.pilot.METHOD
PROTOCOL = "snapshot_execution_policy_v1"
SOURCE_ROOT = snapshots.OUTPUT_ROOT
OUTPUT_ROOT = Path("results/iclr_execution_policy_20260920")
ACTION_NOISE_BASE = 31000000
EXECUTIONS = ("deterministic", "stochastic")


def action_noise_seed(mode, weight_index, episode):
    return ACTION_NOISE_BASE + 1000 * mode + 10 * weight_index + episode


class ExecutionProbe:
    """Actor-only interface to the existing evaluator, with fixed execution mode."""

    def __init__(self, policy, execution, reward_dim=2):
        assert execution in EXECUTIONS
        self.policy, self.execution, self.reward_dim = policy, execution, reward_dim
        self.reset_statistics()

    def reset_statistics(self):
        self.steps = self.saturated = 0
        width = self.policy.mean_head.out_features
        self.sums = np.zeros((5, width), dtype=np.float64)
        self.squares = np.zeros_like(self.sums)

    def select_action(self, state, context, deterministic=True):
        # The shared evaluator requests deterministic actions; this diagnostic
        # explicitly substitutes one frozen execution mode without a critic.
        assert deterministic
        mean, log_std = self.policy(state, context)
        std = log_std.exp()
        mean_action = torch.tanh(mean)
        action = (torch.tanh(torch.distributions.Normal(mean, std).rsample())
                  if self.execution == "stochastic" else mean_action)
        values = torch.stack((mean, std, mean_action, action, action - mean_action))
        values = values.detach().cpu().numpy().reshape(self.sums.shape).astype(np.float64)
        self.sums += values
        self.squares += values ** 2
        self.saturated += int(np.count_nonzero(np.abs(values[3]) >= .95))
        self.steps += 1
        return action

    def statistics(self):
        labels = ("pre_tanh_mean", "pre_tanh_std", "deterministic_action",
                  "executed_action", "noise_displacement")
        result = {name + "_rms": float(np.sqrt(row.mean() / self.steps))
                  for name, row in zip(labels, self.squares)}
        action_mean = self.sums[3] / self.steps
        action_var = np.maximum(self.squares[3] / self.steps - action_mean ** 2, 0.)
        result.update(
            steps=self.steps, pre_tanh_std_mean=float(self.sums[1].mean() / self.steps),
            empirical_action_mean_per_dim=action_mean.tolist(),
            empirical_action_std_per_dim=np.sqrt(action_var).tolist(),
            executed_saturation_fraction=self.saturated / (self.steps * len(action_mean)))
        return result


def evaluate_front(env, actor, encoder, weights, episode_seeds):
    front = []
    for index, weight in enumerate(weights):
        actor.reset_statistics()
        returns, lengths, noise_seeds = [], [], []
        for episode, reset_seed in enumerate(episode_seeds):
            noise_seed = action_noise_seed(env.current_mode_idx, index, episode)
            before = env._total_steps
            with snapshots.pilot.drift.forks.isolated_rng(noise_seed):
                point = training.evaluate_pareto_v2(
                    env, actor, METHOD, encoder, [weight], num_episodes=1,
                    device="cpu", episode_seeds=[reset_seed])[0]
            returns.append(point["return"])
            lengths.append(env._total_steps - before)
            noise_seeds.append(noise_seed)
        assert actor.steps == sum(lengths)
        front.append({"weight": weight.tolist(), "return": np.mean(returns, axis=0).tolist(),
                      "episode_returns": returns, "episode_lengths": lengths,
                      "action_noise_seeds": noise_seeds, "action_statistics": actor.statistics()})
    return front


def deterministic_reference_check(actual, reference):
    checks = {key: actual[key] == reference[key] for key in (
        "evaluation_protocol", "evaluation_weights", "episode_seeds_per_mode", "ref_point")}
    checks["ordered_weights"] = ([[p["weight"] for p in f] for f in actual["pareto_per_mode"]]
                                 == [[p["weight"] for p in f] for f in reference["pareto_per_mode"]])
    error = None
    if all(checks.values()):
        values = lambda r: np.array([[p["return"] for p in f] for f in r["pareto_per_mode"]])
        error = float(np.max(np.abs(values(actual) - values(reference))))
    return {"checks": checks, "max_return_error": error,
            "matched": error is not None and error <= 1e-6}


def comparison(cells):
    deterministic, stochastic = (cells[name] for name in EXECUTIONS)
    delta = []
    for det_front, stoch_front in zip(deterministic["pareto_per_mode"], stochastic["pareto_per_mode"]):
        delta.append([float(np.dot(d["weight"], np.array(s["return"]) - d["return"]))
                      for d, s in zip(det_front, stoch_front)])
    return {
        "stochastic_minus_deterministic_hv_mean_mode": stochastic["hv_mean_mode"] - deterministic["hv_mean_mode"],
        "stochastic_minus_deterministic_hv_per_mode": (
            np.array(stochastic["hv_per_mode"]) - deterministic["hv_per_mode"]).tolist(),
        "paired_preference_utility_delta_per_mode": delta,
        "speed_preference_returns": {name: [f[-1]["return"] for f in c["pareto_per_mode"]]
                                     for name, c in cells.items()},
    }


def run(condition, actor_step, output_root=OUTPUT_ROOT):
    assert condition in snapshots.confirmation.CONDITIONS and actor_step in snapshots.STEPS
    source = SOURCE_ROOT / condition
    reference_report = json.loads((source / "seed31.json").read_text())
    assert reference_report["complete"] and reference_report["protocol"] == snapshots.PROTOCOL
    assert reference_report["training_reproduction"]["matched"]
    reference = next(c for c in reference_report["cells"]
                     if c["actor_step"] == c["encoder_step"] == actor_step)
    states = torch.load(source / "seed31_snapshots.pt", map_location="cpu", weights_only=True)[actor_step]
    output = Path(output_root) / condition / f"step{actor_step}.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    report = {
        "protocol": PROTOCOL, "condition": condition, "seed": 31,
        "actor_step": actor_step, "encoder_step": actor_step, "complete": False,
        "source": str(source), "training_interactions": 0, "extra_updates": 0,
        "evaluation_interactions": 0, "environment_initialization_probe_steps": 1,
        "action_noise_seed_rule": "31000000 + 1000 * mode + 10 * weight_index + episode_index",
        "action_noise_pairing": "Same streams across conditions and snapshot times; action RNG separate from environment-reset RNG.",
        "statistics_scope": "Statistics pool visited steps per preference across three episodes. "
                            "pre_tanh_std is conditional Gaussian spread before tanh; empirical action "
                            "std mixes visited states and action noise. tanh(mean) is not E[tanh(X)].",
        "scope": "Raw undiscounted returns from fixed networks, with fresh episode contexts. "
                 "No critic, optimization, selection feedback, or change to the primary evaluation.",
        "cells": {},
    }

    def save():
        output.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")

    save()
    env = training.make_piecewise_mo_env("mo-halfcheetah-v5", switch_interval=25000, seed=1031)
    try:
        obs_dim, act_dim = env.observation_space.shape[0], env.action_space.shape[0]
        context_dim = training.context_dim_for_method(METHOD, obs_dim, act_dim, env.reward_dim, env.num_modes)
        policy = training.PolicyNet(obs_dim, act_dim, context_dim, hidden=64)
        encoder = training.LatentTransitionEncoder(obs_dim, act_dim, env.reward_dim, latent_dim=8, hidden=64)
        policy.load_state_dict(states["policy"])
        encoder.load_state_dict(states["encoder"])
        policy.eval()
        encoder.eval()
        print(f"EXECUTION_DIAGNOSTIC condition={condition} step={actor_step} training=0", flush=True)
        for execution in EXECUTIONS:
            actor = ExecutionProbe(policy, execution, env.reward_dim)
            before = env._total_steps
            cell = training.evaluate_locked_modes(
                env, lambda weights, seeds: evaluate_front(env, actor, encoder, weights, seeds),
                31, np.array(reference["ref_point"]))
            cell["environment_steps"] = env._total_steps - before
            report["cells"][execution] = cell
            report["evaluation_interactions"] += cell["environment_steps"]
            if execution == "deterministic":
                report["deterministic_reproduction"] = deterministic_reference_check(cell, reference)
            save()
            assert report["deterministic_reproduction"]["matched"], report["deterministic_reproduction"]
            print(f"EXECUTION_EVAL execution={execution} hv={cell['hv_mean_mode']:.6f} "
                  f"interactions={cell['environment_steps']}", flush=True)
    finally:
        env.close()
    report["comparison"] = comparison(report["cells"])
    report["complete"] = True
    save()
    print(f"DONE execution_policy condition={condition} step={actor_step} training=0 "
          f"evaluation={report['evaluation_interactions']}", flush=True)
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--condition", choices=snapshots.confirmation.CONDITIONS, required=True)
    parser.add_argument("--actor-step", type=int, choices=snapshots.STEPS, required=True)
    parser.add_argument("--output-root", type=Path, default=OUTPUT_ROOT)
    args = parser.parse_args()
    run(args.condition, args.actor_step, args.output_root)
