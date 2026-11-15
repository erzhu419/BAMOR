"""Shadow-only fixed-policy raw-return critics and held-out action directions."""

import argparse
import copy
import json
from pathlib import Path

import numpy as np
import torch
from torch import nn

import diagnose_mean_action_gradient as gradient


training = gradient.training
PROTOCOL = "shadow_deterministic_return_critic_v1"
OUTPUT_ROOT = Path("results/iclr_return_critic_20260921")
TRAIN_REPEATS = 5
TRAIN_STEPS = 1000
NOISE_STD = .1
FIT_UPDATES = 20000
BATCH_SIZE = 256
LEARNING_RATE = 3e-4
TAU = .005
BRANCHES = ("base", "soft_plus", "soft_minus", "raw_plus", "raw_minus", "repeat")


def reset_seed(split, seed, mode, repeat):
    return {"train": 42000000, "test": 43000000}[split] + seed * 1000 + mode * 10 + repeat


def noise_seed(seed, mode, weight_index, repeat):
    return 44000000 + seed * 10000 + mode * 1000 + weight_index * 100 + repeat


@torch.no_grad()
def collect_data(env, networks, method, seed):
    rows, episodes = [], []
    before = env._total_steps
    env._lock_mode = True
    for mode in range(env.num_modes):
        env._apply_mode(mode)
        for wi, weight in enumerate(gradient.WEIGHTS):
            for repeat in range(TRAIN_REPEATS):
                rs = reset_seed("train", seed, mode, repeat)
                ns = noise_seed(seed, mode, wi, repeat)
                rng = np.random.default_rng(ns)
                obs, _ = env.reset(seed=rs)
                trackers = (training.TransitionContextTracker(2),
                            training.LearnedTransitionContextTracker(networks["encoder"]))
                for step in range(TRAIN_STEPS):
                    ctx = gradient.context(method, weight, trackers)
                    mean, _ = networks["policy"](torch.as_tensor(obs, dtype=torch.float32)[None],
                                                  torch.as_tensor(ctx)[None])
                    noise = torch.as_tensor(rng.normal(0., NOISE_STD, mean.shape), dtype=torch.float32)
                    action = (mean + noise).tanh().numpy().ravel()
                    sn, reward, term, trunc, _ = env.step(action)
                    trackers[0].update(obs, action, reward, sn)
                    trackers[1].update(obs, action, reward, sn, "cpu")
                    cn = gradient.context(method, weight, trackers)
                    an = networks["policy"].get_action(
                        torch.as_tensor(sn, dtype=torch.float32)[None], torch.as_tensor(cn)[None],
                        deterministic=True).numpy().ravel()
                    rows.append((obs.copy(), action.copy(), reward.copy(), ctx.copy(),
                                 sn.copy(), cn.copy(), an.copy(), [float(term)], [float(trunc)]))
                    obs = sn
                    if term or trunc:
                        assert step + 1 == TRAIN_STEPS, "Unexpected early episode end"
                episodes.append({"mode": mode, "weight_index": wi, "repeat": repeat,
                                 "reset_seed": rs, "noise_seed": ns, "steps": TRAIN_STEPS})
        print(f"COLLECT mode={mode} transitions={len(rows)}", flush=True)
    keys = ("obs", "actions", "rewards", "contexts", "next_obs", "next_contexts",
            "next_actions", "terminated", "truncated")
    data = {key: torch.as_tensor(np.asarray(column), dtype=torch.float32)
            for key, column in zip(keys, zip(*rows))}
    assert env._total_steps - before == len(rows)
    return data, episodes


class RawQ(nn.Module):
    """Train-only normalization; public outputs stay in raw reward units."""

    def __init__(self, data, hidden=64):
        super().__init__()
        for key in ("obs", "actions", "contexts"):
            self.register_buffer(key + "_mean", data[key].mean(0))
            self.register_buffer(key + "_scale", data[key].std(0, unbiased=False).clamp_min(1e-3))
        self.register_buffer("reward_scale", data["rewards"].std(0, unbiased=False).clamp_min(1e-3))
        self.net = training.SoftQNet(data["obs"].shape[1], data["actions"].shape[1],
                                    data["contexts"].shape[1], data["rewards"].shape[1], hidden)

    def forward(self, obs, action, context):
        return self.net((obs - self.obs_mean) / self.obs_scale,
                        (action - self.actions_mean) / self.actions_scale,
                        (context - self.contexts_mean) / self.contexts_scale) * self.reward_scale


@torch.no_grad()
def bellman_target(batch, targets):
    value = torch.minimum(*(q(batch["next_obs"], batch["next_actions"], batch["next_contexts"])
                            for q in targets))
    # A time-limit reset is not an absorbing transition in this continuing task.
    return batch["rewards"] + gradient.GAMMA * (1 - batch["terminated"]) * value


def fit_critics(data, seed, updates=FIT_UPDATES):
    # The fitter receives training tensors only; no held-out outcomes or policy.
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(45000000 + seed)
        critics = nn.ModuleList([RawQ(data), RawQ(data)])
        targets = copy.deepcopy(critics).requires_grad_(False)
        optimizer = torch.optim.Adam(critics.parameters(), lr=LEARNING_RATE)
        rng = torch.Generator().manual_seed(46000000 + seed)
        logs, losses = [], []
        for update in range(1, updates + 1):
            index = torch.randint(len(data["obs"]), (BATCH_SIZE,), generator=rng)
            batch = {key: value[index] for key, value in data.items()}
            target = bellman_target(batch, targets)
            loss = sum(((q(batch["obs"], batch["actions"], batch["contexts"]) - target)
                        / q.reward_scale).square().mean() for q in critics)
            assert torch.isfinite(loss), "Nonfinite shadow critic loss"
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            with torch.no_grad():
                for source, target_param in zip(critics.parameters(), targets.parameters()):
                    target_param.lerp_(source, TAU)
            losses.append(float(loss.detach()))
            if update % 2000 == 0 or update == updates:
                logs.append({"update": update, "mean_normalized_td_loss": float(np.mean(losses))})
                losses.clear()
                print(f"SHADOW_FIT update={update}/{updates} loss={logs[-1]['mean_normalized_td_loss']:.6f}", flush=True)
    critics.zero_grad(set_to_none=True)
    critics.eval().requires_grad_(False)
    return critics, {"updates": updates, "data_rows": len(data["obs"]), "batch_size": BATCH_SIZE,
                     "learning_rate": LEARNING_RATE, "tau": TAU, "losses": logs,
                     "reward_scale": critics[0].reward_scale.tolist()}


def evaluate_case(env, networks, critics, method, weight, seed, epsilon=gradient.EPSILON,
                  score_actions=None):
    obs, _ = env.reset(seed=seed)
    trackers = (training.TransitionContextTracker(2),
                training.LearnedTransitionContextTracker(networks["encoder"]))
    obs, _ = gradient.rollout(env, networks, method, weight, obs, trackers, gradient.WARMUP)
    ctx = gradient.context(method, weight, trackers)
    raw_networks = {"policy": networks["policy"], "q1": critics[0], "q2": critics[1]}
    diagnostics, actions = {}, {}
    for name, scorer in (("soft", networks), ("raw", raw_networks)):
        proposed, diagnostics[name] = gradient.mean_directions(scorer, obs, ctx, weight, epsilon)
        actions.update({name + "_" + sign: proposed[sign] for sign in ("plus", "minus")})
        actions["base"] = proposed["base"]
    # Both scorers rank the same actions; scalar offsets cannot fix contrasts.
    s, c, w = [torch.as_tensor(x, dtype=torch.float32)[None] for x in (obs, ctx, weight)]
    with torch.no_grad():
        predictions = {name: {label: float(gradient.scalar_q(scorer, s, torch.as_tensor(a)[None], c, w))
                              for label, a in actions.items()}
                       for name, scorer in (("soft", networks), ("raw", raw_networks))}
    extra = {} if score_actions is None else {"external_scores": score_actions(obs, ctx, actions)}
    branches, final_obs = {}, {}
    for name in BRANCHES:
        branch = env if name == "repeat" else gradient.fork_env(env)
        copied = copy.deepcopy(trackers, {id(networks["encoder"]): networks["encoder"]})
        try:
            final_obs[name], branches[name] = gradient.rollout(
                branch, networks, method, weight, obs.copy(), copied,
                max(gradient.HORIZONS), gradient.HORIZONS,
                first_action=actions[name] if name not in ("base", "repeat") else None)
        finally:
            if name != "repeat":
                branch.close()
    assert branches["base"] == branches["repeat"]
    np.testing.assert_array_equal(final_obs["base"], final_obs["repeat"])
    return {"reset_seed": seed, "weight": weight.tolist(), "diagnostics": diagnostics,
            "anchor_observation": obs.tolist(), "anchor_context": ctx.tolist(), **extra,
            "predictions": predictions, "branches": branches, "identity_matched": True,
            "evaluation_interactions": gradient.WARMUP + sum(b["steps"] for b in branches.values())}


def summarize_cases(cases):
    h = str(max(gradient.HORIZONS))
    cells = []
    for mode, wi in sorted({(c["mode"], c["weight_index"]) for c in cases}):
        group = [c for c in cases if (c["mode"], c["weight_index"]) == (mode, wi)]
        assert len(group) == gradient.REPEATS
        assert len({c["reset_seed"] for c in group}) == gradient.REPEATS
        cell = {"mode": mode, "weight_index": wi}
        for name in ("soft", "raw"):
            row = {}
            for label, a, b in (("plus_base", name + "_plus", "base"),
                                ("plus_minus", name + "_plus", name + "_minus")):
                values = [float(np.dot(c["weight"], np.asarray(c["branches"][a]["discounted_returns"][h]) -
                                       c["branches"][b]["discounted_returns"][h])) for c in group]
                row[label] = {"per_reset": values, "mean": float(np.mean(values))}
            row["q_plus_minus_mean"] = float(np.mean([
                c["predictions"][name][name + "_plus"] - c["predictions"][name][name + "_minus"] for c in group]))
            cell[name] = row
        cells.append(cell)
    aggregate = {name: {
        "mean_plus_base": float(np.mean([c[name]["plus_base"]["mean"] for c in cells])),
        "mean_plus_minus": float(np.mean([c[name]["plus_minus"]["mean"] for c in cells])),
        "positive_plus_minus_cells": sum(c[name]["plus_minus"]["mean"] > 0 for c in cells),
        "positive_plus_base_cells": sum(c[name]["plus_base"]["mean"] > 0 for c in cells),
        "q_ascent_cells": sum(c[name]["q_plus_minus_mean"] > 0 for c in cells)} for name in ("soft", "raw")}
    contrasts = {}
    for name in ("soft", "raw"):
        errors, agreement = [], []
        for case in cases:
            for a, b in (("soft_plus", "soft_minus"), ("raw_plus", "raw_minus")):
                truth = float(np.dot(case["weight"], np.asarray(case["branches"][a]["discounted_returns"][h]) -
                                     case["branches"][b]["discounted_returns"][h]))
                prediction = case["predictions"][name][a] - case["predictions"][name][b]
                errors.append(prediction - truth)
                agreement.append(int(np.sign(prediction) == np.sign(truth)))
        contrasts[name] = {"rmse": float(np.sqrt(np.mean(np.square(errors)))),
                           "sign_matches": sum(agreement), "comparisons": len(errors)}
    return {"primary_horizon": int(h), "cells": cells, "aggregate": aggregate,
            "same_action_contrast_calibration": contrasts}


def run(condition, seed, output_root=OUTPUT_ROOT):
    assert seed in gradient.SEEDS
    networks, source = gradient.load_models(condition, seed)
    method = gradient.METHODS[condition]
    output = Path(output_root) / condition
    output.mkdir(parents=True, exist_ok=True)
    path = output / f"seed{seed}.json"
    report = {"protocol": PROTOCOL, "condition": condition, "seed": seed, "method": method,
              "source": str(source), "complete": False, "policy_training_interactions": 0,
              "critic_data_interactions": 0, "evaluation_interactions": 0,
              "environment_initialization_probe_steps": 0, "policy_updates": 0,
              "weights": gradient.WEIGHTS.tolist(), "horizons": list(gradient.HORIZONS),
              "warmup": gradient.WARMUP, "epsilon": gradient.EPSILON, "gamma": gradient.GAMMA,
              "collection_noise_std": NOISE_STD, "cases": [],
              "scope": "Independent fixed-policy twin TD critics, no entropy, relabeling or actor updates. "
                       "Train/test reset sets are disjoint; fit length and normalization use training data only. "
                       "Modes configure data collection/evaluation, never critic inputs. "
                       "The continuing Bellman target is tested with finite 512-step raw contrasts. "
                       "Observed model seeds and within-model paired states; shadow diagnostic, not policy improvement."}

    def save():
        path.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")

    save()
    env = training.make_piecewise_mo_env("mo-halfcheetah-v5", switch_interval=25000, seed=seed + 1000)
    report["environment_initialization_probe_steps"] += 1
    try:
        data, episodes = collect_data(env, networks, method, seed)
        report["collection_episodes"] = episodes
        report["critic_data_interactions"] = env._total_steps
    finally:
        env.close()
    save()
    critics, report["fit"] = fit_critics(data, seed, FIT_UPDATES)
    del data
    torch.save(critics.state_dict(), output / f"seed{seed}_raw_critic.pt")
    save()
    # Held-out rollouts start only after the fixed fitting schedule has ended.
    env = training.make_piecewise_mo_env("mo-halfcheetah-v5", switch_interval=25000, seed=seed + 2000)
    report["environment_initialization_probe_steps"] += 1
    env._lock_mode = True
    try:
        for mode in range(env.num_modes):
            env._apply_mode(mode)
            for wi, weight in enumerate(gradient.WEIGHTS):
                for repeat in range(gradient.REPEATS):
                    rs = reset_seed("test", seed, mode, repeat)
                    case = evaluate_case(env, networks, critics, method, weight, rs)
                    case.update(mode=mode, weight_index=wi, repeat=repeat)
                    report["cases"].append(case)
                    report["evaluation_interactions"] += case["evaluation_interactions"]
                save()
                print(f"SHADOW_EVAL mode={mode} weight={wi} cases={len(report['cases'])}", flush=True)
        assert env._total_steps == len(report["cases"]) * (gradient.WARMUP + max(gradient.HORIZONS))
    finally:
        env.close()
    report["summary"] = summarize_cases(report["cases"])
    report["complete"] = True
    save()
    print(f"DONE return_critic condition={condition} seed={seed} data={report['critic_data_interactions']} "
          f"evaluation={report['evaluation_interactions']}", flush=True)
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--condition", choices=gradient.METHODS, required=True)
    parser.add_argument("--seed", type=int, choices=gradient.SEEDS, required=True)
    parser.add_argument("--output-root", type=Path, default=OUTPUT_ROOT)
    args = parser.parse_args()
    run(args.condition, args.seed, args.output_root)
