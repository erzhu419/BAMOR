"""Frozen Ant critic ranking against paired deterministic-policy returns."""

import argparse
import json
from pathlib import Path

import numpy as np
import torch

import train_bamor_mujoco as training


PROTOCOL = "ant_mixed_critic_v1"
METHOD = "bamor_v25_progress_archive_mix"
SOURCE = Path("results/iclr_75k_final_fresh_20260926/mo_ant_v5/v25")
OUTPUT = Path("results/iclr_ant_mixed_critic_20260927")
SEEDS = tuple(range(1201, 1206))
WEIGHTS = np.array([[0., 0., 1.], [.2, .2, .6], [.4, .2, .4]], dtype=np.float32)
VERTICES = np.eye(3, dtype=np.float32)
HORIZONS = (1, 32, 128, 256)
PREFIX_STEPS = 32
STREAMS = 2
GAMMA = .99


def reset_seed(seed, mode, stream):
    return 77000000 + seed * 1000 + mode * 10 + stream


def load_model(env, seed, source=SOURCE):
    path = source / f"{METHOD}_seed{seed}_model.pt"
    state = torch.load(path, map_location="cpu", weights_only=True)
    assert state["policy_archive"][-1]["step"] == 75000
    assert all(torch.equal(value, state["policy_archive"][-1]["policy"][key])
               for key, value in state["policy"].items())
    obs_dim = env.observation_space.shape[0]
    act_dim = env.action_space.shape[0]
    reward_dim = env.reward_dim
    assert (act_dim, reward_dim, env.num_modes) == (8, 3, 3)
    context_dim = training.context_dim_for_method(
        METHOD, obs_dim, act_dim, reward_dim, env.num_modes)
    hidden = state["policy"]["trunk.0.weight"].shape[0]
    model = {
        "policy": training.PolicyNet(obs_dim, act_dim, context_dim, hidden),
        "q1": training.SoftQNet(obs_dim, act_dim, context_dim, reward_dim, hidden),
        "q2": training.SoftQNet(obs_dim, act_dim, context_dim, reward_dim, hidden),
        "encoder": training.LatentTransitionEncoder(
            obs_dim, act_dim, reward_dim, hidden=hidden),
    }
    for name, network in model.items():
        network.load_state_dict(state[name])
        network.eval().requires_grad_(False)
    return model


def trackers(model):
    return (training.TransitionContextTracker(3),
            training.LearnedTransitionContextTracker(model["encoder"]))


def observe(trackers_pair, obs, action, reward, next_obs):
    trackers_pair[0].update(obs, action, reward, next_obs)
    trackers_pair[1].update(obs, action, reward, next_obs, "cpu")


def context(weight, trackers_pair):
    return training.build_mujoco_context(
        METHOD, weight, trans_tracker=trackers_pair[0],
        learned_tracker=trackers_pair[1])


@torch.no_grad()
def action(model, obs, ctx):
    return model["policy"].get_action(
        torch.as_tensor(obs, dtype=torch.float32)[None],
        torch.as_tensor(ctx, dtype=torch.float32)[None],
        deterministic=True).numpy().ravel()


def prefix(env, model, weight, seed, steps):
    obs, _ = env.reset(seed=seed)
    pair = trackers(model)
    history = []
    for _ in range(steps):
        first = action(model, obs, context(weight, pair))
        next_obs, reward, term, trunc, _ = env.step(first)
        history.append((first.copy(), next_obs.copy(), np.asarray(reward).copy()))
        observe(pair, obs, first, reward, next_obs)
        obs = next_obs
        if term or trunc:
            return history, None, None
    return history, obs, pair


@torch.no_grad()
def candidate_actions(model, obs, weight, pair):
    target = np.asarray(context(weight, pair), dtype=np.float32)
    labels = ["own"]
    contexts = [target]
    for index, label in enumerate(("x_velocity", "y_velocity", "control_cost")):
        if np.array_equal(weight, VERTICES[index]):
            continue
        source = target.copy()
        source[:3] = VERTICES[index]
        labels.append(label)
        contexts.append(source)
    s = torch.as_tensor(obs, dtype=torch.float32)[None].expand(len(labels), -1)
    c = torch.as_tensor(np.stack(contexts))
    actions = torch.as_tensor(np.stack([action(model, obs, ctx) for ctx in contexts]))
    own_contexts = c[:1].expand(len(labels), -1)
    w = torch.as_tensor(weight)
    scores = {}
    for key, score_contexts in (("target", own_contexts), ("source", c)):
        values = torch.minimum(model["q1"](s, actions, score_contexts),
                               model["q2"](s, actions, score_contexts))
        scores[key] = (values * w).sum(-1).numpy().tolist()
    return labels, actions.numpy(), scores


def branch(env, model, weight, seed, history, first_action, horizons):
    obs, _ = env.reset(seed=seed)
    pair = trackers(model)
    for replay_action, expected_obs, expected_reward in history:
        next_obs, reward, term, trunc, _ = env.step(replay_action)
        np.testing.assert_allclose(next_obs, expected_obs, rtol=0, atol=1e-10)
        np.testing.assert_allclose(reward, expected_reward, rtol=0, atol=1e-10)
        assert not (term or trunc)
        observe(pair, obs, replay_action, reward, next_obs)
        obs = next_obs
    discounted = np.zeros(3, dtype=np.float64)
    returns = {}
    steps = 0
    ended = False
    for t in range(max(horizons)):
        chosen = first_action if t == 0 else action(model, obs, context(weight, pair))
        next_obs, reward, term, trunc, _ = env.step(chosen)
        discounted += GAMMA ** t * np.asarray(reward, dtype=np.float64)
        observe(pair, obs, chosen, reward, next_obs)
        obs = next_obs
        steps += 1
        if steps in horizons:
            returns[str(steps)] = float(discounted @ weight)
        if term or trunc:
            ended = True
            break
    for horizon in horizons:
        returns.setdefault(str(horizon), float(discounted @ weight))
    return {"returns": returns, "steps": steps, "ended": ended}, len(history) + steps


def effect(case, horizon=256):
    actual = np.asarray([row["returns"][str(horizon)] for row in case["branches"]])
    values = case["q"]["target"]
    selected = int(np.argmax(values))
    source_selected = int(np.argmax(case["q"]["source"]))
    return {"target_choice": case["labels"][selected],
            "target_gain": float(actual[selected] - actual[0]),
            "source_choice": case["labels"][source_selected],
            "source_gain": float(actual[source_selected] - actual[0]),
            "oracle_gain": float(actual.max() - actual[0]),
            "target_q_gain": float(values[selected] - values[0])}


def summarize_cases(cases, horizon=256):
    viable = [case for case in cases if not case.get("prefix_ended")]
    if not viable:
        return {"states": 0, "prefix_ended": len(cases)}
    effects = [effect(case, horizon) for case in viable]
    return {"states": len(viable), "prefix_ended": len(cases) - len(viable),
            "target_nonown": sum(row["target_choice"] != "own" for row in effects),
            "target_gain_mean": float(np.mean([row["target_gain"] for row in effects])),
            "target_gain_positive": sum(row["target_gain"] > 0 for row in effects),
            "target_gain_negative": sum(row["target_gain"] < 0 for row in effects),
            "source_gain_mean": float(np.mean([row["source_gain"] for row in effects])),
            "oracle_gain_mean": float(np.mean([row["oracle_gain"] for row in effects]))}


def run(seed, source=SOURCE, output=OUTPUT, smoke=False):
    assert seed in SEEDS
    torch.set_num_threads(1)
    path = output / f"seed{seed}{'_smoke' if smoke else ''}.json"
    assert not path.exists(), f"Existing output requires inspection: {path}"
    env = training.make_piecewise_mo_env("mo-ant-v5", switch_interval=25000,
                                          seed=seed + 1000)
    env._lock_mode = True
    model = load_model(env, seed, source)
    weights = WEIGHTS[:1] if smoke else WEIGHTS
    horizons = (1, 4) if smoke else HORIZONS
    modes = range(1 if smoke else env.num_modes)
    streams = range(1 if smoke else STREAMS)
    steps = 8 if smoke else PREFIX_STEPS
    report = {"protocol": PROTOCOL, "seed": seed, "complete": False,
              "smoke": smoke, "method": METHOD, "weights": weights.tolist(),
              "horizons": list(horizons), "prefix_steps": steps,
              "streams": len(streams), "gamma": GAMMA,
              "training_interactions": 0, "learner_updates": 0,
              "environment_initialization_probe_steps": 1,
              "evaluation_interactions": 0, "cases": []}
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2) + "\n")
    try:
        for mode in modes:
            env._apply_mode(mode)
            for wi, weight in enumerate(weights):
                for stream in streams:
                    seed_i = reset_seed(seed, mode, stream)
                    history, obs, pair = prefix(env, model, weight, seed_i, steps)
                    report["evaluation_interactions"] += len(history)
                    case = {"mode": mode, "weight_index": wi, "stream": stream,
                            "reset_seed": seed_i, "prefix_ended": obs is None}
                    if obs is not None:
                        labels, actions, scores = candidate_actions(model, obs, weight, pair)
                        case.update({"labels": labels, "q": scores, "branches": []})
                        for candidate in actions:
                            result, used = branch(env, model, weight, seed_i, history,
                                                  candidate, horizons)
                            case["branches"].append(result)
                            report["evaluation_interactions"] += used
                    report["cases"].append(case)
                    path.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
                    print(f"CASE seed={seed} mode={mode} weight={wi} stream={stream} "
                          f"eval={report['evaluation_interactions']}", flush=True)
    finally:
        env.close()
    report["summary"] = {f"mode{mode}_weight{wi}": summarize_cases([
        case for case in report["cases"] if case["mode"] == mode and case["weight_index"] == wi],
        max(horizons))
        for mode in modes for wi in range(len(weights))}
    report["complete"] = True
    path.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    print(f"DONE {PROTOCOL} seed={seed} eval={report['evaluation_interactions']}", flush=True)
    return report


def summarize(output=OUTPUT):
    reports = [json.loads((output / f"seed{seed}.json").read_text()) for seed in SEEDS]
    assert all((row["protocol"], row["seed"], row["complete"], row["smoke"],
                row["weights"], row["horizons"]) ==
               (PROTOCOL, seed, True, False, WEIGHTS.tolist(), list(HORIZONS))
               for seed, row in zip(SEEDS, reports))
    cells = {f"mode{mode}_weight{wi}": summarize_cases([
        case for report in reports for case in report["cases"]
        if case["mode"] == mode and case["weight_index"] == wi])
        for mode in range(3) for wi in range(len(WEIGHTS))}
    primary = summarize_cases([case for report in reports for case in report["cases"]
                               if case["mode"] == 0 and case["weight_index"] in (1, 2)])
    return {"protocol": PROTOCOL, "seeds": list(SEEDS), "weights": WEIGHTS.tolist(),
            "primary": primary, "cells": cells,
            "per_seed_primary": {str(row["seed"]): summarize_cases([
                case for case in row["cases"]
                if case["mode"] == 0 and case["weight_index"] in (1, 2)])
                for row in reports},
            "completed_budget": {key: sum(row[key] for row in reports)
                                 for key in ("training_interactions", "learner_updates",
                                             "evaluation_interactions",
                                             "environment_initialization_probe_steps")}}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--summarize", action="store_true")
    parser.add_argument("--source", type=Path, default=SOURCE)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    args = parser.parse_args()
    if args.summarize:
        result = summarize(args.output)
        path = args.output / "readout.json"
        path.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
        print(f"DONE {PROTOCOL} summary {path}")
    else:
        assert args.seed is not None
        run(args.seed, args.source, args.output, args.smoke)
