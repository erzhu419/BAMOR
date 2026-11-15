"""Frozen mean-output Q-gradient versus one-action raw-return interventions."""

import argparse
import copy
import json
from pathlib import Path

import numpy as np
import torch

import run_mean_value_pilot as pilot


training = pilot.training
PROTOCOL = "mean_action_gradient_v1"
METHODS = {"v27": pilot.BASE, "v28": pilot.METHOD}
SOURCE_ROOTS = {"v27": pilot.REFERENCE_ROOT, "v28": pilot.OUTPUT_ROOT}
OUTPUT_ROOT = Path("results/iclr_mean_action_gradient_20260921")
SEEDS = (31, 33)
WEIGHTS = np.array([[0., 1.], [.5, .5], [1., 0.]], dtype=np.float32)
WARMUP = 128
HORIZONS = (32, 128, 512)
EPSILON = .05
GAMMA = .99
REPEATS = 3
BRANCHES = ("base", "plus", "minus", "repeat")


def reset_seed(seed, mode, repeat):
    return 41000000 + seed * 1000 + mode * 10 + repeat


def load_models(condition, seed):
    method = METHODS[condition]
    source = SOURCE_ROOTS[condition] / f"{method}_seed{seed}"
    reference = json.loads(source.with_suffix(".json").read_text())
    assert (reference["method"], reference["seed"], reference["total_steps"]) == (method, seed, 50000)
    if condition == "v28":
        assert reference["mean_value_pilot"]["complete"]
        assert reference["mean_value_pilot"]["protocol"] == pilot.PROTOCOL
    else:
        assert reference["replay_context_confirmation"]["complete"]
        assert reference["replay_context_confirmation"]["condition"] == "stored"
    path = Path(str(source) + "_model.pt")
    states = torch.load(path, map_location="cpu", weights_only=True)
    last = states["policy_archive"][-1]
    assert last["step"] == 50000
    assert all(torch.equal(v, last["policy"][k]) for k, v in states["policy"].items())
    networks = {"policy": training.PolicyNet(17, 6, 13, hidden=64),
                "encoder": training.LatentTransitionEncoder(17, 6, 2, latent_dim=8, hidden=64),
                "q1": training.SoftQNet(17, 6, 13, 2, hidden=64),
                "q2": training.SoftQNet(17, 6, 13, 2, hidden=64)}
    for name, network in networks.items():
        network.load_state_dict(states[name])
        network.eval().requires_grad_(False)
    return networks, path


def context(method, weight, trackers):
    return training.build_mujoco_context(method, weight, trans_tracker=trackers[0],
                                        learned_tracker=trackers[1])


def scalar_q(networks, obs, action, ctx, weight):
    return (weight * torch.minimum(networks["q1"](obs, action, ctx),
                                   networks["q2"](obs, action, ctx))).sum()


def mean_directions(networks, obs, ctx, weight, epsilon=EPSILON):
    s, c, w = [torch.as_tensor(x, dtype=torch.float32).reshape(1, -1) for x in (obs, ctx, weight)]
    with torch.no_grad():
        mean, _ = networks["policy"](s, c)
    z = mean.detach().requires_grad_(True)
    value = scalar_q(networks, s, z.tanh(), c, w)
    gradient, = torch.autograd.grad(value, z)
    norm = gradient.norm()
    assert torch.isfinite(value) and torch.isfinite(gradient).all()
    direction = gradient / norm if norm > 0 else torch.zeros_like(gradient)
    with torch.no_grad():
        actions = {"base": z.tanh(), "plus": (z + epsilon * direction).tanh(),
                   "minus": (z - epsilon * direction).tanh()}
        qs = {name: float(scalar_q(networks, s, action, c, w)) for name, action in actions.items()}
    actions = {name: action.numpy().ravel() for name, action in actions.items()}
    return actions, {"mean_output": mean.detach().numpy().ravel().tolist(),
                     "gradient": gradient.numpy().ravel().tolist(), "gradient_norm": float(norm),
                     "actions": {k: a.tolist() for k, a in actions.items()}, "q_scalar": qs,
                     "linear_plus_minus_prediction": 2 * epsilon * float(norm),
                     "action_delta_l2": {k: float(np.linalg.norm(actions[k] - actions["base"]))
                                         for k in ("plus", "minus")}}


@torch.no_grad()
def rollout(env, networks, method, weight, obs, trackers, steps, horizons=(), first_action=None):
    discounted, raw = np.zeros(2), np.zeros(2)
    result = {"discounted_returns": {}, "raw_returns": {}}
    before = env._total_steps
    for t in range(steps):
        if t == 0 and first_action is not None:
            action = first_action
        else:
            c = context(method, weight, trackers)
            action = networks["policy"].get_action(
                torch.as_tensor(obs, dtype=torch.float32)[None], torch.as_tensor(c)[None],
                deterministic=True).numpy().ravel()
        next_obs, reward, term, trunc, _ = env.step(action)
        # Convert before discounting: preserve reward precision across horizons.
        reward64 = np.asarray(reward, dtype=np.float64)
        discounted += GAMMA ** t * reward64
        raw += reward64
        trackers[0].update(obs, action, reward, next_obs)
        trackers[1].update(obs, action, reward, next_obs, "cpu")
        obs = next_obs
        if t + 1 in horizons:
            result["discounted_returns"][str(t + 1)] = discounted.tolist()
            result["raw_returns"][str(t + 1)] = raw.tolist()
        assert not (term or trunc), "Frozen HalfCheetah horizon must stay inside the episode"
    result["steps"] = env._total_steps - before
    return obs, result


def fork_env(env):
    branch = copy.deepcopy(env)
    # Gym's pickle reconstruction resets MuJoCo physics and integration state.
    branch._apply_mode(env.current_mode_idx)
    original, copied = env.unwrapped, branch.unwrapped
    # EzPickle also drops the RNG used by subsequent reset(seed=None) calls.
    copied._np_random = copy.deepcopy(original._np_random)
    copied._np_random_seed = original._np_random_seed
    # Preserve cached fields too: Ant's next reward reads pre-step body.xpos,
    # which can lag qpos. Recomputing it with mj_forward changes that reward.
    copied.data = copy.copy(original.data)
    return branch


def run_case(env, networks, method, weight, seed, warmup=WARMUP, horizons=HORIZONS, epsilon=EPSILON):
    before = env._total_steps
    obs, _ = env.reset(seed=seed)
    trackers = (training.TransitionContextTracker(2),
                training.LearnedTransitionContextTracker(networks["encoder"]))
    obs, _ = rollout(env, networks, method, weight, obs, trackers, warmup)
    ctx = context(method, weight, trackers)
    actions, diagnostic = mean_directions(networks, obs, ctx, weight, epsilon)
    branches, final_obs = {}, {}
    # Run the direct, uncopied identity last, after all same-state forks.
    for name in BRANCHES:
        branch = env if name == "repeat" else fork_env(env)
        copied = copy.deepcopy(trackers, {id(networks["encoder"]): networks["encoder"]})
        try:
            final_obs[name], branches[name] = rollout(
                branch, networks, method, weight, obs.copy(), copied, max(horizons), horizons,
                first_action=actions[name] if name in ("plus", "minus") else None)
        finally:
            if name != "repeat":
                branch.close()
    assert branches["base"] == branches["repeat"], "Identity continuation did not reproduce"
    np.testing.assert_array_equal(final_obs["base"], final_obs["repeat"])
    assert env._total_steps - before == warmup + max(horizons)
    return {"reset_seed": seed, "weight": weight.tolist(),
            "anchor_observation": obs.tolist(), "anchor_context": ctx.tolist(),
            "gradient_diagnostic": diagnostic, "branches": branches,
            "identity_matched": True, "warmup_steps": warmup,
            "evaluation_interactions": warmup + sum(b["steps"] for b in branches.values())}


def summarize_cases(cases, horizons=HORIZONS, repeats=REPEATS):
    cells = []
    groups = sorted({(c["mode"], c["weight_index"]) for c in cases})
    for mode, wi in groups:
        group = [c for c in cases if (c["mode"], c["weight_index"]) == (mode, wi)]
        assert len(group) == repeats and len({c["reset_seed"] for c in group}) == repeats
        q_delta = [c["gradient_diagnostic"]["q_scalar"]["plus"] -
                   c["gradient_diagnostic"]["q_scalar"]["minus"] for c in group]
        row = {"mode": mode, "weight_index": wi, "q_plus_minus_mean": float(np.mean(q_delta)),
               "gradient_norm_mean": float(np.mean([c["gradient_diagnostic"]["gradient_norm"] for c in group])),
               "horizons": {}}
        for h in horizons:
            contrasts = {}
            for label, a, b in (("plus_minus", "plus", "minus"), ("plus_base", "plus", "base")):
                values = [float(np.dot(c["weight"], np.asarray(c["branches"][a]["discounted_returns"][str(h)]) -
                                       c["branches"][b]["discounted_returns"][str(h)])) for c in group]
                contrasts[label] = {"per_reset": values, "mean": float(np.mean(values))}
            row["horizons"][str(h)] = contrasts
        cells.append(row)
    h = str(max(horizons))
    positive_q = [c for c in cells if c["q_plus_minus_mean"] > 0]
    return {"cells": cells, "primary_horizon": max(horizons),
            "q_ascent_cells": len(positive_q), "total_cells": len(cells),
            "return_positive_among_q_ascent": sum(c["horizons"][h]["plus_minus"]["mean"] > 0 for c in positive_q),
            "return_zero_among_q_ascent": sum(c["horizons"][h]["plus_minus"]["mean"] == 0 for c in positive_q),
            "mean_plus_minus_return": float(np.mean([c["horizons"][h]["plus_minus"]["mean"] for c in cells])),
            "mean_plus_base_return": float(np.mean([c["horizons"][h]["plus_base"]["mean"] for c in cells]))}


def run(condition, seed, output_root=OUTPUT_ROOT):
    assert seed in SEEDS
    networks, source = load_models(condition, seed)
    method = METHODS[condition]
    path = Path(output_root) / condition / f"seed{seed}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    report = {"protocol": PROTOCOL, "condition": condition, "method": method, "seed": seed,
              "source": str(source), "complete": False, "weights": WEIGHTS.tolist(),
              "warmup_steps": WARMUP, "horizons": list(HORIZONS), "epsilon": EPSILON,
              "perturbation": "L2-normalized gradient in pre-tanh mean-output space, one action only",
              "gamma": GAMMA, "repeats": REPEATS, "training_interactions": 0, "extra_updates": 0,
              "evaluation_interactions": 0, "environment_initialization_probe_steps": 1,
              "scope": "Observed development checkpoints and deterministic visited states. "
                       "Q is the frozen training soft critic, not a raw-return evaluator. "
                       "Raw finite-horizon continuations have no entropy or bootstrap. "
                       "One-action mean-output sensitivity is not a parameter update or a causal "
                       "explanation of training. State pairing is within each model, not across models. "
                       "Modes only configure evaluation physics. "
                       "No fitting, routing feedback, coefficient sweep or performance gate.", "cases": []}

    def save():
        path.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")

    save()
    env = training.make_piecewise_mo_env("mo-halfcheetah-v5", switch_interval=25000, seed=seed + 1000)
    env._lock_mode = True
    print(f"MEAN_GRADIENT condition={condition} seed={seed} training=0", flush=True)
    try:
        for mode in range(env.num_modes):
            env._apply_mode(mode)
            for wi, weight in enumerate(WEIGHTS):
                for repeat in range(REPEATS):
                    case = run_case(env, networks, method, weight, reset_seed(seed, mode, repeat),
                                    WARMUP, HORIZONS, EPSILON)
                    case.update(mode=mode, weight_index=wi, repeat=repeat)
                    report["cases"].append(case)
                    report["evaluation_interactions"] += case["evaluation_interactions"]
                save()
                print(f"GRADIENT_CASE mode={mode} weight={wi} cases={len(report['cases'])} "
                      f"evaluation={report['evaluation_interactions']}", flush=True)
        assert env._total_steps == len(report["cases"]) * (WARMUP + max(HORIZONS))
    finally:
        env.close()
    report["summary"] = summarize_cases(report["cases"], HORIZONS, REPEATS)
    assert report["evaluation_interactions"] == len(report["cases"]) * (WARMUP + 4 * max(HORIZONS))
    report["complete"] = True
    save()
    print(f"DONE mean_action_gradient condition={condition} seed={seed} "
          f"evaluation={report['evaluation_interactions']}", flush=True)
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--condition", choices=METHODS, required=True)
    parser.add_argument("--seed", type=int, choices=SEEDS, required=True)
    parser.add_argument("--output-root", type=Path, default=OUTPUT_ROOT)
    args = parser.parse_args()
    run(args.condition, args.seed, args.output_root)
