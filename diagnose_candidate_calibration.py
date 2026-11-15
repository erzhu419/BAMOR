"""Paired action probes of frozen HalfCheetah models; no training or selection."""

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch

import train_bamor_mujoco as training


METHODS = ("bamor_v24_target_consistent_fixed_archive_mix",
           "bamor_v26_aligned_actor_fixed_archive_mix")
ROOT = Path("results/iclr_v25_progress_pilot_20260914/mo_halfcheetah_v5")
OUTPUT = Path("results/iclr_candidate_calibration_20260919")
HORIZONS = (1, 32, 128, 512)
CONTINUATIONS = ("own", "frozen_online_envelope")
LABELS = ("own", "cross_preference_best", "own_resample_best")


class FrozenModel:
    def __init__(self, checkpoint):
        obs_dim, act_dim, reward_dim = 17, 6, 2
        context_dim = training.context_dim_for_method(METHODS[0], obs_dim, act_dim,
                                                    reward_dim, 4)
        hidden = checkpoint["policy"]["trunk.0.weight"].shape[0]
        self.trainer = training.ConsistentTargetMOSACTrainer(
            obs_dim, act_dim, reward_dim, hidden=hidden, context_dim=context_dim,
            buffer_capacity=1, device="cpu")
        self.encoder = training.LatentTransitionEncoder(
            obs_dim, act_dim, reward_dim, hidden=hidden)
        self.encoder.load_state_dict(checkpoint["encoder"])
        for name in ("policy", "q1", "q2"):
            module = getattr(self.trainer, name)
            module.load_state_dict(checkpoint[name])
            module.eval().requires_grad_(False)
        self.encoder.eval().requires_grad_(False)

    def trackers(self, history=()):
        trackers = (training.TransitionContextTracker(2),
                    training.LearnedTransitionContextTracker(self.encoder))
        for transition in history:
            self.observe(trackers, *transition)
        return trackers

    @staticmethod
    def observe(trackers, obs, action, reward, next_obs):
        trackers[0].update(obs, action, reward, next_obs)
        trackers[1].update(obs, action, reward, next_obs, "cpu")

    @staticmethod
    def context(weight, trackers):
        return torch.from_numpy(training.build_mujoco_context(
            METHODS[0], weight, trans_tracker=trackers[0],
            learned_tracker=trackers[1])).unsqueeze(0)


def load_model(root, method, seed):
    return FrozenModel(torch.load(root / f"{method}_seed{seed}_model.pt",
                                  map_location="cpu", weights_only=True))


def reset_locked(env, mode, seed):
    env._lock_mode = True
    env._apply_mode(mode)
    env._total_steps = env._steps_in_mode = 0
    env.mode_history = [(0, mode)]
    return env.reset(seed=seed)[0]


@torch.no_grad()
def collect_prefix(env, model, mode, reset_seed, weight, steps=128):
    obs = reset_locked(env, mode, reset_seed)
    trackers, history = model.trackers(), []
    for _ in range(steps):
        context = model.context(weight, trackers)
        action = model.trainer.policy.get_action(
            torch.as_tensor(obs, dtype=torch.float32).unsqueeze(0), context,
            deterministic=True).numpy().ravel()
        next_obs, reward, term, trunc, _ = env.step(action)
        if term or trunc:
            raise RuntimeError("Diagnostic prefix ended before its frozen length")
        history.append((obs.copy(), action.copy(), reward.copy(), next_obs.copy()))
        model.observe(trackers, obs, action, reward, next_obs)
        obs = next_obs
    return history


def replay_prefix(env, mode, reset_seed, history):
    obs = reset_locked(env, mode, reset_seed)
    for _, action, _, expected_obs in history:
        obs, _, term, trunc, _ = env.step(action)
        if term or trunc:
            raise RuntimeError("Paired prefix unexpectedly terminated")
    # Replaying actions also reconstructs simulator warm-start and wrapper state.
    np.testing.assert_allclose(obs, expected_obs, rtol=0, atol=1e-10)
    return obs


def candidate_contexts(trainer, context, count=5):
    return torch.cat([context] + [trainer.replace_context_preferences(
        context, trainer.sample_preference_batch(1, "mixed"))
        for _ in range(count - 1)])


def scores(trainer, state, actions, contexts, weight):
    states = state.expand(len(actions), -1)
    return (torch.minimum(trainer.q1(states, actions, contexts),
                          trainer.q2(states, actions, contexts)) * weight).sum(-1)


@torch.no_grad()
def probe_actions(model, obs, weight, trackers, random_seed):
    trainer = model.trainer
    state = torch.as_tensor(obs, dtype=torch.float32).unsqueeze(0)
    context = model.context(weight, trackers)
    with torch.random.fork_rng():
        torch.manual_seed(random_seed)
        contexts = candidate_contexts(trainer, context)
        draw_state = torch.get_rng_state()
        actions, _ = trainer.policy.get_action(state.expand(5, -1), contexts)
        values = scores(trainer, state, actions, context.expand(5, -1),
                        torch.as_tensor(weight))
        # Same Gaussian draws: isolate preference changes from extra sampling.
        torch.set_rng_state(draw_state)
        own_actions, _ = trainer.policy.get_action(state.expand(5, -1),
                                                  context.expand(5, -1))
        own_values = scores(trainer, state, own_actions, context.expand(5, -1),
                            torch.as_tensor(weight))
    best, own_best = int(values.argmax()), int(own_values.argmax())
    return (np.stack([actions[0].numpy(), actions[best].numpy(),
                      own_actions[own_best].numpy()]),
            np.array([float(values[0]), float(values[best]), float(own_values[own_best])]),
            {"candidate_preferences": contexts[:, :2].tolist(),
             "candidate_q": values.tolist(), "own_resample_q": own_values.tolist(),
             "cross_index": best, "own_resample_index": own_best})


@torch.no_grad()
def continuation_action(model, obs, weight, trackers, rule):
    trainer = model.trainer
    state = torch.as_tensor(obs, dtype=torch.float32).unsqueeze(0)
    context = model.context(weight, trackers)
    if rule == "own":
        action, log_prob = trainer.policy.get_action(state, context)
        return action.numpy().ravel(), -trainer.alpha * float(log_prob.item())
    if rule != "frozen_online_envelope":
        raise ValueError(rule)
    contexts = candidate_contexts(trainer, context)
    actions, log_probs = trainer.policy.get_action(state.expand(5, -1), contexts)
    values = scores(trainer, state, actions, contexts, torch.as_tensor(weight))
    best = int((values - trainer.alpha * log_probs.ravel()).argmax())
    return actions[best].numpy(), -trainer.alpha * float(log_probs[best].item())


@torch.no_grad()
def rollout_branch(env, model, mode, reset_seed, history, weight, first_action,
                   rule, random_seed, horizons=HORIZONS):
    obs = replay_prefix(env, mode, reset_seed, history)
    trackers = model.trackers(history)
    vector, bonus, result = np.zeros(2), 0.0, {}
    with torch.random.fork_rng():
        torch.manual_seed(random_seed)
        for step in range(max(horizons)):
            # Q excludes entropy at the forced first action, includes it thereafter.
            action, entropy = ((first_action, 0.0) if step == 0 else
                               continuation_action(model, obs, weight, trackers, rule))
            next_obs, reward, term, trunc, _ = env.step(action)
            discount = model.trainer.gamma ** step
            vector += discount * reward
            bonus += discount * entropy
            model.observe(trackers, obs, action, reward, next_obs)
            obs = next_obs
            if step + 1 in horizons:
                result[str(step + 1)] = {"raw_vector": vector.tolist(),
                                       "raw": float(vector @ weight),
                                       "bonus_adjusted": float(vector @ weight + bonus)}
            if term or trunc:
                raise RuntimeError("HalfCheetah branch ended before diagnostic horizon")
    return result


def paired_statistics(q_values, returns):
    # returns: [action, repeat, metric], with paired repeats, not independent seeds.
    gains = returns[1:] - returns[:1]
    return {"q_gain": (q_values[1:] - q_values[0]).tolist(),
            "mean_gain": gains.mean(axis=1).tolist(),
            "paired_gains": gains.tolist()}


def summarize(cases, horizons=HORIZONS):
    result = {}
    for rule in CONTINUATIONS:
        result[rule] = {}
        for horizon in horizons:
            entries = [case["effects"][rule][str(horizon)] for case in cases]
            q = np.asarray([entry["q_gain"] for entry in entries])
            gains = np.asarray([entry["mean_gain"] for entry in entries])
            result[rule][str(horizon)] = {}
            for index, label in enumerate(LABELS[1:]):
                active = q[:, index] > 0
                result[rule][str(horizon)][label] = {
                    "states": len(cases), "q_positive_states": int(active.sum()),
                    "mean_q_gain": float(q[:, index].mean()),
                    "raw_gain": float(gains[:, index, 0].mean()),
                    "bonus_adjusted_gain": float(gains[:, index, 1].mean()),
                    "raw_positive_fraction_given_q_positive": (
                        float((gains[active, index, 0] > 0).mean()) if active.any() else None),
                    "bonus_positive_fraction_given_q_positive": (
                        float((gains[active, index, 1] > 0).mean()) if active.any() else None),
                }
    return result


def run(args):
    torch.set_num_threads(1)
    started = time.monotonic()
    model = load_model(args.root, args.method, args.seed)
    reference = model if args.method == METHODS[0] else load_model(args.root, METHODS[0], args.seed)
    env = training.make_piecewise_mo_env("mo-halfcheetah-v5", seed=args.seed)
    cases = []
    horizons = (1, 4) if args.smoke else HORIZONS
    repeats = 2 if args.smoke else 4
    prefix_steps = 8 if args.smoke else 128
    try:
        for mode in range(1 if args.smoke else 4):
            for wi, weight in enumerate(np.array([[0., 1.], [.5, .5], [1., 0.]], dtype=np.float32)):
                for stream in range(1 if args.smoke else 2):
                    # Separate from locked-front test seeds and reserved training seeds.
                    reset_seed = 800000 + args.seed * 1000 + mode * 10 + stream
                    random_seed = 900000 + args.seed * 10000 + mode * 1000 + wi * 100 + stream * 10
                    history = collect_prefix(env, reference, mode, reset_seed, weight, prefix_steps)
                    actions, q_values, candidate = probe_actions(
                        model, history[-1][3], weight, model.trackers(history), random_seed)
                    case = {"mode": mode, "weight": weight.tolist(), "reset_seed": reset_seed,
                            "random_seed": random_seed, "q_values": q_values.tolist(),
                            **candidate, "effects": {}}
                    for rule in CONTINUATIONS:
                        runs = [[rollout_branch(env, model, mode, reset_seed, history, weight,
                                               action, rule, random_seed + rep + 1, horizons)
                                 for rep in range(repeats)] for action in actions]
                        case["effects"][rule] = {}
                        for horizon in horizons:
                            returns = np.array([[[rep[str(horizon)][metric]
                                                  for metric in ("raw", "bonus_adjusted")]
                                                 for rep in branch] for branch in runs])
                            case["effects"][rule][str(horizon)] = paired_statistics(q_values, returns)
                    cases.append(case)
                    print(f"case {len(cases)} mode={mode} w={weight.tolist()} stream={stream} "
                          f"seconds={time.monotonic() - started:.1f}", flush=True)
    finally:
        env.close()
    report = {"protocol": "paired_candidate_calibration_v1", "method": args.method,
              "seed": args.seed, "smoke": args.smoke, "prefix_reference": METHODS[0],
              "prefix_steps": prefix_steps, "horizons": list(horizons), "repeats": repeats,
              "candidate_count": 5, "candidate_strategy": "mixed", "gamma": .99, "alpha": .2,
              "limitations": ["Frozen final models and reference-policy states, not training replay.",
                              "No terminal critic bootstrap; finite-horizon ranking diagnostic only.",
                              "Envelope uses online critics; historical target critics were not saved.",
                              "Envelope component log-probability bonus is not selected-policy entropy.",
                              "Fixed envelope recipe is not the changing training operator portfolio.",
                              "Repeated states/rollouts are not independent training seeds."],
              "summary": summarize(cases, horizons),
              "per_mode": {str(m): summarize([c for c in cases if c["mode"] == m], horizons)
                           for m in sorted({c["mode"] for c in cases})},
              "cases": cases, "seconds": time.monotonic() - started}
    args.output.mkdir(parents=True, exist_ok=True)
    path = args.output / f"{args.method}_seed{args.seed}{'_smoke' if args.smoke else ''}.json"
    path.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    print(json.dumps({"output": str(path), "states": len(cases), "seconds": report["seconds"]}), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    parser.add_argument("--method", choices=METHODS, required=True)
    parser.add_argument("--seed", choices=(26, 27, 28), type=int, required=True)
    parser.add_argument("--smoke", action="store_true", help="Short runtime preflight, not evidence")
    run(parser.parse_args())
