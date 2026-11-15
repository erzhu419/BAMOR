"""Live, paired actor-search update probes. No feedback enters main training."""

import argparse
import copy
import json
import time
from contextlib import contextmanager
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

import train_bamor_mujoco as training


METHOD = "bamor_v27_aligned_target_fixed_archive_mix"
REFERENCE_ROOT = Path("results/iclr_v25_progress_pilot_20260914/mo_halfcheetah_v5")
STEPS = tuple(range(10000, 50001, 5000))
WEIGHTS = np.array([[0., 1.], [.5, .5], [1., 0.]], dtype=np.float32)
FIELDS = ("states", "actions", "weights", "contexts", "rewards", "next_states",
          "next_contexts", "dones")
CONFIG = ("preference_relabeling_rate", "preference_relabeling_strategy",
          "critic_relabel_copies", "envelope_target_candidates",
          "vector_critic_coef", "envelope_policy_candidates", "behavior_clone_coef")


@contextmanager
def isolated_rng(seed):
    numpy_state = np.random.get_state()
    with torch.random.fork_rng(devices=[]):
        np.random.seed(seed)
        torch.manual_seed(seed)
        try:
            yield
        finally:
            np.random.set_state(numpy_state)


def batch_at(buffer, indices):
    return tuple(torch.as_tensor(getattr(buffer, name)[indices], device=buffer.device)
                 for name in FIELDS)


class ReplaySubset:
    """Read-only replay view excluding the diagnostic audit transitions."""

    def __init__(self, source, indices):
        self.source = source
        self.indices = indices
        self.size = len(indices)

    def sample(self, batch_size):
        return batch_at(self.source, self.indices[np.random.randint(self.size, size=batch_size)])


def clone_trainer(trainer, replay):
    # Copy optimizer moments and target networks, but not the large replay arrays.
    clone = copy.deepcopy(trainer, {id(trainer.buffer): replay})
    return clone


def actor_loss(trainer, batch, weight, search, critic=None):
    states, actions, _, contexts, *_ = batch
    weights = torch.as_tensor(weight, device=trainer.device).expand(len(states), -1)
    contexts = trainer.replace_context_preferences(contexts, weights)
    if search and trainer.envelope_policy_candidates > 1:
        action, _, loss = trainer.envelope_actor_loss(states, weights, contexts)
    else:
        critic = trainer if critic is None else critic
        action, log_prob = trainer.policy.get_action(states, contexts)
        q = torch.minimum(critic.q1(states, action, contexts),
                          critic.q2(states, action, contexts))
        loss = (trainer.alpha * log_prob - (weights * q).sum(-1, keepdim=True)).mean()
    return loss + trainer.behavior_clone_coef * F.mse_loss(action, actions)


def cosine(a, b):
    denom = float(a.norm() * b.norm())
    return float(torch.dot(a, b) / denom) if denom > 0 else None


def gradient_signals(trainer, batch, seed, draws=4):
    gradients = {}
    for name, search in [("off", False), ("on", True)]:
        gradients[name] = []
        for wi, weight in enumerate(WEIGHTS):
            samples = []
            for draw in range(draws):
                with isolated_rng(seed + wi * 10 + draw):
                    loss = actor_loss(trainer, batch, weight, search)
                    grad = torch.autograd.grad(loss, tuple(trainer.policy.parameters()))
                    samples.append(torch.cat([g.flatten() for g in grad]))
            gradients[name].append(torch.stack(samples).mean(0))
    own, search = gradients["off"], gradients["on"]
    own_mean, search_mean = torch.stack(own).mean(0), torch.stack(search).mean(0)
    return {
        "own_preference_cosines": [[cosine(a, b) for b in own] for a in own],
        "search_to_own_cosines": [[cosine(a, b) for b in own] for a in search],
        "mean_search_vs_own_cosine": cosine(search_mean, own_mean),
        "mean_search_to_each_own_cosine": [cosine(search_mean, g) for g in own],
        "mean_own_to_each_own_cosine": [cosine(own_mean, g) for g in own],
        "gradient_norms": {k: [float(g.norm()) for g in v] for k, v in gradients.items()},
        "draws": draws,
    }


@torch.no_grad()
def audit_losses(trainer, frozen_critic, batch, seed, draws=4):
    losses = []
    for wi, weight in enumerate(WEIGHTS):
        samples = []
        for draw in range(draws):
            with isolated_rng(seed + wi * 10 + draw):
                samples.append(float(actor_loss(trainer, batch, weight, False, frozen_critic)))
        losses.append(float(np.mean(samples)))
    return losses


@torch.no_grad()
def rollout_returns(trainer, encoder, live_env, reset_seeds, horizons):
    returns = []
    for weight in WEIGHTS:
        repeats = []
        for reset_seed in reset_seeds:
            env = copy.deepcopy(live_env)
            try:
                obs, _ = env.reset(seed=reset_seed)
                trans = training.TransitionContextTracker(2)
                latent = training.LearnedTransitionContextTracker(encoder)
                vector, snapshots = np.zeros(2), []
                for t in range(max(horizons)):
                    context = training.build_mujoco_context(
                        METHOD, weight, trans_tracker=trans, learned_tracker=latent)
                    action = trainer.select_action(
                        torch.as_tensor(obs, dtype=torch.float32).unsqueeze(0),
                        torch.as_tensor(context).unsqueeze(0), deterministic=True)
                    action = action.numpy().ravel()
                    next_obs, reward, term, trunc, _ = env.step(action)
                    vector += trainer.gamma ** t * reward
                    trans.update(obs, action, reward, next_obs)
                    latent.update(obs, action, reward, next_obs, "cpu")
                    obs = next_obs
                    if t + 1 in horizons:
                        snapshots.append(vector.copy().tolist())
                    if term or trunc:
                        raise RuntimeError("HalfCheetah probe ended before its frozen horizon")
                repeats.append(snapshots)
            finally:
                env.close()
        returns.append(repeats)
    return returns


def paired_effects(returns):
    arrays = {k: np.asarray(v) for k, v in returns.items()}
    result = {}
    for name, a, b in [("on_minus_off", "on", "off"),
                       ("on_minus_base", "on", "base"),
                       ("off_minus_base", "off", "base")]:
        delta = arrays[a] - arrays[b]
        scalar = np.einsum("wrhd,wd->wrh", delta, WEIGHTS)
        result[name] = {"scalar_by_weight_repeat_horizon": scalar.tolist(),
                        "mean_scalar_by_weight_horizon": scalar.mean(1).tolist(),
                        "mean_vector_by_weight_horizon": delta.mean(1).tolist()}
    return result


def run_forks(trainer, encoder, env, seed, updates=128, repeats=3,
              horizons=(32, 128, 512)):
    with isolated_rng(seed):
        indices = np.random.permutation(trainer.buffer.size)
        audit_size = min(1024, trainer.buffer.size // 4)
        audit = batch_at(trainer.buffer, indices[:min(audit_size, trainer.batch_size)])
        replay = ReplaySubset(trainer.buffer, indices[audit_size:])
        result = {"audit_pool_size": audit_size, "audit_batch_size": len(audit[0]),
                  "replay_pool_size": replay.size, "updates_per_branch": updates,
                  "config": {k: getattr(trainer, k) for k in CONFIG},
                  "signals_before": gradient_signals(trainer, audit, seed + 100)}
        reset_seeds = [seed + 10000 + rep for rep in range(repeats)]
        result["reset_seeds"] = reset_seeds
        returns = {"base": rollout_returns(trainer, encoder, env, reset_seeds, horizons)}
        losses = {"base": audit_losses(trainer, trainer, audit, seed + 200)}
        for label in ("on", "off"):
            branch = clone_trainer(trainer, replay)
            if label == "off":
                branch.envelope_policy_candidates = 1
            for update in range(updates):
                # Recouple every update: actor RNG consumption cannot shift replay
                # or target-proposal randomness on subsequent gradient steps.
                with isolated_rng(seed + 1000 + update):
                    branch.update()
            losses[label] = audit_losses(branch, trainer, audit, seed + 200)
            returns[label] = rollout_returns(branch, encoder, env, reset_seeds, horizons)
            del branch
        result.update({"returns": returns, "effects": paired_effects(returns),
                       "own_soft_loss_with_bc_frozen_critic": losses,
                       "surrogate_on_advantage_by_weight": (
                           np.array(losses["off"]) - losses["on"]).tolist()})
        return result


class TrainingForkProbe:
    def __init__(self, seed, output, steps=STEPS, updates=128, repeats=3,
                 horizons=(32, 128, 512)):
        self.seed, self.output, self.steps = seed, Path(output), steps
        self.updates, self.repeats, self.horizons = updates, repeats, horizons
        self.report = {
            "protocol": "live_paired_update_forks_v1", "method": METHOD, "seed": seed,
            "steps": list(steps), "weights": WEIGHTS.tolist(), "horizons": list(horizons),
            "torch_threads": torch.get_num_threads(),
            "complete": False, "cases": [],
            "scope": "Frozen replay and encoder; contemporaneous training environment copies, "
                     "active mode schedule, fresh resets, deterministic actors, discounted raw "
                     "returns without terminal bootstrap. Extra diagnostic interactions only; "
                     "no main-training feedback or locked-front evaluations.",
        }

    def __call__(self, step, trainer, encoder, env, operator):
        if step not in self.steps:
            return False
        started = time.monotonic()
        case = run_forks(trainer, encoder, env, 10000000 + self.seed * 100000 + step * 100,
                         self.updates, self.repeats, self.horizons)
        case.update({"step": step, "operator": operator,
                     "seconds": time.monotonic() - started,
                     "extra_environment_steps": 3 * len(WEIGHTS) * self.repeats * max(self.horizons)})
        self.report["cases"].append(case)
        self.report["complete"] = step == self.steps[-1]
        self.output.parent.mkdir(parents=True, exist_ok=True)
        self.output.write_text(json.dumps(self.report, indent=2) + "\n")
        print(f"FORK step={step} actor_candidates={trainer.envelope_policy_candidates} "
              f"seconds={case['seconds']:.1f}", flush=True)
        return self.report["complete"]


def reproduction_check(result, reference):
    actual, expected = result["episode_rewards"], reference["episode_rewards"]
    aligned = ([x["step"] for x in actual] == [x["step"] for x in expected]
               and np.array_equal(result["ground_truth"]["mode_history"],
                                  reference["ground_truth"]["mode_history"]))
    error = (float(np.max(np.abs(np.array([x["return"] for x in actual])
                                - np.array([x["return"] for x in expected]))))
             if aligned else None)
    return {"matched_steps_and_schedule": aligned, "max_episode_return_error": error,
            "matched": aligned and error <= 1e-6}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    reference = json.loads((REFERENCE_ROOT / f"{METHOD}_seed{args.seed}.json").read_text())
    probe = TrainingForkProbe(args.seed, args.output)
    result = training.train_method(
        "mo-halfcheetah-v5", METHOD, 50000, 25000, args.seed,
        str(args.output.parent), hidden=64, eval_freq=100000,
        device="cpu", training_probe=probe)
    probe.report["base_reproduction"] = reproduction_check(result, reference)
    probe.report["complete"] = probe.report["base_reproduction"]["matched"]
    args.output.write_text(json.dumps(probe.report, indent=2) + "\n")
    if not probe.report["complete"]:
        raise RuntimeError("Main trajectory did not reproduce the existing v27 control")


if __name__ == "__main__":
    main()
