"""Reconstruct live forks, decompose their losses, and reuse frozen rollout labels."""

import argparse
import copy
import json
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

import diagnose_update_forks as forks
import train_bamor_mujoco as training


CACHED_ROOT = Path("results/iclr_update_forks_20260919")
COMPONENTS = ("negative_q", "entropy_loss", "bc_loss", "total")


def scalar_q(critic, states, actions, contexts, weights):
    vector = torch.minimum(critic.q1(states, actions, contexts),
                           critic.q2(states, actions, contexts))
    return (weights * vector).sum(-1, keepdim=True)


@torch.no_grad()
def audit_components(trainer, critic, batch, seed, draws=4):
    states, actions, _, contexts, *_ = batch
    result = {k: [] for k in COMPONENTS}
    result["deterministic_replay_q"] = []
    for wi, weight in enumerate(forks.WEIGHTS):
        weights = torch.as_tensor(weight, device=trainer.device).expand(len(states), -1)
        context = trainer.replace_context_preferences(contexts, weights)
        samples = {k: [] for k in COMPONENTS}
        for draw in range(draws):
            with forks.isolated_rng(seed + wi * 10 + draw):
                action, log_prob = trainer.policy.get_action(states, context)
                q = scalar_q(critic, states, action, context, weights)
                entropy = trainer.alpha * log_prob
                bc = trainer.behavior_clone_coef * F.mse_loss(action, actions)
                values = {"negative_q": -q.mean(), "entropy_loss": entropy.mean(),
                          "bc_loss": bc, "total": (entropy - q).mean() + bc}
                for key, value in values.items():
                    samples[key].append(float(value))
        for key in COMPONENTS:
            result[key].append(float(np.mean(samples[key])))
        action = trainer.select_action(states, context, deterministic=True)
        result["deterministic_replay_q"].append(
            float(scalar_q(critic, states, action, context, weights).mean()))
    np.testing.assert_allclose(
        np.array(result["negative_q"]) + result["entropy_loss"] + result["bc_loss"],
        result["total"], rtol=2e-6, atol=2e-6)
    return result


def reset_inputs(encoder, live_env, seeds):
    # Only reset the copied simulator; never step it or expose mode labels.
    env = copy.deepcopy(live_env)
    try:
        states = torch.as_tensor(np.stack([env.reset(seed=s)[0] for s in seeds]),
                                 dtype=torch.float32)
    finally:
        env.close()
    contexts = []
    for weight in forks.WEIGHTS:
        context = training.build_mujoco_context(
            forks.METHOD, weight, trans_tracker=training.TransitionContextTracker(2),
            learned_tracker=training.LearnedTransitionContextTracker(encoder))
        contexts.append(torch.as_tensor(context).unsqueeze(0).expand(len(seeds), -1))
    return states, contexts


@torch.no_grad()
def reset_action_q(trainer, critic, inputs):
    states, contexts = inputs
    result = []
    for weight, context in zip(forks.WEIGHTS, contexts):
        weights = torch.as_tensor(weight).unsqueeze(0)
        # Match the original rollout's batch-one first-action evaluation.
        result.append([float(scalar_q(
            critic, states[i:i + 1], trainer.select_action(
                states[i:i + 1], context[i:i + 1], deterministic=True),
            context[i:i + 1], weights).item()) for i in range(len(states))])
    return result


def advantages(measurements):
    on, off = measurements["on"], measurements["off"]
    result = {key: (np.array(off[key]) - on[key]).tolist() for key in COMPONENTS}
    result["deterministic_replay_q"] = (
        np.array(on["deterministic_replay_q"]) - off["deterministic_replay_q"]).tolist()
    reset_delta = np.array(on["deterministic_reset_q"]) - off["deterministic_reset_q"]
    result["deterministic_reset_q"] = reset_delta.mean(1).tolist()
    result["deterministic_reset_q_paired"] = reset_delta.tolist()
    return result


def reconstruct(trainer, encoder, env, seed, cached):
    config = {k: getattr(trainer, k) for k in forks.CONFIG}
    assert config == cached["config"], "Fork operator configuration changed"
    with forks.isolated_rng(seed):
        indices = np.random.permutation(trainer.buffer.size)
        audit_size = min(1024, trainer.buffer.size // 4)
        audit = forks.batch_at(trainer.buffer, indices[:min(audit_size, trainer.batch_size)])
        replay = forks.ReplaySubset(trainer.buffer, indices[audit_size:])
        assert (audit_size, len(audit[0]), replay.size) == (
            cached["audit_pool_size"], cached["audit_batch_size"], cached["replay_pool_size"])
        inputs = reset_inputs(encoder, env, cached["reset_seeds"])
        measurements = {}
        for label in ("base", "on", "off"):
            branch = trainer if label == "base" else forks.clone_trainer(trainer, replay)
            if label == "off":
                branch.envelope_policy_candidates = 1
            if label != "base":
                for update in range(cached["updates_per_branch"]):
                    with forks.isolated_rng(seed + 1000 + update):
                        branch.update()
            measured = audit_components(branch, trainer, audit, seed + 200)
            np.testing.assert_array_equal(
                measured["total"], cached["own_soft_loss_with_bc_frozen_critic"][label],
                err_msg=f"Reconstructed {label} fork does not match frozen rollout labels")
            measured["deterministic_reset_q"] = reset_action_q(branch, trainer, inputs)
            measurements[label] = measured
            del branch
        deltas = advantages(measurements)
        np.testing.assert_array_equal(deltas["total"], cached["surrogate_on_advantage_by_weight"])
        if config["envelope_policy_candidates"] == 1:
            assert measurements["on"] == measurements["off"], "Identity control diverged"
        return {"config": config, "matched_original_losses": True,
                "measurements": measurements, "on_advantages": deltas,
                "cached_return_effects": cached["effects"],
                "extra_environment_steps": 0, "diagnostic_resets": len(cached["reset_seeds"])}


class LossComponentProbe:
    def __init__(self, seed, output, cached):
        if (cached["protocol"] != "live_paired_update_forks_v1"
                or cached["method"] != forks.METHOD or cached["seed"] != seed
                or not cached["complete"] or not cached["base_reproduction"]["matched"]):
            raise ValueError("Need a completed, reproduced fork run from the same seed")
        assert cached["steps"] == [c["step"] for c in cached["cases"]]
        self.seed, self.output = seed, Path(output)
        self.cases = {c["step"]: c for c in cached["cases"]}
        self.report = {
            "protocol": "fork_loss_components_v1", "method": forks.METHOD, "seed": seed,
            "steps": cached["steps"], "weights": cached["weights"],
            "horizons": cached["horizons"], "torch_threads": torch.get_num_threads(),
            "complete": False, "cases": [],
            "scope": "All values use the pre-fork soft critics. Component losses use the "
                     "original held-out replay and four paired action draws. Deterministic "
                     "Q uses replay and the original rollout reset states separately. "
                     "Rollout labels are reused, not new confirmation data; no selector feedback.",
        }

    def __call__(self, step, trainer, encoder, env, operator):
        if step not in self.cases:
            return False
        cached = self.cases[step]
        assert operator == cached["operator"]
        started = time.monotonic()
        case = reconstruct(trainer, encoder, env,
                           10000000 + self.seed * 100000 + step * 100, cached)
        case.update({"step": step, "operator": operator, "seconds": time.monotonic() - started})
        self.report["cases"].append(case)
        self.output.parent.mkdir(parents=True, exist_ok=True)
        self.output.write_text(json.dumps(self.report, indent=2, allow_nan=False) + "\n")
        print(f"COMPONENTS step={step} matched_losses=True seconds={case['seconds']:.1f}", flush=True)
        return step == self.report["steps"][-1]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    cached_path = CACHED_ROOT / f"seed{args.seed}.json"
    cached = json.loads(cached_path.read_text())
    assert cached["steps"] == list(forks.STEPS)
    assert all(c["updates_per_branch"] == 128 for c in cached["cases"])
    reference = json.loads((forks.REFERENCE_ROOT / f"{forks.METHOD}_seed{args.seed}.json").read_text())
    probe = LossComponentProbe(args.seed, args.output, cached)
    probe.report["cached_source"] = str(cached_path)
    result = training.train_method(
        "mo-halfcheetah-v5", forks.METHOD, 50000, 25000, args.seed,
        str(args.output.parent), hidden=64, eval_freq=100000, device="cpu", training_probe=probe)
    probe.report["base_reproduction"] = forks.reproduction_check(result, reference)
    probe.report["complete"] = probe.report["base_reproduction"]["matched"]
    args.output.write_text(json.dumps(probe.report, indent=2, allow_nan=False) + "\n")
    if not probe.report["complete"]:
        raise RuntimeError("Main trajectory did not reproduce the existing v27 control")


if __name__ == "__main__":
    main()
