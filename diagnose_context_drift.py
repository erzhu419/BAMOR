"""Read-only replay-context diagnostics on reproduced v27 training trajectories."""

import argparse
import json
from pathlib import Path
import time

import numpy as np
import torch

import diagnose_update_forks as forks
import train_bamor_mujoco as training


STEPS = (10000, 20000, 30000, 35000, 40000, 42500, 45000, 47500, 50000)
COHORT_WIDTH = 10000
SAMPLES = 256
PROTOCOL = "causal_replay_context_drift_v1"


@torch.no_grad()
def reencode_history(buffer, encoder, chunk_size=512):
    """Entry i is the latent before transition i; entry i+1 is its successor."""
    n = buffer.size
    assert 0 < n < buffer.capacity and buffer.ptr == n, "Diagnostic needs the unwrapped training prefix"
    np.testing.assert_array_equal(buffer.timesteps[:n, 0], np.arange(1, n + 1))
    tracker = training.LearnedTransitionContextTracker(encoder, latent_dim=encoder.latent_dim)
    latent = np.zeros((n + 1, encoder.latent_dim), dtype=np.float32)
    was_training = encoder.training
    encoder.eval()
    try:
        for start in range(0, n, chunk_size):
            end = min(n, start + chunk_size)
            tensors = [torch.as_tensor(x, device=buffer.device) for x in (
                buffer.states[start:end], buffer.actions[start:end],
                np.tanh(buffer.rewards[start:end] / tracker.reward_scale), buffer.next_states[start:end])]
            encoded = encoder.encode(*tensors).cpu().numpy()
            for offset, z in enumerate(encoded):
                i = start + offset
                # v27 keeps the tracker across episode resets; dones must not reset this EMA.
                latent[i + 1] = z if i == 0 else (
                    (1. - tracker.ema_alpha) * latent[i] + tracker.ema_alpha * z)
    finally:
        encoder.train(was_training)
    return latent


def replace_latent(context, latent):
    result = context.copy()
    result[:, -latent.shape[1]:] = latent
    return result


def delta_stats(value):
    value = np.asarray(value, dtype=np.float64)
    assert np.isfinite(value).all()
    return {"mean": float(value.mean()), "rms": float(np.sqrt(np.mean(value ** 2))),
            "p95_abs": float(np.quantile(np.abs(value), .95))}


@torch.no_grad()
def panel_outputs(trainer, states, contexts, actions=None):
    states = torch.as_tensor(states, device=trainer.device)
    contexts = torch.as_tensor(contexts, device=trainer.device)
    output_actions, values = [], []
    for wi, weight in enumerate(forks.WEIGHTS):
        weights = torch.as_tensor(weight, device=trainer.device).expand(len(states), -1)
        ctx = trainer.replace_context_preferences(contexts, weights)
        action = (trainer.select_action(states, ctx, deterministic=True) if actions is None
                  else torch.as_tensor(actions[wi], device=trainer.device))
        q = torch.minimum(trainer.q1(states, action, ctx), trainer.q2(states, action, ctx))
        output_actions.append(action.cpu().numpy())
        values.append((q * weights).sum(-1).cpu().numpy())
    return np.asarray(output_actions), np.asarray(values)


def capture_panel(trainer, latent, indices, step):
    buffer = trainer.buffer
    anchor_context = replace_latent(buffer.contexts[indices], latent[indices])
    actions, values = panel_outputs(trainer, buffer.states[indices], anchor_context)
    return {"indices": indices, "anchor_step": step, "anchor_context": anchor_context,
            "anchor_latent_next": latent[indices + 1].copy(),
            "anchor_actions": actions, "anchor_values": values}


def measure_panel(trainer, latent, panel):
    buffer, idx = trainer.buffer, panel["indices"]
    stored = buffer.contexts[idx]
    fresh = replace_latent(stored, latent[idx])
    state = buffer.states[idx]
    a_stored, q_ss = panel_outputs(trainer, state, stored)
    a_fresh, q_ff = panel_outputs(trainer, state, fresh)
    _, q_fs = panel_outputs(trainer, state, fresh, a_stored)
    _, q_sf = panel_outputs(trainer, state, stored, a_fresh)
    a_anchor, _ = panel_outputs(trainer, state, panel["anchor_context"])
    _, q_anchor = panel_outputs(trainer, state, panel["anchor_context"], panel["anchor_actions"])
    width = latent.shape[1]
    return {
        "anchor_step": panel["anchor_step"], "samples": len(idx),
        "latent_current_minus_stored": delta_stats(latent[idx] - stored[:, -width:]),
        "next_latent_current_minus_stored": delta_stats(latent[idx + 1] - buffer.next_contexts[idx, -width:]),
        "latent_current_minus_anchor": delta_stats(latent[idx] - panel["anchor_context"][:, -width:]),
        "next_latent_current_minus_anchor": delta_stats(latent[idx + 1] - panel["anchor_latent_next"]),
        "preferences": [{
            "weight": weight.tolist(),
            "actor_context_effect": delta_stats(a_fresh[wi] - a_stored[wi]),
            "actor_weight_drift_fixed_anchor_context": delta_stats(a_anchor[wi] - panel["anchor_actions"][wi]),
            "critic_weight_drift_fixed_anchor_inputs": delta_stats(q_anchor[wi] - panel["anchor_values"][wi]),
            "q_context_effect_fixed_stored_action": delta_stats(q_fs[wi] - q_ss[wi]),
            "q_action_effect_fixed_stored_context": delta_stats(q_sf[wi] - q_ss[wi]),
            "q_joint_effect": delta_stats(q_ff[wi] - q_ss[wi]),
            "q_means": {k: float(v[wi].mean()) for k, v in (
                ("stored_context_stored_action", q_ss), ("fresh_context_stored_action", q_fs),
                ("stored_context_fresh_action", q_sf), ("fresh_context_fresh_action", q_ff))},
        } for wi, weight in enumerate(forks.WEIGHTS)],
    }


def annotate_modes(report, ground_truth):
    # Ground truth is added only after training to describe preselected time cohorts.
    history = ground_truth["mode_history"]
    starts, modes = np.asarray(history).T
    for cohort in report["cohorts"]:
        timesteps = np.asarray(cohort["sample_timesteps"])
        labels = modes[np.searchsorted(starts, timesteps - 1, side="right") - 1]
        cohort["posthoc_sample_counts_per_mode"] = [
            int(np.sum(labels == m)) for m in range(ground_truth["num_modes"])]
    report["ground_truth_mode_history"] = history


class ContextDriftProbe:
    def __init__(self, seed, output, steps=STEPS, cohort_width=COHORT_WIDTH, samples=SAMPLES):
        self.seed, self.output, self.steps = seed, Path(output), tuple(steps)
        self.cohort_width, self.samples, self.panels = cohort_width, samples, {}
        self.report = {
            "protocol": PROTOCOL, "method": forks.METHOD, "seed": seed,
            "steps": list(self.steps), "weights": forks.WEIGHTS.tolist(),
            "cohort_width": cohort_width, "samples_per_cohort": samples,
            "complete": False, "measurements_complete": False, "cases": [], "cohorts": [],
            "extra_environment_steps": 0, "extra_updates": 0,
            "scope": "Observation only, no training feedback or environment trial. Current encoder "
                     "re-encodes the observed training prefix; pre-transition context excludes that "
                     "transition. Reproduce continuous EMA across episode boundaries. Only the latent "
                     "suffix changes; stored transition features remain fixed. Samples come from fixed "
                     "time bins, not mode labels. Anchor inputs/actions stay fixed across probes. "
                     "Q measurements are learned soft-critic sensitivity, not calibrated returns or "
                     "evidence that refreshing contexts improves policy quality. This is a retrospective "
                     "encoding diagnostic at the current training prefix, not a historical online encoder.",
        }

    def __call__(self, step, trainer, encoder, env, operator):
        if step not in self.steps:
            return False
        started = time.monotonic()
        assert trainer.buffer.size == step
        assert trainer.context_dim == 2 + 3 + encoder.latent_dim
        with forks.isolated_rng(60000000 + self.seed * 100000 + step):
            latent = reencode_history(trainer.buffer, encoder)
            for start in range(0, step - self.cohort_width + 1, self.cohort_width):
                if start not in self.panels:
                    rng = np.random.default_rng(61000000 + self.seed * 100000 + start)
                    indices = np.sort(rng.choice(np.arange(start, start + self.cohort_width),
                                                 min(self.samples, self.cohort_width), replace=False))
                    self.panels[start] = capture_panel(trainer, latent, indices, step)
                    self.report["cohorts"].append({
                        "start_transition_index": start, "end_transition_index": start + self.cohort_width,
                        "anchor_step": step, "sample_timesteps": (indices + 1).tolist()})
            panels = [{"start_transition_index": start, **measure_panel(trainer, latent, panel)}
                      for start, panel in self.panels.items()]
        self.report["cases"].append({"step": step, "operator": operator, "panels": panels,
                                     "seconds": time.monotonic() - started})
        self.report["measurements_complete"] = step == self.steps[-1]
        self.output.parent.mkdir(parents=True, exist_ok=True)
        self.output.write_text(json.dumps(self.report, indent=2, allow_nan=False) + "\n")
        print(f"CONTEXT_DRIFT step={step} cohorts={len(panels)} extra_steps=0 "
              f"seconds={time.monotonic() - started:.2f}", flush=True)
        return self.report["measurements_complete"]


def run(seed, output):
    reference = json.loads((forks.REFERENCE_ROOT / f"{forks.METHOD}_seed{seed}.json").read_text())
    assert reference["total_steps"] == STEPS[-1]
    probe = ContextDriftProbe(seed, output)
    result = training.train_method(
        "mo-halfcheetah-v5", forks.METHOD, STEPS[-1], 25000, seed, str(Path(output).parent),
        hidden=64, eval_freq=100000, device="cpu", training_probe=probe)
    probe.report["base_reproduction"] = forks.reproduction_check(result, reference)
    annotate_modes(probe.report, result["ground_truth"])
    probe.report["main_environment_steps"] = STEPS[-1]
    probe.report["total_training_interactions"] = STEPS[-1]
    probe.report["complete"] = (probe.report["measurements_complete"]
                                 and probe.report["base_reproduction"]["matched"])
    Path(output).write_text(json.dumps(probe.report, indent=2, allow_nan=False) + "\n")
    assert probe.report["complete"], "Main trajectory failed to reproduce the frozen v27 reference"
    print(f"DONE context_drift seed={seed} interactions={STEPS[-1]} "
          f"return_error={probe.report['base_reproduction']['max_episode_return_error']}", flush=True)
    return probe.report


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, choices=(26, 27, 28), required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    run(args.seed, args.output)
