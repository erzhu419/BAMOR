"""Pair late actors with contemporaneous/final encoders, without training feedback."""

import argparse
import json
from pathlib import Path

import numpy as np
import torch

import run_replay_context_confirmation as confirmation


pilot = confirmation.pilot
training = pilot.training
PROTOCOL = "snapshot_encoder_pairing_v1"
SEED = 31
STEPS = confirmation.LATE_STEPS
OUTPUT_ROOT = Path("results/iclr_snapshot_encoders_20260920")


class SnapshotEncoderProbe:
    def __init__(self, refresh=None, steps=STEPS):
        self.refresh, self.steps = refresh, tuple(steps)
        self.snapshots = {}
        self.trainer = self.encoder = None
        self.training_steps = 0

    def __call__(self, step, trainer, encoder, env, operator):
        if self.refresh is not None:
            self.refresh(step, trainer, encoder, env, operator)
        if step in self.steps:
            self.snapshots[step] = {
                name: {k: v.detach().cpu().clone() for k, v in net.state_dict().items()}
                for name, net in (("policy", trainer.policy), ("encoder", encoder))}
            print(f"SNAPSHOT_PAIR step={step}", flush=True)
        if step == self.steps[-1]:
            self.trainer, self.encoder = trainer, encoder
            self.training_steps = env._total_steps
            return True
        return False


def evaluate_pairs(probe, env, seed, ref_point):
    """Yield five distinct cells; the final matched/final pair is evaluated once."""
    trainer, encoder = probe.trainer, probe.encoder
    assert not trainer.critic_select_policy
    final_step = probe.steps[-1]
    assert tuple(probe.snapshots) == probe.steps
    try:
        for actor_step in probe.steps:
            trainer.policy.load_state_dict(probe.snapshots[actor_step]["policy"])
            for encoder_step in dict.fromkeys((actor_step, final_step)):
                encoder.load_state_dict(probe.snapshots[encoder_step]["encoder"])
                before = env._total_steps
                metrics = training.evaluate_locked_modes(
                    env, lambda weights, seeds: training.evaluate_pareto_v2(
                        env, trainer, pilot.METHOD, encoder, weights,
                        num_episodes=len(seeds), device=trainer.device,
                        episode_seeds=seeds), seed, ref_point)
                yield {"actor_step": actor_step, "encoder_step": encoder_step,
                       "environment_steps": env._total_steps - before, **metrics}
    finally:
        trainer.policy.load_state_dict(probe.snapshots[final_step]["policy"])
        encoder.load_state_dict(probe.snapshots[final_step]["encoder"])


def reference_front_check(cell, reference):
    checks = {key: cell[key] == reference[key] for key in (
        "evaluation_protocol", "evaluation_weights", "episode_seeds_per_mode", "ref_point")}
    expected = [[p for p in front if p["archive_step"] == cell["actor_step"]]
                for front in reference["pareto_per_mode"]]
    actual = cell["pareto_per_mode"]
    checks["ordered_weights"] = ([[p["weight"] for p in f] for f in actual]
                                 == [[p["weight"] for p in f] for f in expected])
    error = None
    if all(checks.values()):
        error = float(np.max(np.abs(
            np.asarray([[p["return"] for p in f] for f in actual])
            - np.asarray([[p["return"] for p in f] for f in expected]))))
    return {"checks": checks, "max_return_error": error,
            "matched": error is not None and error <= 1e-6}


def paired_metrics(cells, steps=STEPS):
    lookup = {(c["actor_step"], c["encoder_step"]): c for c in cells}
    pairs = []
    for step in steps:
        matched, final = lookup[step, step], lookup[step, steps[-1]]
        pairs.append({
            "actor_step": step,
            "contemporaneous_hv_mean_mode": matched["hv_mean_mode"],
            "final_encoder_hv_mean_mode": final["hv_mean_mode"],
            "contemporaneous_minus_final_hv": matched["hv_mean_mode"] - final["hv_mean_mode"],
            "contemporaneous_hv_per_mode": matched["hv_per_mode"],
            "final_encoder_hv_per_mode": final["hv_per_mode"],
            "contemporaneous_max_speed_per_mode": [max(p["return"][0] for p in f)
                                                   for f in matched["pareto_per_mode"]],
            "final_encoder_max_speed_per_mode": [max(p["return"][0] for p in f)
                                                 for f in final["pareto_per_mode"]],
        })
    return pairs


def run(condition, output_root=OUTPUT_ROOT):
    assert condition in confirmation.CONDITIONS
    reference_path = confirmation.OUTPUT_ROOT / condition / f"{pilot.METHOD}_seed{SEED}.json"
    reference = json.loads(reference_path.read_text())
    assert reference["replay_context_confirmation"]["complete"]
    assert reference["replay_context_confirmation"]["protocol"] == confirmation.PROTOCOL
    assert reference["archive_context"] == "final_encoder"
    output = Path(output_root) / condition
    output.mkdir(parents=True, exist_ok=True)
    refresh = (pilot.ReplayContextRefresh(output / f"seed{SEED}_refresh.json")
               if condition == "refresh" else None)
    probe = SnapshotEncoderProbe(refresh)
    report = {
        "protocol": PROTOCOL, "condition": condition, "seed": SEED, "complete": False,
        "reference": str(reference_path), "snapshot_steps": list(STEPS), "cells": [],
        "extra_training_updates": 0, "evaluation_feedback_to_training": False,
        "scope": "Same actor, own-time versus final encoder; deterministic actions, raw "
                 "undiscounted returns, reset transition context per evaluation episode. "
                 "These are offline locked-mode evaluations, not online training returns.",
    }

    def save():
        (output / f"seed{SEED}.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")

    save()
    print(f"ENCODER_DIAGNOSTIC condition={condition} seed={SEED} steps={pilot.MAIN_STEPS}", flush=True)
    result = training.train_method(
        "mo-halfcheetah-v5", pilot.METHOD, pilot.MAIN_STEPS, 25000, SEED, str(output),
        hidden=64, learning_starts=pilot.LEARNING_STARTS, eval_freq=100000, device="cpu",
        training_probe=probe)
    # Keep only the small paired network states remotely, not replay/optimizer dumps.
    torch.save(probe.snapshots, output / f"seed{SEED}_snapshots.pt")
    reproduction = pilot.drift.forks.reproduction_check(result, reference)
    reproduction["training_settings"] = {key: result[key] == reference[key] for key in (
        "method", "env", "seed", "total_steps", "switch_interval", "context_dim",
        "critic_target_rule", "bootstrap_terminal_rule", "critic_candidate_context_rule", "v2_context")}
    report["training_reproduction"] = reproduction
    report["training_interactions"] = probe.training_steps
    report["episode_rewards"] = result["episode_rewards"]
    report["ground_truth"] = result["ground_truth"]
    if refresh is not None:
        report["replay_context_refresh"] = refresh.state_dict(complete=True)
        refresh.save(complete=True)
        assert [c["step"] for c in refresh.cases] == list(pilot.REFRESH_STEPS)
    save()
    assert probe.training_steps == pilot.MAIN_STEPS
    assert reproduction["matched"] and all(reproduction["training_settings"].values()), reproduction

    env = training.make_piecewise_mo_env("mo-halfcheetah-v5", switch_interval=25000, seed=SEED + 1000)
    report["environment_initialization_probe_steps"] = 3  # Two training envs plus this eval env.
    report["evaluation_interactions"] = 0
    try:
        for cell in evaluate_pairs(probe, env, SEED, np.asarray(reference["ref_point"])):
            if cell["encoder_step"] == STEPS[-1]:
                cell["reference_reproduction"] = reference_front_check(cell, reference)
            report["cells"].append(cell)
            report["evaluation_interactions"] += cell["environment_steps"]
            save()
            assert cell.get("reference_reproduction", {"matched": True})["matched"], cell
            print(f"PAIR_EVAL actor={cell['actor_step']} encoder={cell['encoder_step']} "
                  f"hv={cell['hv_mean_mode']:.6f} interactions={cell['environment_steps']}", flush=True)
    finally:
        env.close()
    report["pairs"] = paired_metrics(report["cells"])
    report["complete"] = True
    save()
    print(f"DONE snapshot_encoder_pairing condition={condition} seed={SEED} "
          f"training={report['training_interactions']} evaluation={report['evaluation_interactions']}", flush=True)
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--condition", choices=confirmation.CONDITIONS, required=True)
    parser.add_argument("--output-root", type=Path, default=OUTPUT_ROOT)
    args = parser.parse_args()
    run(args.condition, args.output_root)
