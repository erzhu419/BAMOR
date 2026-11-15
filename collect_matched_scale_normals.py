"""Fresh normal trajectories from the existing frozen75000-step policy sources."""

import argparse
import json
from pathlib import Path
import time

import numpy as np
import torch

from run_conditional_change_probe import (
    ENVIRONMENTS, SEEDS, CHANGE_AT, METHODS, SOURCE as CHECKPOINTS,
    training, collect, load_models)
from run_episode_blocked_change_probe import source_folder, write


PROTOCOL = "matched_scale_fresh_normals_v1"
ROOT = Path("results/iclr_matched_scale_normals_20261001")
COLLECTION_STEPS = dict(stationary=3000, sampler_change=1000, actor_change=1000)


def seeds(source_seed, case_index):
    return dict(environment_seed=130000000 + 10 * source_seed + case_index,
                reset_seed=90000000 + 50000 * source_seed + 10000 * case_index,
                weight_seed=120000000 + 10 * source_seed + case_index)


def run(env_name, source_seed, root=ROOT, source=CHECKPOINTS):
    folder = source_folder(root, env_name, source_seed)
    output = folder / "readout.json"
    if output.exists():
        saved = json.loads(output.read_text())
        assert (saved["protocol"], saved["env"], saved["seed"], saved["collection_steps"]) == (
            PROTOCOL, env_name, source_seed, COLLECTION_STEPS)
        if saved["complete"]:
            return saved
        raise RuntimeError("Incomplete collection retained; inspect its interactions before retrying")
    started = time.monotonic()
    report = dict(protocol=PROTOCOL, env=env_name, seed=source_seed, complete=False,
        source=str(source), source_methods=METHODS, source_step=75000,
        collection_steps=COLLECTION_STEPS, collection_interactions=0, initialization_probes=0,
        additional_interactions=0, policy_optimizer_steps=0, critic_optimizer_steps=0,
        encoder_optimizer_steps=0, cases={}, development_only=True)
    write(output, report)
    stored, policies, encoder = {}, None, None
    for index, (case, steps) in enumerate(COLLECTION_STEPS.items()):
        rng = seeds(source_seed, index)
        env = training.make_piecewise_mo_env(env_name, switch_interval=10**9, seed=rng["environment_seed"])
        env._lock_mode = True
        report["initialization_probes"] += 1
        try:
            if policies is None:
                policies, encoder = load_models(env, source_seed, source)
                assert all(not parameter.requires_grad for model in [*policies.values(), encoder]
                           for parameter in model.parameters())
            data = collect(env, policies, encoder, steps, rng["reset_seed"], rng["weight_seed"], case)
            assert env.mode_history == [(0, 0)] and env.current_mode_idx == 0
            row = dict(steps=steps, seeds=rng, ground_truth_mode_history_audit_only=[[0, 0]],
                       completed_episodes=int(data["terminals"].any(1).sum()))
            if case == "sampler_change":
                original = np.random.default_rng(rng["weight_seed"]).dirichlet(
                    np.ones(env.reward_dim), size=steps).astype(np.float32)
                row["intervention_active"] = bool(np.any(data["weights"][CHANGE_AT:] != original[CHANGE_AT:]))
            if case == "actor_change":
                obs_dim = env.observation_space.shape[0]
                with torch.no_grad():
                    fixed_action = policies["fixed"].get_action(
                        torch.as_tensor(data["inputs"][CHANGE_AT:, :obs_dim]),
                        torch.as_tensor(data["contexts"][CHANGE_AT:]), deterministic=True).numpy()
                delta = np.abs(fixed_action - data["inputs"][CHANGE_AT:, obs_dim:])
                row.update(intervention_active=bool(np.any(delta > 0)),
                           same_state_context_actor_action_max_delta=float(delta.max()))
            if case != "stationary":
                assert row["intervention_active"], "Normal policy/sampler coverage must actually change"
            stored.update({f"{case}_{key}": value for key, value in data.items()})
            report["cases"][case] = row
        finally:
            report["collection_interactions"] += int(env._total_steps)
            report["additional_interactions"] = report["collection_interactions"] + report["initialization_probes"]
            env.close()
            write(output, report)
        print(f"NORMAL {env_name} seed={source_seed} {case} steps={steps}", flush=True)
    assert report["collection_interactions"] == sum(COLLECTION_STEPS.values())
    np.savez_compressed(folder / "transitions.npz", **stored)
    report.update(complete=True, seconds=time.monotonic() - started)
    write(output, report)
    print(f"DONE {PROTOCOL} {env_name} seed={source_seed} interactions={report['additional_interactions']}", flush=True)
    return report


def normal_data(root, env_name, seed):
    folder = source_folder(root, env_name, seed)
    row = json.loads((folder / "readout.json").read_text())
    assert (row["protocol"], row["env"], row["seed"], row["complete"], row["collection_steps"]) == (
        PROTOCOL, env_name, seed, True, COLLECTION_STEPS)
    assert row["collection_interactions"] == sum(COLLECTION_STEPS.values())
    assert row["initialization_probes"] == len(COLLECTION_STEPS)
    assert row["additional_interactions"] == row["collection_interactions"] + row["initialization_probes"]
    assert all(row["cases"][case]["ground_truth_mode_history_audit_only"] == [[0, 0]] for case in COLLECTION_STEPS)
    assert all(row["cases"][case]["intervention_active"] for case in COLLECTION_STEPS if case != "stationary")
    with np.load(folder / "transitions.npz") as data:
        inputs = np.concatenate([data[f"{case}_inputs"] for case in COLLECTION_STEPS])
        targets = np.concatenate([data[f"{case}_targets"] for case in COLLECTION_STEPS])
    assert len(inputs) == row["collection_interactions"]
    return inputs, targets, row


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--env", choices=ENVIRONMENTS, required=True)
    parser.add_argument("--seed", type=int, choices=SEEDS, required=True)
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--source", type=Path, default=CHECKPOINTS)
    args = parser.parse_args()
    torch.set_num_threads(2)
    run(args.env, args.seed, args.root, args.source)
