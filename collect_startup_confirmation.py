"""Collect new startup and paired evaluation streams for the frozen detector."""

import json
from pathlib import Path

import numpy as np
import torch

from collect_matched_scale_normals import COLLECTION_STEPS
from run_conditional_change_probe import (
    CASES, CASE_STEPS, CHANGE_AT, METHODS, SOURCE as CHECKPOINTS,
    training, collect, load_models)
from run_episode_blocked_change_probe import write


PROTOCOL = "fresh_startup_confirmation_collection_v1"
ROOT = Path("results/iclr_startup_confirmation_20261001/collection")
STARTUP_STEPS = dict(COLLECTION_STEPS)


def seeds(seed, group, index=0):
    if group == "offset":
        return dict(environment_seed=310000000 + 10 * seed,
            reset_seed=1200000000 + 50000 * seed, weight_seed=320000000 + 10 * seed)
    if group == "startup":
        return dict(environment_seed=230000000 + 10 * seed + index,
            reset_seed=400000000 + 50000 * seed + 10000 * index,
            weight_seed=240000000 + 10 * seed + index)
    assert group == "evaluation"
    return dict(environment_seed=250000000 + 10 * seed,
        reset_seed=600000000 + 50000 * seed, weight_seed=260000000 + 10 * seed)


def steps(group):
    assert group in ("offset", "startup", "evaluation")
    if group == "offset":
        return dict(stationary=STARTUP_STEPS["stationary"])
    return STARTUP_STEPS if group == "startup" else dict.fromkeys(CASES, CASE_STEPS)


def budget(group):
    counts = steps(group)
    return dict(collection_interactions=sum(counts.values()), initialization_probes=len(counts),
        additional_interactions=sum(counts.values()) + len(counts))


def folder(root, env, seed, group):
    return root / env.replace("-", "_") / f"seed{seed}" / group


def audit(data, policies, env, case, rng):
    result = dict(ground_truth_mode_history_audit_only=[list(entry) for entry in env.mode_history])
    if case == "sampler_change":
        original = np.random.default_rng(rng["weight_seed"]).dirichlet(
            np.ones(env.reward_dim), size=len(data["inputs"])).astype(np.float32)
        result["intervention_active"] = bool(np.any(data["weights"][CHANGE_AT:] != original[CHANGE_AT:]))
    if case == "actor_change":
        obs_dim = env.observation_space.shape[0]
        with torch.no_grad():
            fixed = policies["fixed"].get_action(torch.as_tensor(data["inputs"][CHANGE_AT:, :obs_dim]),
                torch.as_tensor(data["contexts"][CHANGE_AT:]), deterministic=True).numpy()
        delta = np.abs(fixed - data["inputs"][CHANGE_AT:, obs_dim:])
        result.update(intervention_active=bool(np.any(delta > 0)),
            same_state_context_actor_action_max_delta=float(delta.max()))
    if case in ("sampler_change", "actor_change"):
        assert result["intervention_active"]
    expected = [[0, 0], [CHANGE_AT, int(case[-1])]] if case.startswith("mode") else [[0, 0]]
    assert result["ground_truth_mode_history_audit_only"] == expected
    return result


def validate(row, env, seed, group, seed_fn=seeds):
    assert (row["protocol"], row["env"], row["seed"], row["group"], row["complete"], row["collection_steps"]) == (
        PROTOCOL, env, seed, group, True, steps(group))
    assert all(row[key] == value for key, value in budget(group).items())
    assert row["source_methods"] == METHODS and row["source_step"] == 75000
    assert all(row[key] == 0 for key in ("policy_optimizer_steps", "critic_optimizer_steps", "encoder_optimizer_steps"))
    for index, case in enumerate(steps(group)):
        assert row["cases"][case]["seeds"] == seed_fn(seed, group, index)
        expected = [[0, 0], [CHANGE_AT, int(case[-1])]] if case.startswith("mode") else [[0, 0]]
        assert row["cases"][case]["ground_truth_mode_history_audit_only"] == expected
    if group == "evaluation":
        assert row["paired_prefixes_verified"] == len(CASES) - 1


def run_group(env_name, seed, group, root=ROOT, source=CHECKPOINTS, seed_fn=seeds):
    target = folder(root, env_name, seed, group)
    output = target / "readout.json"
    if output.exists():
        saved = json.loads(output.read_text())
        if not saved["complete"]:
            raise RuntimeError("Incomplete acquisition retained; inspect spent interactions before retrying")
        validate(saved, env_name, seed, group, seed_fn)
        return saved
    report = dict(protocol=PROTOCOL, env=env_name, seed=seed, group=group, complete=False,
        collection_steps=steps(group), source=str(source), source_methods=METHODS, source_step=75000,
        collection_interactions=0, initialization_probes=0, additional_interactions=0,
        policy_optimizer_steps=0, critic_optimizer_steps=0, encoder_optimizer_steps=0,
        cases={}, paired_prefixes_verified=0)
    write(output, report)
    stored, policies, encoder, reference = {}, None, None, None
    for index, (case, count) in enumerate(steps(group).items()):
        rng = seed_fn(seed, group, index)
        env = training.make_piecewise_mo_env(env_name, switch_interval=10**9, seed=rng["environment_seed"])
        env._lock_mode = True
        report["initialization_probes"] += 1
        try:
            if policies is None:
                policies, encoder = load_models(env, seed, source)
                assert all(not p.requires_grad for model in [*policies.values(), encoder] for p in model.parameters())
            data = collect(env, policies, encoder, count, rng["reset_seed"], rng["weight_seed"], case)
            if group == "evaluation":
                if reference is None:
                    reference = data
                else:
                    for key in data:
                        np.testing.assert_array_equal(data[key][:CHANGE_AT], reference[key][:CHANGE_AT])
                    report["paired_prefixes_verified"] += 1
            report["cases"][case] = dict(steps=count, seeds=rng, **audit(data, policies, env, case, rng))
            stored.update({f"{case}_{key}": value for key, value in data.items()})
        finally:
            report["collection_interactions"] += int(env._total_steps)
            report["additional_interactions"] = report["collection_interactions"] + report["initialization_probes"]
            env.close()
            write(output, report)
        print(f"COLLECT {env_name} seed={seed} {group}/{case} interactions={report['additional_interactions']}", flush=True)
    np.savez_compressed(target / "transitions.npz", **stored)
    report["complete"] = True
    validate(report, env_name, seed, group, seed_fn)
    write(output, report)
    return report


def data(root, env, seed, group, seed_fn=seeds):
    target = folder(root, env, seed, group)
    row = json.loads((target / "readout.json").read_text())
    validate(row, env, seed, group, seed_fn)
    with np.load(target / "transitions.npz") as raw:
        cases = {case: (raw[f"{case}_inputs"].copy(), raw[f"{case}_targets"].copy()) for case in steps(group)}
    assert all(len(x) == len(y) == steps(group)[case] for case, (x, y) in cases.items())
    return cases, row
