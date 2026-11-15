"""Paired data-collection forks with a common learner and frozen representation."""

import argparse
import copy
import json
from pathlib import Path
import time

import numpy as np
import torch

import run_episode_forecast_pilot as previous
from diagnose_mean_action_gradient import fork_env
from diagnose_update_forks import FIELDS, batch_at, clone_trainer, isolated_rng
from run_final_policy_fresh import direct_utility


PROTOCOL = "paired_collection_value_probe_v1"
ROOT = Path("results/iclr_collection_value_20261001")
SEEDS = tuple(range(2301, 2306))
ENVIRONMENTS = previous.ENVIRONMENTS
METHOD = previous.METHODS["fixed"]
CUT, COLLECTION_STEPS, UPDATES = 25000, 1500, 1500
REPEATS, HORIZON = 2, 1000
SAMPLERS = ("uniform", "stratified", "mixed", "uniform_repeat")
BRANCHES = ("old_only", *SAMPLERS)
EVALUATED = ("base", "old_only", "uniform", "stratified", "mixed")
CONTRASTS = (("mixed", "old_only"), ("uniform", "old_only"), ("stratified", "old_only"),
             ("mixed", "uniform"), ("stratified", "uniform"), ("mixed", "stratified"),
             ("old_only", "base"))
KWARGS = previous.COMMON_KWARGS
training, write_report = previous.training, previous.write_report
STORAGE_FIELDS = (*FIELDS, "timesteps", "regimes")


def specification():
    return dict(method=METHOD, seeds=list(SEEDS), environments=list(ENVIRONMENTS),
        prefix_steps=CUT, collection_steps=COLLECTION_STEPS, updates_per_branch=UPDATES,
        evaluation_repeats=REPEATS, evaluation_horizon=HORIZON,
        training_kwargs=KWARGS, samplers=list(SAMPLERS), evaluated=list(EVALUATED),
        common_learning_contract=previous.LEARNING_CONTRACT)


def budget(env):
    modes = previous.previous.previous.extension.ENVIRONMENTS[env]["num_modes"]
    updates = CUT - KWARGS["learning_starts"] + 1
    return dict(training_interactions=CUT + len(SAMPLERS) * COLLECTION_STEPS,
        evaluation_interactions=len(EVALUATED) * modes * 21 * REPEATS * HORIZON,
        environment_initialization_probe_steps=2,
        actor_optimizer_steps=updates + len(BRANCHES) * UPDATES,
        critic_optimizer_steps=updates + len(BRANCHES) * UPDATES,
        encoder_optimizer_steps=updates)


def empty_buffer(template, capacity, cls=training.ReplayBuffer):
    return cls(capacity, template.states.shape[1], template.actions.shape[1],
        template.weights.shape[1], template.contexts.shape[1], template.rewards.shape[1], template.device)


class PairedReplay(training.ReplayBuffer):
    def sample(self, batch_size):
        # A common uniform draw pairs different pool sizes without changing the law.
        indices = (np.random.random(batch_size) * self.size).astype(np.int64)
        return batch_at(self, indices)


def combined_buffer(old, new=None):
    pools = (old,) if new is None else (old, new)
    result = empty_buffer(old, sum(p.size for p in pools), PairedReplay)
    for pool in pools:
        for field in STORAGE_FIELDS:
            getattr(result, field)[result.size:result.size + pool.size] = getattr(pool, field)[:pool.size]
        result.size += pool.size
    result.ptr = result.size % result.capacity
    return result


class Capture:
    def __init__(self):
        self.step, self.state = CUT, None

    def __call__(self, state):
        # Training stops here; retain live learner state and copy only simulators.
        self.state = dict(state, env=fork_env(state["env"]), eval_env=fork_env(state["eval_env"]))


def collect(state, sampler, seed):
    started = time.monotonic()
    env = fork_env(state["env"])
    encoder, trainer = state["encoder"], state["trainer"]
    trans = copy.deepcopy(state["trans_tracker"])
    learned = copy.deepcopy(state["learned_tracker"], {id(encoder): encoder})
    data = empty_buffer(trainer.buffer, COLLECTION_STEPS)
    obs = state["obs"].copy()
    total_reward = np.zeros(trainer.reward_dim)
    completed, terminals, truncations = 0, 0, 0
    operator = "uniform" if sampler == "uniform_repeat" else sampler
    try:
        with isolated_rng(seed):
            for index in range(COLLECTION_STEPS):
                step = state["step"] + index + 1
                weight, _ = state["adaptive_selector"].sample_weight(operator, trainer.reward_dim, step)
                context = training.build_mujoco_context(METHOD, weight, trans_tracker=trans, learned_tracker=learned)
                with torch.no_grad():
                    action, _ = trainer.select_action(torch.as_tensor(obs).float().unsqueeze(0),
                                                      torch.as_tensor(context).float().unsqueeze(0))
                action = action.numpy().flatten()
                next_obs, reward, terminated, truncated, _ = env.step(action)
                trans.update(obs, action, reward, next_obs)
                learned.update(obs, action, reward, next_obs, "cpu")
                next_context = training.build_mujoco_context(METHOD, weight, trans_tracker=trans, learned_tracker=learned)
                data.push(obs, action, weight, context, reward, next_obs, next_context,
                          trainer.replay_terminal(terminated, truncated), t=step)
                total_reward += reward
                completed += terminated or truncated
                terminals += terminated
                truncations += truncated
                obs = env.reset()[0] if terminated or truncated else next_obs
        assert env._total_steps - state["env"]._total_steps == COLLECTION_STEPS
        return data, dict(interactions=COLLECTION_STEPS, completed_episodes=int(completed),
            terminated=int(terminals), truncated=int(truncations),
            seconds=time.monotonic() - started,
            total_raw_reward=total_reward.tolist(), mean_preference=data.weights.mean(axis=0).tolist(),
            mode_history=[list(row) for row in env.mode_history])
    finally:
        env.close()


def learn(base, data, seed):
    branch = clone_trainer(base, combined_buffer(base.buffer, data))
    before = dict(branch.update_work)
    for update in range(UPDATES):
        with isolated_rng(seed + update):
            branch.update()
    work = {key: branch.update_work[key] - value for key, value in before.items()}
    assert work == dict(critic_optimizer_steps=UPDATES, actor_optimizer_steps=UPDATES,
        critic_rows=UPDATES * branch.batch_size, target_candidate_rows=UPDATES * branch.batch_size,
        actor_candidate_rows=UPDATES * branch.batch_size)
    return branch, work


def same_learner(a, b):
    for name in ("policy", "q1", "q2", "q1_target", "q2_target"):
        assert all(torch.equal(value, getattr(b, name).state_dict()[key])
                   for key, value in getattr(a, name).state_dict().items()), name
    for name in ("policy_opt", "q_opt"):
        x, y = getattr(a, name).state_dict(), getattr(b, name).state_dict()
        assert x["param_groups"] == y["param_groups"] and x["state"].keys() == y["state"].keys()
        for param in x["state"]:
            assert all(torch.equal(value, y["state"][param][key])
                       for key, value in x["state"][param].items()), name
    assert a.update_work == b.update_work


def evaluate(state, trainer, seed):
    env = fork_env(state["eval_env"])
    weights = training.full_simplex_grid(trainer.reward_dim)
    ref = training.reference_point_for_env(state["run_config"]["env_name"], trainer.reward_dim)
    result = dict(evaluation_weights=weights.tolist(), episode_seeds_per_mode=[],
                  ref_point=ref.tolist(), pareto_per_mode=[], hv_per_mode=[])
    before = env._total_steps
    started = time.monotonic()
    try:
        env._lock_mode = True
        for mode in range(env.num_modes):
            env._apply_mode(mode)
            seeds = [72000000 + seed * 1000 + mode * 10 + i for i in range(REPEATS)]
            # Reuse the production evaluator; HORIZON is1000 in formal runs.
            front = training.evaluate_pareto_v2(env, trainer, METHOD, state["encoder"], weights,
                num_episodes=REPEATS, episode_seeds=seeds, device="cpu")
            result["episode_seeds_per_mode"].append(seeds)
            result["pareto_per_mode"].append(front)
            result["hv_per_mode"].append(training.compute_hv_nd(np.asarray([p["return"] for p in front]), ref))
        result.update(direct_utility=direct_utility(result), final_hv=float(np.mean(result["hv_per_mode"])),
            interactions=int(env._total_steps - before), seconds=time.monotonic() - started)
        assert 0 < result["interactions"] <= env.num_modes * len(weights) * REPEATS * HORIZON
        return result
    finally:
        env.close()


def run(env, seed, root=ROOT):
    root = Path(root)
    frozen = json.loads((root / "protocol.json").read_text())
    assert frozen["protocol"] == PROTOCOL and frozen["experiment"] == specification()
    assert env in ENVIRONMENTS and seed in SEEDS
    output = root / env.replace("-", "_") / f"seed{seed}.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.exists():
        saved = json.loads(output.read_text())
        assert (saved["protocol"], saved["env"], saved["seed"], saved["experiment"]) == (PROTOCOL, env, seed, specification())
        if saved["complete"]:
            return saved
        raise RuntimeError("Incomplete attempt preserved; inspect before a new-root retry")
    report = dict(protocol=PROTOCOL, env=env, seed=seed, experiment=specification(), complete=False,
        planned_budget_upper_bound=budget(env), completed_budget=dict.fromkeys(budget(env), 0),
        collection={}, branches={}, controls_passed=False)
    write_report(output, report)
    capture = Capture()
    training.train_method(env, METHOD, CUT, 25000, seed, str(output.parent / f"prefix{seed}"),
                          state_probe=capture, **KWARGS)
    state = capture.state
    costs = report["completed_budget"]
    updates = CUT - KWARGS["learning_starts"] + 1
    costs.update(training_interactions=CUT, environment_initialization_probe_steps=2,
        actor_optimizer_steps=state["trainer"].update_work["actor_optimizer_steps"],
        critic_optimizer_steps=state["trainer"].update_work["critic_optimizer_steps"],
        encoder_optimizer_steps=training.optimizer_step_count(state["encoder_opt"]))
    assert all(costs[key] == updates for key in ("actor_optimizer_steps", "critic_optimizer_steps", "encoder_optimizer_steps"))
    prefix_work = dict(state["trainer"].update_work)
    encoder_before = copy.deepcopy(state["encoder"].state_dict())
    report["prefix"] = dict(step=state["step"], replay_size=state["trainer"].buffer.size,
        training_seconds=state["training_seconds"],
        episode_steps=state["episode_steps"], mode_history=[list(r) for r in state["env"].mode_history],
        learner_work=prefix_work, network_parameters=training.network_parameter_counts(
            state["trainer"].policy, (state["trainer"].q1, state["trainer"].q2), state["encoder"]))
    write_report(output, report)
    try:
        data = {}
        for sampler in SAMPLERS:
            data[sampler], report["collection"][sampler] = collect(state, sampler, 52000000 + seed * 1000)
            costs["training_interactions"] += COLLECTION_STEPS
            assert report["collection"][sampler]["mode_history"] == report["collection"]["uniform"]["mode_history"]
            write_report(output, report)
        for field in STORAGE_FIELDS:
            np.testing.assert_array_equal(getattr(data["uniform"], field), getattr(data["uniform_repeat"], field))
        assert state["trainer"].buffer.size == CUT and state["trainer"].update_work == prefix_work
        report["collection_identity_passed"] = True
        baseline = evaluate(state, state["trainer"], seed)
        costs["evaluation_interactions"] += baseline["interactions"]
        report["branches"]["base"] = dict(evaluation=baseline, learner_work=dict.fromkeys(prefix_work, 0))
        identity = None
        for name in BRANCHES:
            started = time.monotonic()
            branch, work = learn(state["trainer"], data.get(name), 62000000 + seed * 10000)
            learner_seconds = time.monotonic() - started
            costs["actor_optimizer_steps"] += work["actor_optimizer_steps"]
            costs["critic_optimizer_steps"] += work["critic_optimizer_steps"]
            if name == "uniform_repeat":
                same_learner(identity, branch)
                report["learner_identity_passed"] = True
            else:
                evaluated = evaluate(state, branch, seed)
                costs["evaluation_interactions"] += evaluated["interactions"]
                report["branches"][name] = dict(evaluation=evaluated, learner_work=work,
                    learner_seconds=learner_seconds,
                    replay_size=branch.buffer.size, new_replay_fraction=(0. if name == "old_only" else
                        COLLECTION_STEPS / branch.buffer.size))
                for key in ("evaluation_weights", "episode_seeds_per_mode", "ref_point"):
                    assert evaluated[key] == baseline[key]
                print(f"COLLECTION_VALUE {env} seed={seed} branch={name} "
                      f"utility={evaluated['direct_utility']['mean']:.6f}", flush=True)
            if name == "uniform":
                identity = branch
            else:
                del branch
            write_report(output, report)
        assert state["trainer"].update_work == prefix_work
        assert all(torch.equal(value, state["encoder"].state_dict()[key]) for key, value in encoder_before.items())
        assert training.optimizer_step_count(state["encoder_opt"]) == updates
        for key, upper in budget(env).items():
            assert (0 < costs[key] <= upper if key == "evaluation_interactions" else costs[key] == upper)
        report.update(complete=True, controls_passed=True, representation_frozen=True)
        write_report(output, report)
        print(f"DONE {PROTOCOL} {env} seed={seed} cost={json.dumps(costs)}", flush=True)
        return report
    finally:
        state["env"].close()
        state["eval_env"].close()


def summarize(root=ROOT):
    result = dict(protocol=PROTOCOL, seeds=list(SEEDS), experiment=specification(), complete=True,
        environments={}, completed_budget=dict.fromkeys(budget(ENVIRONMENTS[0]), 0),
        uncertainty="20000 paired-seed percentile bootstrap resamples,RNG0; descriptive95% intervals")
    indices = np.random.default_rng(0).integers(0, len(SEEDS), (20000, len(SEEDS)))
    for env in ENVIRONMENTS:
        runs = []
        for seed in SEEDS:
            record = json.loads((Path(root) / env.replace("-", "_") / f"seed{seed}.json").read_text())
            assert (record["protocol"], record["env"], record["seed"], record["complete"],
                record["controls_passed"], record["experiment"], record["representation_frozen"]) == (
                PROTOCOL, env, seed, True, True, specification(), True)
            assert record["collection_identity_passed"] and record["learner_identity_passed"]
            for key, upper in budget(env).items():
                cost = record["completed_budget"][key]
                assert (0 < cost <= upper if key == "evaluation_interactions" else cost == upper)
                result["completed_budget"][key] += cost
            compact = dict(seed=seed, collection=record["collection"], completed_budget=record["completed_budget"],
                prefix=record["prefix"], branches={})
            baseline = record["branches"]["base"]["evaluation"]
            for name in EVALUATED:
                branch = record["branches"][name]
                value = branch["evaluation"]
                for key in ("evaluation_weights", "episode_seeds_per_mode", "ref_point"):
                    assert value[key] == baseline[key]
                if name != "base":
                    assert branch["learner_work"] == record["branches"]["old_only"]["learner_work"]
                compact["branches"][name] = dict(direct_utility=value["direct_utility"], final_hv=value["final_hv"],
                    mode_hv=value["hv_per_mode"], evaluation_interactions=value["interactions"],
                    evaluation_seconds=value["seconds"], learner_seconds=branch.get("learner_seconds", 0.),
                    learner_work=branch["learner_work"])
            runs.append(compact)
        contrasts = {}
        for a, b in CONTRASTS:
            measured = {}
            for metric in ("direct_utility", "final_hv"):
                values = lambda m: np.asarray([r["branches"][m][metric]["mean"] if metric == "direct_utility"
                    else r["branches"][m][metric] for r in runs])
                av, bv = values(a), values(b)
                assert np.isfinite(av).all() and np.isfinite(bv).all()
                measured[metric] = dict(previous.previous.previous.paired(av.tolist(), bv.tolist()),
                    paired_mean_ci95=np.quantile((av - bv)[indices].mean(axis=1), [.025, .975]).tolist())
            measured["per_mode_utility_mean_delta"] = np.mean([np.asarray(r["branches"][a]["direct_utility"]["per_mode"]) -
                np.asarray(r["branches"][b]["direct_utility"]["per_mode"]) for r in runs], axis=0).tolist()
            contrasts[f"{a}_minus_{b}"] = measured
        methods = {name: dict(direct_utility_mean=float(np.mean([r["branches"][name]["direct_utility"]["mean"] for r in runs])),
            minimum_direct_utility=min(r["branches"][name]["direct_utility"]["mean"] for r in runs),
            final_hv_mean=float(np.mean([r["branches"][name]["final_hv"] for r in runs])),
            per_mode_utility_mean=np.mean([r["branches"][name]["direct_utility"]["per_mode"] for r in runs], axis=0).tolist(),
            learner_seconds_mean=float(np.mean([r["branches"][name]["learner_seconds"] for r in runs])),
            evaluation_seconds_mean=float(np.mean([r["branches"][name]["evaluation_seconds"] for r in runs])))
            for name in EVALUATED}
        result["environments"][env] = dict(runs=runs, paired=contrasts, methods=methods)
    result["learning_value_gate_passed"] = all(v["paired"]["mixed_minus_old_only"]["direct_utility"]["mean_delta"] > 0
        for v in result["environments"].values())
    result["hopper_hv_guard_passed"] = result["environments"]["mo-hopper-v5"]["paired"]["mixed_minus_old_only"]["final_hv"]["mean_delta"] >= 0
    result["mixture_advantage_gate_passed"] = all(v["paired"]["mixed_minus_uniform"]["direct_utility"]["mean_delta"] > 0
        for v in result["environments"].values())
    result["decision"] = ("collection_learning_value_lead_online_selection_unproven"
        if result["learning_value_gate_passed"] and result["hopper_hv_guard_passed"] else "hold_collection_learning_value")
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--env", choices=ENVIRONMENTS)
    parser.add_argument("--seed", type=int, choices=SEEDS)
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--summarize", action="store_true")
    args = parser.parse_args()
    if args.summarize:
        report = summarize(args.root)
        write_report(args.root / "compact_readout.json", report)
        print(json.dumps(dict(decision=report["decision"], budget=report["completed_budget"])))
    else:
        assert args.env and args.seed is not None
        run(args.env, args.seed, args.root)
