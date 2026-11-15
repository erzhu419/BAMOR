"""Costed causal episode routing between independently trained objectives."""

import argparse
import copy
import json
from pathlib import Path

import numpy as np
import torch

import run_objective_from_start as source


training, base, write = source.training, source.base, source.write_report
PROTOCOL = "frozen_objective_episode_router_v1"
ROOT = Path("results/iclr_frozen_objective_router_20261002")
ENVIRONMENTS, SEEDS = source.ENVIRONMENTS, (2601, 2602, 2603)
STEPS, SWITCH_INTERVAL, PREFIX, HORIZON, REPEATS = 75000, 25000, 32, 1000, 2
KWARGS = dict(source.KWARGS, eval_freq=1000000)
ROUTES = ("soft", "raw_after_prefix", "risk", "budget_soft")


def specification():
    return dict(environments=list(ENVIRONMENTS), seeds=list(SEEDS), candidate_steps=STEPS,
        switch_interval=SWITCH_INTERVAL, training_kwargs=KWARGS, method=source.METHOD,
        candidates=source.specification()["trainer_classes"], prefix_steps=PREFIX,
        horizon=HORIZON, repeats=REPEATS, calibration_episodes=42,
        calibration="two_randomized_sequential_episodes_per_full21_weight; natural_unlocked_schedule",
        features="weight,prefix_reward_mean,prefix_reward_std,soft_mean_action_square,raw_mean_action_square",
        label="undiscounted_weighted_suffix_return/(horizon-prefix)",
        estimator="two_standardized_ridge_regressions; lambda1; unpenalized_intercept",
        admission="mu_raw-mu_soft > sum_of_residual_RMS_times_sqrt(1+ridge_leverage); else_soft",
        control="continue_original_soft_prefix_to_2*candidate_steps+actual_calibration_interactions",
        evaluation="four_routes_full21_weights_four_locked_modes_paired_resets; final_actors_only",
        privilege="router_uses_no_environment_id_mode_id_critic_or_future_return")


class ReturnGate:
    def fit(self, rows):
        self.ready = all(sum(r["candidate"] == name for r in rows) >= 2 for name in ("soft", "raw"))
        self.models = {}
        if not self.ready:
            return self
        x = np.asarray([r["features"] for r in rows], dtype=float)
        self.center, self.scale = x.mean(0), x.std(0)
        self.scale[self.scale == 0] = 1.
        z = self.design(x)
        penalty = np.eye(z.shape[1])
        penalty[0, 0] = 0.
        for name in ("soft", "raw"):
            mask = np.asarray([r["candidate"] == name for r in rows])
            a, y = z[mask], np.asarray([r["target"] for r in rows])[mask]
            inverse = np.linalg.inv(a.T @ a + penalty)
            beta = inverse @ a.T @ y
            rms = float(np.sqrt(np.mean((y - a @ beta) ** 2)))
            self.models[name] = dict(beta=beta, inverse=inverse, rms=rms, samples=int(mask.sum()))
        return self

    def design(self, x):
        x = np.atleast_2d(x)
        return np.column_stack((np.ones(len(x)), (x - self.center) / self.scale))

    def decide(self, features):
        if not self.ready:
            return "soft", dict(margin=None, insufficient_completed_prefixes=True)
        x = self.design(features)[0]
        values, penalties = {}, {}
        for name, model in self.models.items():
            values[name] = float(x @ model["beta"])
            penalties[name] = model["rms"] * float(np.sqrt(1 + x @ model["inverse"] @ x))
        margin = values["raw"] - values["soft"] - sum(penalties.values())
        return ("raw" if margin > 0 else "soft"), dict(predictions=values, penalties=penalties, margin=margin)

    def state_dict(self):
        out = dict(ready=self.ready, models={})
        if self.ready:
            out.update(center=self.center.tolist(), scale=self.scale.tolist())
            out["models"] = {name: {k: v.tolist() if isinstance(v, np.ndarray) else v for k, v in model.items()}
                             for name, model in self.models.items()}
        return out


def snapshot(state):
    trainer = state["trainer"]
    return dict(policy=copy.deepcopy(trainer.policy).eval().requires_grad_(False),
        encoder=copy.deepcopy(state["encoder"]).eval().requires_grad_(False), reward_dim=trainer.reward_dim)


@torch.no_grad()
def episode(env, policies, weight, reset_seed, route, gate=None):
    obs, _ = env.reset(seed=reset_seed)
    dimension = len(weight)
    trans = training.TransitionContextTracker(dimension)
    learned = {name: training.LearnedTransitionContextTracker(model["encoder"], latent_dim=training.LATENT_CONTEXT_DIM)
               for name, model in policies.items()}
    prefix_rewards, total, suffix = [], np.zeros(dimension), np.zeros(dimension)
    chosen, features, decision = "soft", None, None
    before = env._total_steps
    for step in range(HORIZON):
        contexts = {name: training.build_mujoco_context(source.METHOD, weight,
            trans_tracker=trans, learned_tracker=tracker) for name, tracker in learned.items()}

        def action(name):
            return policies[name]["policy"].get_action(torch.as_tensor(obs).float()[None],
                torch.as_tensor(contexts[name]).float()[None], deterministic=True).numpy().ravel()

        if step == PREFIX and route != "budget_soft":
            actions = {name: action(name) for name in ("soft", "raw")}
            features = np.concatenate((weight, np.mean(prefix_rewards, axis=0), np.std(prefix_rewards, axis=0),
                [np.mean(actions[name] ** 2) for name in ("soft", "raw")])).tolist()
            if route == "risk":
                chosen, decision = gate.decide(features)
            elif route == "raw_after_prefix":
                chosen = "raw"
        executed = action(chosen)
        next_obs, reward, terminated, truncated, _ = env.step(executed)
        total += reward
        if step < PREFIX:
            prefix_rewards.append(np.asarray(reward).copy())
        else:
            suffix += reward
        trans.update(obs, executed, reward, next_obs)
        for tracker in learned.values():
            tracker.update(obs, executed, reward, next_obs, "cpu")
        obs = next_obs
        if terminated or truncated:
            break
    return dict(return_vector=total.tolist(), suffix_return=suffix.tolist(), features=features,
        chosen=chosen, decision=decision, interactions=env._total_steps - before)


def calibrate(state, policies, seed):
    env = base.fork_env(state["env"])
    assert not env._lock_mode
    rows, interactions, prefix_terminations = [], 0, 0
    rng = np.random.default_rng(83000000 + seed)
    weights = training.full_simplex_grid(policies["soft"]["reward_dim"])
    index = 0
    try:
        for weight_index in rng.permutation(len(weights)):
            for candidate in rng.permutation(["soft", "raw"]):
                reset_seed = 84000000 + seed * 1000 + index
                result = episode(env, policies, weights[weight_index], reset_seed,
                    "soft" if candidate == "soft" else "raw_after_prefix")
                interactions += result["interactions"]
                index += 1
                if result["features"] is None:
                    prefix_terminations += 1
                    continue
                rows.append(dict(candidate=str(candidate), features=result["features"], reset_seed=reset_seed,
                    target=float(np.dot(weights[weight_index], result["suffix_return"]) / (HORIZON - PREFIX))))
        return ReturnGate().fit(rows), dict(rows=rows, interactions=interactions,
            prefix_terminations=prefix_terminations, episodes=index)
    finally:
        env.close()


def evaluate(state, policies, seed, route, gate=None):
    env = base.fork_env(state["eval_env"])
    weights = training.full_simplex_grid(policies["soft"]["reward_dim"])
    ref = training.reference_point_for_env(state["run_config"]["env_name"], len(weights[0]))
    result = dict(evaluation_weights=weights.tolist(), ref_point=ref.tolist(), episode_seeds_per_mode=[],
        pareto_per_mode=[], hv_per_mode=[], interactions=0, raw_choices=0, soft_choices=0, prefix_terminations=0)
    try:
        env._lock_mode = True
        for mode in range(env.num_modes):
            env._apply_mode(mode)
            seeds = [85000000 + seed * 1000 + mode * 10 + repeat for repeat in range(REPEATS)]
            front = []
            for weight in weights:
                returns = []
                for reset_seed in seeds:
                    row = episode(env, policies, weight, reset_seed, route, gate)
                    returns.append(row["return_vector"])
                    result["interactions"] += row["interactions"]
                    result[row["chosen"] + "_choices"] += 1
                    result["prefix_terminations"] += row["interactions"] <= PREFIX
                front.append({"weight": weight.tolist(), "return": np.mean(returns, axis=0).tolist()})
            result["pareto_per_mode"].append(front)
            result["episode_seeds_per_mode"].append(seeds)
            result["hv_per_mode"].append(training.compute_hv_nd(np.asarray([p["return"] for p in front]), ref))
        result.update(direct_utility=base.direct_utility(result), final_hv=float(np.mean(result["hv_per_mode"])))
        return result
    finally:
        env.close()


def train(env, seed, output, factory=None, resume=None, steps=None):
    steps = STEPS if steps is None else steps
    capture = base.Capture()
    capture.step = steps
    training.train_method(env, source.METHOD, steps, SWITCH_INTERVAL, seed, str(output.parent),
        trainer_factory=factory, resume_state=resume, state_probe=capture, **KWARGS)
    return capture.state


def clocks(state):
    t = state["trainer"]
    return dict(actor=training.optimizer_step_count(t.policy_opt), critic=training.optimizer_step_count(t.q_opt),
        encoder=training.optimizer_step_count(state["encoder_opt"]))


def check(record):
    assert (record["protocol"], record["experiment"], record["complete"]) == (PROTOCOL, specification(), True)
    assert record["env"] in ENVIRONMENTS and record["seed"] in SEEDS and record["controls_passed"]
    assert record["cold_initializations"] == ["soft", "raw"]
    u = STEPS - KWARGS["learning_starts"] + 1
    n = record["calibration"]["interactions"]
    assert record["calibration"]["episodes"] == 42 and 0 < n <= 42 * HORIZON
    assert record["candidate_clocks"] == dict.fromkeys(("soft", "raw"), dict(actor=u, critic=u, encoder=u))
    b = 2 * STEPS + n
    assert record["baseline_total_steps"] == b
    assert record["baseline_clocks"] == dict.fromkeys(("actor", "critic", "encoder"), b - KWARGS["learning_starts"] + 1)
    assert record["candidate_price"] == dict(training_interactions=2 * STEPS, calibration_interactions=n,
        initialization_probes=4, actor_steps=2 * u, critic_steps=2 * u, encoder_steps=2 * u)
    assert record["baseline_price"] == dict(training_interactions=b, calibration_interactions=0,
        initialization_probes=4, actor_steps=b - KWARGS["learning_starts"] + 1,
        critic_steps=b - KWARGS["learning_starts"] + 1, encoder_steps=b - KWARGS["learning_starts"] + 1)
    costs = record["actual_work"]
    assert costs == dict(training_interactions=3 * STEPS + n, calibration_interactions=n,
        evaluation_interactions=sum(v["interactions"] for v in record["evaluations"].values()),
        initialization_probes=6, actor_steps=3 * STEPS + n - 2 * KWARGS["learning_starts"] + 2,
        critic_steps=3 * STEPS + n - 2 * KWARGS["learning_starts"] + 2,
        encoder_steps=3 * STEPS + n - 2 * KWARGS["learning_starts"] + 2, ridge_fits=2 if record["gate"]["ready"] else 0)
    assert set(record["evaluations"]) == set(ROUTES)
    paired = record["evaluations"]["soft"]["evaluation"]
    for value in record["evaluations"].values():
        assert value["evaluation"] == paired
        assert 0 < value["interactions"] <= 4 * 21 * REPEATS * HORIZON
        assert np.isfinite(value["endpoint"]["utility"]) and np.isfinite(value["endpoint"]["hv"])


def run(env, seed, root=ROOT):
    root = Path(root)
    frozen = json.loads((root / "protocol.json").read_text())
    assert (frozen["protocol"], frozen["experiment"]) == (PROTOCOL, specification())
    assert env in ENVIRONMENTS and seed in SEEDS
    output = root / env.replace("-", "_") / f"seed{seed}.json"
    if output.exists():
        previous = json.loads(output.read_text())
        if previous["complete"]:
            check(previous)
            assert (previous["env"], previous["seed"]) == (env, seed)
            return previous
        raise RuntimeError("Incomplete attempt retained; inspect before a new-root retry")
    output.parent.mkdir(parents=True, exist_ok=True)
    report = dict(protocol=PROTOCOL, experiment=specification(), env=env, seed=seed, complete=False,
        controls_passed=False, evaluations={}, candidate_clocks={}, cold_initializations=[])
    write(output, report)
    states = {}
    try:
        for name, factory in (("soft", source.TRAINERS["soft"]), ("raw", source.TRAINERS["mean_raw"])):
            def cold(*args, **kwargs):
                trainer = factory(*args, **kwargs)
                assert not trainer.policy_opt.state and not trainer.q_opt.state
                assert all(torch.equal(v, getattr(trainer, name + "_target").state_dict()[key])
                    for name in ("q1", "q2") for key, v in getattr(trainer, name).state_dict().items())
                report["cold_initializations"].append(name)
                report["phase"] = "training_" + name
                write(output, report)
                print(f"ROUTER_COLD_START env={env} seed={seed} candidate={name} trainer={type(trainer).__name__}", flush=True)
                return trainer
            states[name] = train(env, seed, output, cold)
            source.local.crossed.common_contract(states[name]["trainer"])
            assert (states[name]["trainer"].buffer.regimes[:STEPS] == -1).all()
            report["candidate_clocks"][name] = clocks(states[name])
            report["phase"] = name + "_trained"
            write(output, report)
            print(f"ROUTER_SOURCE_READY env={env} seed={seed} candidate={name} clocks={report['candidate_clocks'][name]}", flush=True)
        histories = [[list(row) for row in states[name]["env"].mode_history] for name in ("soft", "raw")]
        assert histories[0] == histories[1]
        report["candidate_mode_history"] = histories[0]
        report["network_parameters_per_candidate"] = {name: training.network_parameter_counts(
            state["trainer"].policy, (state["trainer"].q1, state["trainer"].q2), state["encoder"])
            for name, state in states.items()}
        assert report["network_parameters_per_candidate"]["soft"] == report["network_parameters_per_candidate"]["raw"]
        assert states["raw"]["trainer"].objective_updates == STEPS - KWARGS["learning_starts"] + 1
        policies = {name: snapshot(state) for name, state in states.items()}
        gate, acquisition = calibrate(states["soft"], policies, seed)
        report.update(calibration=acquisition, gate=gate.state_dict(), phase="calibrated")
        write(output, report)
        for route in ROUTES[:-1]:
            evaluated = evaluate(states["soft"], policies, seed, route, gate)
            report["evaluations"][route] = dict(endpoint=source.local.source.endpoint(dict(evaluation=evaluated)),
                evaluation={k: evaluated[k] for k in ("evaluation_weights", "episode_seeds_per_mode", "ref_point")},
                **{k: evaluated[k] for k in ("interactions", "raw_choices", "soft_choices", "prefix_terminations")})
            write(output, report)
        n = acquisition["interactions"]
        b = 2 * STEPS + n
        report["baseline_total_steps"] = b
        print(f"ROUTER_PRICE env={env} seed={seed} calibration={n} baseline_steps={b}", flush=True)
        states["baseline"] = train(env, seed, output, resume=states["soft"], steps=b)
        report["baseline_clocks"] = clocks(states["baseline"])
        assert states["baseline"]["env"].mode_history[:len(histories[0])] == [tuple(r) for r in histories[0]]
        budget_policy = {"soft": snapshot(states["baseline"])}
        evaluated = evaluate(states["baseline"], budget_policy, seed, "budget_soft")
        report["evaluations"]["budget_soft"] = dict(endpoint=source.local.source.endpoint(dict(evaluation=evaluated)),
            evaluation={k: evaluated[k] for k in ("evaluation_weights", "episode_seeds_per_mode", "ref_point")},
            **{k: evaluated[k] for k in ("interactions", "raw_choices", "soft_choices", "prefix_terminations")})
        u = STEPS - KWARGS["learning_starts"] + 1
        report.update(candidate_price=dict(training_interactions=2 * STEPS, calibration_interactions=n,
                initialization_probes=4, actor_steps=2 * u, critic_steps=2 * u, encoder_steps=2 * u),
            baseline_price=dict(training_interactions=b, calibration_interactions=0, initialization_probes=4,
                actor_steps=report["baseline_clocks"]["actor"], critic_steps=report["baseline_clocks"]["critic"],
                encoder_steps=report["baseline_clocks"]["encoder"]),
            actual_work=dict(training_interactions=3 * STEPS + n, calibration_interactions=n,
                evaluation_interactions=sum(v["interactions"] for v in report["evaluations"].values()),
                initialization_probes=6, actor_steps=report["baseline_clocks"]["actor"] + u,
                critic_steps=report["baseline_clocks"]["critic"] + u,
                encoder_steps=report["baseline_clocks"]["encoder"] + u, ridge_fits=2 if gate.ready else 0),
            controls_passed=True, complete=True, phase="complete")
        check(report)
        torch.save({name: {k: model[k].state_dict() for k in ("policy", "encoder")}
                    for name, model in dict(policies, budget_soft=budget_policy["soft"]).items()},
                   output.with_suffix(".actors.pt"))
        write(output, report)
        print(f"DONE {PROTOCOL} env={env} seed={seed} cost={json.dumps(report['actual_work'])}", flush=True)
        return report
    finally:
        for state in states.values():
            state["env"].close()
            state["eval_env"].close()


def summarize(root=ROOT):
    result = dict(protocol=PROTOCOL, experiment=specification(), complete=True, environments={}, actual_work={})
    for env in ENVIRONMENTS:
        rows = [json.loads((Path(root) / env.replace("-", "_") / f"seed{seed}.json").read_text()) for seed in SEEDS]
        for seed, row in zip(SEEDS, rows):
            check(row)
            assert (row["env"], row["seed"]) == (env, seed)
            for key, value in row["actual_work"].items():
                result["actual_work"][key] = result["actual_work"].get(key, 0) + value
        methods = {route: {metric: dict(mean=float(np.mean(values)), minimum_seed=float(min(values)), per_seed=values)
            for metric in ("utility", "hv")
            for values in [[r["evaluations"][route]["endpoint"][metric] for r in rows]]} for route in ROUTES}
        mode_delta = np.mean([np.subtract(r["evaluations"]["risk"]["endpoint"]["per_mode_utility"],
            r["evaluations"]["budget_soft"]["endpoint"]["per_mode_utility"]) for r in rows], axis=0).tolist()
        contrasts = {metric: dict(mean=methods["risk"][metric]["mean"] - methods["budget_soft"][metric]["mean"],
            per_seed=[r["evaluations"]["risk"]["endpoint"][metric] - r["evaluations"]["budget_soft"]["endpoint"][metric] for r in rows],
            minimum_seed_difference=methods["risk"][metric]["minimum_seed"] - methods["budget_soft"][metric]["minimum_seed"])
            for metric in ("utility", "hv")}
        passed = contrasts["utility"]["mean"] > 0 and contrasts["hv"]["mean"] >= 0
        passed = passed and all(c["minimum_seed_difference"] >= 0 for c in contrasts.values()) and min(mode_delta) >= 0
        raw_choices = sum(r["evaluations"]["risk"]["raw_choices"] for r in rows)
        value = raw_choices > 0 and methods["risk"]["utility"]["mean"] > methods["soft"]["utility"]["mean"]
        result["environments"][env] = dict(methods=methods, risk_minus_budget_soft=contrasts,
            mean_per_mode_utility_difference=mode_delta, raw_choices=raw_choices,
            priced_screen_passed=passed, routing_value_passed=value, runs=rows)
    passed = all(r["priced_screen_passed"] for r in result["environments"].values())
    value = any(r["routing_value_passed"] for r in result["environments"].values())
    result["decision"] = "fresh_confirmation_required" if passed and value else "hold_frozen_objective_router"
    write(Path(root) / "readout.json", result)
    print("DONE", PROTOCOL, "summary", result["decision"], flush=True)
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--env", choices=ENVIRONMENTS)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--summarize", action="store_true")
    args = parser.parse_args()
    if args.summarize:
        summarize(args.root)
    else:
        run(args.env, args.seed, args.root)
