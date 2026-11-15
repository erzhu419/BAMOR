"""Costed whole-policy acceptance from observable historical simulator states."""

import argparse
import json
from pathlib import Path

import numpy as np
import torch

import run_return_labelled_guard as source


training, router, base, shared, write = source.training, source.router, source.base, source.shared, source.write
PROTOCOL = "historical_state_whole_policy_return_gate_v1"
ROOT = Path("results/iclr_history_return_gate_20261003")
ENVIRONMENTS, SEEDS = source.ENVIRONMENTS, (2901, 2902, 2903)
CUT, PROPOSAL, CAPTURE_EVERY, SLOTS = 75000, 100000, 6250, 4
HISTORY_WINDOW, HORIZON = 64, 1000
KWARGS, SWITCH_INTERVAL = source.KWARGS, source.SWITCH_INTERVAL
MODEL_NAMES = ("prefix", "proposal", "priced_history", "priced_current")
METHOD_NAMES = ("prefix", "proposal", "history", "current", "blind", "priced_history", "priced_current")


def specification():
    return dict(environments=list(ENVIRONMENTS), seeds=list(SEEDS), method=shared.source.METHOD,
        training_kwargs=KWARGS, switch_interval=SWITCH_INTERVAL, incumbent_steps=CUT, proposal_steps=PROPOSAL,
        capture_every=CAPTURE_EVERY, selected_states=SLOTS, history_window=HISTORY_WINDOW, horizon=HORIZON,
        pool="fixed_training_clock_snapshots; factual_physics_observation_episode_age_and_recent_transitions; no_reset",
        selection="newest_first_then_farthest_minimum_squared_distance_in_pool_standardized_observable_transition_features; "
                  "ties_latest; no_learned_latent_versions_mode_labels_or_returns",
        preferences="same_four_distinct_full21_indices_in_both_arms; RNG104100000+seed",
        context="each_frozen_encoder_reencodes_recent_factual_transitions; fresh_trackers_then_paid_branch_transitions",
        trials="four_paired_historical_states_vs_four_paired_current_state_trials; own_deterministic_policy_encoder; "
               "natural_unlocked_schedule; stop_at_episode_end_or_horizon; no_critic_bootstrap",
        gate="positive_equal_pair_mean_actual_undiscounted_weighted_proposal_minus_prefix_return_accepts_entire_"
             "proposal_actor_encoder; else_prefix; identical_gate_in_both_arms",
        blind="whole_policy_fair_coin_RNG104300000+seed; charged_history_trials_ignored",
        native_price="per_arm_proposal_steps+actual_trial_interactions-2_training_plus_four_probes; "
                     "candidate_proposal_steps+trials_plus_two_probes; native_has_more_learner_updates",
        evaluation="choices_frozen_before_full21_four_locked_modes_two_paired_resets; four_actor_encoder_endpoints_"
                   "reused_for_all_whole_policy_choices",
        screen="history_mean_utility_HV_minimum_seed_utility_HV_all_mean_modes_at_least_own_priced_native; "
               "mean_utility_HV_at_least_current_and_blind; no_harmful_accepted_proposal; "
               "HC_low_speed_at_least_prefix; genuine_positive_coverage_value_required",
        privilege="paid_simulator_state_forks; mode_ids_only_post_choice_coverage_audit_and_evaluation; "
                  "partial_episode_and_historical_state_distribution_are_explicit")


def select_slots(features, count=SLOTS):
    x = np.asarray(features, dtype=np.float64)
    scale = x.std(0)
    z = (x - x.mean(0)) / np.where(scale == 0, 1., scale)
    selected = [len(x) - 1]
    while len(selected) < count:
        distances = ((z[:, None] - z[selected][None]) ** 2).sum(-1).min(1)
        distances[selected] = -np.inf
        selected.append(len(x) - 1 - int(np.argmax(distances[::-1])))
    return selected


class HistoryCapture:
    def __init__(self):
        self.times = sorted({*range(CAPTURE_EVERY, PROPOSAL + 1, CAPTURE_EVERY), CUT, PROPOSAL})
        self.step, self.pool, self.models, self.state = self.times[0], [], {}, None

    def __call__(self, state):
        step, buffer = state["step"], state["trainer"].buffer
        if step % CAPTURE_EVERY == 0 or step == PROPOSAL:
            n = min(HISTORY_WINDOW, buffer.size)
            indices = (buffer.ptr - n + np.arange(n)) % buffer.capacity
            self.pool.append(dict(step=step, env=base.fork_env(state["env"]), obs=state["obs"].copy(),
                episode_age=state["episode_steps"], feature=state["trans_tracker"].features().copy(),
                history={k: getattr(buffer, k)[indices].copy() for k in ("states", "actions", "rewards", "next_states")}))
        if step == CUT:
            self.models["prefix"] = router.snapshot(state)
        if step == PROPOSAL:
            self.models["proposal"] = router.snapshot(state)
            self.state = dict(state, env=base.fork_env(state["env"]), eval_env=base.fork_env(state["eval_env"]))
            return True
        self.step = self.times[self.times.index(step) + 1]
        return False

    def close(self):
        for slot in self.pool:
            slot["env"].close()


@torch.no_grad()
def continuation(env, model, slot, weight):
    assert not env._lock_mode
    trans = training.TransitionContextTracker(len(weight))
    learned = training.LearnedTransitionContextTracker(model["encoder"], latent_dim=training.LATENT_CONTEXT_DIM)
    history = slot["history"]
    for values in zip(*(history[k] for k in ("states", "actions", "rewards", "next_states"))):
        trans.update(*values)
        learned.update(*values, "cpu")
    obs, total, before = slot["obs"].copy(), np.zeros(len(weight)), env._total_steps
    initial_context = training.build_mujoco_context(shared.source.METHOD, weight, trans_tracker=trans, learned_tracker=learned)
    for _ in range(HORIZON):
        context = training.build_mujoco_context(shared.source.METHOD, weight, trans_tracker=trans, learned_tracker=learned)
        action = model["policy"].get_action(torch.as_tensor(obs).float()[None],
            torch.as_tensor(context).float()[None], deterministic=True).numpy().ravel()
        obs_new, reward, terminated, truncated, _ = env.step(action)
        total += reward
        trans.update(obs, action, reward, obs_new)
        learned.update(obs, action, reward, obs_new, "cpu")
        obs = obs_new
        if terminated or truncated:
            break
    return dict(return_vector=total.tolist(), steps=env._total_steps - before,
        initial_context=initial_context.tolist(), final_observation=obs.tolist(),
        terminated=bool(terminated), truncated=bool(truncated),
        reencoded_rows=len(history["states"]))


def gate(rows):
    margins = [float(np.dot(r["weight"], np.subtract(r["branches"]["proposal"]["return_vector"],
                                                r["branches"]["prefix"]["return_vector"]))) for r in rows]
    score = float(np.mean(margins))
    return dict(chosen="proposal" if score > 0 else "prefix", score=score, margins=margins)


def collect(capture, indices, weight_indices, seed):
    weights = training.full_simplex_grid(capture.models["prefix"]["reward_dim"])
    rows = []
    with base.isolated_rng(104200000 + seed):
        for index, wi in zip(indices, weight_indices):
            slot = capture.pool[index]
            branches = {name: base.fork_env(slot["env"]) for name in ("prefix", "proposal")}
            try:
                for branch in branches.values():
                    np.testing.assert_array_equal(branch.unwrapped.data.qpos, slot["env"].unwrapped.data.qpos)
                    np.testing.assert_array_equal(branch.unwrapped.data.qvel, slot["env"].unwrapped.data.qvel)
                    assert branch.unwrapped.data.time == slot["env"].unwrapped.data.time
                    assert branch._total_steps == slot["step"]
                outcomes = {name: continuation(branch, capture.models[name], slot, weights[wi])
                            for name, branch in branches.items()}
                rows.append(dict(capture_step=slot["step"], episode_age=slot["episode_age"],
                    weight_index=int(wi), weight=weights[wi].tolist(), branches=outcomes, paired_physics=True))
            finally:
                for branch in branches.values():
                    branch.close()
    return dict(rows=rows, decision=gate(rows),
        interactions=sum(b["steps"] for r in rows for b in r["branches"].values()),
        reencoded_rows=sum(b["reencoded_rows"] for r in rows for b in r["branches"].values()))


def path_prices(trials):
    u = PROPOSAL - KWARGS["learning_starts"] + 1
    prices = {}
    for arm in ("history", "current"):
        n = trials[arm]["interactions"]
        prices[arm] = dict(training_interactions=PROPOSAL, calibration_interactions=n, initialization_probes=2,
            actor_steps=u, critic_steps=u, encoder_steps=u)
        prices["priced_" + arm] = dict(training_interactions=PROPOSAL + n - 2, calibration_interactions=0,
            initialization_probes=4, actor_steps=u + n - 2, critic_steps=u + n - 2, encoder_steps=u + n - 2)
    prices["blind"] = dict(prices["history"])
    return prices


def train(env, seed, output, steps, resume=None, capture=None):
    if capture is None:
        capture = base.Capture()
        capture.step = steps
    training.train_method(env, shared.source.METHOD, steps, SWITCH_INTERVAL, seed, str(output.parent),
                          resume_state=resume, state_probe=capture, **KWARGS)
    return capture.state


def check(record):
    assert (record["protocol"], record["experiment"], record["complete"]) == (PROTOCOL, specification(), True)
    assert record["env"] in ENVIRONMENTS and record["seed"] in SEEDS and record["controls_passed"]
    assert set(record["evaluations"]) == set(MODEL_NAMES)
    for arm in ("history", "current"):
        trial = record["trials"][arm]
        assert len(trial["rows"]) == SLOTS and trial["decision"] == gate(trial["rows"])
        assert trial["interactions"] == sum(b["steps"] for r in trial["rows"] for b in r["branches"].values())
        assert 2 * SLOTS <= trial["interactions"] <= 2 * SLOTS * HORIZON
        assert all(r["paired_physics"] for r in trial["rows"])
        assert record["clocks"]["priced_" + arm] == {k: PROPOSAL + trial["interactions"] - 2 - KWARGS["learning_starts"] + 1
                                                    for k in ("actor", "critic", "encoder")}
        assert record["choices"][arm] == trial["decision"]["chosen"]
    assert [r["weight_index"] for r in record["trials"]["history"]["rows"]] == [r["weight_index"] for r in record["trials"]["current"]["rows"]]
    assert record["path_prices"] == path_prices(record["trials"])
    n = sum(t["interactions"] for t in record["trials"].values())
    u = PROPOSAL + n - 4 - KWARGS["learning_starts"] + 1
    assert record["actual_work"] == dict(training_interactions=PROPOSAL + n - 4, calibration_interactions=n,
        initialization_probes=6, actor_steps=u, critic_steps=u, encoder_steps=u,
        reencoded_rows=sum(t["reencoded_rows"] for t in record["trials"].values()),
        evaluation_interactions=sum(e["interactions"] for e in record["evaluations"].values()))
    assert record["clocks"]["proposal"] == {k: PROPOSAL - KWARGS["learning_starts"] + 1 for k in ("actor", "critic", "encoder")}
    for value in record["evaluations"].values():
        assert value["evaluation"] == record["evaluations"]["prefix"]["evaluation"]


def run(env, seed, root=ROOT):
    root = Path(root)
    frozen = json.loads((root / "protocol.json").read_text())
    assert (frozen["protocol"], frozen["experiment"]) == (PROTOCOL, specification())
    assert env in ENVIRONMENTS and seed in SEEDS
    output = root / env.replace("-", "_") / f"seed{seed}.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.exists():
        previous = json.loads(output.read_text())
        if previous["complete"]:
            check(previous)
            assert (previous["env"], previous["seed"]) == (env, seed)
            return previous
        raise RuntimeError("Incomplete attempt retained; inspect before a new-root retry")
    report = dict(protocol=PROTOCOL, experiment=specification(), env=env, seed=seed,
        complete=False, controls_passed=False, phase="factual_training", evaluations={}, clocks={}, trials={})
    write(output, report)
    capture, states = HistoryCapture(), {}
    try:
        state = train(env, seed, output, PROPOSAL, capture=capture)
        models = capture.models
        states["proposal"] = state
        report["clocks"]["proposal"] = router.clocks(state)
        selected = select_slots([s["feature"] for s in capture.pool], SLOTS)
        weights = training.full_simplex_grid(models["prefix"]["reward_dim"])
        wi = np.random.default_rng(104100000 + seed).permutation(len(weights))[:SLOTS]
        report["phase"] = "return_trials"
        write(output, report)
        for arm, indices in (("history", selected), ("current", [len(capture.pool) - 1] * SLOTS)):
            report["trials"][arm] = collect(capture, indices, wi, seed)
            write(output, report)
        report["choices"] = {arm: report["trials"][arm]["decision"]["chosen"] for arm in ("history", "current")}
        report["choices"]["blind"] = ("prefix", "proposal")[np.random.default_rng(104300000 + seed).integers(2)]
        # These labels are an audit after the choices, never a source-selection input.
        report["coverage"] = dict(pool_steps=[s["step"] for s in capture.pool],
            selected_indices=selected, selected_steps=[capture.pool[i]["step"] for i in selected],
            pool_features=[s["feature"].tolist() for s in capture.pool],
            observed_mode_ids=[s["env"].current_mode_idx for s in capture.pool],
            selected_mode_ids=[capture.pool[i]["env"].current_mode_idx for i in selected])
        write(output, report)
        for arm in ("history", "current"):
            name = "priced_" + arm
            stop = PROPOSAL + report["trials"][arm]["interactions"] - 2
            report["phase"] = name
            write(output, report)
            states[name] = train(env, seed, output, stop, resume=source.clone_state(state))
            models[name] = router.snapshot(states[name])
            report["clocks"][name] = router.clocks(states[name])
            shared.source.local.crossed.common_contract(states[name]["trainer"])
            assert (states[name]["trainer"].buffer.regimes[:stop] == -1).all()
            write(output, report)
        report["phase"] = "evaluation"
        write(output, report)
        for name in MODEL_NAMES:
            report["evaluations"][name] = shared.endpoint_record(router.evaluate(state, {"soft": models[name]}, seed, "budget_soft"))
            write(output, report)
        n = sum(t["interactions"] for t in report["trials"].values())
        u = PROPOSAL + n - 4 - KWARGS["learning_starts"] + 1
        report.update(path_prices=path_prices(report["trials"]),
            actual_work=dict(training_interactions=PROPOSAL + n - 4, calibration_interactions=n,
                initialization_probes=6, actor_steps=u, critic_steps=u, encoder_steps=u,
                reencoded_rows=sum(t["reencoded_rows"] for t in report["trials"].values()),
                evaluation_interactions=sum(e["interactions"] for e in report["evaluations"].values())),
            complete=True, controls_passed=True, phase="complete")
        check(report)
        torch.save({name: {k: models[name][k].state_dict() for k in ("policy", "encoder")} for name in MODEL_NAMES}, output.with_suffix(".actors.pt"))
        write(output, report)
        print(f"DONE {PROTOCOL} env={env} seed={seed} choices={report['choices']} cost={report['actual_work']}", flush=True)
        return report
    finally:
        capture.close()
        for state in states.values():
            state["env"].close()
            state["eval_env"].close()


def summarize(root=ROOT):
    result = dict(protocol=PROTOCOL, experiment=specification(), complete=True, environments={}, actual_work={})
    coverage_value = False
    for env in ENVIRONMENTS:
        rows = [json.loads((Path(root) / env.replace("-", "_") / f"seed{seed}.json").read_text()) for seed in SEEDS]
        methods, harmful = {}, []
        for seed, row in zip(SEEDS, rows):
            check(row)
            assert (row["env"], row["seed"]) == (env, seed)
            for k, v in row["actual_work"].items():
                result["actual_work"][k] = result["actual_work"].get(k, 0) + v
            if row["choices"]["history"] == "proposal":
                old, new = [row["evaluations"][m]["endpoint"] for m in ("prefix", "proposal")]
                if any(new[m] < old[m] for m in ("utility", "hv")) or any(a < b for a, b in zip(new["per_mode_utility"], old["per_mode_utility"])):
                    harmful.append(seed)
        for name in METHOD_NAMES:
            values = [r["evaluations"][r["choices"].get(name, name)]["endpoint"] for r in rows]
            methods[name] = {m: dict(mean=float(np.mean(v)), minimum_seed=min(v), per_seed=v) for m in ("utility", "hv")
                             for v in [[e[m] for e in values]]}
            methods[name]["per_seed_mode_utility"] = [e["per_mode_utility"] for e in values]
            methods[name]["mean_mode_utility"] = np.mean(methods[name]["per_seed_mode_utility"], 0).tolist()
        contrasts = {}
        for a, b in (("history", "priced_history"), ("current", "priced_current"), ("history", "current"), ("history", "blind")):
            metrics = {m: dict(mean=methods[a][m]["mean"] - methods[b][m]["mean"],
                minimum_seed_difference=methods[a][m]["minimum_seed"] - methods[b][m]["minimum_seed"],
                per_seed=np.subtract(methods[a][m]["per_seed"], methods[b][m]["per_seed"]).tolist()) for m in ("utility", "hv")}
            modes = np.subtract(methods[a]["mean_mode_utility"], methods[b]["mean_mode_utility"]).tolist()
            passed = all(v["mean"] >= 0 for v in metrics.values())
            if b.startswith("priced_"):
                passed = passed and min(modes) >= 0 and all(v["minimum_seed_difference"] >= 0 for v in metrics.values())
            contrasts[a + "_minus_" + b] = dict(metrics=metrics, mean_mode_utility_difference=modes, passed=passed)
        gain = contrasts["history_minus_current"]["metrics"]
        coverage_value = coverage_value or (all(v["mean"] >= 0 for v in gain.values()) and any(v["mean"] > 0 for v in gain.values()))
        retention = env != "mo-halfcheetah-v5" or methods["history"]["mean_mode_utility"][1] >= methods["prefix"]["mean_mode_utility"][1]
        passed = retention and not harmful and all(contrasts[k]["passed"] for k in (
            "history_minus_priced_history", "history_minus_current", "history_minus_blind"))
        result["environments"][env] = dict(methods=methods, contrasts=contrasts, harmful_history_acceptances=harmful,
            retention_passed=retention, passed=passed,
            trials=[r["trials"] for r in rows], choices=[r["choices"] for r in rows],
            coverage=[r["coverage"] for r in rows], path_prices=[r["path_prices"] for r in rows])
    result["coverage_value_seen"] = coverage_value
    result["decision"] = "fresh_confirmation_required" if coverage_value and all(e["passed"] for e in result["environments"].values()) else "hold_historical_state_return_gate"
    write(Path(root) / "compact_readout.json", result)
    print("DONE", PROTOCOL, "summary", result["decision"], flush=True)
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--env", choices=ENVIRONMENTS)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--summarize", action="store_true")
    args = parser.parse_args()
    summarize(args.root) if args.summarize else run(args.env, args.seed, args.root)
