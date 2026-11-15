"""Crossed reset/preference measurements for costed scene-mean acceptance."""

import argparse
import json
from pathlib import Path

import numpy as np
import torch

import run_history_return_gate as history


training, router, base, shared, write = history.training, history.router, history.base, history.shared, history.write
PROTOCOL = "crossed_scene_mean_return_gate_v1"
ROOT = Path("results/iclr_crossed_scene_return_20261003")
ENVIRONMENTS, SEEDS = history.ENVIRONMENTS, (3001, 3002, 3003)
PREFERENCES, REPEATS = 4, 2
MODEL_NAMES = ("prefix", "proposal", "priced")
METHOD_NAMES = (*MODEL_NAMES, "scene", "mean", "blind")


def specification():
    return dict(environments=list(ENVIRONMENTS), seeds=list(SEEDS), method=shared.source.METHOD,
        training_kwargs=history.KWARGS, switch_interval=history.SWITCH_INTERVAL,
        incumbent_steps=history.CUT, proposal_steps=history.PROPOSAL, capture_every=history.CAPTURE_EVERY,
        scenes=history.SLOTS, preferences=PREFERENCES, reset_repeats=REPEATS, horizon=history.HORIZON,
        source="uninterrupted_native_trajectory; newest_then_farthest_standardized_observable_features; no_mode_labels",
        measurements="full_scene_preference_reset_cartesian_product; four_uniform_without_replacement_full21_weights; "
                     "same_weights_all_scenes; RNG105100000+seed",
        resets="105200000+1000*seed+10*scene+repeat; shared_across_preferences_and_actor_pairs; "
               "fresh_trackers_own_frozen_encoder; actual_episode_end_or_horizon",
        scene_privilege="fork_observed_simulator_instance_then_lock_existing_physics_and_reset; "
                        "no_mode_id_selection_or_teacher_labels; labels_after_choices_only",
        gate="equal_preference_reset_mean_per_scene; all_observed_scene_means_nonnegative_and_overall_positive_"
             "accept_whole_proposal; else_prefix; preference_tradeoffs_allowed; no_confidence_bound",
        controls="overall_mean_gate_and_whole_policy_coin_RNG105300000+seed_use_same_paid_measurements",
        native_price="proposal_steps+actual_trial_interactions-2_training_plus_four_probes; "
                     "candidate_proposal_steps+trials_plus_two_probes; native_has_more_learner_updates",
        evaluation="freeze_choices_then_full21_four_locked_modes_two_paired_resets; three_endpoints_reused",
        screen="both_envs_scene_mean_utility_HV_minimum_seed_utility_HV_all_mean_modes_at_least_priced; "
               "mean_utility_HV_at_least_mean_and_blind; HC_low_speed_at_least_prefix; "
               "no_accepted_proposal_regression_in_utility_HV_or_mode_mean; strict_value_over_mean_in_one_env")


@torch.no_grad()
def episode(env, model, weight, seed):
    before = env._total_steps
    obs, _ = env.reset(seed=seed)
    assert env._total_steps == before and env._lock_mode
    start = dict(observation=obs.tolist(), qpos=env.unwrapped.data.qpos.tolist(),
                 qvel=env.unwrapped.data.qvel.tolist(), time=float(env.unwrapped.data.time))
    trans = training.TransitionContextTracker(len(weight))
    learned = training.LearnedTransitionContextTracker(model["encoder"], latent_dim=training.LATENT_CONTEXT_DIM)
    total = np.zeros(len(weight))
    for _ in range(history.HORIZON):
        context = training.build_mujoco_context(shared.source.METHOD, weight, trans_tracker=trans, learned_tracker=learned)
        action = model["policy"].get_action(torch.as_tensor(obs).float()[None],
            torch.as_tensor(context).float()[None], deterministic=True).numpy().ravel()
        ns, reward, terminated, truncated, _ = env.step(action)
        total += reward
        trans.update(obs, action, reward, ns)
        learned.update(obs, action, reward, ns, "cpu")
        obs = ns
        if terminated or truncated:
            break
    return dict(start=start, return_vector=total.tolist(), steps=env._total_steps - before,
                terminated=bool(terminated), truncated=bool(truncated))


def decisions(rows):
    cells, margins = set(), [[] for _ in range(history.SLOTS)]
    for row in rows:
        cell = (row["scene"], row["preference"], row["repeat"])
        assert cell not in cells
        cells.add(cell)
        a, b = [row["branches"][k]["return_vector"] for k in ("prefix", "proposal")]
        margins[row["scene"]].append(float(np.dot(row["weight"], np.subtract(b, a))))
    assert cells == {(s, p, r) for s in range(history.SLOTS) for p in range(PREFERENCES) for r in range(REPEATS)}
    means = [float(np.mean(m)) for m in margins]
    overall = float(np.mean(means))
    return dict(scene="proposal" if overall > 0 and min(means) >= 0 else "prefix",
                mean="proposal" if overall > 0 else "prefix", scene_means=means, overall=overall)


def collect(capture, selected, weight_indices, seed):
    weights = training.full_simplex_grid(capture.models["prefix"]["reward_dim"])
    rows = []
    with base.isolated_rng(105400000 + seed):
        for scene, index in enumerate(selected):
            slot = capture.pool[index]
            for preference, wi in enumerate(weight_indices):
                for repeat in range(REPEATS):
                    reset_seed = 105200000 + 1000 * seed + 10 * scene + repeat
                    branches = {k: base.fork_env(slot["env"]) for k in ("prefix", "proposal")}
                    try:
                        results = {}
                        for name, env in branches.items():
                            assert env._total_steps == slot["step"] and not env._lock_mode
                            env._lock_mode = True
                            mode_events = len(env.mode_history)
                            results[name] = episode(env, capture.models[name], weights[wi], reset_seed)
                            assert len(env.mode_history) == mode_events
                        assert results["prefix"]["start"] == results["proposal"]["start"]
                        rows.append(dict(scene=scene, preference=preference, repeat=repeat,
                            capture_step=slot["step"], weight_index=int(wi), weight=weights[wi].tolist(),
                            reset_seed=reset_seed, branches=results, paired_reset_physics=True))
                    finally:
                        for env in branches.values():
                            env.close()
    return dict(rows=rows, decisions=decisions(rows),
        interactions=sum(b["steps"] for r in rows for b in r["branches"].values()), resets=2 * len(rows))


def prices(n):
    u = history.PROPOSAL - history.KWARGS["learning_starts"] + 1
    candidate = dict(training_interactions=history.PROPOSAL, calibration_interactions=n,
        initialization_probes=2, actor_steps=u, critic_steps=u, encoder_steps=u)
    return dict(candidate=candidate, priced=dict(training_interactions=history.PROPOSAL + n - 2,
        calibration_interactions=0, initialization_probes=4,
        actor_steps=u + n - 2, critic_steps=u + n - 2, encoder_steps=u + n - 2))


def check(record):
    assert (record["protocol"], record["experiment"], record["complete"]) == (PROTOCOL, specification(), True)
    assert record["env"] in ENVIRONMENTS and record["seed"] in SEEDS and record["controls_passed"]
    trial = record["trial"]
    assert trial["decisions"] == decisions(trial["rows"])
    assert all(r["paired_reset_physics"] and r["branches"]["prefix"]["start"] == r["branches"]["proposal"]["start"]
               for r in trial["rows"])
    n = sum(b["steps"] for r in trial["rows"] for b in r["branches"].values())
    assert n == trial["interactions"] and trial["resets"] == 2 * len(trial["rows"])
    assert trial["resets"] <= n <= trial["resets"] * history.HORIZON
    for name in ("scene", "mean"):
        assert record["choices"][name] == trial["decisions"][name]
    assert record["path_prices"] == prices(n)
    assert record["clocks"]["proposal"] == {k: history.PROPOSAL - history.KWARGS["learning_starts"] + 1 for k in ("actor", "critic", "encoder")}
    u = history.PROPOSAL + n - 2 - history.KWARGS["learning_starts"] + 1
    assert record["clocks"]["priced"] == {k: u for k in ("actor", "critic", "encoder")}
    assert set(record["evaluations"]) == set(MODEL_NAMES)
    assert all(e["evaluation"] == record["evaluations"]["prefix"]["evaluation"] for e in record["evaluations"].values())
    assert record["actual_work"] == dict(training_interactions=history.PROPOSAL + n - 2,
        calibration_interactions=n, calibration_resets=trial["resets"], initialization_probes=4,
        actor_steps=u, critic_steps=u, encoder_steps=u,
        evaluation_interactions=sum(e["interactions"] for e in record["evaluations"].values()))


def run(env, seed, root=ROOT):
    root = Path(root)
    assert json.loads((root / "protocol.json").read_text()) == dict(protocol=PROTOCOL, experiment=specification())
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
        complete=False, controls_passed=False, phase="factual_training", evaluations={}, clocks={})
    write(output, report)
    capture, states = history.HistoryCapture(), {}
    try:
        state = history.train(env, seed, output, history.PROPOSAL, capture=capture)
        states["proposal"] = state
        models = capture.models
        report["clocks"]["proposal"] = router.clocks(state)
        selected = history.select_slots([s["feature"] for s in capture.pool], history.SLOTS)
        weights = training.full_simplex_grid(models["prefix"]["reward_dim"])
        wi = np.random.default_rng(105100000 + seed).permutation(len(weights))[:PREFERENCES]
        report["phase"] = "crossed_trials"
        write(output, report)
        report["trial"] = collect(capture, selected, wi, seed)
        report["choices"] = {name: report["trial"]["decisions"][name] for name in ("scene", "mean")}
        report["choices"]["blind"] = ("prefix", "proposal")[np.random.default_rng(105300000 + seed).integers(2)]
        # Simulator labels are only a post-choice coverage audit.
        report["coverage"] = dict(selected_steps=[capture.pool[i]["step"] for i in selected],
            selected_features=[capture.pool[i]["feature"].tolist() for i in selected],
            selected_mode_ids=[capture.pool[i]["env"].current_mode_idx for i in selected],
            observed_mode_ids=[s["env"].current_mode_idx for s in capture.pool])
        n = report["trial"]["interactions"]
        report["phase"] = "priced_training"
        write(output, report)
        states["priced"] = history.train(env, seed, output, history.PROPOSAL + n - 2,
                                        resume=history.source.clone_state(state))
        models["priced"] = router.snapshot(states["priced"])
        report["clocks"]["priced"] = router.clocks(states["priced"])
        shared.source.local.crossed.common_contract(states["priced"]["trainer"])
        assert (states["priced"]["trainer"].buffer.regimes[:history.PROPOSAL + n - 2] == -1).all()
        report["phase"] = "evaluation"
        write(output, report)
        for name in MODEL_NAMES:
            report["evaluations"][name] = shared.endpoint_record(router.evaluate(state, {"soft": models[name]}, seed, "budget_soft"))
            write(output, report)
        u = history.PROPOSAL + n - 2 - history.KWARGS["learning_starts"] + 1
        report.update(path_prices=prices(n), actual_work=dict(training_interactions=history.PROPOSAL + n - 2,
            calibration_interactions=n, calibration_resets=report["trial"]["resets"], initialization_probes=4,
            actor_steps=u, critic_steps=u, encoder_steps=u,
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
    added_value = False
    for env in ENVIRONMENTS:
        records = [json.loads((Path(root) / env.replace("-", "_") / f"seed{seed}.json").read_text()) for seed in SEEDS]
        methods, harmful = {}, []
        for seed, record in zip(SEEDS, records):
            check(record)
            assert (record["env"], record["seed"]) == (env, seed)
            for k, v in record["actual_work"].items():
                result["actual_work"][k] = result["actual_work"].get(k, 0) + v
            if record["choices"]["scene"] == "proposal":
                old, new = [record["evaluations"][k]["endpoint"] for k in ("prefix", "proposal")]
                if any(new[k] < old[k] for k in ("utility", "hv")) or any(a < b for a, b in zip(new["per_mode_utility"], old["per_mode_utility"])):
                    harmful.append(seed)
        for name in METHOD_NAMES:
            values = [r["evaluations"][r["choices"].get(name, name)]["endpoint"] for r in records]
            methods[name] = {k: dict(mean=float(np.mean(v)), minimum_seed=min(v), per_seed=v) for k in ("utility", "hv")
                             for v in [[e[k] for e in values]]}
            methods[name]["mean_mode_utility"] = np.mean([v["per_mode_utility"] for v in values], 0).tolist()
            methods[name]["per_seed_mode_utility"] = [v["per_mode_utility"] for v in values]
        comparisons = {}
        for other in ("priced", "mean", "blind"):
            metrics = {k: dict(mean=methods["scene"][k]["mean"] - methods[other][k]["mean"],
                minimum_seed_difference=methods["scene"][k]["minimum_seed"] - methods[other][k]["minimum_seed"],
                per_seed=np.subtract(methods["scene"][k]["per_seed"], methods[other][k]["per_seed"]).tolist()) for k in ("utility", "hv")}
            modes = np.subtract(methods["scene"]["mean_mode_utility"], methods[other]["mean_mode_utility"]).tolist()
            passed = all(v["mean"] >= 0 for v in metrics.values())
            if other == "priced":
                passed = passed and min(modes) >= 0 and all(v["minimum_seed_difference"] >= 0 for v in metrics.values())
            comparisons[other] = dict(metrics=metrics, mean_mode_difference=modes, passed=passed)
        gain = comparisons["mean"]["metrics"]
        added_value = added_value or (all(v["mean"] >= 0 for v in gain.values()) and any(v["mean"] > 0 for v in gain.values()))
        retention = env != "mo-halfcheetah-v5" or methods["scene"]["mean_mode_utility"][1] >= methods["prefix"]["mean_mode_utility"][1]
        result["environments"][env] = dict(methods=methods, comparisons=comparisons,
            harmful_acceptances=harmful, retention_passed=retention,
            passed=retention and not harmful and all(c["passed"] for c in comparisons.values()),
            choices=[r["choices"] for r in records], scene_statistics=[r["trial"]["decisions"] for r in records],
            coverage=[r["coverage"] for r in records], path_prices=[r["path_prices"] for r in records])
    result["added_value_seen"] = added_value
    result["decision"] = "fresh_confirmation_required" if added_value and all(e["passed"] for e in result["environments"].values()) else "hold_crossed_scene_mean_gate"
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
