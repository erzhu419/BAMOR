"""Fresh matched full-training comparison of soft versus mean/raw objectives."""

import argparse
import json
from pathlib import Path
import time

import numpy as np
import torch

import run_mean_raw_objective_probe as local


base, training, write_report = local.base, local.training, local.write_report
PROTOCOL = "objective_from_initialization_v1"
ROOT = Path("results/iclr_objective_from_start_20261002")
SEEDS = tuple(range(2501, 2506))
ENVIRONMENTS = local.source.ENVIRONMENTS
METHOD = "bamor_common_target_projected_fixed"
CONDITIONS = ("soft", "mean_raw")
TOTAL_STEPS, SWITCH_INTERVAL = 75000, 25000
KWARGS = dict(local.KWARGS, log_freq=5000)


class MeanRawTrainer(training.ProjectedCommonTargetMOSACTrainer):
    critic_target_rule = "mean_action_entropy_free_discounted_vector"
    actor_update_rule = "mean_action_raw_projected_value"

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        local.crossed.common_contract(self)
        assert type(self.policy) is training.PolicyNet
        self.objective_updates = 0

    def update(self, w_anchor=None, anchor_eta=0.):
        if self.buffer.size < self.batch_size:
            return None, None
        local.raw_update(self)
        self.objective_updates += 1
        return None, None


TRAINERS = {"soft": training.ProjectedCommonTargetMOSACTrainer, "mean_raw": MeanRawTrainer}


def specification():
    return dict(environments=list(ENVIRONMENTS), seeds=list(SEEDS), conditions=list(CONDITIONS),
        method=METHOD, total_steps=TOTAL_STEPS, switch_interval=SWITCH_INTERVAL,
        training_kwargs=KWARGS, trainer_classes={k: v.__name__ for k, v in TRAINERS.items()},
        initialization="same_seed_constructor_from_scratch; no_checkpoint_or_optimizer_reuse",
        raw_target=local.specification()["target"], raw_actor=local.specification()["actor"],
        critic_loss="unchanged_trace_normalized_vector_plus_projection_mse",
        collection="existing_Gaussian_policy_behavior_after_shared_random_action_warmup",
        allocation="unchanged_fixed_common_target_uniform_stratified_mixed_distribution",
        evaluation="locked_final_actor_full21_simplex; no_archive_or_checkpoint_selection",
        evaluation_repeats=base.REPEATS, evaluation_horizon=base.HORIZON)


def budget(env):
    updates = TOTAL_STEPS - KWARGS["learning_starts"] + 1
    modes = base.previous.previous.previous.extension.ENVIRONMENTS[env]["num_modes"]
    return dict(training_interactions=TOTAL_STEPS,
        evaluation_interactions=modes * 21 * base.REPEATS * base.HORIZON,
        environment_initialization_probe_steps=2,
        actor_optimizer_steps=updates, critic_optimizer_steps=updates,
        encoder_optimizer_steps=updates)


def check_record(record):
    assert (record["protocol"], record["experiment"], record["complete"]) == (PROTOCOL, specification(), True)
    assert record["env"] in ENVIRONMENTS and record["condition"] in CONDITIONS and record["seed"] in SEEDS
    assert record["controls_passed"] and record["cold_initialization_passed"]
    updates = TOTAL_STEPS - KWARGS["learning_starts"] + 1
    for key, upper in budget(record["env"]).items():
        actual = record["completed_budget"][key]
        assert 0 < actual <= upper if key == "evaluation_interactions" else actual == upper
    work = record["learner_work"]
    assert work == dict(critic_optimizer_steps=updates, actor_optimizer_steps=updates,
        critic_rows=updates * KWARGS["batch_size"], target_candidate_rows=updates * KWARGS["batch_size"],
        actor_candidate_rows=updates * KWARGS["batch_size"])
    assert record["optimizer_clocks"] == dict(actor=updates, critic=updates, encoder=updates)
    assert record["trainer_class"] == TRAINERS[record["condition"]].__name__
    assert record["raw_objective_updates"] == (updates if record["condition"] == "mean_raw" else 0)
    assert record["first_update_step"] == KWARGS["learning_starts"]
    assert record["representation_replay_labels_unprivileged"]
    allocation = record["allocation"]
    assert record["selector_class"] == "FixedCommonTargetSamplingSelector"
    assert not allocation["selection_uses_scores"]
    operators = allocation["operators"]
    if sum(allocation["counts"].values()) < len(operators):
        assert allocation["last_probs"] == {op: float(op == allocation["current_operator"]) for op in operators}
    else:
        assert all(p == 1 / 3 for p in allocation["last_probs"].values())
    expected = dict(preference_relabeling_strategy="uniform", critic_relabel_copies=1,
        envelope_target_candidates=1, envelope_policy_candidates=1,
        vector_critic_coef=0.0, behavior_clone_coef=0.0)
    assert all(config == expected for config in allocation["operator_trainer_configs"].values())
    assert record["evaluation"]["evaluation_weights"] == training.full_simplex_grid(
        len(record["evaluation"]["evaluation_weights"][0])).tolist()
    for key in ("utility", "hv"):
        assert np.isfinite(record["endpoint"][key])


def run(env, condition, seed, root=ROOT):
    root = Path(root)
    frozen = json.loads((root / "protocol.json").read_text())
    assert (frozen["protocol"], frozen["experiment"]) == (PROTOCOL, specification())
    assert env in ENVIRONMENTS and condition in CONDITIONS and seed in SEEDS
    output = root / env.replace("-", "_") / condition / f"seed{seed}.json"
    if output.exists():
        saved = json.loads(output.read_text())
        assert (saved["protocol"], saved["env"], saved["condition"], saved["seed"], saved["experiment"]) == (
            PROTOCOL, env, condition, seed, specification())
        if saved["complete"]:
            check_record(saved)
            return saved
        raise RuntimeError("Incomplete attempt preserved; inspect before a new-root retry")
    output.parent.mkdir(parents=True, exist_ok=True)
    report = dict(protocol=PROTOCOL, env=env, condition=condition, seed=seed,
        experiment=specification(), complete=False, planned_budget_upper_bound=budget(env),
        completed_budget=dict.fromkeys(budget(env), 0), controls_passed=False)
    write_report(output, report)
    capture = base.Capture()
    capture.step = TOTAL_STEPS
    first_updates = []

    def factory(*args, **kwargs):
        trainer = TRAINERS[condition](*args, **kwargs)
        assert not trainer.q_opt.state and not trainer.policy_opt.state
        assert all(torch.equal(value, getattr(trainer, name + "_target").state_dict()[key])
            for name in ("q1", "q2") for key, value in getattr(trainer, name).state_dict().items())
        report["cold_initialization_passed"] = True
        write_report(output, report)
        return trainer

    def first_update(step, trainer, encoder, live_env, operator):
        if step == KWARGS["learning_starts"]:
            assert trainer.update_work["actor_optimizer_steps"] == trainer.update_work["critic_optimizer_steps"] == 1
            assert training.optimizer_step_count(trainer.policy_opt) == 1
            assert type(trainer) is TRAINERS[condition]
            if condition == "mean_raw":
                assert trainer.objective_updates == 1
            first_updates.append(step)
            report["first_update_step"] = step
            write_report(output, report)
            print(f"FROM_START_FIRST_UPDATE env={env} condition={condition} seed={seed} step={step}", flush=True)
        return False

    started = time.monotonic()
    training.train_method(env, METHOD, TOTAL_STEPS, SWITCH_INTERVAL, seed, str(output.parent),
        trainer_factory=factory, state_probe=capture, training_probe=first_update, **KWARGS)
    state, trainer = capture.state, capture.state["trainer"]
    assert first_updates == [KWARGS["learning_starts"]]
    try:
        local.crossed.common_contract(trainer)
        work = dict(trainer.update_work)
        clocks = dict(actor=training.optimizer_step_count(trainer.policy_opt),
            critic=training.optimizer_step_count(trainer.q_opt),
            encoder=training.optimizer_step_count(state["encoder_opt"]))
        report.update(trainer_class=type(trainer).__name__, learner_work=work, optimizer_clocks=clocks,
            raw_objective_updates=getattr(trainer, "objective_updates", 0),
            representation_replay_labels_unprivileged=bool((trainer.buffer.regimes[:trainer.buffer.size] == -1).all()),
            network_parameters=training.network_parameter_counts(trainer.policy, (trainer.q1, trainer.q2), state["encoder"]),
            selector_class=type(state["adaptive_selector"]).__name__,
            allocation=state["adaptive_selector"].state_dict(),
            mode_history=[list(row) for row in state["env"].mode_history],
            training_seconds=time.monotonic() - started)
        costs = report["completed_budget"]
        costs.update(training_interactions=state["env"]._total_steps,
            environment_initialization_probe_steps=2, actor_optimizer_steps=clocks["actor"],
            critic_optimizer_steps=clocks["critic"], encoder_optimizer_steps=clocks["encoder"])
        write_report(output, report)
        evaluated = base.evaluate(state, trainer, seed)
        costs["evaluation_interactions"] = evaluated["interactions"]
        report.update(evaluation={k: evaluated[k] for k in ("evaluation_weights", "episode_seeds_per_mode", "ref_point")},
            endpoint=local.source.endpoint(dict(evaluation=evaluated)), evaluation_seconds=evaluated["seconds"],
            controls_passed=True, complete=True)
        check_record(report)
        write_report(output, report)
        print(f"DONE {PROTOCOL} env={env} condition={condition} seed={seed} "
              f"utility={report['endpoint']['utility']:.6f} cost={json.dumps(costs)}", flush=True)
        return report
    finally:
        state["env"].close()
        state["eval_env"].close()


def summarize(root=ROOT):
    indices = np.random.default_rng(0).integers(0, len(SEEDS), (20000, len(SEEDS)))
    result = dict(protocol=PROTOCOL, experiment=specification(), environments={},
        completed_budget=dict.fromkeys(budget(ENVIRONMENTS[0]), 0))
    for env in ENVIRONMENTS:
        runs = []
        for seed in SEEDS:
            pair = {c: json.loads((Path(root) / env.replace("-", "_") / c / f"seed{seed}.json").read_text())
                    for c in CONDITIONS}
            for condition, record in pair.items():
                check_record(record)
                assert (record["env"], record["condition"], record["seed"]) == (env, condition, seed)
                for key, value in record["completed_budget"].items():
                    result["completed_budget"][key] += value
            for key in ("evaluation", "mode_history", "learner_work", "optimizer_clocks", "network_parameters"):
                assert pair["soft"][key] == pair["mean_raw"][key], (env, seed, key)
            runs.append(dict(seed=seed, **pair))
        methods, contrasts = {}, {}
        for condition in CONDITIONS:
            methods[condition] = {}
            for metric in ("utility", "hv"):
                values = [row[condition]["endpoint"][metric] for row in runs]
                methods[condition][metric] = dict(mean=float(np.mean(values)), per_seed=values, minimum_seed=min(values))
            methods[condition]["per_mode_mean_utility"] = np.mean([
                row[condition]["endpoint"]["per_mode_utility"] for row in runs], axis=0).tolist()
            methods[condition]["training_seconds_mean"] = float(np.mean([row[condition]["training_seconds"] for row in runs]))
        for metric in ("utility", "hv"):
            delta = np.asarray([row["mean_raw"]["endpoint"][metric] - row["soft"]["endpoint"][metric] for row in runs])
            contrasts[metric] = dict(mean=float(delta.mean()), per_seed=delta.tolist(),
                descriptive_ci95=np.quantile(delta[indices].mean(1), [.025, .975]).tolist(),
                minimum_seed_difference=methods["mean_raw"][metric]["minimum_seed"] - methods["soft"][metric]["minimum_seed"])
        passed = contrasts["utility"]["mean"] > 0 and contrasts["hv"]["mean"] >= 0
        passed = passed and all(contrasts[m]["minimum_seed_difference"] >= 0 for m in ("utility", "hv"))
        result["environments"][env] = dict(runs=runs, methods=methods, mean_raw_minus_soft=contrasts,
            robust_lead_screen_passed=passed)
    lead = all(value["robust_lead_screen_passed"] for value in result["environments"].values())
    result.update(complete=True, decision="from_start_robust_lead_requires_confirmation" if lead else "hold_from_start_objective")
    write_report(Path(root) / "compact_readout.json", result)
    print("DONE", PROTOCOL, "summary", json.dumps(dict(decision=result["decision"], completed_budget=result["completed_budget"])), flush=True)
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--env", choices=ENVIRONMENTS)
    parser.add_argument("--condition", choices=CONDITIONS)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--summarize", action="store_true")
    args = parser.parse_args()
    if args.summarize:
        summarize(args.root)
    else:
        assert args.env is not None and args.condition is not None and args.seed is not None
        run(args.env, args.condition, args.seed, args.root)
