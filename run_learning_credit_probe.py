"""Fresh paired collection/learning forks with a held-out fixed-target credit."""

import argparse
import copy
import json
from pathlib import Path
import time

import numpy as np
import torch

import diagnose_collection_credit as screen
import run_collection_value_probe as base


PROTOCOL = "heldout_learning_credit_probe_v1"
ROOT = Path("results/iclr_learning_credit_20261002")
SEEDS = tuple(range(2401, 2406))
ENVIRONMENTS = base.ENVIRONMENTS
METHOD, CUT, COLLECTION_STEPS, KWARGS = base.METHOD, base.CUT, base.COLLECTION_STEPS, base.KWARGS
UPDATES = 128
BRANCHES = ("old_only", "uniform", "mixed", "uniform_repeat")
EVALUATED = BRANCHES[:-1]
training, write_report = base.training, base.write_report


def specification():
    return dict(method=METHOD, seeds=list(SEEDS), environments=list(ENVIRONMENTS),
        prefix_steps=CUT, collection_steps=COLLECTION_STEPS, updates_per_branch=UPDATES,
        training_kwargs=KWARGS, branches=list(BRANCHES), evaluated=list(EVALUATED),
        evaluation_repeats=base.REPEATS, evaluation_horizon=base.HORIZON,
        audit_weights="unchanged21_full_simplex", audit_rule="fixed_prefix_soft_target_projected_loss",
        common_learning_contract=base.previous.LEARNING_CONTRACT)


def budget(env):
    modes = base.previous.previous.previous.extension.ENVIRONMENTS[env]["num_modes"]
    updates = CUT - KWARGS["learning_starts"] + 1
    audit_rows = 21 * COLLECTION_STEPS
    return dict(training_interactions=CUT + 4 * COLLECTION_STEPS,
        evaluation_interactions=len(EVALUATED) * modes * 21 * base.REPEATS * base.HORIZON,
        environment_initialization_probe_steps=2,
        actor_optimizer_steps=updates + len(BRANCHES) * UPDATES,
        critic_optimizer_steps=updates + len(BRANCHES) * UPDATES,
        encoder_optimizer_steps=updates, audit_target_rows=audit_rows,
        audit_loss_rows=(2 + len(BRANCHES)) * audit_rows)


@torch.no_grad()
def make_audit(trainer, data, seed):
    batch = base.batch_at(data, np.arange(data.size))
    states, actions, _, contexts, reward, next_states, next_contexts, done = batch
    weights = training.full_simplex_grid(trainer.reward_dim)
    targets = []
    for index, weight in enumerate(weights):
        w = torch.as_tensor(weight, device=trainer.device).expand(data.size, -1)
        context = trainer.replace_context_preferences(next_contexts, w)
        with base.isolated_rng(seed + index):
            scalar, vector = trainer.envelope_next_values(next_states, w, context)
            _, target = trainer.critic_targets(reward, w, done, scalar, vector)
        targets.append(target.detach().clone())
    return dict(states=states, actions=actions, contexts=contexts, weights=weights,
                targets=torch.stack(targets), rows=len(weights) * data.size)


@torch.no_grad()
def audit_loss(trainer, audit):
    losses = []
    for weight, target in zip(audit["weights"], audit["targets"]):
        w = torch.as_tensor(weight, device=trainer.device).expand(len(audit["states"]), -1)
        context = trainer.replace_context_preferences(audit["contexts"], w)
        q1 = trainer.q1(audit["states"], audit["actions"], context)
        q2 = trainer.q2(audit["states"], audit["actions"], context)
        loss = trainer.critic_objective(q1, q2, w, (w * target).sum(-1, keepdim=True), target)
        losses.append(float(loss))
    return dict(mean=float(np.mean(losses)), per_weight=losses, rows=audit["rows"])


def credit_choice(losses):
    return "mixed" if losses["mixed"]["mean"] < losses["uniform"]["mean"] else "uniform"


def learn(trainer, data, seed):
    branch = base.clone_trainer(trainer, base.combined_buffer(trainer.buffer, data))
    before = dict(branch.update_work)
    for update in range(UPDATES):
        with base.isolated_rng(seed + update):
            branch.update()
    work = {key: branch.update_work[key] - value for key, value in before.items()}
    assert work == dict(critic_optimizer_steps=UPDATES, actor_optimizer_steps=UPDATES,
        critic_rows=UPDATES * branch.batch_size, target_candidate_rows=UPDATES * branch.batch_size,
        actor_candidate_rows=UPDATES * branch.batch_size)
    return branch, work


def run(env, seed, root=ROOT):
    root = Path(root)
    frozen = json.loads((root / "protocol.json").read_text())
    assert frozen["protocol"] == PROTOCOL and frozen["experiment"] == specification()
    assert (CUT, COLLECTION_STEPS, KWARGS) == (base.CUT, base.COLLECTION_STEPS, base.KWARGS)
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
        collection={}, audit_losses={}, branches={}, controls_passed=False)
    write_report(output, report)
    capture = base.Capture()
    training.train_method(env, METHOD, CUT, 25000, seed, str(output.parent / f"prefix{seed}"),
                          state_probe=capture, **KWARGS)
    state = capture.state
    trainer = state["trainer"]
    prefix_work = dict(trainer.update_work)
    encoder_before = copy.deepcopy(state["encoder"].state_dict())
    costs = report["completed_budget"]
    updates = CUT - KWARGS["learning_starts"] + 1
    costs.update(training_interactions=CUT, environment_initialization_probe_steps=2,
        actor_optimizer_steps=prefix_work["actor_optimizer_steps"],
        critic_optimizer_steps=prefix_work["critic_optimizer_steps"],
        encoder_optimizer_steps=training.optimizer_step_count(state["encoder_opt"]))
    assert all(costs[key] == updates for key in ("actor_optimizer_steps", "critic_optimizer_steps", "encoder_optimizer_steps"))
    report["prefix"] = dict(step=state["step"], replay_size=trainer.buffer.size,
        training_seconds=state["training_seconds"], learner_work=prefix_work,
        mode_history=[list(row) for row in state["env"].mode_history])
    try:
        data = {}
        for name in ("uniform", "mixed", "uniform_repeat", "audit_uniform"):
            sampler = "uniform" if name == "audit_uniform" else name
            rng_seed = (53000000 if name == "audit_uniform" else 52000000) + seed * 1000
            data[name], report["collection"][name] = base.collect(state, sampler, rng_seed)
            costs["training_interactions"] += COLLECTION_STEPS
            assert report["collection"][name]["mode_history"] == report["collection"]["uniform"]["mode_history"]
            write_report(output, report)
        for field in base.STORAGE_FIELDS:
            np.testing.assert_array_equal(getattr(data["uniform"], field), getattr(data["uniform_repeat"], field))
        report["collection_identity_passed"] = True
        audit = make_audit(trainer, data["audit_uniform"], 82000000 + seed * 1000)
        costs["audit_target_rows"] += audit["rows"]
        report["audit_losses"]["base"] = audit_loss(trainer, audit)
        costs["audit_loss_rows"] += audit["rows"]
        branches = {}
        # Record the learning-side credit decision before any endpoint rollout.
        for name in BRANCHES:
            started = time.monotonic()
            branch, work = learn(trainer, data.get(name), 62000000 + seed * 10000)
            branches[name] = branch
            report["branches"][name] = dict(learner_work=work, learner_seconds=time.monotonic() - started,
                replay_size=branch.buffer.size, new_replay_fraction=0. if name == "old_only" else COLLECTION_STEPS / branch.buffer.size)
            costs["actor_optimizer_steps"] += work["actor_optimizer_steps"]
            costs["critic_optimizer_steps"] += work["critic_optimizer_steps"]
            report["audit_losses"][name] = audit_loss(branch, audit)
            costs["audit_loss_rows"] += audit["rows"]
            write_report(output, report)
        base.same_learner(branches["uniform"], branches["uniform_repeat"])
        assert report["audit_losses"]["uniform"] == report["audit_losses"]["uniform_repeat"]
        report.update(learner_identity_passed=True, learning_credit_choice=credit_choice(report["audit_losses"]),
            reward_proxy_choice=screen.choices_from_collection(report["collection"])[1]["reward_rate"],
            choice_recorded_before_evaluation=True)
        write_report(output, report)
        for name in EVALUATED:
            evaluation = base.evaluate(state, branches[name], seed)
            costs["evaluation_interactions"] += evaluation["interactions"]
            report["branches"][name]["evaluation"] = evaluation
            baseline = report["branches"]["old_only"]["evaluation"]
            for key in ("evaluation_weights", "episode_seeds_per_mode", "ref_point"):
                assert evaluation[key] == baseline[key]
            print(f"LEARNING_CREDIT {env} seed={seed} branch={name} "
                  f"audit_loss={report['audit_losses'][name]['mean']:.6f} utility={evaluation['direct_utility']['mean']:.6f}", flush=True)
            write_report(output, report)
        assert trainer.buffer.size == CUT and trainer.update_work == prefix_work
        assert report["audit_losses"]["base"] == audit_loss(trainer, audit)
        costs["audit_loss_rows"] += audit["rows"]
        assert all(torch.equal(value, state["encoder"].state_dict()[key]) for key, value in encoder_before.items())
        assert training.optimizer_step_count(state["encoder_opt"]) == updates
        for key, upper in budget(env).items():
            assert 0 < costs[key] <= upper if key == "evaluation_interactions" else costs[key] == upper
        report.update(complete=True, controls_passed=True, representation_frozen=True)
        write_report(output, report)
        print(f"DONE {PROTOCOL} {env} seed={seed} cost={json.dumps(costs)}", flush=True)
        return report
    finally:
        state["env"].close()
        state["eval_env"].close()


def summarize(root=ROOT):
    indices = np.random.default_rng(0).integers(0, len(SEEDS), (20000, len(SEEDS)))
    result = dict(protocol=PROTOCOL, experiment=specification(), complete=True,
        completed_budget=dict.fromkeys(budget(ENVIRONMENTS[0]), 0), environments={})
    for env in ENVIRONMENTS:
        rows = []
        for seed in SEEDS:
            record = json.loads((Path(root) / env.replace("-", "_") / f"seed{seed}.json").read_text())
            assert (record["protocol"], record["env"], record["seed"], record["experiment"], record["complete"],
                record["controls_passed"], record["representation_frozen"]) == (PROTOCOL, env, seed, specification(), True, True, True)
            assert record["collection_identity_passed"] and record["learner_identity_passed"] and record["choice_recorded_before_evaluation"]
            assert record["learning_credit_choice"] == credit_choice(record["audit_losses"])
            assert record["audit_losses"]["uniform"] == record["audit_losses"]["uniform_repeat"]
            for key, upper in budget(env).items():
                actual = record["completed_budget"][key]
                assert 0 < actual <= upper if key == "evaluation_interactions" else actual == upper
                result["completed_budget"][key] += actual
            baseline = record["branches"]["old_only"]
            for name in EVALUATED:
                branch = record["branches"][name]
                assert branch["learner_work"] == baseline["learner_work"]
                for key in ("evaluation_weights", "episode_seeds_per_mode", "ref_point"):
                    assert branch["evaluation"][key] == baseline["evaluation"][key]
            row = {key: record[key] for key in ("seed", "prefix", "collection", "audit_losses", "learning_credit_choice",
                "reward_proxy_choice", "completed_budget")}
            row["endpoints"] = {name: dict(utility=record["branches"][name]["evaluation"]["direct_utility"]["mean"],
                per_mode_utility=record["branches"][name]["evaluation"]["direct_utility"]["per_mode"],
                hv=record["branches"][name]["evaluation"]["final_hv"],
                learner_seconds=record["branches"][name]["learner_seconds"],
                evaluation_seconds=record["branches"][name]["evaluation"]["seconds"])
                for name in EVALUATED}
            rows.append(row)
        gain = [row["endpoints"]["mixed"]["utility"] - row["endpoints"]["uniform"]["utility"] for row in rows]
        policies = {name: screen.policy_metrics(rows, choices, indices) for name, choices in {
            "learning_credit": [row["learning_credit_choice"] for row in rows],
            "reward_proxy": [row["reward_proxy_choice"] for row in rows],
            "always_uniform": ["uniform"] * len(rows), "always_mixed": ["mixed"] * len(rows),
            "hindsight_oracle": ["mixed" if value > 0 else "uniform" for value in gain]}.items()}
        for name in ("learning_credit", "reward_proxy"):
            choices = policies[name]["choices"]
            policies[name].update(sign_matches=sum((choice == "mixed") == (value > 0) for choice, value in zip(choices, gain) if value != 0),
                informative_pairs=sum(value != 0 for value in gain),
                harmful_mixed_choices=sum(choice == "mixed" and value < 0 for choice, value in zip(choices, gain)),
                mean_minus_best_fixed=policies[name]["utility_mean"] - max(policies["always_uniform"]["utility_mean"], policies["always_mixed"]["utility_mean"]))
        paired = {f"{arm}_minus_old_only": {metric: screen.delta_stats([row["endpoints"][arm][metric] - row["endpoints"]["old_only"][metric] for row in rows], indices)
            for metric in ("utility", "hv")} for arm in ("uniform", "mixed")}
        result["environments"][env] = dict(runs=rows, paired=paired, policies=policies)
    primary = {env: data["policies"]["learning_credit"] for env, data in result["environments"].items()}
    result["headroom_observed"] = all(data["policies"]["hindsight_oracle"]["utility_minus_uniform"]["mean"] > 0 for data in result["environments"].values())
    result["credit_screen_passed"] = (primary["mo-halfcheetah-v5"]["utility_minus_uniform"]["mean"] >= 0
        and primary["mo-hopper-v5"]["utility_minus_uniform"]["mean"] > 0
        and primary["mo-hopper-v5"]["hv_minus_uniform"]["mean"] >= 0)
    result["decision"] = ("hold_no_observed_selection_headroom" if not result["headroom_observed"] else
        "learning_side_credit_lead_online_unproven" if result["credit_screen_passed"] else "hold_learning_side_credit")
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
