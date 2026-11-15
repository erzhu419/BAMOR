"""A single-policy actor constraint labelled by paid, paired episode returns."""

import argparse
import copy
import json
from pathlib import Path

import numpy as np
import torch

import run_shared_objective_router as shared
from run_terminal_component_forks import clone_state


training, router, base, write = shared.training, shared.router, shared.base, shared.write
PROTOCOL = "return_labelled_actor_tangent_guard_v1"
ROOT = Path("results/iclr_return_labelled_guard_20261003")
ENVIRONMENTS, SEEDS = shared.ENVIRONMENTS, (2801, 2802, 2803)
CUT, PROPOSAL, END = 75000, 100000, 125000
HORIZON, SWITCH_INTERVAL, KWARGS = 1000, 25000, shared.KWARGS
CONDITIONS = ("scored", "blind", "priced")


def specification():
    return dict(environments=list(ENVIRONMENTS), seeds=list(SEEDS), method=shared.source.METHOD,
        prefix_steps=CUT, proposal_steps=PROPOSAL, end_steps=END, training_kwargs=KWARGS,
        switch_interval=SWITCH_INTERVAL, calibration_horizon=HORIZON, conditions=list(CONDITIONS),
        calibration="one_full_episode_per_teacher_per_full21_weight; paired_reset_and_forked_observed_state; "
                    "unlocked_natural_schedule_advances_with_proposal_trajectory",
        label="actual_undiscounted_weighted_episode_return; choose_proposal_on_ties",
        blind="one_fixed_fair_coin_per_pair; RNG97000000+seed; no_return_ordering",
        constraint="one_preference_balanced_winning_trajectory_per_update; batch_size_factual_state_context_action_rows; "
                   "mean_action_MSE_gradient; remove_outward_component_from_actual_Adam_parameter_displacement",
        moments="retain_native_proposal_Adam_moments; no_radius_coefficient_or_critic_rank",
        auxiliary_rng="96000000+seed; private_generator_shared_draw_law_for_scored_and_blind",
        priced_steps="end_steps+actual_calibration_interactions; same_six_initialization_probes_per_path",
        work_match="guard_offline_actor_critic_encoder_topups_equal_actual_calibration_interactions; "
                   "optimizer_steps_match_priced_native; extra_constraint_forward_backward_work_recorded",
        top_up_rng="98000000+100000*seed+update; no_extra_interactions",
        evaluation="frozen_prefix_proposal_scored_blind_priced_actors; full21_four_modes_two_paired_resets; no_archive_selection",
        screen="both_envs_scored_mean_utility_HV_minimum_seed_utility_HV_and_all_mode_utilities_at_least_priced; "
               "both_envs_mean_utility_HV_at_least_blind; HC_low_speed_at_least_prefix_and_strictly_above_blind",
        privilege="no_mode_id_anchor_caps_or_held_evaluation_labels_in_guard; simulator_branching_is_explicitly_charged")


def choose_teacher(old_return, new_return, weight):
    return "new" if np.dot(weight, new_return) >= np.dot(weight, old_return) else "old"


@torch.no_grad()
def trajectory(env, model, weight, seed):
    obs, _ = env.reset(seed=seed)
    trans = training.TransitionContextTracker(len(weight))
    learned = training.LearnedTransitionContextTracker(model["encoder"], latent_dim=training.LATENT_CONTEXT_DIM)
    states, contexts, actions, total = [], [], [], np.zeros(len(weight))
    for _ in range(HORIZON):
        context = training.build_mujoco_context(shared.source.METHOD, weight, trans_tracker=trans, learned_tracker=learned)
        action = model["policy"].get_action(torch.as_tensor(obs).float()[None],
            torch.as_tensor(context).float()[None], deterministic=True).numpy().ravel()
        states.append(obs.copy())
        contexts.append(context.copy())
        actions.append(action.copy())
        ns, reward, terminal, truncated, _ = env.step(action)
        total += reward
        trans.update(obs, action, reward, ns)
        learned.update(obs, action, reward, ns, "cpu")
        obs = ns
        if terminal or truncated:
            break
    return dict(states=np.asarray(states, dtype=np.float32), contexts=np.asarray(contexts, dtype=np.float32),
                actions=np.asarray(actions, dtype=np.float32)), total


def collect(state, models, seed):
    env = base.fork_env(state["env"])
    assert not env._lock_mode
    weights = training.full_simplex_grid(models["old"]["reward_dim"])
    order = np.random.default_rng(99000000 + seed).permutation(len(weights))
    blind = np.random.default_rng(97000000 + seed).integers(0, 2, len(weights))
    pairs, rows, interactions = [], [], 0
    try:
        with base.isolated_rng(95000000 + seed):
            for index, wi in enumerate(order):
                reset_seed = 94000000 + seed * 1000 + index
                branches = {name: base.fork_env(env) for name in ("old", "new")}
                data, returns = {}, {}
                try:
                    for name in branches:
                        before = branches[name]._total_steps
                        data[name], returns[name] = trajectory(branches[name], models[name], weights[wi], reset_seed)
                        assert branches[name]._total_steps - before == len(data[name]["states"])
                        interactions += len(data[name]["states"])
                    np.testing.assert_array_equal(data["old"]["states"][0], data["new"]["states"][0])
                    chosen = choose_teacher(returns["old"], returns["new"], weights[wi])
                    blind_choice = ("old", "new")[blind[index]]
                    pairs.append(dict(trajectories=data, scored=chosen, blind=blind_choice))
                    rows.append(dict(weight_index=int(wi), weight=weights[wi].tolist(), reset_seed=reset_seed,
                        returns={k: v.tolist() for k, v in returns.items()},
                        utility_difference=float(weights[wi] @ (returns["new"] - returns["old"])),
                        steps={k: len(v["states"]) for k, v in data.items()}, scored=chosen, blind=blind_choice,
                        paired_initial_observation=True))
                    env.close()
                    env = branches.pop("new")
                finally:
                    for branch in branches.values():
                        branch.close()
        return pairs, dict(rows=rows, interactions=interactions, episodes=2 * len(weights),
            schedule_start=PROPOSAL, schedule_end=env._total_steps,
            selected_old=sum(r["scored"] == "old" for r in rows), selected_new=sum(r["scored"] == "new" for r in rows))
    finally:
        env.close()


class TangentAdam(torch.optim.Adam):
    """Constrain the displacement after Adam preconditioning, not the raw gradient."""

    def __init__(self, params, **options):
        super().__init__(params, **options)
        self.constraint = None
        self.projections, self.max_positive_inner_product = 0, 0.

    def step(self, closure=None):
        params = [p for group in self.param_groups for p in group["params"]]
        before = [p.detach().clone() for p in params]
        result = super().step(closure)
        if self.constraint is not None:
            with torch.no_grad():
                active = [(p, old, h) for p, old, h in zip(params, before, self.constraint) if h is not None]
                dot = sum((h.double() * (p.double() - old.double())).sum() for p, old, h in active)
                norm = sum(h.double().square().sum() for _, _, h in active)
                if norm > 0 and dot > 0:
                    scale = dot / norm
                    for p, _, h in active:
                        p.copy_(p.double() - scale * h.double())
                    self.projections += 1
                residual = sum((h.double() * (p.double() - old.double())).sum() for p, old, h in active)
                self.max_positive_inner_product = max(self.max_positive_inner_product, float(residual))
        self.constraint = None
        return result


class Guard:
    def __init__(self, condition, pairs, seed, calibration_steps):
        self.condition, self.pairs = condition, pairs
        self.calibration_steps = calibration_steps
        self.rng = np.random.default_rng(96000000 + seed)
        self.trainer = None
        self.updates, self.rows, self.loss_sum = 0, 0, 0.

    def configure(self, step, trainer):
        if self.trainer is not None:
            return
        self.trainer, self.native_objective = trainer, trainer.actor_objective
        old = trainer.policy_opt
        trainer.policy_opt = TangentAdam(trainer.policy.parameters(), **old.defaults)
        trainer.policy_opt.load_state_dict(old.state_dict())
        trainer.actor_objective = self.objective

    def objective(self, s, w, c, q_scalar, log_prob):
        loss = self.native_objective(s, w, c, q_scalar, log_prob)
        pair = self.pairs[int(self.rng.integers(len(self.pairs)))]
        data = pair["trajectories"][pair[self.condition]]
        indices = (self.rng.random(self.trainer.batch_size) * len(data["states"])).astype(np.int64)
        states, contexts, actions = [torch.as_tensor(data[k][indices], device=self.trainer.device)
                                    for k in ("states", "contexts", "actions")]
        actual = self.trainer.policy.get_action(states, contexts, deterministic=True)
        retention = (actual - actions).square().mean()
        params = tuple(self.trainer.policy.parameters())
        self.trainer.policy_opt.constraint = torch.autograd.grad(retention, params, allow_unused=True)
        self.updates += 1
        self.rows += len(indices)
        self.loss_sum += float(retention.detach())
        return loss

    def after_step(self, step, trainer, encoder, env, operator):
        pass

    def interaction_step(self, step):
        return int(step + self.calibration_steps)

    def state_dict(self):
        return dict(condition=self.condition, constraint_gradient_calls=self.updates, auxiliary_rows=self.rows,
            mean_constraint_loss=self.loss_sum / max(1, self.updates),
            projected_actor_steps=self.trainer.policy_opt.projections,
            max_positive_floating_inner_product=self.trainer.policy_opt.max_positive_inner_product)


def train(env, seed, output, steps, resume=None, guard=None):
    capture = base.Capture()
    capture.step = steps
    training.train_method(env, shared.source.METHOD, steps, SWITCH_INTERVAL, seed, str(output.parent),
        resume_state=resume, update_router=guard, state_probe=capture, **KWARGS)
    return capture.state


def check(record):
    assert (record["protocol"], record["experiment"], record["complete"]) == (PROTOCOL, specification(), True)
    assert record["env"] in ENVIRONMENTS and record["seed"] in SEEDS and record["controls_passed"]
    n = record["calibration"]["interactions"]
    assert 0 < n <= 42 * HORIZON and record["calibration"]["episodes"] == 42
    rows = record["calibration"]["rows"]
    assert sorted(r["weight_index"] for r in rows) == list(range(21))
    assert n == sum(sum(r["steps"].values()) for r in rows)
    for row in rows:
        assert row["paired_initial_observation"]
        assert row["scored"] == choose_teacher(row["returns"]["old"], row["returns"]["new"], row["weight"])
    assert record["priced_total_steps"] == END + n
    u = END + n - KWARGS["learning_starts"] + 1
    assert record["clocks"] == {c: dict(actor=u, critic=u, encoder=u) for c in CONDITIONS}
    for c in ("scored", "blind"):
        g = record["guards"][c]
        assert g["condition"] == c and g["constraint_gradient_calls"] == END - PROPOSAL + n
        assert g["auxiliary_rows"] == g["constraint_gradient_calls"] * KWARGS["batch_size"]
    assert record["path_prices"] == {c: dict(training_interactions=END if c != "priced" else END + n,
        calibration_interactions=n if c != "priced" else 0, initialization_probes=6,
        actor_steps=u, critic_steps=u, encoder_steps=u) for c in CONDITIONS}
    assert record["actual_work"] == dict(training_interactions=3 * END - 2 * PROPOSAL + n,
        calibration_interactions=n, initialization_probes=10,
        actor_steps=3 * END - 2 * PROPOSAL + 3 * n - KWARGS["learning_starts"] + 1,
        critic_steps=3 * END - 2 * PROPOSAL + 3 * n - KWARGS["learning_starts"] + 1,
        encoder_steps=3 * END - 2 * PROPOSAL + 3 * n - KWARGS["learning_starts"] + 1,
        constraint_gradient_calls=2 * (END - PROPOSAL + n),
        auxiliary_rows=2 * (END - PROPOSAL + n) * KWARGS["batch_size"],
        evaluation_interactions=sum(r["interactions"] for r in record["evaluations"].values()))
    assert set(record["evaluations"]) == {"prefix", "proposal", *CONDITIONS}
    for row in record["evaluations"].values():
        assert row["evaluation"] == record["evaluations"]["prefix"]["evaluation"]


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
        complete=False, controls_passed=False, phase="prefix", evaluations={}, guards={}, clocks={})
    write(output, report)
    states, models = {}, {}
    try:
        states["prefix"] = train(env, seed, output, CUT)
        models["prefix"] = router.snapshot(states["prefix"])
        report["phase"] = "proposal"
        write(output, report)
        states["proposal"] = train(env, seed, output, PROPOSAL, resume=states["prefix"])
        models["proposal"] = router.snapshot(states["proposal"])
        report["phase"] = "calibration"
        write(output, report)
        pairs, calibration = collect(states["proposal"], {"old": models["prefix"], "new": models["proposal"]}, seed)
        torch.save([{k: {f: torch.as_tensor(a) for f, a in d.items()} for k, d in p["trajectories"].items()}
                    for p in pairs], output.with_suffix(".calibration.pt"))
        n = calibration["interactions"]
        report.update(calibration=calibration, priced_total_steps=END + n,
                      proposal_mode_history=[list(r) for r in states["proposal"]["env"].mode_history])
        write(output, report)
        for condition in CONDITIONS:
            report["phase"] = "training_" + condition
            write(output, report)
            branch = clone_state(states["proposal"])
            guard = Guard(condition, pairs, seed, n) if condition != "priced" else None
            stop = END if guard else END + n
            states[condition] = train(env, seed, output, stop, resume=branch, guard=guard)
            if guard:
                for update in range(n):
                    with base.isolated_rng(98000000 + seed * 100000 + update):
                        states[condition]["trainer"].update()
                        shared.encoder_update(states[condition])
                report["guards"][condition] = guard.state_dict()
            shared.source.local.crossed.common_contract(states[condition]["trainer"])
            assert (states[condition]["trainer"].buffer.regimes[:stop] == -1).all()
            report["clocks"][condition] = router.clocks(states[condition])
            models[condition] = router.snapshot(states[condition])
            write(output, report)
            print(f"GUARD_TRAINED env={env} seed={seed} condition={condition} clocks={report['clocks'][condition]}", flush=True)
        report["phase"] = "evaluation"
        write(output, report)
        for name, model in models.items():
            report["evaluations"][name] = shared.endpoint_record(router.evaluate(
                states["proposal"], {"soft": model}, seed, "budget_soft"))
            write(output, report)
        u = END + n - KWARGS["learning_starts"] + 1
        report.update(path_prices={c: dict(training_interactions=END if c != "priced" else END + n,
            calibration_interactions=n if c != "priced" else 0, initialization_probes=6,
            actor_steps=u, critic_steps=u, encoder_steps=u) for c in CONDITIONS},
            actual_work=dict(training_interactions=3 * END - 2 * PROPOSAL + n, calibration_interactions=n,
                initialization_probes=10, actor_steps=3 * END - 2 * PROPOSAL + 3 * n - KWARGS["learning_starts"] + 1,
                critic_steps=3 * END - 2 * PROPOSAL + 3 * n - KWARGS["learning_starts"] + 1,
                encoder_steps=3 * END - 2 * PROPOSAL + 3 * n - KWARGS["learning_starts"] + 1,
                constraint_gradient_calls=2 * (END - PROPOSAL + n),
                auxiliary_rows=2 * (END - PROPOSAL + n) * KWARGS["batch_size"],
                evaluation_interactions=sum(r["interactions"] for r in report["evaluations"].values())),
            complete=True, controls_passed=True, phase="complete")
        check(report)
        torch.save({name: {k: model[k].state_dict() for k in ("policy", "encoder")}
                    for name, model in models.items()}, output.with_suffix(".actors.pt"))
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
        methods = {}
        for name in ("prefix", "proposal", *CONDITIONS):
            metrics = {m: dict(mean=float(np.mean(v)), minimum_seed=min(v), per_seed=v) for m in ("utility", "hv")
                for v in [[r["evaluations"][name]["endpoint"][m] for r in rows]]}
            metrics["mean_mode_utility"] = np.mean([r["evaluations"][name]["endpoint"]["per_mode_utility"] for r in rows], 0).tolist()
            methods[name] = metrics
        contrasts = {}
        for control in ("priced", "blind"):
            metrics = {m: dict(mean=methods["scored"][m]["mean"] - methods[control][m]["mean"],
                minimum_seed_difference=methods["scored"][m]["minimum_seed"] - methods[control][m]["minimum_seed"],
                per_seed=[r["evaluations"]["scored"]["endpoint"][m] - r["evaluations"][control]["endpoint"][m] for r in rows])
                for m in ("utility", "hv")}
            modes = np.subtract(methods["scored"]["mean_mode_utility"], methods[control]["mean_mode_utility"]).tolist()
            passed = all(v["mean"] >= 0 for v in metrics.values())
            if control == "priced":
                passed = passed and min(modes) >= 0 and all(v["minimum_seed_difference"] >= 0 for v in metrics.values())
            contrasts[control] = dict(metrics=metrics, mean_mode_utility_difference=modes, passed=passed)
        retention = env != "mo-halfcheetah-v5" or (methods["scored"]["mean_mode_utility"][1] >=
            methods["prefix"]["mean_mode_utility"][1] and contrasts["blind"]["mean_mode_utility_difference"][1] > 0)
        result["environments"][env] = dict(methods=methods, contrasts=contrasts, retention_passed=retention,
            passed=retention and all(v["passed"] for v in contrasts.values()),
            calibration=[r["calibration"] for r in rows], guards=[r["guards"] for r in rows],
            path_prices=[r["path_prices"] for r in rows])
    result["decision"] = "fresh_confirmation_required" if all(r["passed"] for r in result["environments"].values()) else "hold_return_labelled_tangent_guard"
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
