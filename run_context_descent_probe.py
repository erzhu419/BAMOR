"""A costed actor-only context-group descent intervention on the fixed substrate."""

import argparse
import json
from pathlib import Path
import time

import numpy as np
from scipy.optimize import nnls
import torch

import run_history_return_gate as history


training, router, base, shared, write = history.training, history.router, history.base, history.shared, history.write
PROTOCOL = "context_group_adam_descent_v1"
ROOT = Path("results/iclr_context_descent_20261003")
ENVIRONMENTS, SEEDS = history.ENVIRONMENTS, (3101, 3102, 3103)
CUT, END, GROUPS, FIT_ROWS = 75000, 100000, 4, 256
KWARGS, SWITCH_INTERVAL = history.KWARGS, history.SWITCH_INTERVAL
MODEL_NAMES = ("prefix", "native", "grouped")


def specification():
    return dict(environments=list(ENVIRONMENTS), seeds=list(SEEDS), method=shared.source.METHOD,
        training_kwargs=KWARGS, switch_interval=SWITCH_INTERVAL, prefix_steps=CUT, end_steps=END,
        groups=GROUPS, fit_rows=FIT_ROWS,
        grouping="fixed_prefix_farthest_centers; evenly_spaced_replay_rows; standardized_observable_transition_"
                 "features_only; no_preferences_learned_latents_or_mode_labels; nearest_center_assignment",
        intervention="native_SAC_row_losses_on_unchanged_batch; one_gradient_per_present_group; "
                     "nearest_actual_warm_Adam_displacement_with_all_group_inner_products_nonpositive; NNLS_dual",
        unchanged="critic_target_loss_and_updates; encoder_updates; replay_sampling; preference_relabeling; "
                  "collection_allocation; native_actor_objective; Adam_moment_updates",
        controls="two_continuations_of_exact_same_prefix; native_vs_grouped; no_teacher_or_acceptance_gate",
        path_budget="100000_training_plus_four_initialization_probes_per_path; no_calibration; "
                    "same_actor_critic_encoder_step_counts; extra_group_backwards_and_projection_time_recorded",
        evaluation="choices_fixed_before_full21_four_locked_modes_two_paired_resets; final_actor_encoder_pairs; no_archive",
        screen="both_envs_grouped_mean_utility_HV_minimum_seed_utility_HV_all_mean_mode_utilities_at_least_native; "
               "strict_mean_utility_gain_in_both; HC_low_speed_at_least_prefix",
        theory_scope="first_order_current_training_batch_group_losses_only; no_return_or_training_convergence_guarantee")


class ObservableGroups:
    def __init__(self, trainer):
        r = trainer.reward_dim
        self.start, self.stop = r, 2 * r + 1
        indices = np.linspace(0, trainer.buffer.size - 1, min(FIT_ROWS, trainer.buffer.size), dtype=int)
        features = trainer.buffer.contexts[indices, self.start:self.stop].astype(np.float64)
        self.mean, scale = features.mean(0), features.std(0)
        self.scale = np.where(scale == 0, 1., scale)
        selected = history.select_slots(features, GROUPS)
        self.centers = (features[selected] - self.mean) / self.scale
        self.fit_indices = indices.tolist()

    def assign(self, context):
        features = context.detach().cpu().numpy()[:, self.start:self.stop]
        z = (features - self.mean) / self.scale
        return ((z[:, None] - self.centers[None]) ** 2).sum(-1).argmin(1)

    def state_dict(self):
        return dict(feature_slice=[self.start, self.stop], mean=self.mean.tolist(), scale=self.scale.tolist(),
                    centers=self.centers.tolist(), fit_indices=self.fit_indices)


def project_displacement(delta, gradients):
    """Project onto g_i dot d <= 0; NNLS solves the nonnegative dual multipliers."""
    norms = gradients.norm(dim=1)
    directions = gradients[norms > 0].double() / norms[norms > 0, None].double()
    if len(directions) == 0 or (directions @ delta <= 0).all():
        return delta, directions, False
    coefficients, _ = nnls(directions.cpu().numpy().T, delta.cpu().numpy(), maxiter=100 * len(directions))
    projected = delta - directions.T @ torch.as_tensor(coefficients, device=delta.device)
    tolerance = 1e-8 * max(float(delta.norm()), 1e-12)
    assert float((directions @ projected).max()) <= tolerance, "NNLS failed the cone constraint"
    return projected, directions, True


class ContextAdam(torch.optim.Adam):
    def __init__(self, params, **options):
        super().__init__(params, **options)
        self.constraints = None
        self.steps, self.projections, self.seconds = 0, 0, 0.
        self.native_norm_sum, self.accepted_norm_sum, self.max_actual_increase = 0., 0., 0.

    def step(self, closure=None):
        params = [p for group in self.param_groups for p in group["params"]]
        before = torch.cat([p.detach().reshape(-1) for p in params]).double()
        result = super().step(closure)
        assert self.constraints is not None
        started = time.perf_counter()
        with torch.no_grad():
            native = torch.cat([p.detach().reshape(-1) for p in params]).double() - before
            accepted, directions, changed = project_displacement(native, self.constraints)
            if changed:
                offset = 0
                for p in params:
                    p.copy_((before[offset:offset + p.numel()] + accepted[offset:offset + p.numel()]).reshape_as(p))
                    offset += p.numel()
                self.projections += 1
            actual = torch.cat([p.detach().reshape(-1) for p in params]).double() - before
            if len(directions):
                self.max_actual_increase = max(self.max_actual_increase, float((directions @ actual).max()))
            self.native_norm_sum += float(native.norm())
            self.accepted_norm_sum += float(actual.norm())
        self.seconds += time.perf_counter() - started
        self.steps += 1
        self.constraints = None
        return result


class ContextDescent:
    def __init__(self, prefix_trainer):
        self.groups = ObservableGroups(prefix_trainer)
        self.trainer = None
        self.updates, self.gradient_calls, self.gradient_rows, self.seconds = 0, 0, 0, 0.
        self.group_rows = np.zeros(GROUPS, dtype=np.int64)
        self.present_group_counts = np.zeros(GROUPS + 1, dtype=np.int64)

    def configure(self, step, trainer):
        if self.trainer is not None:
            return
        self.trainer = trainer
        old = trainer.policy_opt
        trainer.policy_opt = ContextAdam(trainer.policy.parameters(), **old.defaults)
        trainer.policy_opt.load_state_dict(old.state_dict())
        trainer.actor_objective = self.objective

    def objective(self, s, w, c, q_scalar, log_prob):
        rows = (self.trainer.alpha * log_prob - q_scalar).reshape(-1)
        started = time.perf_counter()
        labels = self.groups.assign(c)
        gradients = []
        params = tuple(self.trainer.policy.parameters())
        for group in np.unique(labels):
            mask = torch.as_tensor(labels == group, device=rows.device)
            grad = torch.autograd.grad(rows[mask].mean(), params, retain_graph=True)
            gradients.append(torch.cat([g.detach().reshape(-1) for g in grad]).double())
            count = int(mask.sum())
            self.group_rows[group] += count
            self.gradient_rows += count
            self.gradient_calls += 1
        self.trainer.policy_opt.constraints = torch.stack(gradients)
        self.present_group_counts[len(gradients)] += 1
        self.updates += 1
        self.seconds += time.perf_counter() - started
        return rows.mean()

    def after_step(self, step, trainer, encoder, env, operator):
        pass

    def interaction_step(self, step):
        return step

    def state_dict(self):
        opt = self.trainer.policy_opt
        return dict(grouping=self.groups.state_dict(), updates=self.updates, group_gradient_calls=self.gradient_calls,
            group_gradient_rows=self.gradient_rows, group_rows=self.group_rows.tolist(),
            present_group_counts=self.present_group_counts.tolist(), gradient_seconds=self.seconds,
            projection_seconds=opt.seconds, projected_steps=opt.projections,
            sum_native_displacement_norm=opt.native_norm_sum, sum_accepted_displacement_norm=opt.accepted_norm_sum,
            max_positive_actual_normalized_group_inner_product=opt.max_actual_increase)


def train(env, seed, output, steps, resume=None, intervention=None):
    capture = base.Capture()
    capture.step = steps
    training.train_method(env, shared.source.METHOD, steps, SWITCH_INTERVAL, seed, str(output.parent),
        resume_state=resume, update_router=intervention, state_probe=capture, **KWARGS)
    return capture.state


def check(record):
    assert (record["protocol"], record["experiment"], record["complete"], record["controls_passed"]) == (PROTOCOL, specification(), True, True)
    assert record["env"] in ENVIRONMENTS and record["seed"] in SEEDS
    u = END - KWARGS["learning_starts"] + 1
    assert record["clocks"] == {k: dict(actor=u, critic=u, encoder=u) for k in ("native", "grouped")}
    price = dict(training_interactions=END, calibration_interactions=0, initialization_probes=4,
                 actor_steps=u, critic_steps=u, encoder_steps=u)
    assert record["path_prices"] == dict(native=price, grouped=price)
    extra = record["intervention"]
    assert extra["updates"] == END - CUT and extra["group_gradient_rows"] == (END - CUT) * KWARGS["batch_size"]
    assert 1 <= extra["group_gradient_calls"] / (END - CUT) <= GROUPS
    assert sum(extra["present_group_counts"]) == END - CUT
    assert sum(extra["group_rows"]) == extra["group_gradient_rows"]
    assert extra["projected_steps"] <= END - CUT
    evaluations = record["evaluations"]
    assert set(evaluations) == set(MODEL_NAMES)
    assert all(e["evaluation"] == evaluations["prefix"]["evaluation"] for e in evaluations.values())
    physical_u = CUT - KWARGS["learning_starts"] + 1 + 2 * (END - CUT)
    assert record["actual_work"] == dict(training_interactions=CUT + 2 * (END - CUT),
        calibration_interactions=0, initialization_probes=6, actor_steps=physical_u, critic_steps=physical_u,
        encoder_steps=physical_u, group_gradient_calls=extra["group_gradient_calls"],
        group_gradient_rows=extra["group_gradient_rows"],
        evaluation_interactions=sum(e["interactions"] for e in evaluations.values()))


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
                  complete=False, controls_passed=False, phase="prefix", evaluations={}, clocks={})
    write(output, report)
    states = {}
    try:
        prefix = train(env, seed, output, CUT)
        states["prefix"] = prefix
        models = dict(prefix=router.snapshot(prefix))
        intervention = ContextDescent(prefix["trainer"])
        for name in ("native", "grouped"):
            report["phase"] = name
            write(output, report)
            started = time.perf_counter()
            states[name] = train(env, seed, output, END, resume=history.source.clone_state(prefix),
                                 intervention=intervention if name == "grouped" else None)
            report.setdefault("training_seconds", {})[name] = time.perf_counter() - started
            models[name] = router.snapshot(states[name])
            report["clocks"][name] = router.clocks(states[name])
            shared.source.local.crossed.common_contract(states[name]["trainer"])
            assert (states[name]["trainer"].buffer.regimes[:END] == -1).all()
        report["intervention"] = intervention.state_dict()
        report["phase"] = "evaluation"
        write(output, report)
        for name in MODEL_NAMES:
            report["evaluations"][name] = shared.endpoint_record(router.evaluate(prefix, {"soft": models[name]}, seed, "budget_soft"))
            write(output, report)
        u = END - KWARGS["learning_starts"] + 1
        price = dict(training_interactions=END, calibration_interactions=0, initialization_probes=4,
                     actor_steps=u, critic_steps=u, encoder_steps=u)
        physical_u = CUT - KWARGS["learning_starts"] + 1 + 2 * (END - CUT)
        report.update(path_prices=dict(native=price, grouped=price), actual_work=dict(
            training_interactions=CUT + 2 * (END - CUT), calibration_interactions=0, initialization_probes=6,
            actor_steps=physical_u, critic_steps=physical_u, encoder_steps=physical_u,
            group_gradient_calls=report["intervention"]["group_gradient_calls"],
            group_gradient_rows=report["intervention"]["group_gradient_rows"],
            evaluation_interactions=sum(e["interactions"] for e in report["evaluations"].values())),
            complete=True, controls_passed=True, phase="complete")
        check(report)
        torch.save({name: {k: models[name][k].state_dict() for k in ("policy", "encoder")} for name in MODEL_NAMES}, output.with_suffix(".actors.pt"))
        write(output, report)
        print("DONE", PROTOCOL, env, seed, json.dumps(report["actual_work"]), flush=True)
        return report
    finally:
        for state in states.values():
            state["env"].close()
            state["eval_env"].close()


def summarize(root=ROOT):
    result = dict(protocol=PROTOCOL, experiment=specification(), complete=True, environments={}, actual_work={})
    for env in ENVIRONMENTS:
        records = [json.loads((Path(root) / env.replace("-", "_") / f"seed{s}.json").read_text()) for s in SEEDS]
        methods = {}
        for seed, record in zip(SEEDS, records):
            check(record)
            assert (record["env"], record["seed"]) == (env, seed)
            for k, v in record["actual_work"].items():
                result["actual_work"][k] = result["actual_work"].get(k, 0) + v
        for name in MODEL_NAMES:
            endpoints = [r["evaluations"][name]["endpoint"] for r in records]
            methods[name] = {k: dict(mean=float(np.mean(values)), minimum_seed=min(values), per_seed=values)
                             for k in ("utility", "hv") for values in [[e[k] for e in endpoints]]}
            methods[name]["mean_mode_utility"] = np.mean([e["per_mode_utility"] for e in endpoints], 0).tolist()
        deltas = {k: dict(mean=methods["grouped"][k]["mean"] - methods["native"][k]["mean"],
            minimum_seed_difference=methods["grouped"][k]["minimum_seed"] - methods["native"][k]["minimum_seed"],
            per_seed=np.subtract(methods["grouped"][k]["per_seed"], methods["native"][k]["per_seed"]).tolist())
            for k in ("utility", "hv")}
        modes = np.subtract(methods["grouped"]["mean_mode_utility"], methods["native"]["mean_mode_utility"]).tolist()
        retention = env != "mo-halfcheetah-v5" or methods["grouped"]["mean_mode_utility"][1] >= methods["prefix"]["mean_mode_utility"][1]
        passed = retention and min(modes) >= 0 and deltas["utility"]["mean"] > 0 and all(
            v["mean"] >= 0 and v["minimum_seed_difference"] >= 0 for v in deltas.values())
        result["environments"][env] = dict(methods=methods, grouped_minus_native=deltas,
            mean_mode_difference=modes, retention_passed=retention, passed=passed,
            interventions=[r["intervention"] for r in records], training_seconds=[r["training_seconds"] for r in records])
    result["decision"] = "fresh_confirmation_required" if all(e["passed"] for e in result["environments"].values()) else "hold_context_group_descent"
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
