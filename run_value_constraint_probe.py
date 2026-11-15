"""Change only the group constraint gradients; keep the native SAC objective."""

import argparse
import json
from pathlib import Path
import time

import numpy as np
import torch

import run_transition_group_probe as source


PROTOCOL = "value_constraint_group_adam_v1"
ROOT = Path("results/iclr_value_constraint_20261003")
ENVIRONMENTS, SEEDS = source.ENVIRONMENTS, (3301, 3302, 3303)
CUT, END, GROUPS = source.CUT, source.END, source.GROUPS
KWARGS, SWITCH_INTERVAL = source.KWARGS, source.SWITCH_INTERVAL
MODEL_NAMES = ("prefix", "native", "soft", "value")
DIAG_UPDATES = source.DIAG_UPDATES
write = source.write


def specification():
    original = source.specification()
    return dict(environments=list(ENVIRONMENTS), seeds=list(SEEDS), method=source.shared.source.METHOD,
        training_kwargs=KWARGS, switch_interval=SWITCH_INTERVAL, prefix_steps=CUT, end_steps=END,
        models=list(MODEL_NAMES), grouping="fixed_prefix_evenly_spaced_rows_standardization_farthest_centers_"
            "nearest_assignment; " + source.FEATURE_RULE,
        groups=GROUPS, fit_rows=source.FIT_ROWS, privilege=original["privilege"],
        intervention="soft_arm_group_mean(alpha_log_prob_minus_Q)_gradients; value_arm_group_mean(minus_Q)_gradients; "
                     "same_native_SAC_actor_objective_and_warm_Adam_moments; same_nearest_displacement_NNLS_projection",
        unchanged=original["unchanged"] + "; fixed_transition_partition_input_and_native_entropy_coefficient",
        controls="three_continuations_of_exact_same_prefix; soft_vs_value_changes_only_constraint_gradients",
        diagnostics=dict(updates=list(DIAG_UPDATES), components="same_native_action_entropy_value_soft_gradients; "
            "derivatives_along_actual_displacement; two_extra_group_backwards_per_diagnostic; no_extra_rng",
            audit_rows=source.AUDIT_ROWS, saved="server_only_models_critics_Adam_state_and_evenly_spaced_replay_rows"),
        path_budget=original["path_budget"], evaluation=original["evaluation"],
        screen="both_envs_value_mean_utility_HV_minimum_seed_utility_HV_all_mean_mode_utilities_at_least_"
               "both_native_and_soft; strictly_higher_mean_utility_than_both; HC_low_speed_at_least_prefix",
        theory_scope="first_order_current_training_critic_value_component_only; critic_still_has_soft_targets; "
                     "no_independent_raw_return_or_neural_training_convergence_guarantee")


class ValueConstraintDescent(source.MeasuredDescent):
    def __init__(self, prefix_trainer, constraint):
        super().__init__(prefix_trainer, "transition")
        assert constraint in ("soft", "value")
        self.constraint = constraint

    def objective(self, s, w, c, q_scalar, log_prob):
        rows = dict(entropy=(self.trainer.alpha * log_prob).reshape(-1), value=-q_scalar.reshape(-1),
                    soft=(self.trainer.alpha * log_prob - q_scalar).reshape(-1))
        started = time.perf_counter()
        labels = self.groups.assign(c)
        groups = np.unique(labels)
        params = tuple(self.trainer.policy.parameters())
        gradients = []
        for group in groups:
            mask = torch.as_tensor(labels == group, device=q_scalar.device)
            grad = torch.autograd.grad(rows[self.constraint][mask].mean(), params, retain_graph=True)
            gradients.append(torch.cat([g.detach().reshape(-1) for g in grad]).double())
            count = int(mask.sum())
            self.group_rows[group] += count
            self.gradient_rows += count
            self.gradient_calls += 1
        constraints = torch.stack(gradients)
        self.trainer.policy_opt.constraints = constraints
        self.present_group_counts[len(groups)] += 1
        self.updates += 1
        self.seconds += time.perf_counter() - started
        if self.updates in DIAG_UPDATES:
            started = time.perf_counter()
            components = {self.constraint: constraints}
            for name in rows:
                if name == self.constraint:
                    continue
                gradients = []
                for group in groups:
                    mask = torch.as_tensor(labels == group, device=q_scalar.device)
                    grad = torch.autograd.grad(rows[name][mask].mean(), params, retain_graph=True)
                    gradients.append(torch.cat([g.detach().reshape(-1) for g in grad]).double())
                    self.component_calls += 1
                    self.component_rows += int(mask.sum())
                components[name] = torch.stack(gradients)
            before = torch.cat([p.detach().reshape(-1) for p in params]).double()
            self.pending = (before, groups, components["soft"], components["entropy"], components["value"])
            self.component_seconds += time.perf_counter() - started
        return rows["soft"].mean()

    def state_dict(self):
        return dict(super().state_dict(), constraint=self.constraint)


def save_audit(output, states, models):
    artifact = dict(protocol=PROTOCOL, models={})
    for name, state in states.items():
        t, b = state["trainer"], state["trainer"].buffer
        indices = np.linspace(0, b.size - 1, min(source.AUDIT_ROWS, b.size), dtype=int)
        artifact["models"][name] = dict(alpha=float(t.alpha), step=state["step"],
            policy=models[name]["policy"].state_dict(), encoder=models[name]["encoder"].state_dict(),
            q1=t.q1.state_dict(), q2=t.q2.state_dict(), policy_opt=t.policy_opt.state_dict(),
            replay_indices=indices.tolist(), torch_rng=state["torch_rng"],
            replay={k: torch.from_numpy(getattr(b, k)[indices].copy()) for k in
                ("states", "actions", "weights", "contexts", "rewards", "next_states", "next_contexts", "dones", "timesteps")})
    torch.save(artifact, output.with_suffix(".audit.pt"))


def train(env, seed, output, steps, resume=None, intervention=None):
    capture = source.base.Capture()
    capture.step = steps
    source.training.train_method(env, source.shared.source.METHOD, steps, SWITCH_INTERVAL, seed, str(output.parent),
        resume_state=resume, update_router=intervention, state_probe=capture, **KWARGS)
    return capture.state


def check(record):
    assert (record["protocol"], record["experiment"], record["complete"], record["controls_passed"]) == (PROTOCOL, specification(), True, True)
    assert record["env"] in ENVIRONMENTS and record["seed"] in SEEDS
    u = END - KWARGS["learning_starts"] + 1
    price = dict(training_interactions=END, calibration_interactions=0, initialization_probes=4,
                 actor_steps=u, critic_steps=u, encoder_steps=u)
    assert record["path_prices"] == {name: price for name in MODEL_NAMES[1:]}
    assert record["clocks"] == {name: dict(actor=u, critic=u, encoder=u) for name in MODEL_NAMES[1:]}
    for name, extra in record["interventions"].items():
        assert name in ("soft", "value") and extra["constraint"] == name and extra["partition"] == "transition"
        assert extra["updates"] == END - CUT
        assert extra["group_gradient_rows"] == (END - CUT) * KWARGS["batch_size"] == sum(extra["group_rows"])
        assert sum(extra["present_group_counts"]) == END - CUT
        assert END - CUT <= extra["group_gradient_calls"] <= GROUPS * (END - CUT)
        assert extra["component_gradient_rows"] == 2 * len(DIAG_UPDATES) * KWARGS["batch_size"]
        assert [r["update"] for r in extra["component_records"]] == list(DIAG_UPDATES)
        assert extra["component_gradient_calls"] == 2 * sum(len(r["groups"]) for r in extra["component_records"])
    assert set(record["interventions"]) == {"soft", "value"}
    assert record["interventions"]["soft"]["grouping"] == record["interventions"]["value"]["grouping"]
    assert set(record["evaluations"]) == set(MODEL_NAMES)
    assert all(e["evaluation"] == record["evaluations"]["prefix"]["evaluation"] for e in record["evaluations"].values())
    physical_u = CUT - KWARGS["learning_starts"] + 1 + 3 * (END - CUT)
    expected = dict(training_interactions=CUT + 3 * (END - CUT), calibration_interactions=0,
        initialization_probes=8, actor_steps=physical_u, critic_steps=physical_u, encoder_steps=physical_u,
        evaluation_interactions=sum(e["interactions"] for e in record["evaluations"].values()))
    for key in ("group_gradient_calls", "group_gradient_rows", "component_gradient_calls", "component_gradient_rows"):
        expected[key] = sum(e[key] for e in record["interventions"].values())
    assert record["actual_work"] == expected
    assert all(record["replay_audits"][n]["mode_history"] == record["replay_audits"]["native"]["mode_history"]
               for n in MODEL_NAMES[1:])


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
                  complete=False, controls_passed=False, phase="prefix", clocks={}, evaluations={})
    write(output, report)
    states, interventions = {}, {}
    try:
        prefix = train(env, seed, output, CUT)
        states["prefix"] = prefix
        models = dict(prefix=source.router.snapshot(prefix))
        groups = dict(context=source.source.ObservableGroups(prefix["trainer"]), transition=source.TransitionGroups(prefix["trainer"]))
        for name in MODEL_NAMES[1:]:
            report["phase"] = name
            write(output, report)
            intervention = None if name == "native" else ValueConstraintDescent(prefix["trainer"], name)
            started = time.perf_counter()
            states[name] = train(env, seed, output, END, resume=source.source.history.source.clone_state(prefix), intervention=intervention)
            report.setdefault("training_seconds", {})[name] = time.perf_counter() - started
            models[name] = source.router.snapshot(states[name])
            report["clocks"][name] = source.router.clocks(states[name])
            source.shared.source.local.crossed.common_contract(states[name]["trainer"])
            assert (states[name]["trainer"].buffer.regimes[:END] == -1).all()
            if intervention is not None:
                interventions[name] = intervention.state_dict()
        report.update(interventions=interventions, phase="evaluation",
            replay_audits={name: source.replay_audit(state, groups) for name, state in states.items()})
        save_audit(output, states, models)
        write(output, report)
        for name in MODEL_NAMES:
            report["evaluations"][name] = source.shared.endpoint_record(source.router.evaluate(prefix, {"soft": models[name]}, seed, "budget_soft"))
            write(output, report)
        u, physical_u = END - KWARGS["learning_starts"] + 1, CUT - KWARGS["learning_starts"] + 1 + 3 * (END - CUT)
        price = dict(training_interactions=END, calibration_interactions=0, initialization_probes=4,
                     actor_steps=u, critic_steps=u, encoder_steps=u)
        actual = dict(training_interactions=CUT + 3 * (END - CUT), calibration_interactions=0, initialization_probes=8,
            actor_steps=physical_u, critic_steps=physical_u, encoder_steps=physical_u,
            evaluation_interactions=sum(e["interactions"] for e in report["evaluations"].values()))
        for key in ("group_gradient_calls", "group_gradient_rows", "component_gradient_calls", "component_gradient_rows"):
            actual[key] = sum(e[key] for e in interventions.values())
        report.update(path_prices={name: price for name in MODEL_NAMES[1:]}, actual_work=actual,
                      complete=True, controls_passed=True, phase="complete")
        check(report)
        write(output, report)
        print("DONE", PROTOCOL, env, seed, json.dumps(actual), flush=True)
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
        contrasts, passed = {}, True
        for control in ("native", "soft"):
            delta = {k: dict(mean=methods["value"][k]["mean"] - methods[control][k]["mean"],
                minimum_seed_difference=methods["value"][k]["minimum_seed"] - methods[control][k]["minimum_seed"],
                per_seed=np.subtract(methods["value"][k]["per_seed"], methods[control][k]["per_seed"]).tolist())
                for k in ("utility", "hv")}
            modes = np.subtract(methods["value"]["mean_mode_utility"], methods[control]["mean_mode_utility"]).tolist()
            passed = passed and delta["utility"]["mean"] > 0 and min(modes) >= 0 and all(
                v["mean"] >= 0 and v["minimum_seed_difference"] >= 0 for v in delta.values())
            contrasts[control] = dict(metrics=delta, mean_mode_difference=modes)
        retention = env != "mo-halfcheetah-v5" or methods["value"]["mean_mode_utility"][1] >= methods["prefix"]["mean_mode_utility"][1]
        result["environments"][env] = dict(methods=methods, value_minus_control=contrasts,
            retention_passed=retention, passed=bool(passed and retention),
            interventions=[r["interventions"] for r in records], replay_audits=[r["replay_audits"] for r in records],
            training_seconds=[r["training_seconds"] for r in records])
    result["decision"] = "fresh_confirmation_required" if all(e["passed"] for e in result["environments"].values()) else "hold_value_constraint"
    write(Path(root) / "compact_readout.json", result)
    print("DONE", PROTOCOL, "summary", result["decision"], flush=True)
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--env", choices=ENVIRONMENTS)
    parser.add_argument("--seed", type=int, choices=SEEDS)
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--summarize", action="store_true")
    args = parser.parse_args()
    summarize(args.root) if args.summarize else run(args.env, args.seed, args.root)
