"""Matched partition-input intervention with same-action soft/value gradient audits."""

import argparse
import json
from pathlib import Path
import time

import numpy as np
import torch

import run_context_descent_probe as source


training, router, base, shared, write = source.training, source.router, source.base, source.shared, source.write
PROTOCOL = "transition_input_group_adam_descent_v1"
ROOT = Path("results/iclr_transition_group_20261003")
ENVIRONMENTS, SEEDS = source.ENVIRONMENTS, (3201, 3202, 3203)
CUT, END, GROUPS, FIT_ROWS, AUDIT_ROWS = 75000, 100000, 4, 256, 256
KWARGS, SWITCH_INTERVAL = source.KWARGS, source.SWITCH_INTERVAL
MODEL_NAMES = ("prefix", "native", "context", "transition")
DIAG_UPDATES = (1, *range(1000, END - CUT + 1, 1000))
FEATURE_RULE = "one_step_GenericTransitionContextTracker; tanh(delta/3),tanh(delta/max(action_L2,.1)/3),tanh(action),tanh(reward/3)"


def specification():
    return dict(environments=list(ENVIRONMENTS), seeds=list(SEEDS), method=shared.source.METHOD,
        training_kwargs=KWARGS, switch_interval=SWITCH_INTERVAL, prefix_steps=CUT, end_steps=END,
        models=list(MODEL_NAMES), groups=GROUPS, fit_rows=FIT_ROWS,
        grouping="same_fixed_prefix_evenly_spaced_rows_standardization_farthest_centers_nearest_assignment; "
                 "context_arm_original_observable_history_slice; transition_arm_" + FEATURE_RULE,
        privilege="recorded_replay_transitions_only; no_preferences_learned_latents_or_mode_labels_in_grouping; "
                  "physical_labels_reconstructed_only_in_post_training_replay_audit",
        intervention=source.specification()["intervention"], unchanged=source.specification()["unchanged"],
        controls="three_independent_continuations_of_exact_same_prefix; context_vs_transition_changes_only_partition_input",
        diagnostics=dict(updates=list(DIAG_UPDATES), components="alpha_log_prob_and_minus_Q; same_native_sampled_actions; "
            "no_extra_rng_draws_or_optimizer_updates; pair_cosines_and_accepted_displacement_derivatives",
            rows=AUDIT_ROWS, saved="server_only_actor_encoder_critics_and_evenly_spaced_replay_rows_for_each_endpoint"),
        path_budget="100000_training_plus_four_initialization_probes_each; equal_actor_critic_encoder_clocks; "
                    "physical150000_training_plus_eight_probes_per_env_seed; extra_backwards_separately_counted",
        evaluation=source.specification()["evaluation"],
        screen="both_envs_transition_mean_utility_HV_minimum_seed_utility_HV_all_mean_mode_utilities_at_least_"
               "both_native_and_context; strictly_higher_mean_utility_than_both; HC_low_speed_at_least_prefix",
        theory_scope=source.specification()["theory_scope"])


def transition_features(s, a, r, sn):
    # Vectorized one-step version of the repository's generic transition statistics.
    delta = sn - s
    response = delta / np.maximum(np.linalg.norm(a, axis=-1, keepdims=True), .1)
    return np.concatenate((np.tanh(delta / 3.), np.tanh(response / 3.), np.tanh(a), np.tanh(r / 3.)), -1)


class TransitionGroups:
    def __init__(self, trainer):
        b = trainer.buffer
        indices = np.linspace(0, b.size - 1, min(FIT_ROWS, b.size), dtype=int)
        x = transition_features(b.states[indices], b.actions[indices], b.rewards[indices], b.next_states[indices]).astype(np.float64)
        self.mean, scale = x.mean(0), x.std(0)
        self.scale = np.where(scale == 0, 1., scale)
        self.centers = (x[source.history.select_slots(x, GROUPS)] - self.mean) / self.scale
        self.fit_indices, self.sampled_features = indices.tolist(), None

    def labels(self, features):
        z = (features - self.mean) / self.scale
        return ((z[:, None] - self.centers[None]) ** 2).sum(-1).argmin(1)

    def assign(self, context):
        assert len(context) == len(self.sampled_features)
        return self.labels(self.sampled_features)

    def state_dict(self):
        return dict(feature_rule=FEATURE_RULE, mean=self.mean.tolist(), scale=self.scale.tolist(),
                    centers=self.centers.tolist(), fit_indices=self.fit_indices)


def cosine(a, b):
    denominator = float(a.norm() * b.norm())
    return float(a @ b) / denominator if denominator else None


def component_record(groups, soft, entropy, value, delta):
    error = float((soft - entropy - value).norm()) / max(float(entropy.norm() + value.norm()), 1e-12)
    assert error < 1e-5, "component gradients do not reconstruct the native soft gradient"
    return dict(component_sum_relative_error=error, accepted_displacement_norm=float(delta.norm()),
        groups=[dict(group=int(g), entropy_norm=float(e.norm()), value_norm=float(v.norm()), soft_norm=float(h.norm()),
            entropy_value_cosine=cosine(e, v), entropy_derivative=float(e @ delta),
            value_derivative=float(v @ delta), soft_derivative=float(h @ delta))
            for g, h, e, v in zip(groups, soft, entropy, value)],
        pairs=[dict(groups=[int(groups[i]), int(groups[j])], entropy_cosine=cosine(entropy[i], entropy[j]),
            value_cosine=cosine(value[i], value[j]), soft_cosine=cosine(soft[i], soft[j]))
            for i in range(len(groups)) for j in range(i + 1, len(groups))])


class MeasuredDescent(source.ContextDescent):
    def __init__(self, prefix_trainer, partition):
        super().__init__(prefix_trainer)
        assert partition in ("context", "transition")
        self.partition = partition
        if partition == "transition":
            self.groups = TransitionGroups(prefix_trainer)
        self.pending, self.records = None, []
        self.component_calls, self.component_rows, self.component_seconds = 0, 0, 0.

    def configure(self, step, trainer):
        if self.trainer is not None:
            return
        super().configure(step, trainer)
        if self.partition == "transition":
            sample = trainer.buffer.sample

            def observe_sample(count):
                batch = sample(count)
                self.groups.sampled_features = transition_features(
                    *(batch[i].detach().cpu().numpy() for i in (0, 1, 4, 5)))
                return batch

            trainer.buffer.sample = observe_sample

    def objective(self, s, w, c, q_scalar, log_prob):
        loss = super().objective(s, w, c, q_scalar, log_prob)
        if self.updates not in DIAG_UPDATES:
            return loss
        started = time.perf_counter()
        labels = self.groups.assign(c)
        groups = np.unique(labels)
        params = tuple(self.trainer.policy.parameters())
        components = []
        for rows in ((self.trainer.alpha * log_prob).reshape(-1), -q_scalar.reshape(-1)):
            gradients = []
            for group in groups:
                mask = torch.as_tensor(labels == group, device=rows.device)
                grad = torch.autograd.grad(rows[mask].mean(), params, retain_graph=True)
                gradients.append(torch.cat([g.detach().reshape(-1) for g in grad]).double())
                self.component_calls += 1
                self.component_rows += int(mask.sum())
            components.append(torch.stack(gradients))
        before = torch.cat([p.detach().reshape(-1) for p in params]).double()
        self.pending = (before, groups, self.trainer.policy_opt.constraints, *components)
        self.component_seconds += time.perf_counter() - started
        return loss

    def after_step(self, step, trainer, encoder, env, operator):
        if self.pending is not None:
            before, groups, soft, entropy, value = self.pending
            after = torch.cat([p.detach().reshape(-1) for p in trainer.policy.parameters()]).double()
            self.records.append(dict(step=step, update=self.updates,
                **component_record(groups, soft, entropy, value, after - before)))
            self.pending = None

    def state_dict(self):
        return dict(super().state_dict(), partition=self.partition, component_gradient_calls=self.component_calls,
            component_gradient_rows=self.component_rows, component_gradient_seconds=self.component_seconds,
            component_records=self.records)


def train(env, seed, output, steps, resume=None, intervention=None):
    capture = base.Capture()
    capture.step = steps
    training.train_method(env, shared.source.METHOD, steps, SWITCH_INTERVAL, seed, str(output.parent),
        resume_state=resume, update_router=intervention, state_probe=capture, **KWARGS)
    return capture.state


def replay_audit(state, groups):
    b = state["trainer"].buffer
    indices = np.linspace(0, b.size - 1, min(AUDIT_ROWS, b.size), dtype=int)
    events = np.asarray(state["env"].mode_history)
    # Switch events are recorded after transition t, so new physics starts at t+1.
    modes = events[np.searchsorted(events[:, 0], b.timesteps[indices, 0] - 1, side="right") - 1, 1]
    x = transition_features(b.states[indices], b.actions[indices], b.rewards[indices], b.next_states[indices])
    labels = {"context": groups["context"].assign(torch.as_tensor(b.contexts[indices])),
              "transition": groups["transition"].labels(x)}
    return dict(indices=indices.tolist(), mode_history=events.tolist(),
        mode_rows=np.bincount(modes, minlength=4).tolist(),
        mode_by_group={name: [[int(((modes == m) & (g == j)).sum()) for j in range(GROUPS)]
                             for m in range(4)] for name, g in labels.items()})


def save_audit(output, states, models):
    artifact = dict(protocol=PROTOCOL, models={})
    for name, state in states.items():
        t, b = state["trainer"], state["trainer"].buffer
        indices = np.linspace(0, b.size - 1, min(AUDIT_ROWS, b.size), dtype=int)
        artifact["models"][name] = dict(alpha=float(t.alpha), step=state["step"], policy=models[name]["policy"].state_dict(),
            encoder=models[name]["encoder"].state_dict(), q1=t.q1.state_dict(), q2=t.q2.state_dict(),
            replay_indices=indices.tolist(), torch_rng=state["torch_rng"],
            replay={k: torch.from_numpy(getattr(b, k)[indices].copy()) for k in
                ("states", "actions", "weights", "contexts", "rewards", "next_states", "next_contexts", "dones", "timesteps")})
    torch.save(artifact, output.with_suffix(".audit.pt"))


def check(record):
    assert (record["protocol"], record["experiment"], record["complete"], record["controls_passed"]) == (PROTOCOL, specification(), True, True)
    assert record["env"] in ENVIRONMENTS and record["seed"] in SEEDS
    u = END - KWARGS["learning_starts"] + 1
    price = dict(training_interactions=END, calibration_interactions=0, initialization_probes=4,
                 actor_steps=u, critic_steps=u, encoder_steps=u)
    assert record["path_prices"] == {name: price for name in MODEL_NAMES[1:]}
    assert record["clocks"] == {name: dict(actor=u, critic=u, encoder=u) for name in MODEL_NAMES[1:]}
    for name in ("context", "transition"):
        extra = record["interventions"][name]
        assert extra["partition"] == name and extra["updates"] == END - CUT
        assert extra["group_gradient_rows"] == (END - CUT) * KWARGS["batch_size"] == sum(extra["group_rows"])
        assert sum(extra["present_group_counts"]) == END - CUT
        assert END - CUT <= extra["group_gradient_calls"] <= GROUPS * (END - CUT)
        assert extra["component_gradient_rows"] == 2 * len(DIAG_UPDATES) * KWARGS["batch_size"]
        assert [r["update"] for r in extra["component_records"]] == list(DIAG_UPDATES)
        assert extra["component_gradient_calls"] == 2 * sum(len(r["groups"]) for r in extra["component_records"])
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
        models = dict(prefix=router.snapshot(prefix))
        groups = dict(context=source.ObservableGroups(prefix["trainer"]), transition=TransitionGroups(prefix["trainer"]))
        for name in MODEL_NAMES[1:]:
            report["phase"] = name
            write(output, report)
            intervention = None if name == "native" else MeasuredDescent(prefix["trainer"], name)
            started = time.perf_counter()
            states[name] = train(env, seed, output, END, resume=source.history.source.clone_state(prefix), intervention=intervention)
            report.setdefault("training_seconds", {})[name] = time.perf_counter() - started
            models[name] = router.snapshot(states[name])
            report["clocks"][name] = router.clocks(states[name])
            shared.source.local.crossed.common_contract(states[name]["trainer"])
            assert (states[name]["trainer"].buffer.regimes[:END] == -1).all()
            if intervention is not None:
                interventions[name] = intervention.state_dict()
        report.update(interventions=interventions, phase="evaluation",
            replay_audits={name: replay_audit(state, groups) for name, state in states.items()})
        save_audit(output, states, models)
        write(output, report)
        for name in MODEL_NAMES:
            report["evaluations"][name] = shared.endpoint_record(router.evaluate(prefix, {"soft": models[name]}, seed, "budget_soft"))
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
        for control in ("native", "context"):
            delta = {k: dict(mean=methods["transition"][k]["mean"] - methods[control][k]["mean"],
                minimum_seed_difference=methods["transition"][k]["minimum_seed"] - methods[control][k]["minimum_seed"],
                per_seed=np.subtract(methods["transition"][k]["per_seed"], methods[control][k]["per_seed"]).tolist())
                for k in ("utility", "hv")}
            modes = np.subtract(methods["transition"]["mean_mode_utility"], methods[control]["mean_mode_utility"]).tolist()
            passed = passed and delta["utility"]["mean"] > 0 and min(modes) >= 0 and all(
                v["mean"] >= 0 and v["minimum_seed_difference"] >= 0 for v in delta.values())
            contrasts[control] = dict(metrics=delta, mean_mode_difference=modes)
        retention = env != "mo-halfcheetah-v5" or methods["transition"]["mean_mode_utility"][1] >= methods["prefix"]["mean_mode_utility"][1]
        result["environments"][env] = dict(methods=methods, transition_minus_control=contrasts,
            retention_passed=retention, passed=bool(passed and retention),
            interventions=[r["interventions"] for r in records], replay_audits=[r["replay_audits"] for r in records],
            training_seconds=[r["training_seconds"] for r in records])
    result["decision"] = "fresh_confirmation_required" if all(e["passed"] for e in result["environments"].values()) else "hold_transition_group_descent"
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
