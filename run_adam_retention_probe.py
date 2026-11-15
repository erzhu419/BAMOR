"""Isolate retained Adam state under the unchanged mass-scaled parameter bound."""

import argparse
import copy
import json
from pathlib import Path
import time

import numpy as np
import torch

import run_update_residual_probe as parent


base, training, write_report = parent.base, parent.training, parent.write_report
PROTOCOL = "adam_retention_probe_v1"
ROOT = Path("results/iclr_adam_retention_20261002")
SOURCE = parent.ROOT
ENV, SEEDS, STREAMS = parent.ENV, parent.SEEDS, parent.STREAMS
METHOD, CUT, COLLECTION_STEPS, UPDATES, KWARGS = (
    parent.METHOD, parent.CUT, parent.COLLECTION_STEPS, parent.UPDATES, parent.KWARGS)
CACHED = {"old": "old", "uniform": "uniform", "bounded_uniform_mom": "bounded"}
NEW = "bounded_old_mom"
ARMS = (*CACHED, NEW)


def specification():
    return dict(source_root=str(SOURCE), source_experiment=parent.specification(),
        env=ENV, seeds=list(SEEDS), streams=list(STREAMS), method=METHOD,
        prefix_steps=CUT, collection_steps=COLLECTION_STEPS, updates_per_arm=UPDATES,
        training_kwargs=KWARGS, arms=list(ARMS), proposals_per_update=2,
        parameter_rule=parent.specification()["radius"], projection=parent.specification()["projection"],
        intervention="after_parameter_acceptance_copy_both_old_reference_Adam_state_dicts",
        proposal_rule="resync_reference_to_current_accepted_parameters_targets_and_optimizer_state_each_step",
        update_rng="62000000+10000*seed+10000000*stream+update",
        evaluation_repeats=base.REPEATS, evaluation_horizon=base.HORIZON)


def budget():
    prefix = CUT - KWARGS["learning_starts"] + 1
    modes = base.previous.previous.previous.extension.ENVIRONMENTS[ENV]["num_modes"]
    proposals = len(STREAMS) * len(ARMS) * 2 * UPDATES
    audit_rows = 21 * COLLECTION_STEPS
    return dict(training_interactions=CUT + 2 * COLLECTION_STEPS,
        evaluation_interactions=(len(STREAMS) + 1) * modes * 21 * base.REPEATS * base.HORIZON,
        environment_initialization_probe_steps=2,
        actor_optimizer_steps=prefix + proposals, critic_optimizer_steps=prefix + proposals,
        encoder_optimizer_steps=prefix, audit_target_rows=audit_rows,
        audit_loss_rows=(2 + len(STREAMS) * len(ARMS)) * audit_rows)


def same_optimizer(first, second):
    a, b = first.state_dict(), second.state_dict()
    assert a["param_groups"] == b["param_groups"] and a["state"].keys() == b["state"].keys()
    for index, state in a["state"].items():
        assert all(torch.equal(value, b["state"][index][key]) for key, value in state.items())


def learn(trainer, pool, arm, seed):
    assert arm in ARMS and UPDATES == parent.UPDATES
    if arm in CACHED:
        return parent.learn(trainer, pool, CACHED[arm], seed)
    parent.crossed.common_contract(trainer)
    branch, reference = (base.clone_trainer(trainer, pool) for _ in range(2))
    reference_replay = parent.crossed.CrossedReplay(pool, trainer.buffer.size)
    active_replay = parent.crossed.CrossedReplay(pool, trainer.buffer.size)
    new_mass = (pool.size - trainer.buffer.size) / pool.size
    before_work = dict(branch.update_work)
    diagnostics = {group: [] for group in parent.GROUPS}
    for update in range(UPDATES):
        parent.sync_reference(reference, branch)
        before = {key: [p.detach().clone() for p in parent.parameters(branch, names)]
                  for key, names in parent.GROUPS.items()}
        targets = {name: [p.detach().clone() for p in getattr(branch, name).parameters()]
                   for name in ("q1_target", "q2_target")}
        with base.isolated_rng(seed + update):
            parent.crossed.crossed_update(reference, reference_replay, False, False)
        with base.isolated_rng(seed + update):
            parent.crossed.crossed_update(branch, active_replay, True, True)
        for group, names in parent.GROUPS.items():
            diagnostics[group].append(parent.bound_residual(
                parent.parameters(branch, names), parent.parameters(reference, names), before[group], new_mass))
        with torch.no_grad():
            for name, target in (("q1", "q1_target"), ("q2", "q2_target")):
                for value, accepted, start in zip(getattr(branch, target).parameters(),
                        getattr(branch, name).parameters(), targets[target]):
                    value.copy_(start).lerp_(accepted, branch.tau)
        # Retain the old proposal's memory, not its parameters or target networks.
        for name in ("q_opt", "policy_opt"):
            getattr(branch, name).load_state_dict(copy.deepcopy(getattr(reference, name).state_dict()))
    for name in ("q_opt", "policy_opt"):
        same_optimizer(getattr(branch, name), getattr(reference, name))
    logical = {key: branch.update_work[key] - value for key, value in before_work.items()}
    assert logical == dict(critic_optimizer_steps=UPDATES, actor_optimizer_steps=UPDATES,
        critic_rows=UPDATES * branch.batch_size, target_candidate_rows=UPDATES * branch.batch_size,
        actor_candidate_rows=UPDATES * branch.batch_size)
    counts = dict(reference_rows=reference_replay.sampled_rows,
        active_rows=active_replay.sampled_rows, new_rows=active_replay.critic_new_rows)
    assert counts["reference_rows"] == counts["active_rows"] == UPDATES * branch.batch_size
    assert reference_replay.critic_new_rows == reference_replay.actor_new_rows == 0
    assert active_replay.critic_new_rows == active_replay.actor_new_rows
    return branch, {key: 2 * value for key, value in logical.items()}, dict(
        logical_work=logical, new_mass=new_mass, **counts, diagnostics=diagnostics,
        moment_retention_passed=True)


def check_budget(record):
    for key, upper in budget().items():
        actual = record["completed_budget"][key]
        assert 0 < actual <= upper if key == "evaluation_interactions" else actual == upper


def run(seed, root=ROOT):
    root = Path(root)
    frozen = json.loads((root / "protocol.json").read_text())
    assert frozen["protocol"] == PROTOCOL and frozen["experiment"] == specification()
    assert seed in SEEDS and (CUT, COLLECTION_STEPS, UPDATES, KWARGS) == (
        parent.CUT, parent.COLLECTION_STEPS, parent.UPDATES, parent.KWARGS)
    output = root / f"seed{seed}.json"
    if output.exists():
        saved = json.loads(output.read_text())
        assert (saved["protocol"], saved["seed"], saved["experiment"]) == (PROTOCOL, seed, specification())
        if saved["complete"]:
            return saved
        raise RuntimeError("Incomplete attempt preserved; inspect before a new-root retry")
    cached = json.loads((SOURCE / f"seed{seed}.json").read_text())
    assert (cached["protocol"], cached["seed"], cached["experiment"], cached["complete"]) == (
        parent.PROTOCOL, seed, parent.specification(), True)
    assert all(cached[key] for key in ("controls_passed", "representation_frozen",
        "source_reconstruction_passed", "endpoint_reconstruction_passed"))
    report = dict(protocol=PROTOCOL, seed=seed, experiment=specification(), complete=False,
        controls_passed=False, planned_budget_upper_bound=budget(),
        completed_budget=dict.fromkeys(budget(), 0), collection={}, streams={})
    write_report(output, report)
    capture = base.Capture()
    training.train_method(ENV, METHOD, CUT, 25000, seed, str(root / f"prefix{seed}"),
                          state_probe=capture, **KWARGS)
    state, costs = capture.state, report["completed_budget"]
    trainer = state["trainer"]
    prefix_work = dict(trainer.update_work)
    encoder_before = copy.deepcopy(state["encoder"].state_dict())
    costs.update(training_interactions=CUT, environment_initialization_probe_steps=2,
        actor_optimizer_steps=prefix_work["actor_optimizer_steps"],
        critic_optimizer_steps=prefix_work["critic_optimizer_steps"],
        encoder_optimizer_steps=training.optimizer_step_count(state["encoder_opt"]))
    report["prefix"] = dict(step=state["step"], replay_size=trainer.buffer.size,
        learner_work=prefix_work, mode_history=[list(row) for row in state["env"].mode_history])
    write_report(output, report)
    try:
        assert report["prefix"] == cached["prefix"]
        data = {}
        for name in ("uniform", "audit_uniform"):
            rng = (53000000 if name == "audit_uniform" else 52000000) + seed * 1000
            data[name], profile = base.collect(state, "uniform", rng)
            costs["training_interactions"] += COLLECTION_STEPS
            report["collection"][name] = profile
            assert all(value == cached["collection"][name][key]
                       for key, value in profile.items() if key != "seconds")
            write_report(output, report)
        pool = parent.source.coupled_buffer(trainer.buffer, data["uniform"])
        audit = parent.source.source.make_audit(trainer, data["audit_uniform"], 82000000 + seed * 1000)
        costs["audit_target_rows"] += audit["rows"]

        def measure(model):
            value = parent.source.source.audit_loss(model, audit)
            costs["audit_loss_rows"] += audit["rows"]
            return value

        report["base_audit_loss"] = measure(trainer)
        assert report["base_audit_loss"] == cached["base_audit_loss"]
        report["source_reconstruction_passed"] = True
        write_report(output, report)
        for stream in STREAMS:
            rng = 62000000 + seed * 10000 + stream * 10000000
            branches, rows = {}, {}
            for arm in ARMS:
                started = time.monotonic()
                model, work, details = learn(trainer, pool, arm, rng)
                for key in ("actor_optimizer_steps", "critic_optimizer_steps"):
                    costs[key] += work[key]
                rows[arm] = dict(learner_work=work, **details, audit_loss=measure(model),
                    learner_seconds=time.monotonic() - started)
                branches[arm] = model
                if arm in CACHED:
                    reference = cached["streams"][str(stream)]["arms"][CACHED[arm]]
                    for key in ("learner_work", "logical_work", "new_mass", "reference_rows",
                                "active_rows", "new_rows", "diagnostics", "audit_loss"):
                        assert rows[arm][key] == reference[key], (seed, stream, arm, key)
                    rows[arm].update(evaluation=reference["evaluation"], evaluation_reused=True)
                write_report(output, report)
            assert rows[NEW]["new_rows"] == rows["uniform"]["new_rows"] == rows["bounded_uniform_mom"]["new_rows"]
            assert rows["old"]["new_rows"] == 0
            report["streams"][str(stream)] = dict(control_reconstruction_passed=True, arms=rows)
            write_report(output, report)
            if stream == 0:
                sentinel = base.evaluate(state, branches["bounded_uniform_mom"], seed)
                costs["evaluation_interactions"] += sentinel["interactions"]
                assert all(value == rows["bounded_uniform_mom"]["evaluation"][key]
                           for key, value in sentinel.items() if key != "seconds")
                report["endpoint_reconstruction_passed"] = True
                write_report(output, report)
            evaluated = base.evaluate(state, branches[NEW], seed)
            costs["evaluation_interactions"] += evaluated["interactions"]
            rows[NEW].update(evaluation=evaluated, evaluation_reused=False)
            for key in ("evaluation_weights", "episode_seeds_per_mode", "ref_point"):
                assert evaluated[key] == rows["old"]["evaluation"][key]
            print(f"ADAM_RETENTION seed={seed} stream={stream} "
                  f"utility={evaluated['direct_utility']['mean']:.6f}", flush=True)
            write_report(output, report)
            del branches, model
        assert trainer.buffer.size == CUT and trainer.update_work == prefix_work
        assert measure(trainer) == report["base_audit_loss"]
        assert all(torch.equal(value, state["encoder"].state_dict()[key])
                   for key, value in encoder_before.items())
        assert training.optimizer_step_count(state["encoder_opt"]) == costs["encoder_optimizer_steps"]
        check_budget(report)
        report.update(complete=True, controls_passed=True, representation_frozen=True)
        write_report(output, report)
        print(f"DONE {PROTOCOL} seed={seed} cost={json.dumps(costs)}", flush=True)
        return report
    finally:
        state["env"].close()
        state["eval_env"].close()


def summarize(root=ROOT):
    bootstrap = np.random.default_rng(0).integers(0, len(SEEDS), (20000, len(SEEDS)))
    result = dict(protocol=PROTOCOL, experiment=specification(), runs=[],
        completed_budget=dict.fromkeys(budget(), 0))
    for seed in SEEDS:
        record = json.loads((Path(root) / f"seed{seed}.json").read_text())
        assert (record["protocol"], record["seed"], record["experiment"], record["complete"]) == (
            PROTOCOL, seed, specification(), True)
        assert all(record[key] for key in ("controls_passed", "representation_frozen",
            "source_reconstruction_passed", "endpoint_reconstruction_passed"))
        check_budget(record)
        for key, value in record["completed_budget"].items():
            result["completed_budget"][key] += value
        row = dict(seed=seed, streams={})
        for stream in STREAMS:
            saved = record["streams"][str(stream)]
            assert saved["control_reconstruction_passed"]
            arms = saved["arms"]
            assert arms[NEW]["moment_retention_passed"]
            for arm in ARMS:
                assert arms[arm]["learner_work"] == arms["old"]["learner_work"]
                assert arms[arm]["evaluation_reused"] == (arm in CACHED)
                for key in ("evaluation_weights", "episode_seeds_per_mode", "ref_point"):
                    assert arms[arm]["evaluation"][key] == arms["old"]["evaluation"][key]
            assert arms[NEW]["new_rows"] == arms["uniform"]["new_rows"] == arms["bounded_uniform_mom"]["new_rows"]
            row["streams"][str(stream)] = {arm: parent.source.endpoint(arms[arm]) for arm in ARMS}
            for arm in ("bounded_uniform_mom", NEW):
                clipping = {}
                for group, values in arms[arm]["diagnostics"].items():
                    assert len(values) == UPDATES
                    assert all(v["accepted_residual_norm"] <= v["radius"] +
                               1e-7 + 1e-4 * v["radius"] for v in values)
                    clipping[group] = dict(clipped_steps=sum(v["scale"] < 1 for v in values),
                        mean_scale=float(np.mean([v["scale"] for v in values])))
                row["streams"][str(stream)][arm]["clipping"] = clipping
        result["runs"].append(row)
    result["arm_summaries"], result["contrasts"] = {}, {}
    for arm in ARMS:
        summary = {}
        for metric in ("utility", "hv"):
            values = np.asarray([[row["streams"][str(s)][arm][metric] for s in STREAMS]
                                 for row in result["runs"]])
            means = values.mean(1)
            summary[metric] = dict(mean=float(means.mean()), per_seed_means=means.tolist(),
                minimum_seed_mean=float(means.min()), minimum_stream_endpoint=float(values.min()))
        summary["per_mode_mean_utility"] = np.mean([[row["streams"][str(s)][arm]["per_mode_utility"]
            for s in STREAMS] for row in result["runs"]], axis=(0, 1)).tolist()
        result["arm_summaries"][arm] = summary
    for control in CACHED:
        contrasts = {}
        for metric in ("utility", "hv"):
            deltas = np.asarray([[row["streams"][str(s)][NEW][metric] -
                row["streams"][str(s)][control][metric] for s in STREAMS] for row in result["runs"]])
            means = deltas.mean(1)
            contrasts[metric] = dict(mean=float(means.mean()), per_seed_means=means.tolist(),
                per_seed_stream_deltas=deltas.tolist(),
                descriptive_ci95=np.quantile(means[bootstrap].mean(1), [.025, .975]).tolist())
        result["contrasts"][f"{NEW}_minus_{control}"] = contrasts
    contrasts = result["contrasts"]
    passed = all(contrasts[f"{NEW}_minus_{control}"]["utility"]["mean"] > 0 for control in CACHED)
    passed = passed and contrasts[f"{NEW}_minus_old"]["hv"]["mean"] >= 0
    result.update(complete=True, decision="old_moment_lead_requires_fresh_validation" if passed else "hold_old_moment_retention")
    write_report(Path(root) / "compact_readout.json", result)
    print(json.dumps(dict(decision=result["decision"], completed_budget=result["completed_budget"])), flush=True)
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int)
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--summarize", action="store_true")
    args = parser.parse_args()
    if args.summarize:
        summarize(args.root)
    else:
        assert args.seed is not None
        run(args.seed, args.root)
