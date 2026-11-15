"""Matched Adam-proposal residual bounds on the inspected Hopper prefixes."""

import argparse
import copy
import json
from pathlib import Path
import time

import numpy as np
import torch

import run_crossed_replay_probe as crossed


source = crossed.source
base, training, write_report = crossed.base, crossed.training, crossed.write_report
PROTOCOL = "update_residual_probe_v2"
ROOT = Path("results/iclr_update_residual_20261002_repair")
SOURCE = source.ROOT
ENV, SEEDS, STREAMS = crossed.ENV, crossed.SEEDS, crossed.STREAMS
METHOD, CUT, COLLECTION_STEPS, UPDATES, KWARGS = (
    crossed.METHOD, crossed.CUT, crossed.COLLECTION_STEPS, crossed.UPDATES, crossed.KWARGS)
ARMS = ("old", "uniform", "bounded")
NETWORKS = ("policy", "q1", "q2", "q1_target", "q2_target")
GROUPS = {"actor": ("policy",), "critic": ("q1", "q2")}


def specification():
    return dict(source_root=str(SOURCE), source_experiment=source.specification(),
        env=ENV, seeds=list(SEEDS), streams=list(STREAMS), method=METHOD,
        prefix_steps=CUT, collection_steps=COLLECTION_STEPS, updates_per_arm=UPDATES,
        training_kwargs=KWARGS, arms=list(ARMS), proposals_per_update=2,
        radius="new_replay_mass_times_current_old_replay_Adam_step_norm_per_parameter_group",
        projection="float64_displacements_and_interpolation; round_outward_coordinates_one_ulp_toward_reference",
        moments="retain_selected_proposal_Adam_moments; reference_resynced_each_update",
        targets="interpolate_once_from_pre_update_targets_to_accepted_critic",
        update_rng="62000000+10000*seed+10000000*stream+update",
        evaluation_repeats=base.REPEATS, evaluation_horizon=base.HORIZON)


def budget():
    prefix = CUT - KWARGS["learning_starts"] + 1
    modes = base.previous.previous.previous.extension.ENVIRONMENTS[ENV]["num_modes"]
    proposals = len(STREAMS) * (2 + 2 * len(ARMS)) * UPDATES
    audit_rows = 21 * COLLECTION_STEPS
    return dict(training_interactions=CUT + 2 * COLLECTION_STEPS,
        evaluation_interactions=(len(STREAMS) + 1) * modes * 21 * base.REPEATS * base.HORIZON,
        environment_initialization_probe_steps=2,
        actor_optimizer_steps=prefix + proposals, critic_optimizer_steps=prefix + proposals,
        encoder_optimizer_steps=prefix, audit_target_rows=audit_rows,
        audit_loss_rows=(2 + len(STREAMS) * (2 + len(ARMS))) * audit_rows)


def parameters(trainer, names):
    return [parameter for name in names for parameter in getattr(trainer, name).parameters()]


def sync_reference(reference, current):
    for name in NETWORKS:
        getattr(reference, name).load_state_dict(getattr(current, name).state_dict())
    for name in ("q_opt", "policy_opt"):
        getattr(reference, name).load_state_dict(copy.deepcopy(getattr(current, name).state_dict()))
    reference.update_work = dict(current.update_work)


def norm(tensors):
    return float(torch.sqrt(sum(tensor.double().square().sum() for tensor in tensors)))


@torch.no_grad()
def bound_residual(current, reference, before, new_mass):
    residual = [value.double() - old.double() for value, old in zip(current, reference)]
    reference_norm = norm([old.double() - start.double() for old, start in zip(reference, before)])
    radius = new_mass * reference_norm
    raw_norm = norm(residual)
    scale = min(1., radius / raw_norm) if raw_norm else 1.
    if scale < 1.:
        for value, old, delta in zip(current, reference, residual):
            projected = old.double() + scale * delta
            rounded = projected.to(value.dtype)
            outward = (rounded.double() - old.double()).abs() > (projected - old.double()).abs()
            # Quantized coordinates must remain inside the continuous projection.
            value.copy_(torch.where(outward, torch.nextafter(rounded, old), rounded))
    accepted = norm([value.double() - old.double() for value, old in zip(current, reference)])
    assert accepted <= radius + 1e-7 + 1e-4 * radius
    return dict(reference_step_norm=reference_norm, radius=radius, raw_residual_norm=raw_norm,
                accepted_residual_norm=accepted, scale=scale)


def learn(trainer, pool, arm, seed):
    assert arm in ARMS
    crossed.common_contract(trainer)
    branch = base.clone_trainer(trainer, pool)
    reference = base.clone_trainer(trainer, pool)
    reference_replay = crossed.CrossedReplay(pool, trainer.buffer.size)
    active_replay = crossed.CrossedReplay(pool, trainer.buffer.size)
    new_mass = (pool.size - trainer.buffer.size) / pool.size
    before_work = dict(branch.update_work)
    diagnostics = {group: [] for group in GROUPS}
    for update in range(UPDATES):
        sync_reference(reference, branch)
        if arm == "bounded":
            before = {key: [p.detach().clone() for p in parameters(branch, names)]
                      for key, names in GROUPS.items()}
            targets = {name: [p.detach().clone() for p in getattr(branch, name).parameters()]
                       for name in ("q1_target", "q2_target")}
        with base.isolated_rng(seed + update):
            crossed.crossed_update(reference, reference_replay, False, False)
        with base.isolated_rng(seed + update):
            crossed.crossed_update(branch, active_replay, arm != "old", arm != "old")
        if arm == "bounded":
            for group, names in GROUPS.items():
                diagnostics[group].append(bound_residual(
                    parameters(branch, names), parameters(reference, names), before[group], new_mass))
            # The tentative target update must not survive a rejected critic displacement.
            with torch.no_grad():
                for name, target in (("q1", "q1_target"), ("q2", "q2_target")):
                    for value, accepted, start in zip(getattr(branch, target).parameters(),
                            getattr(branch, name).parameters(), targets[target]):
                        value.copy_(start).lerp_(accepted, branch.tau)
    logical_work = {key: branch.update_work[key] - value for key, value in before_work.items()}
    assert logical_work == dict(critic_optimizer_steps=UPDATES, actor_optimizer_steps=UPDATES,
        critic_rows=UPDATES * branch.batch_size, target_candidate_rows=UPDATES * branch.batch_size,
        actor_candidate_rows=UPDATES * branch.batch_size)
    work = {key: 2 * value for key, value in logical_work.items()}
    counts = dict(reference_rows=reference_replay.sampled_rows,
        active_rows=active_replay.sampled_rows, new_rows=active_replay.critic_new_rows)
    assert counts["reference_rows"] == counts["active_rows"] == UPDATES * branch.batch_size
    assert reference_replay.critic_new_rows == reference_replay.actor_new_rows == 0
    assert active_replay.critic_new_rows == active_replay.actor_new_rows
    return branch, work, dict(logical_work=logical_work, new_mass=new_mass, **counts, diagnostics=diagnostics)


def check_budget(record):
    for key, upper in budget().items():
        actual = record["completed_budget"][key]
        assert 0 < actual <= upper if key == "evaluation_interactions" else actual == upper


def run(seed, root=ROOT):
    root = Path(root)
    frozen = json.loads((root / "protocol.json").read_text())
    assert frozen["protocol"] == PROTOCOL and frozen["experiment"] == specification()
    assert seed in SEEDS and (CUT, COLLECTION_STEPS, UPDATES, KWARGS) == (
        base.CUT, base.COLLECTION_STEPS, source.UPDATES, base.KWARGS)
    output = root / f"seed{seed}.json"
    if output.exists():
        saved = json.loads(output.read_text())
        assert (saved["protocol"], saved["seed"], saved["experiment"]) == (PROTOCOL, seed, specification())
        if saved["complete"]:
            return saved
        raise RuntimeError("Incomplete attempt preserved; inspect before a new-root retry")
    cached = json.loads((SOURCE / ENV.replace("-", "_") / f"seed{seed}.json").read_text())
    assert (cached["protocol"], cached["seed"], cached["experiment"], cached["complete"]) == (
        source.PROTOCOL, seed, source.specification(), True)
    assert all(cached[key] for key in ("controls_passed", "representation_frozen",
        "source_reconstruction_passed", "learner_identity_passed"))
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
        pool = source.coupled_buffer(trainer.buffer, data["uniform"])
        audit = source.source.make_audit(trainer, data["audit_uniform"], 82000000 + seed * 1000)
        costs["audit_target_rows"] += audit["rows"]

        def measure(model):
            value = source.source.audit_loss(model, audit)
            costs["audit_loss_rows"] += audit["rows"]
            return value

        def add_work(work):
            for key in ("actor_optimizer_steps", "critic_optimizer_steps"):
                costs[key] += work[key]

        report["base_audit_loss"] = measure(trainer)
        assert report["base_audit_loss"] == cached["base_audit_loss"]
        report["source_reconstruction_passed"] = True
        for stream in STREAMS:
            rng = 62000000 + seed * 10000 + stream * 10000000
            native = {}
            for arm, source_arm in (("old", "old_only"), ("uniform", "uniform")):
                model, work = source.learn(trainer, None if arm == "old" else data["uniform"], rng)
                add_work(work)
                assert measure(model) == cached["streams"][str(stream)][source_arm]["audit_loss"]
                native[arm] = model
            branches, rows = {}, {}
            for arm in ARMS:
                started = time.monotonic()
                model, work, details = learn(trainer, pool, arm, rng)
                add_work(work)
                rows[arm] = dict(learner_work=work, **details, audit_loss=measure(model),
                    learner_seconds=time.monotonic() - started)
                branches[arm] = model
                if arm != "bounded":
                    base.same_learner(model, native[arm])
                    reference = cached["streams"][str(stream)]["old_only" if arm == "old" else arm]
                    assert rows[arm]["audit_loss"] == reference["audit_loss"]
                    rows[arm].update(evaluation=reference["evaluation"], evaluation_reused=True)
            assert rows["uniform"]["new_rows"] == rows["bounded"]["new_rows"]
            assert rows["old"]["new_rows"] == 0
            report["streams"][str(stream)] = dict(control_identity_passed=True, arms=rows)
            write_report(output, report)
            if stream == 0:
                sentinel = base.evaluate(state, branches["old"], seed)
                costs["evaluation_interactions"] += sentinel["interactions"]
                assert all(value == rows["old"]["evaluation"][key]
                           for key, value in sentinel.items() if key != "seconds")
                report["endpoint_reconstruction_passed"] = True
                write_report(output, report)
            evaluated = base.evaluate(state, branches["bounded"], seed)
            costs["evaluation_interactions"] += evaluated["interactions"]
            rows["bounded"].update(evaluation=evaluated, evaluation_reused=False)
            for key in ("evaluation_weights", "episode_seeds_per_mode", "ref_point"):
                assert evaluated[key] == rows["old"]["evaluation"][key]
            print(f"UPDATE_RESIDUAL seed={seed} stream={stream} "
                  f"utility={evaluated['direct_utility']['mean']:.6f}", flush=True)
            write_report(output, report)
            del branches, native, model
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
            assert saved["control_identity_passed"]
            arms = saved["arms"]
            for arm in ARMS:
                assert arms[arm]["learner_work"] == arms["old"]["learner_work"]
                assert arms[arm]["evaluation_reused"] == (arm != "bounded")
                for key in ("evaluation_weights", "episode_seeds_per_mode", "ref_point"):
                    assert arms[arm]["evaluation"][key] == arms["old"]["evaluation"][key]
            row["streams"][str(stream)] = {arm: source.endpoint(arms[arm]) for arm in ARMS}
            clipping = {}
            for group, values in arms["bounded"]["diagnostics"].items():
                assert len(values) == UPDATES
                assert all(v["accepted_residual_norm"] <= v["radius"] +
                           1e-7 + 1e-4 * v["radius"] for v in values)
                clipping[group] = dict(clipped_steps=sum(v["scale"] < 1 for v in values),
                    mean_scale=float(np.mean([v["scale"] for v in values])),
                    minimum_scale=min(v["scale"] for v in values))
            row["streams"][str(stream)]["bounded"]["clipping"] = clipping
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
    for control in ("old", "uniform"):
        contrasts = {}
        for metric in ("utility", "hv"):
            deltas = np.asarray([[row["streams"][str(s)]["bounded"][metric] -
                row["streams"][str(s)][control][metric] for s in STREAMS] for row in result["runs"]])
            means = deltas.mean(1)
            contrasts[metric] = dict(mean=float(means.mean()), per_seed_means=means.tolist(),
                per_seed_stream_deltas=deltas.tolist(),
                descriptive_ci95=np.quantile(means[bootstrap].mean(1), [.025, .975]).tolist())
        result["contrasts"][f"bounded_minus_{control}"] = contrasts
    activated = any(v["clipped_steps"] for row in result["runs"] for s in STREAMS
                    for v in row["streams"][str(s)]["bounded"]["clipping"].values())
    old = result["contrasts"]["bounded_minus_old"]
    uniform = result["contrasts"]["bounded_minus_uniform"]
    passed = activated and old["utility"]["mean"] > 0 and old["hv"]["mean"] >= 0 and uniform["utility"]["mean"] > 0
    result.update(complete=True, constraint_activated=activated,
        decision="bounded_update_lead_requires_fresh_validation" if passed else "hold_bounded_update_rule")
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
