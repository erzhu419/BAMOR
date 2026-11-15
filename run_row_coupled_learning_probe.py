"""Retrospective data-value diagnostic with row-stable uniform replay coupling."""

import argparse
import copy
import json
from pathlib import Path
import time

import numpy as np
import torch

import run_learning_credit_probe as source


base, training, write_report = source.base, source.training, source.write_report
PROTOCOL = "row_coupled_learning_probe_v1"
ROOT = Path("results/iclr_row_coupled_learning_20261002")
SOURCE = source.ROOT
SEEDS, ENVIRONMENTS = source.SEEDS, source.ENVIRONMENTS
METHOD, CUT, COLLECTION_STEPS, KWARGS = source.METHOD, source.CUT, source.COLLECTION_STEPS, source.KWARGS
UPDATES = source.UPDATES
STREAMS = (0, 1, 2)
ARMS = ("old_only", "uniform", "mixed")
CONTRASTS = (("uniform", "old_only"), ("mixed", "old_only"), ("mixed", "uniform"))


def specification():
    return dict(source_root=str(SOURCE), source_experiment=source.specification(),
        seeds=list(SEEDS), environments=list(ENVIRONMENTS), method=METHOD,
        prefix_steps=CUT, collection_steps=COLLECTION_STEPS, training_kwargs=KWARGS,
        updates_per_branch=UPDATES, update_streams=list(STREAMS), arms=list(ARMS),
        update_rng="62000000+10000*seed+10000000*stream+update",
        replay_coupling="independent_row_and_tail_gate_draws_same_old_row_unless_tail",
        evaluation_repeats=base.REPEATS, evaluation_horizon=base.HORIZON)


def budget(env):
    modes = base.previous.previous.previous.extension.ENVIRONMENTS[env]["num_modes"]
    prefix = CUT - KWARGS["learning_starts"] + 1
    forks = len(ARMS) * (1 + len(STREAMS)) + 1
    audit_rows = 21 * COLLECTION_STEPS
    return dict(training_interactions=CUT + 3 * COLLECTION_STEPS,
        evaluation_interactions=len(ARMS) * len(STREAMS) * modes * 21 * base.REPEATS * base.HORIZON,
        environment_initialization_probe_steps=2,
        actor_optimizer_steps=prefix + forks * UPDATES,
        critic_optimizer_steps=prefix + forks * UPDATES,
        encoder_optimizer_steps=prefix, audit_target_rows=audit_rows,
        audit_loss_rows=(2 + forks) * audit_rows)


def row_indices(index_u, gate_u, old_size, new_size):
    indices = (index_u * old_size).astype(np.int64)
    tail = gate_u < new_size / (old_size + new_size)
    if new_size:
        indices[tail] = old_size + (index_u[tail] * new_size).astype(np.int64)
    return indices, tail


class RowCoupledReplay(training.ReplayBuffer):
    def sample(self, batch_size):
        # Both arms consume the gate draw; unchanged old rows share index_u.
        index_u, gate_u = np.random.random(batch_size), np.random.random(batch_size)
        indices, tail = row_indices(index_u, gate_u, self.old_size, self.size - self.old_size)
        self.sampled_rows += batch_size
        self.new_rows += int(tail.sum())
        return base.batch_at(self, indices)


def coupled_buffer(old, new=None):
    pools = (old,) if new is None else (old, new)
    replay = base.empty_buffer(old, sum(pool.size for pool in pools), RowCoupledReplay)
    replay.old_size, replay.sampled_rows, replay.new_rows = old.size, 0, 0
    for pool in pools:
        for field in base.STORAGE_FIELDS:
            getattr(replay, field)[replay.size:replay.size + pool.size] = getattr(pool, field)[:pool.size]
        replay.size += pool.size
    replay.ptr = replay.size % replay.capacity
    return replay


def learn(trainer, data, seed):
    branch = base.clone_trainer(trainer, coupled_buffer(trainer.buffer, data))
    before = dict(branch.update_work)
    for update in range(UPDATES):
        with base.isolated_rng(seed + update):
            branch.update()
    work = {key: branch.update_work[key] - value for key, value in before.items()}
    assert work == dict(critic_optimizer_steps=UPDATES, actor_optimizer_steps=UPDATES,
        critic_rows=UPDATES * branch.batch_size, target_candidate_rows=UPDATES * branch.batch_size,
        actor_candidate_rows=UPDATES * branch.batch_size)
    assert branch.buffer.sampled_rows == UPDATES * branch.batch_size
    return branch, work


def check_budget(record):
    for key, upper in budget(record["env"]).items():
        actual = record["completed_budget"][key]
        assert 0 < actual <= upper if key == "evaluation_interactions" else actual == upper


def run(env, seed, root=ROOT):
    root = Path(root)
    frozen = json.loads((root / "protocol.json").read_text())
    assert frozen["protocol"] == PROTOCOL and frozen["experiment"] == specification()
    assert (CUT, COLLECTION_STEPS, UPDATES, KWARGS) == (base.CUT, base.COLLECTION_STEPS, source.UPDATES, base.KWARGS)
    assert env in ENVIRONMENTS and seed in SEEDS
    output = root / env.replace("-", "_") / f"seed{seed}.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.exists():
        saved = json.loads(output.read_text())
        assert (saved["protocol"], saved["env"], saved["seed"], saved["experiment"]) == (PROTOCOL, env, seed, specification())
        if saved["complete"]:
            return saved
        raise RuntimeError("Incomplete attempt preserved; inspect before a new-root retry")
    cached = json.loads((SOURCE / env.replace("-", "_") / f"seed{seed}.json").read_text())
    assert (cached["protocol"], cached["env"], cached["seed"], cached["experiment"], cached["complete"]) == (
        source.PROTOCOL, env, seed, source.specification(), True)
    assert all(cached[key] for key in ("controls_passed", "representation_frozen",
        "collection_identity_passed", "learner_identity_passed", "choice_recorded_before_evaluation"))
    report = dict(protocol=PROTOCOL, env=env, seed=seed, experiment=specification(), complete=False,
        controls_passed=False, planned_budget_upper_bound=budget(env),
        completed_budget=dict.fromkeys(budget(env), 0), collection={}, legacy={}, streams={})
    write_report(output, report)
    capture = base.Capture()
    training.train_method(env, METHOD, CUT, 25000, seed, str(output.parent / f"prefix{seed}"),
                          state_probe=capture, **KWARGS)
    state, costs = capture.state, report["completed_budget"]
    trainer = state["trainer"]
    prefix_work = dict(trainer.update_work)
    encoder_before = copy.deepcopy(state["encoder"].state_dict())
    costs.update(training_interactions=CUT, environment_initialization_probe_steps=2,
        actor_optimizer_steps=prefix_work["actor_optimizer_steps"],
        critic_optimizer_steps=prefix_work["critic_optimizer_steps"],
        encoder_optimizer_steps=training.optimizer_step_count(state["encoder_opt"]))
    report["prefix"] = dict(step=state["step"], replay_size=trainer.buffer.size, learner_work=prefix_work,
        mode_history=[list(row) for row in state["env"].mode_history])
    try:
        assert all(value == cached["prefix"][key] for key, value in report["prefix"].items())
        data = {}
        for name in ("uniform", "mixed", "audit_uniform"):
            rng = (53000000 if name == "audit_uniform" else 52000000) + seed * 1000
            data[name], profile = base.collect(state, "uniform" if name == "audit_uniform" else name, rng)
            costs["training_interactions"] += COLLECTION_STEPS
            report["collection"][name] = profile
            assert all(value == cached["collection"][name][key] for key, value in profile.items() if key != "seconds")
            write_report(output, report)
        audit = source.make_audit(trainer, data["audit_uniform"], 82000000 + seed * 1000)
        costs["audit_target_rows"] += audit["rows"]

        def measure(model):
            measured = source.audit_loss(model, audit)
            costs["audit_loss_rows"] += audit["rows"]
            return measured

        def add_work(work):
            for key in ("actor_optimizer_steps", "critic_optimizer_steps"):
                costs[key] += work[key]

        report["base_audit_loss"] = measure(trainer)
        assert report["base_audit_loss"] == cached["audit_losses"]["base"]
        legacy_old = None
        for name in ARMS:
            branch, work = source.learn(trainer, data.get(name), 62000000 + seed * 10000)
            add_work(work)
            measured = measure(branch)
            assert work == cached["branches"][name]["learner_work"]
            assert measured == cached["audit_losses"][name], (env, seed, name, "legacy reconstruction")
            report["legacy"][name] = dict(audit_loss=measured, learner_work=work,
                evaluation=cached["branches"][name]["evaluation"])
            if name == "old_only":
                legacy_old = branch
            del branch
        report["source_reconstruction_passed"] = True
        write_report(output, report)
        for stream in STREAMS:
            rng = 62000000 + seed * 10000 + stream * 10000000
            rows, branches = {}, {}
            for name in ARMS:
                started = time.monotonic()
                branch, work = learn(trainer, data.get(name), rng)
                add_work(work)
                branches[name] = branch
                rows[name] = dict(learner_work=work, learner_seconds=time.monotonic() - started,
                    audit_loss=measure(branch), replay_size=branch.buffer.size,
                    sampled_rows=branch.buffer.sampled_rows, new_rows=branch.buffer.new_rows)
            assert rows["uniform"]["new_rows"] == rows["mixed"]["new_rows"]
            if stream == 0:
                base.same_learner(legacy_old, branches["old_only"])
                assert rows["old_only"]["audit_loss"] == report["legacy"]["old_only"]["audit_loss"]
                repeated, work = learn(trainer, data["uniform"], rng)
                add_work(work)
                base.same_learner(repeated, branches["uniform"])
                assert measure(repeated) == rows["uniform"]["audit_loss"]
                report["learner_identity_passed"] = True
                del repeated, legacy_old
            report["streams"][str(stream)] = rows
            write_report(output, report)
            for name in ARMS:
                evaluated = base.evaluate(state, branches[name], seed)
                costs["evaluation_interactions"] += evaluated["interactions"]
                rows[name]["evaluation"] = evaluated
                for key in ("evaluation_weights", "episode_seeds_per_mode", "ref_point"):
                    assert evaluated[key] == cached["branches"]["old_only"]["evaluation"][key]
                if stream == 0 and name == "old_only":
                    assert all(value == cached["branches"][name]["evaluation"][key]
                               for key, value in evaluated.items() if key != "seconds")
                print(f"ROW_COUPLED {env} seed={seed} stream={stream} arm={name} "
                      f"utility={evaluated['direct_utility']['mean']:.6f}", flush=True)
                write_report(output, report)
            del branches
        assert trainer.buffer.size == CUT and trainer.update_work == prefix_work
        assert measure(trainer) == report["base_audit_loss"]
        assert all(torch.equal(value, state["encoder"].state_dict()[key]) for key, value in encoder_before.items())
        assert training.optimizer_step_count(state["encoder_opt"]) == costs["encoder_optimizer_steps"]
        check_budget(report)
        report.update(complete=True, controls_passed=True, representation_frozen=True)
        write_report(output, report)
        print(f"DONE {PROTOCOL} {env} seed={seed} cost={json.dumps(costs)}", flush=True)
        return report
    finally:
        state["env"].close()
        state["eval_env"].close()


def endpoint(branch):
    evaluated = branch["evaluation"]
    return dict(utility=evaluated["direct_utility"]["mean"],
        per_mode_utility=evaluated["direct_utility"]["per_mode"], hv=evaluated["final_hv"],
        hv_per_mode=evaluated["hv_per_mode"])


def summarize(root=ROOT):
    bootstrap = np.random.default_rng(0).integers(0, len(SEEDS), (20000, len(SEEDS)))
    result = dict(protocol=PROTOCOL, experiment=specification(), complete=True,
        completed_budget=dict.fromkeys(budget(ENVIRONMENTS[0]), 0), environments={})
    for env in ENVIRONMENTS:
        runs = []
        for seed in SEEDS:
            record = json.loads((Path(root) / env.replace("-", "_") / f"seed{seed}.json").read_text())
            assert (record["protocol"], record["env"], record["seed"], record["experiment"], record["complete"]) == (
                PROTOCOL, env, seed, specification(), True)
            assert all(record[key] for key in ("controls_passed", "representation_frozen",
                "source_reconstruction_passed", "learner_identity_passed"))
            check_budget(record)
            for key, actual in record["completed_budget"].items():
                result["completed_budget"][key] += actual
            row = {key: record[key] for key in ("seed", "prefix", "collection", "completed_budget")}
            row["legacy"] = {name: endpoint(record["legacy"][name]) for name in ARMS}
            row["streams"] = {}
            for stream in STREAMS:
                branches = record["streams"][str(stream)]
                old = branches["old_only"]
                assert branches["uniform"]["new_rows"] == branches["mixed"]["new_rows"]
                for name in ARMS:
                    assert branches[name]["learner_work"] == old["learner_work"] == record["legacy"][name]["learner_work"]
                    assert branches[name]["sampled_rows"] == UPDATES * KWARGS["batch_size"]
                    for key in ("evaluation_weights", "episode_seeds_per_mode", "ref_point"):
                        assert branches[name]["evaluation"][key] == old["evaluation"][key]
                row["streams"][str(stream)] = {name: dict(endpoint(branches[name]),
                    audit_loss=branches[name]["audit_loss"], sampled_rows=branches[name]["sampled_rows"],
                    new_rows=branches[name]["new_rows"], learner_seconds=branches[name]["learner_seconds"])
                    for name in ARMS}
            runs.append(row)
        stats = {}
        for a, b in CONTRASTS:
            measures = {}
            for metric in ("utility", "hv"):
                deltas = np.asarray([[row["streams"][str(stream)][a][metric] -
                    row["streams"][str(stream)][b][metric] for stream in STREAMS] for row in runs])
                seed_means = deltas.mean(1)
                legacy = np.asarray([row["legacy"][a][metric] - row["legacy"][b][metric] for row in runs])
                measures[metric] = dict(mean=float(seed_means.mean()),
                    descriptive_ci95=np.quantile(seed_means[bootstrap].mean(1), [.025, .975]).tolist(),
                    per_seed_stream_deltas=deltas.tolist(), per_seed_means=seed_means.tolist(),
                    within_seed_stream_std=deltas.std(1, ddof=1).tolist(),
                    seeds_with_both_signs=int(((deltas.min(1) < 0) & (deltas.max(1) > 0)).sum()),
                    legacy_per_seed_deltas=legacy.tolist(), legacy_mean=float(legacy.mean()),
                    first_stream_coupling_change=(deltas[:, 0] - legacy).tolist())
            stats[f"{a}_minus_{b}"] = measures
        result["environments"][env] = dict(runs=runs, contrasts=stats)
    positive = all(result["environments"][env]["contrasts"]["uniform_minus_old_only"]["utility"]["mean"] > 0
                   for env in ENVIRONMENTS)
    hv_ok = result["environments"]["mo-hopper-v5"]["contrasts"]["uniform_minus_old_only"]["hv"]["mean"] >= 0
    result["decision"] = "positive_paired_data_value_requires_new_validation" if positive and hv_ok else "hold_new_data_value_under_row_coupling"
    write_report(Path(root) / "compact_readout.json", result)
    print(json.dumps(dict(decision=result["decision"], completed_budget=result["completed_budget"])), flush=True)
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--env", choices=ENVIRONMENTS)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--summarize", action="store_true")
    args = parser.parse_args()
    if args.summarize:
        summarize(args.root)
    else:
        assert args.env and args.seed is not None
        run(args.env, args.seed, args.root)
