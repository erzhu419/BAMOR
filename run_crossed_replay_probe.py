"""Hopper factorial diagnosis of critic replay versus actor state/context inputs."""

import argparse
import copy
import json
from pathlib import Path
import time

import numpy as np
import torch

import run_row_coupled_learning_probe as source


base, training, write_report = source.base, source.training, source.write_report
PROTOCOL = "crossed_replay_probe_v1"
ROOT = Path("results/iclr_crossed_replay_20261002")
SOURCE = source.ROOT
ENV = "mo-hopper-v5"
SEEDS, STREAMS = source.SEEDS, source.STREAMS
METHOD, CUT, COLLECTION_STEPS, UPDATES, KWARGS = source.METHOD, source.CUT, source.COLLECTION_STEPS, source.UPDATES, source.KWARGS
CELLS = {"old_old": (False, False), "new_old": (True, False),
         "old_new": (False, True), "new_new": (True, True)}
DIAGONALS = {"old_old": "old_only", "new_new": "uniform"}
CONTRASTS = {
    "critic_effect_actor_old": {"new_old": 1, "old_old": -1},
    "critic_effect_actor_new": {"new_new": 1, "old_new": -1},
    "actor_effect_critic_old": {"old_new": 1, "old_old": -1},
    "actor_effect_critic_new": {"new_new": 1, "new_old": -1},
    "total_new_data_effect": {"new_new": 1, "old_old": -1},
    "interaction": {"new_new": 1, "new_old": -1, "old_new": -1, "old_old": 1}}


def specification():
    return dict(source_root=str(SOURCE), source_experiment=source.specification(),
        env=ENV, seeds=list(SEEDS), streams=list(STREAMS), method=METHOD,
        prefix_steps=CUT, collection_steps=COLLECTION_STEPS, updates_per_cell=UPDATES,
        training_kwargs=KWARGS, cells={key: list(value) for key, value in CELLS.items()},
        axes=["critic_receives_added_uniform", "actor_receives_added_uniform_states_and_contexts"],
        replay_draws="same_two_arrays_and_tail_mask_for_both_components",
        actor_preferences="reuse_critic_uniform_relabel_weights_without_extra_rng",
        update_rng="62000000+10000*seed+10000000*stream+update",
        evaluation_repeats=base.REPEATS, evaluation_horizon=base.HORIZON)


def budget():
    modes = base.previous.previous.previous.extension.ENVIRONMENTS[ENV]["num_modes"]
    prefix = CUT - KWARGS["learning_starts"] + 1
    forks = len(STREAMS) * (len(CELLS) + len(DIAGONALS))
    audits = 21 * COLLECTION_STEPS
    endpoints = len(STREAMS) * (len(CELLS) - len(DIAGONALS)) + 1
    return dict(training_interactions=CUT + 2 * COLLECTION_STEPS,
        evaluation_interactions=endpoints * modes * 21 * base.REPEATS * base.HORIZON,
        environment_initialization_probe_steps=2,
        actor_optimizer_steps=prefix + forks * UPDATES,
        critic_optimizer_steps=prefix + forks * UPDATES,
        encoder_optimizer_steps=prefix, audit_target_rows=audits,
        audit_loss_rows=(2 + forks) * audits)


class CrossedReplay:
    def __init__(self, pool, old_size):
        self.pool, self.old_size = pool, old_size
        self.sampled_rows = self.critic_new_rows = self.actor_new_rows = 0

    def sample_pair(self, batch_size, critic_new, actor_new):
        index_u, gate_u = np.random.random(batch_size), np.random.random(batch_size)
        new_size = self.pool.size - self.old_size
        critic_indices, critic_tail = source.row_indices(index_u, gate_u, self.old_size, new_size if critic_new else 0)
        actor_indices, actor_tail = source.row_indices(index_u, gate_u, self.old_size, new_size if actor_new else 0)
        self.sampled_rows += batch_size
        self.critic_new_rows += int(critic_tail.sum())
        self.actor_new_rows += int(actor_tail.sum())
        actor = tuple(torch.as_tensor(getattr(self.pool, name)[actor_indices], device=self.pool.device)
                      for name in ("states", "contexts"))
        return base.batch_at(self.pool, critic_indices), actor


def common_contract(trainer):
    assert isinstance(trainer, training.ProjectedCommonTargetMOSACTrainer)
    assert trainer.preference_relabeling and trainer.preference_relabeling_rate == 1.
    assert trainer.preference_relabeling_strategy == "uniform"
    assert trainer.critic_relabel_copies == trainer.envelope_target_candidates == trainer.envelope_policy_candidates == 1
    assert trainer.behavior_clone_coef == 0. and not trainer.dual_branch_update and not trainer.reward_normalization


def crossed_update(trainer, replay, critic_new, actor_new):
    critic, actor = replay.sample_pair(trainer.batch_size, critic_new, actor_new)
    s, a, w, c, reward, next_s, next_c, done = critic
    w, c, next_c = trainer.relabel_preferences(w, c, next_c)
    actor_s, actor_c = actor
    actor_c = trainer.replace_context_preferences(actor_c, w)
    with torch.no_grad():
        scalar, vector = trainer.envelope_next_values(next_s, w, next_c)
        target_scalar, target_vector = trainer.critic_targets(reward, w, done, scalar, vector)
    q1, q2 = trainer.q1(s, a, c), trainer.q2(s, a, c)
    critic_loss = trainer.critic_objective(q1, q2, w, target_scalar, target_vector)
    trainer.q_opt.zero_grad()
    critic_loss.backward()
    trainer.q_opt.step()
    trainer.update_work["critic_optimizer_steps"] += 1
    trainer.update_work["critic_rows"] += len(s)
    trainer.update_work["target_candidate_rows"] += len(s)

    action, log_prob = trainer.policy.get_action(actor_s, actor_c)
    q = torch.min(trainer.q1(actor_s, action, actor_c), trainer.q2(actor_s, action, actor_c))
    scalar = (w * q).sum(-1, keepdim=True)
    policy_loss = trainer.actor_objective(actor_s, w, actor_c, scalar, log_prob)
    trainer.policy_opt.zero_grad()
    policy_loss.backward()
    trainer.policy_opt.step()
    trainer.update_work["actor_optimizer_steps"] += 1
    trainer.update_work["actor_candidate_rows"] += len(actor_s)
    for network, target in ((trainer.q1, trainer.q1_target), (trainer.q2, trainer.q2_target)):
        for parameter, target_parameter in zip(network.parameters(), target.parameters()):
            target_parameter.data.lerp_(parameter.data, trainer.tau)


def learn(trainer, pool, cell, seed):
    common_contract(trainer)
    branch = base.clone_trainer(trainer, pool)
    replay = CrossedReplay(pool, trainer.buffer.size)
    before = dict(branch.update_work)
    for update in range(UPDATES):
        with base.isolated_rng(seed + update):
            crossed_update(branch, replay, *CELLS[cell])
    work = {key: branch.update_work[key] - value for key, value in before.items()}
    assert work == dict(critic_optimizer_steps=UPDATES, actor_optimizer_steps=UPDATES,
        critic_rows=UPDATES * branch.batch_size, target_candidate_rows=UPDATES * branch.batch_size,
        actor_candidate_rows=UPDATES * branch.batch_size)
    counts = dict(sampled_rows=replay.sampled_rows, critic_new_rows=replay.critic_new_rows,
                  actor_new_rows=replay.actor_new_rows)
    assert counts["sampled_rows"] == UPDATES * branch.batch_size
    return branch, work, counts


def check_budget(record):
    for key, upper in budget().items():
        actual = record["completed_budget"][key]
        assert 0 < actual <= upper if key == "evaluation_interactions" else actual == upper


def run(seed, root=ROOT):
    root = Path(root)
    frozen = json.loads((root / "protocol.json").read_text())
    assert frozen["protocol"] == PROTOCOL and frozen["experiment"] == specification()
    assert (CUT, COLLECTION_STEPS, UPDATES, KWARGS) == (base.CUT, base.COLLECTION_STEPS, source.UPDATES, base.KWARGS)
    assert seed in SEEDS
    output = root / f"seed{seed}.json"
    if output.exists():
        saved = json.loads(output.read_text())
        assert (saved["protocol"], saved["seed"], saved["experiment"]) == (PROTOCOL, seed, specification())
        if saved["complete"]:
            return saved
        raise RuntimeError("Incomplete attempt preserved; inspect before a new-root retry")
    cached = json.loads((SOURCE / ENV.replace("-", "_") / f"seed{seed}.json").read_text())
    assert (cached["protocol"], cached["env"], cached["seed"], cached["experiment"], cached["complete"]) == (
        source.PROTOCOL, ENV, seed, source.specification(), True)
    assert all(cached[key] for key in ("controls_passed", "representation_frozen",
        "source_reconstruction_passed", "learner_identity_passed"))
    report = dict(protocol=PROTOCOL, env=ENV, seed=seed, experiment=specification(), complete=False,
        controls_passed=False, planned_budget_upper_bound=budget(),
        completed_budget=dict.fromkeys(budget(), 0), collection={}, streams={})
    write_report(output, report)
    capture = base.Capture()
    training.train_method(ENV, METHOD, CUT, 25000, seed, str(root / f"prefix{seed}"), state_probe=capture, **KWARGS)
    state, costs = capture.state, report["completed_budget"]
    trainer = state["trainer"]
    prefix_work = dict(trainer.update_work)
    encoder_before = copy.deepcopy(state["encoder"].state_dict())
    costs.update(training_interactions=CUT, environment_initialization_probe_steps=2,
        actor_optimizer_steps=prefix_work["actor_optimizer_steps"], critic_optimizer_steps=prefix_work["critic_optimizer_steps"],
        encoder_optimizer_steps=training.optimizer_step_count(state["encoder_opt"]))
    report["prefix"] = dict(step=state["step"], replay_size=trainer.buffer.size, learner_work=prefix_work,
        mode_history=[list(row) for row in state["env"].mode_history])
    try:
        assert report["prefix"] == cached["prefix"]
        data = {}
        for name in ("uniform", "audit_uniform"):
            rng = (53000000 if name == "audit_uniform" else 52000000) + seed * 1000
            data[name], profile = base.collect(state, "uniform", rng)
            costs["training_interactions"] += COLLECTION_STEPS
            report["collection"][name] = profile
            assert all(value == cached["collection"][name][key] for key, value in profile.items() if key != "seconds")
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
            for cell, arm in DIAGONALS.items():
                branch, work = source.learn(trainer, None if arm == "old_only" else data["uniform"], rng)
                add_work(work)
                assert work == cached["streams"][str(stream)][arm]["learner_work"]
                assert measure(branch) == cached["streams"][str(stream)][arm]["audit_loss"]
                native[cell] = branch
            branches, rows = {}, {}
            for cell in CELLS:
                started = time.monotonic()
                branch, work, counts = learn(trainer, pool, cell, rng)
                add_work(work)
                branches[cell] = branch
                rows[cell] = dict(learner_work=work, learner_seconds=time.monotonic() - started,
                    audit_loss=measure(branch), **counts)
                if cell in DIAGONALS:
                    base.same_learner(branch, native[cell])
                    reference = cached["streams"][str(stream)][DIAGONALS[cell]]
                    assert rows[cell]["audit_loss"] == reference["audit_loss"]
                    rows[cell].update(evaluation=reference["evaluation"], evaluation_reused=True)
            tail_rows = rows["new_new"]["critic_new_rows"]
            assert rows["new_new"]["actor_new_rows"] == rows["new_old"]["critic_new_rows"] == rows["old_new"]["actor_new_rows"] == tail_rows
            assert rows["old_old"]["critic_new_rows"] == rows["old_old"]["actor_new_rows"] == rows["new_old"]["actor_new_rows"] == rows["old_new"]["critic_new_rows"] == 0
            report["streams"][str(stream)] = dict(diagonal_identity_passed=True, cells=rows)
            write_report(output, report)
            if stream == 0:
                sentinel = base.evaluate(state, branches["old_old"], seed)
                costs["evaluation_interactions"] += sentinel["interactions"]
                assert all(value == rows["old_old"]["evaluation"][key] for key, value in sentinel.items() if key != "seconds")
                report["endpoint_reconstruction_passed"] = True
                write_report(output, report)
            for cell in ("new_old", "old_new"):
                evaluated = base.evaluate(state, branches[cell], seed)
                costs["evaluation_interactions"] += evaluated["interactions"]
                rows[cell].update(evaluation=evaluated, evaluation_reused=False)
                for key in ("evaluation_weights", "episode_seeds_per_mode", "ref_point"):
                    assert evaluated[key] == rows["old_old"]["evaluation"][key]
                print(f"CROSSED_REPLAY seed={seed} stream={stream} cell={cell} "
                      f"utility={evaluated['direct_utility']['mean']:.6f}", flush=True)
                write_report(output, report)
            del branches, native, branch
        assert trainer.buffer.size == CUT and trainer.update_work == prefix_work
        assert measure(trainer) == report["base_audit_loss"]
        assert all(torch.equal(value, state["encoder"].state_dict()[key]) for key, value in encoder_before.items())
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
    result = dict(protocol=PROTOCOL, experiment=specification(), complete=True,
        completed_budget=dict.fromkeys(budget(), 0), runs=[])
    for seed in SEEDS:
        record = json.loads((Path(root) / f"seed{seed}.json").read_text())
        assert (record["protocol"], record["seed"], record["experiment"], record["complete"]) == (PROTOCOL, seed, specification(), True)
        assert all(record[key] for key in ("controls_passed", "representation_frozen", "source_reconstruction_passed", "endpoint_reconstruction_passed"))
        check_budget(record)
        for key, value in record["completed_budget"].items():
            result["completed_budget"][key] += value
        row = {key: record[key] for key in ("seed", "prefix", "collection", "completed_budget")}
        row["streams"] = {}
        for stream in STREAMS:
            saved = record["streams"][str(stream)]
            assert saved["diagonal_identity_passed"]
            cells = saved["cells"]
            for cell in CELLS:
                assert cells[cell]["learner_work"] == cells["old_old"]["learner_work"]
                assert cells[cell]["sampled_rows"] == UPDATES * KWARGS["batch_size"]
                assert cells[cell]["evaluation_reused"] == (cell in DIAGONALS)
                for key in ("evaluation_weights", "episode_seeds_per_mode", "ref_point"):
                    assert cells[cell]["evaluation"][key] == cells["old_old"]["evaluation"][key]
            row["streams"][str(stream)] = {cell: dict(source.endpoint(cells[cell]),
                audit_loss=cells[cell]["audit_loss"], critic_new_rows=cells[cell]["critic_new_rows"],
                actor_new_rows=cells[cell]["actor_new_rows"], learner_seconds=cells[cell]["learner_seconds"])
                for cell in CELLS}
        result["runs"].append(row)
    result["cell_summaries"] = {}
    for cell in CELLS:
        summary = {}
        for metric in ("utility", "hv"):
            values = np.asarray([[row["streams"][str(stream)][cell][metric] for stream in STREAMS]
                                 for row in result["runs"]])
            seed_means = values.mean(1)
            summary[metric] = dict(mean=float(seed_means.mean()), per_seed_means=seed_means.tolist(),
                minimum_seed_mean=float(seed_means.min()), minimum_stream_endpoint=float(values.min()))
        summary["per_mode_mean_utility"] = np.mean([[row["streams"][str(stream)][cell]["per_mode_utility"]
            for stream in STREAMS] for row in result["runs"]], axis=(0, 1)).tolist()
        result["cell_summaries"][cell] = summary
    stats = {}
    for name, coefficients in CONTRASTS.items():
        stats[name] = {}
        for metric in ("utility", "hv"):
            delta = np.asarray([[sum(coef * row["streams"][str(stream)][cell][metric]
                for cell, coef in coefficients.items()) for stream in STREAMS] for row in result["runs"]])
            means = delta.mean(1)
            stats[name][metric] = dict(mean=float(means.mean()), per_seed_means=means.tolist(),
                per_seed_stream_deltas=delta.tolist(), within_seed_stream_std=delta.std(1, ddof=1).tolist(),
                descriptive_ci95=np.quantile(means[bootstrap].mean(1), [.025, .975]).tolist())
    result["contrasts"] = stats
    screens = {}
    for cell in ("new_old", "old_new"):
        values = {}
        for metric in ("utility", "hv"):
            values[metric] = {base_cell: float(np.mean([row["streams"][str(stream)][cell][metric] -
                row["streams"][str(stream)][base_cell][metric] for row in result["runs"] for stream in STREAMS]))
                for base_cell in ("old_old", "new_new")}
        screens[cell] = dict(differences=values, passed=values["utility"]["new_new"] > 0 and
            values["utility"]["old_old"] >= 0 and values["hv"]["old_old"] >= 0)
    result["route_screens"] = screens
    passed = [cell for cell, screen in screens.items() if screen["passed"]]
    result["decision"] = "route_lead_requires_new_validation:" + ",".join(passed) if passed else "hold_no_crossed_route_recovers_data_value"
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
