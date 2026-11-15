"""Cross the mean-action entropy-free objective with added uniform replay."""

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
PROTOCOL = "mean_raw_objective_probe_v1"
ROOT = Path("results/iclr_mean_raw_objective_20261002")
SOURCE = source.ROOT
ENV, SEEDS, STREAMS = crossed.ENV, crossed.SEEDS, crossed.STREAMS
METHOD, CUT, COLLECTION_STEPS, UPDATES, KWARGS = (
    crossed.METHOD, crossed.CUT, crossed.COLLECTION_STEPS, crossed.UPDATES, crossed.KWARGS)
CELLS = {"soft_old": (False, False), "soft_uniform": (False, True),
         "raw_old": (True, False), "raw_uniform": (True, True)}
CACHED = {"soft_old": "old_only", "soft_uniform": "uniform"}
CONTRASTS = {
    "soft_data": {"soft_uniform": 1, "soft_old": -1},
    "raw_data": {"raw_uniform": 1, "raw_old": -1},
    "objective_old": {"raw_old": 1, "soft_old": -1},
    "objective_uniform": {"raw_uniform": 1, "soft_uniform": -1},
    "overall": {"raw_uniform": 1, "soft_old": -1},
    "interaction": {"raw_uniform": 1, "raw_old": -1, "soft_uniform": -1, "soft_old": 1}}


def specification():
    return dict(source_root=str(SOURCE), source_experiment=source.specification(),
        env=ENV, seeds=list(SEEDS), streams=list(STREAMS), method=METHOD,
        prefix_steps=CUT, collection_steps=COLLECTION_STEPS, updates_per_cell=UPDATES,
        training_kwargs=KWARGS, cells={k: list(v) for k, v in CELLS.items()},
        target="r+gamma*(1-terminated)*min_componentwise_Q_targets(s_next,tanh_mean_policy,c_next)",
        actor="minus_mean_weighted_min_componentwise_Q(s,tanh_mean_policy,c)",
        critic_loss="unchanged_trace_normalized_vector_plus_projection_mse",
        optimizer="warm_Adam_without_reset; log_std_head_receives_no_raw_objective_gradient",
        replay="unchanged_row_coupled_uniform; same_rows_and_relabel_preferences_across_objectives",
        update_rng="62000000+10000*seed+10000000*stream+update",
        evaluation_repeats=base.REPEATS, evaluation_horizon=base.HORIZON)


def budget():
    prefix = CUT - KWARGS["learning_starts"] + 1
    modes = base.previous.previous.previous.extension.ENVIRONMENTS[ENV]["num_modes"]
    forks = len(STREAMS) * len(CELLS) * UPDATES
    audit_rows = 21 * COLLECTION_STEPS
    return dict(training_interactions=CUT + 2 * COLLECTION_STEPS,
        evaluation_interactions=(2 * len(STREAMS) + 1) * modes * 21 * base.REPEATS * base.HORIZON,
        environment_initialization_probe_steps=2,
        actor_optimizer_steps=prefix + forks, critic_optimizer_steps=prefix + forks,
        encoder_optimizer_steps=prefix, audit_target_rows=audit_rows,
        audit_loss_rows=(2 + len(STREAMS) * len(CELLS)) * audit_rows)


@torch.no_grad()
def mean_raw_targets(trainer, reward, weights, next_s, next_c, done):
    action = trainer.policy.get_action(next_s, next_c, deterministic=True)
    vector = torch.min(trainer.q1_target(next_s, action, next_c),
                       trainer.q2_target(next_s, action, next_c))
    target = reward + trainer.gamma * (1 - done) * vector
    return (weights * target).sum(-1, keepdim=True), target


def raw_update(trainer):
    s, a, w, c, reward, next_s, next_c, done = trainer.buffer.sample(trainer.batch_size)
    w, c, next_c = trainer.relabel_preferences(w, c, next_c)
    scalar, vector = mean_raw_targets(trainer, reward, w, next_s, next_c, done)
    loss = trainer.critic_objective(trainer.q1(s, a, c), trainer.q2(s, a, c), w, scalar, vector)
    trainer.q_opt.zero_grad()
    loss.backward()
    trainer.q_opt.step()
    trainer.update_work["critic_optimizer_steps"] += 1
    trainer.update_work["critic_rows"] += len(s)
    trainer.update_work["target_candidate_rows"] += len(s)

    action = trainer.policy.get_action(s, c, deterministic=True)
    q = torch.min(trainer.q1(s, action, c), trainer.q2(s, action, c))
    loss = -(w * q).sum(-1).mean()
    trainer.policy_opt.zero_grad(set_to_none=True)
    loss.backward()
    trainer.policy_opt.step()
    trainer.update_work["actor_optimizer_steps"] += 1
    trainer.update_work["actor_candidate_rows"] += len(s)
    with torch.no_grad():
        for network, target in ((trainer.q1, trainer.q1_target), (trainer.q2, trainer.q2_target)):
            for parameter, target_parameter in zip(network.parameters(), target.parameters()):
                target_parameter.lerp_(parameter, trainer.tau)


def learn(trainer, data, cell, seed):
    crossed.common_contract(trainer)
    assert type(trainer.policy) is training.PolicyNet
    raw, added = CELLS[cell]
    if not raw:
        return source.learn(trainer, data if added else None, seed)
    branch = base.clone_trainer(trainer, source.coupled_buffer(trainer.buffer, data if added else None))
    before = dict(branch.update_work)
    for update in range(UPDATES):
        with base.isolated_rng(seed + update):
            raw_update(branch)
    work = {key: branch.update_work[key] - value for key, value in before.items()}
    assert work == dict(critic_optimizer_steps=UPDATES, actor_optimizer_steps=UPDATES,
        critic_rows=UPDATES * branch.batch_size, target_candidate_rows=UPDATES * branch.batch_size,
        actor_candidate_rows=UPDATES * branch.batch_size)
    assert branch.buffer.sampled_rows == UPDATES * branch.batch_size
    return branch, work


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
    report = dict(protocol=PROTOCOL, seed=seed, experiment=specification(), complete=False,
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
    report["prefix"] = dict(step=state["step"], replay_size=trainer.buffer.size,
        learner_work=prefix_work, mode_history=[list(row) for row in state["env"].mode_history])
    write_report(output, report)
    try:
        assert all(value == cached["prefix"][key] for key, value in report["prefix"].items())
        data = {}
        for name in ("uniform", "audit_uniform"):
            rng = (53000000 if name == "audit_uniform" else 52000000) + seed * 1000
            data[name], profile = base.collect(state, "uniform", rng)
            costs["training_interactions"] += COLLECTION_STEPS
            report["collection"][name] = profile
            assert all(value == cached["collection"][name][key] for key, value in profile.items() if key != "seconds")
            write_report(output, report)
        audit = source.source.make_audit(trainer, data["audit_uniform"], 82000000 + seed * 1000)
        costs["audit_target_rows"] += audit["rows"]

        def measure(model):
            value = source.source.audit_loss(model, audit)
            costs["audit_loss_rows"] += audit["rows"]
            return value

        report["base_soft_audit_loss"] = measure(trainer)
        assert report["base_soft_audit_loss"] == cached["base_audit_loss"]
        report["source_reconstruction_passed"] = True
        write_report(output, report)
        for stream in STREAMS:
            rng = 62000000 + seed * 10000 + stream * 10000000
            branches, rows = {}, {}
            for cell in CELLS:
                started = time.monotonic()
                model, work = learn(trainer, data["uniform"], cell, rng)
                for key in ("actor_optimizer_steps", "critic_optimizer_steps"):
                    costs[key] += work[key]
                rows[cell] = dict(learner_work=work, sampled_rows=model.buffer.sampled_rows,
                    new_rows=model.buffer.new_rows, soft_audit_loss=measure(model),
                    learner_seconds=time.monotonic() - started)
                branches[cell] = model
                if cell in CACHED:
                    reference = cached["streams"][str(stream)][CACHED[cell]]
                    assert work == reference["learner_work"]
                    assert rows[cell]["soft_audit_loss"] == reference["audit_loss"]
                    assert rows[cell]["sampled_rows"] == reference["sampled_rows"]
                    assert rows[cell]["new_rows"] == reference["new_rows"]
                    rows[cell].update(evaluation=reference["evaluation"], evaluation_reused=True)
            assert rows["soft_uniform"]["new_rows"] == rows["raw_uniform"]["new_rows"]
            assert rows["soft_old"]["new_rows"] == rows["raw_old"]["new_rows"] == 0
            report["streams"][str(stream)] = dict(control_reconstruction_passed=True, cells=rows)
            write_report(output, report)
            if stream == 0:
                sentinel = base.evaluate(state, branches["soft_old"], seed)
                costs["evaluation_interactions"] += sentinel["interactions"]
                assert all(value == rows["soft_old"]["evaluation"][key] for key, value in sentinel.items() if key != "seconds")
                report["endpoint_reconstruction_passed"] = True
                write_report(output, report)
            for cell in ("raw_old", "raw_uniform"):
                evaluation = base.evaluate(state, branches[cell], seed)
                costs["evaluation_interactions"] += evaluation["interactions"]
                rows[cell].update(evaluation=evaluation, evaluation_reused=False)
                for key in ("evaluation_weights", "episode_seeds_per_mode", "ref_point"):
                    assert evaluation[key] == rows["soft_old"]["evaluation"][key]
                print(f"MEAN_RAW seed={seed} stream={stream} cell={cell} "
                      f"utility={evaluation['direct_utility']['mean']:.6f}", flush=True)
                write_report(output, report)
            del branches, model
        assert trainer.buffer.size == CUT and trainer.update_work == prefix_work
        assert measure(trainer) == report["base_soft_audit_loss"]
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
    indices = np.random.default_rng(0).integers(0, len(SEEDS), (20000, len(SEEDS)))
    result = dict(protocol=PROTOCOL, experiment=specification(), runs=[], completed_budget=dict.fromkeys(budget(), 0))
    for seed in SEEDS:
        record = json.loads((Path(root) / f"seed{seed}.json").read_text())
        assert (record["protocol"], record["seed"], record["experiment"], record["complete"]) == (PROTOCOL, seed, specification(), True)
        assert all(record[key] for key in ("controls_passed", "representation_frozen",
            "source_reconstruction_passed", "endpoint_reconstruction_passed"))
        check_budget(record)
        for key, value in record["completed_budget"].items():
            result["completed_budget"][key] += value
        row = dict(seed=seed, streams={})
        for stream in STREAMS:
            saved = record["streams"][str(stream)]
            assert saved["control_reconstruction_passed"]
            cells = saved["cells"]
            for cell in CELLS:
                assert cells[cell]["learner_work"] == cells["soft_old"]["learner_work"]
                assert cells[cell]["sampled_rows"] == UPDATES * KWARGS["batch_size"]
                assert cells[cell]["evaluation_reused"] == (cell in CACHED)
                for key in ("evaluation_weights", "episode_seeds_per_mode", "ref_point"):
                    assert cells[cell]["evaluation"][key] == cells["soft_old"]["evaluation"][key]
            assert cells["soft_old"]["new_rows"] == cells["raw_old"]["new_rows"] == 0
            assert cells["soft_uniform"]["new_rows"] == cells["raw_uniform"]["new_rows"]
            row["streams"][str(stream)] = {cell: source.endpoint(cells[cell]) for cell in CELLS}
        result["runs"].append(row)
    result["cell_summaries"], result["contrasts"] = {}, {}
    for cell in CELLS:
        summary = {}
        for metric in ("utility", "hv"):
            values = np.asarray([[row["streams"][str(s)][cell][metric] for s in STREAMS] for row in result["runs"]])
            means = values.mean(1)
            summary[metric] = dict(mean=float(means.mean()), per_seed_means=means.tolist(),
                minimum_seed_mean=float(means.min()), minimum_stream_endpoint=float(values.min()))
        summary["per_mode_mean_utility"] = np.mean([[row["streams"][str(s)][cell]["per_mode_utility"]
            for s in STREAMS] for row in result["runs"]], axis=(0, 1)).tolist()
        result["cell_summaries"][cell] = summary
    for name, coefficients in CONTRASTS.items():
        contrasts = {}
        for metric in ("utility", "hv"):
            values = np.asarray([[sum(coef * row["streams"][str(s)][cell][metric]
                for cell, coef in coefficients.items()) for s in STREAMS] for row in result["runs"]])
            means = values.mean(1)
            contrasts[metric] = dict(mean=float(means.mean()), per_seed_means=means.tolist(),
                per_seed_stream_deltas=values.tolist(),
                descriptive_ci95=np.quantile(means[indices].mean(1), [.025, .975]).tolist())
        result["contrasts"][name] = contrasts
    c = result["contrasts"]
    passed = all(c[name]["utility"]["mean"] > 0 and c[name]["hv"]["mean"] >= 0
                 for name in ("raw_data", "overall")) and c["interaction"]["utility"]["mean"] > 0
    result.update(complete=True, decision="mean_raw_lead_requires_fresh_validation" if passed else "hold_mean_raw_objective")
    write_report(Path(root) / "compact_readout.json", result)
    print("DONE", PROTOCOL, "summary", json.dumps(dict(decision=result["decision"], completed_budget=result["completed_budget"])), flush=True)
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
