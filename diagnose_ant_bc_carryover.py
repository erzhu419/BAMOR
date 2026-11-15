"""Read-only continuation diagnosis of the five completed Ant BC pairs."""

import argparse
import json
from pathlib import Path

import numpy as np

import run_ant_component_replication as experiment


PROTOCOL = "ant_bc_carryover_logs_v1"
PAIRS = ("full", "no_bc")


def reward_mode_counts(history, start, end):
    # The switch is logged after step(t)'s reward; the new mode starts at t+1.
    counts = {}
    for i, (switch, mode) in enumerate(history):
        next_switch = history[i + 1][0] if i + 1 < len(history) else end
        count = min(end, next_switch) - max(start, switch + 1) + 1
        if count > 0:
            counts[str(mode)] = counts.get(str(mode), 0) + count
    assert sum(counts.values()) == end - start + 1
    return counts


def feedback(block):
    if not block["complete"]:
        assert "half_mean_rewards" not in block and "progress_score" not in block
        return None
    rewards, scale = np.asarray(block["half_mean_rewards"]), np.asarray(block["scale_before_block"])
    terms = np.tanh((rewards[1] - rewards[0]) / scale) / 3
    np.testing.assert_allclose(terms.sum(), block["progress_score"], atol=1e-12)
    return dict(half_mean_rewards=rewards.tolist(), scale_before_block=scale.tolist(),
                objective_score_terms=terms.tolist(), progress_score=block["progress_score"])


def endpoint(full, no_bc, weights):
    assert full["step"] == no_bc["step"]
    mixed = (weights[:, 2] > 0) & (weights[:, 2] < 1)
    zero, pure = weights[:, 2] == 0, weights[:, 2] == 1
    assert (mixed.sum(), zero.sum(), pure.sum()) == (14, 6, 1)
    delta = np.asarray(no_bc["returns"]) - full["returns"]
    assert delta.shape == (3, 21, 3) and np.isfinite(delta).all()
    terms = delta * weights[None, :, :]
    utility = terms.sum(axis=-1)
    contributions = terms[:, mixed].mean(axis=1)
    primary = contributions[0].sum()
    np.testing.assert_allclose(primary, no_bc[experiment.PRIMARY] - full[experiment.PRIMARY], atol=1e-8)
    return dict(step=full["step"], primary_delta=float(primary),
        mixed_weighted_objective_contributions_per_mode=contributions.tolist(),
        group_utility_delta_per_mode={name: utility[:, mask].mean(axis=1).tolist()
                                     for name, mask in (("mixed", mixed), ("zero_energy", zero), ("pure_energy", pure))},
        mixed_mode0_positive_weights=int((utility[0, mixed] > 0).sum()),
        mixed_mode0_negative_weights=int((utility[0, mixed] < 0).sum()),
        mode0_by_weight=[dict(weight=w.tolist(), objective_return_delta=d.tolist(),
                             weighted_contributions=t.tolist(), utility_delta=float(t.sum()))
                         for w, d, t in zip(weights, delta[0], terms[0])])


def diagnose(raw):
    assert raw["protocol"] == experiment.PROTOCOL and raw["complete"]
    assert raw["seed"] in experiment.SEEDS
    full, no_bc = [raw["branches"][name] for name in PAIRS]
    assert full["complete"] and no_bc["complete"]
    history = full["mode_history"]
    assert history == no_bc["mode_history"]
    blocks = {name: [b for b in raw["branches"][name]["block_history"]
                     if b["end_step"] > experiment.CUT] for name in PAIRS}
    assert len(blocks["full"]) == len(blocks["no_bc"]) == 7
    selector, timeline = experiment.NativeSelector(3), []
    for a, b in zip(blocks["full"], blocks["no_bc"]):
        assert all(a[k] == b[k] for k in ("end_step", "operator", "steps", "complete", "half_counts"))
        start, end = a["end_step"] - a["steps"] + 1, a["end_step"]
        assert sum(a["half_counts"]) == a["steps"]
        boundary = start + a["half_counts"][0] - 1
        config = selector.trainer_config(a["operator"])
        coef = config["behavior_clone_coef"]
        timeline.append(dict(start=start, end=end, operator=a["operator"],
            phase="intervention" if end == experiment.BLOCK_END else "continuation",
            bc_coef=dict(full=coef, no_bc=0. if end == experiment.BLOCK_END else coef),
            copies=config["critic_relabel_copies"], half_counts=a["half_counts"],
            half_reward_mode_counts=[reward_mode_counts(history, start, boundary),
                                    reward_mode_counts(history, boundary + 1, end)],
            feedback=dict(full=feedback(a), no_bc=feedback(b))))
    weights = np.asarray(raw["evaluation"]["evaluation_weights"])
    endpoints = [endpoint(a, b, weights) for a, b in zip(full["measurements"][-2:], no_bc["measurements"])]
    assert [e["step"] for e in endpoints] == [experiment.BLOCK_END, experiment.END]
    continuations = timeline[1:]
    return dict(seed=raw["seed"], mode_history=history, timeline=timeline, endpoints=endpoints,
        postwindow_bc_steps=sum(r["end"] - r["start"] + 1 for r in continuations if r["bc_coef"]["full"] > 0),
        postwindow_switches=[dict(after_step=t, new_mode=m) for t, m in history
                             if experiment.BLOCK_END <= t < experiment.END],
        postwindow_natural_draw_mismatches={name: sum(d["drawn"] != d["executed"]
            for d in raw["branches"][name]["selector_draws"] if d["start"] > experiment.BLOCK_END) for name in PAIRS},
        missing_feedback_steps=sum(r["end"] - r["start"] + 1 for r in timeline if r["feedback"]["full"] is None),
        continuation_primary_change={name: rows[-1][experiment.PRIMARY] - rows[-2][experiment.PRIMARY]
                                     for name, rows in (("full", full["measurements"]), ("no_bc", no_bc["measurements"]))},
        change_in_bc_effect=endpoints[1]["primary_delta"] - endpoints[0]["primary_delta"])


def summarize(root=experiment.ROOT):
    rows = [diagnose(json.loads((root / f"seed{seed}.json").read_text())) for seed in experiment.SEEDS]
    return dict(protocol=PROTOCOL, source_protocol=experiment.PROTOCOL, seeds=list(experiment.SEEDS), runs=rows,
        reward_definition=dict(components=["x_velocity", "y_velocity", "negative_unweighted_control_cost"],
            shared_additions=["reward_survive", "reward_contact"], mode_scales=[[.1, .1, 5.], [.1, 5., .1], [5., .1, .2]]),
        new_budget=dict(training_interactions=0, learner_updates=0, evaluation_interactions=0),
        limitations="Retrospective descriptive analysis of five seeds. Only 66500/75000 held-out endpoints; "
                    "training block means are not held-out utility. Last1000 steps lack reward summaries. "
                    "No within-block mode separation for a mixed half. Modes annotate logs, not deployment decisions.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=experiment.ROOT)
    args = parser.parse_args()
    result = summarize(args.root)
    output = args.root / "bc_carryover.json"
    experiment.base.write_report(output, result)
    print(json.dumps(dict(path=str(output), bytes=output.stat().st_size, new_budget=result["new_budget"])))
    for row in result["runs"]:
        print(json.dumps({k: row[k] for k in ("seed", "postwindow_bc_steps", "postwindow_switches",
            "postwindow_natural_draw_mismatches", "missing_feedback_steps", "change_in_bc_effect")}, separators=(",", ":")))
