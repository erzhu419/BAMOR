"""Read-only attribution of seed1405's sole shared-credit schedule difference."""

import argparse
import json
from pathlib import Path

import numpy as np

import diagnose_ant_allocation as allocation
import run_ant_shared_credit as experiment


SEED, INDEX = 1405, 40
CONDITIONS = ("per_label", "shared")
PROTOCOL = "ant_shared_credit_single_block_v1"


def prefix_selector(blocks, stop, shared):
    cls = experiment.SharedCreditOwnPreferenceSelector if shared else allocation.CLASSES["base_adaptive"]
    selector = cls(3)
    for block in blocks[:stop]:
        assert block["complete"]
        op = block["operator"]
        selector.counts[op] += 1
        selector.step_counts[op] += block["steps"]
        selector.scores[op] = (1 - selector.lr) * selector.scores[op] + selector.lr * block["progress_score"]
        if shared:
            for alias in selector.credit_aliases[op]:
                selector.scores[alias] = selector.scores[op]
        np.testing.assert_allclose(selector.scores[op], block["score_after"], rtol=1e-12, atol=1e-12)
    return selector


def score_sources(blocks, stop, selector, op, history):
    aliases = selector.credit_aliases[op] if hasattr(selector, "credit_aliases") else [op]
    uses = [b for b in blocks[:stop] if b["operator"] in aliases]
    start = blocks[stop]["end_step"] - blocks[stop]["steps"] + 1
    rows = [{"operator": b["operator"], "end_step": b["end_step"],
             "age_steps": start - b["end_step"], "progress": b["progress_score"],
             "ema_contribution": selector.lr * (1 - selector.lr) ** (len(uses) - i - 1) * b["progress_score"],
             "mode_at_end": allocation.replay.mode_at(history, b["end_step"]),
             "switches_inside": allocation.switch_crossings(b, history)} for i, b in enumerate(uses)]
    np.testing.assert_allclose(sum(r["ema_contribution"] for r in rows), selector.scores[op], atol=1e-12)
    return rows


def family_intervals(distribution, operators):
    probabilities = np.asarray([distribution[op] for op in operators])
    conditional = probabilities / probabilities.sum()
    bounds = np.r_[0., np.cumsum(conditional)]
    return {op: {"probability": float(p), "conditional_probability": float(q),
                 "conditional_cdf_interval": [float(bounds[i]), float(bounds[i + 1])]}
            for i, (op, p, q) in enumerate(zip(operators, probabilities, conditional))}


def common_prefix(raws, index):
    a, b = [raws[c] for c in CONDITIONS]
    blocks_a, blocks_b = [r["adaptive_operator_final"]["block_history"] for r in (a, b)]
    differences = [i for i, (x, y) in enumerate(zip(blocks_a, blocks_b)) if x["operator"] != y["operator"]]
    assert len(blocks_a) == len(blocks_b) == 47 and differences == [index]
    # Score records deliberately differ; rewards, allocation and scales must not.
    keys = ("end_step", "operator", "steps", "complete", "half_counts", "half_mean_rewards",
            "scale_before_block", "normalized_vector_gain", "progress_score")
    assert [{k: r[k] for k in keys} for r in blocks_a[:index]] == [
        {k: r[k] for k in keys} for r in blocks_b[:index]]
    start = blocks_a[index]["end_step"] - blocks_a[index]["steps"] + 1
    episodes = [[r for r in data["episode_rewards"] if r["step"] < start] for data in (a, b)]
    assert episodes[0] == episodes[1]
    assert blocks_a[index]["scale_before_block"] == blocks_b[index]["scale_before_block"]
    return {"complete_blocks": index, "episodes": len(episodes[0]),
            "start_step": start, "end_step": blocks_a[index]["end_step"],
            "subsequent_identical_operator_blocks": len(blocks_a) - index - 1}


def endpoint(raw):
    fronts = raw["pareto_per_mode"]
    returns = np.asarray([[p["return"] for p in front] for front in fronts])
    weights = np.asarray(raw["evaluation_weights"])
    assert returns.shape == (3, 21, 3) and weights.shape == (21, 3)
    utility = (returns * weights[None, :, :]).sum(axis=-1)
    return {"metrics": experiment.previous.metrics(raw), "direct_utility_per_mode": utility.mean(axis=1).tolist(),
            "final_hv_per_mode": raw["final_hv_per_mode"]}


def diagnose(raws):
    first = common_prefix(raws, INDEX)
    a, b = [raws[c] for c in CONDITIONS]
    for key in ("evaluation_weights", "episode_seeds_per_mode", "ref_point"):
        assert a[key] == b[key]
    history = a["ground_truth"]["mode_history"]
    assert history == b["ground_truth"]["mode_history"]
    result = {"protocol": PROTOCOL, "seed": SEED, "block": INDEX + 1, "common_prefix": first,
              "mode_history": history, "conditions": {}}
    chosen = {c: raw["adaptive_operator_final"]["block_history"][INDEX]["operator"] for c, raw in raws.items()}
    for condition, raw in raws.items():
        assert (raw["seed"], raw["env"], raw["method"]) == (SEED, experiment.previous.ENV,
            experiment.previous.METHODS["base_adaptive"])
        experiment.credit_check(raw, condition)
        blocks = raw["adaptive_operator_final"]["block_history"]
        selector = prefix_selector(blocks, INDEX, shared=condition == "shared")
        state = raw["adaptive_operator_final"]
        for key in ("operators", "families", "warmup_sequence", "epsilon", "temperature",
                    "family_epsilon", "family_temperature", "min_family_fraction"):
            assert getattr(selector, key) == state[key]
        distribution = allocation.replay.selection_distribution(selector)
        assert distribution["route"] == "free_family_sampling"
        family = chosen[condition].split(":")[0]
        assert family == "extreme"
        intervals = family_intervals(distribution["operator_probabilities"], selector.families[family])
        block = blocks[INDEX]
        config = allocation.effective_config(selector, chosen[condition])
        assert selector.trainer_config(chosen[condition]) == state["operator_trainer_configs"][chosen[condition]]
        result["conditions"][condition] = {
            "chosen": chosen[condition], "config": config, "distribution": distribution,
            "family_scores": selector._family_scores(), "family_steps": selector._family_step_counts(),
            "extreme_choices": {op: {**row, "score_before": selector.scores[op],
                "sources": score_sources(blocks, INDEX, selector, op, history)} for op, row in intervals.items()},
            "block_feedback": block,
            "objective_score_terms": (np.tanh(block["normalized_vector_gain"]) / 3).tolist(),
            "switches_inside_block": allocation.switch_crossings(block, history),
            "block_work": {"learner_updates": block["steps"],
                "expanded_rows_actor_and_critic_each": block["steps"] * 256 * config["critic_relabel_copies"],
                "bc_active_rows": block["steps"] * 256 * config["critic_relabel_copies"]
                    if config["behavior_clone_coef"] else 0},
            "endpoint": endpoint(raw)}
    left, right = [result["conditions"][c] for c in CONDITIONS]
    result["changed_config"] = {k: {c: result["conditions"][c]["config"][k] for c in CONDITIONS}
                                for k, v in left["config"].items() if v != right["config"][k]}
    result["shared_minus_per_label"] = {
        "block_progress": right["block_feedback"]["progress_score"] - left["block_feedback"]["progress_score"],
        "metrics": {k: right["endpoint"]["metrics"][k] - v for k, v in left["endpoint"]["metrics"].items()},
        "direct_utility_per_mode": (np.asarray(right["endpoint"]["direct_utility_per_mode"]) -
                                    left["endpoint"]["direct_utility_per_mode"]).tolist()}
    # A compatibility interval is not a recovered historical random draw.
    intervals = [result["conditions"][c]["extreme_choices"][chosen[c]]["conditional_cdf_interval"] for c in CONDITIONS]
    overlap = [max(i[0] for i in intervals), min(i[1] for i in intervals)]
    result["same_within_family_uniform_compatible_interval"] = overlap if overlap[0] < overlap[1] else None
    result["new_budget"] = {"training_interactions": 0, "evaluation_interactions": 0, "learner_updates": 0}
    result["limitations"] = (
        "One posthoc seed, not an independent causal test of any component. Actual sampler RNG and full "
        "optimizer/replay states were not saved. CDF overlap is conditional compatibility, not recovered RNG. "
        "Three update settings and their random draws change together. Subsequent arm labels match, not "
        "subsequent states or draws. Block reward progress is not immediate held-out policy improvement; "
        "only the 75k actor was evaluated. Modes annotate sources, never drive selection.")
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=experiment.ROOT)
    args = parser.parse_args()
    raws = {c: json.loads((args.root / c / f"seed{SEED}" /
                f"{experiment.previous.METHODS['base_adaptive']}_seed{SEED}.json").read_text()) for c in CONDITIONS}
    result = diagnose(raws)
    path = args.root / "seed1405_diagnostic.json"
    path.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    print(json.dumps({"protocol": PROTOCOL, "output_bytes": path.stat().st_size,
                      "delta": result["shared_minus_per_label"]}))
