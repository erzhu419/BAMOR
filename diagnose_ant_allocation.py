"""Prefix-only feedback diagnosis of the completed both-searches-off Ant pairs."""

import argparse
from collections import Counter
import json
from pathlib import Path
from statistics import mean

import numpy as np

import diagnose_progress_terminal_block as replay
import run_ant_pairwise_final75 as experiment


PROTOCOL = "ant_ownpref_allocation_readout_v1"
ROOT = Path("results/iclr_ant_allocation_20260927")
CONDITIONS = ("base_adaptive", "base_blind")
CLASSES = {
    "base_adaptive": replay.training.OwnPreferenceBothProgressExtremeOperatorSelector,
    "base_blind": replay.training.OwnPreferenceBothFixedProgressExtremeOperatorSelector,
}


def selector_prefix(blocks, stop, blind=False):
    selector = CLASSES["base_blind" if blind else "base_adaptive"](3)
    for block in blocks[:stop]:
        assert block["complete"], "Incomplete tails must not supply feedback"
        op = block["operator"]
        selector.counts[op] += 1
        selector.step_counts[op] += block["steps"]
        selector.scores[op] = block["score_after"]
    return selector


def effective_config(selector, op):
    return {"collection_sampler": op.split(":")[1].split("_")[0],
            "relabel_rate": selector.relabel_rate(op), **selector.trainer_config(op)}


def switch_crossings(block, history):
    start = block["end_step"] - block["steps"] + 1
    return [t for t, _ in history if start <= t < block["end_step"]]


def validate_raw(raw, condition, seed):
    assert (raw["env"], raw["seed"], raw["method"], raw["total_steps"],
            raw["training_interactions"]) == (
                experiment.ENV, seed, experiment.METHODS[condition], 75000, 75000)
    state = raw["adaptive_operator_final"]
    selector = CLASSES[condition](3)
    expected = selector.state_dict()
    for key in ("operators", "operator_trainer_configs", "families", "warmup_sequence",
                "score_signal", "learning_starts", "block_steps", "epsilon", "temperature",
                "family_epsilon", "family_temperature", "min_family_fraction",
                "actor_update_rule", "critic_candidate_rule"):
        assert state[key] == expected[key], key
    blocks = state["block_history"]
    assert [b["end_step"] for b in blocks] == list(range(6500, 74001, 1500)) + [75000]
    assert [b["operator"] for b in blocks[:10]] == selector.warmup_sequence
    for i, block in enumerate(blocks):
        op = block["operator"]
        complete = i < 46
        assert block["complete"] == complete
        assert block["steps"] == (1500 if complete else 1000)
        assert block["half_counts"] == ([750, 750] if complete else [750, 250])
        if complete:
            rewards = np.asarray(block["half_mean_rewards"])
            scale = np.asarray(block["scale_before_block"])
            assert rewards.shape == (2, 3) and scale.shape == (3,)
            assert np.all(scale >= 1e-6)
            gain = (rewards[1] - rewards[0]) / scale
            np.testing.assert_allclose(gain, block["normalized_vector_gain"], atol=1e-12)
            score = float(np.tanh(gain).mean())
            np.testing.assert_allclose(score, block["progress_score"], atol=1e-12)
            selector.scores[op] = (1 - selector.lr) * selector.scores[op] + selector.lr * score
            selector.counts[op] += 1
        else:
            assert "progress_score" not in block
        selector.step_counts[op] += block["steps"]
        np.testing.assert_allclose(selector.scores[op], block["score_after"], atol=1e-12)
    assert selector.counts == state["counts"]
    assert selector.step_counts == state["step_counts"]
    np.testing.assert_allclose(list(selector.scores.values()),
                               [state["scores"][op] for op in selector.operators], atol=1e-12)


def credit_sources(blocks, stop, op, lr, history):
    uses = [b for b in blocks[:stop] if b["operator"] == op]
    return [{"end_step": b["end_step"],
             "ema_contribution": lr * (1 - lr) ** (len(uses) - i - 1) * b["progress_score"],
             "objective_terms": (np.tanh(b["normalized_vector_gain"]) / 3).tolist(),
             "switch_crossings": switch_crossings(b, history)}
            for i, b in enumerate(uses)]


def tv(a, b):
    return sum(abs(a[k] - b[k]) for k in a) / 2


def decision_record(blocks, index, history):
    block = blocks[index]
    start = block["end_step"] - block["steps"] + 1
    selector = selector_prefix(blocks, index)
    adaptive = replay.selection_distribution(selector)
    blind = replay.selection_distribution(selector_prefix(blocks, index, blind=True))
    a, b = adaptive["operator_probabilities"], blind["operator_probabilities"]
    observed = {op: [x for x in blocks[:index] if x["operator"] == op]
                for op in selector.operators}
    last_switch = max(t for t, _ in history if t < start)
    ages = {op: start - rows[-1]["end_step"] for op, rows in observed.items() if rows}
    chosen = block["operator"]
    aliases = [op for op in selector.operators
               if effective_config(selector, op) == effective_config(selector, chosen)]
    return {"block": index + 1, "start_step": start, "operator": chosen,
            "route": adaptive["route"], "adaptive_blind_tv": tv(a, b),
            "chosen_probability_adaptive": a[chosen], "chosen_probability_blind": b[chosen],
            "score_before": selector.scores[chosen], "feedback_age": ages.get(chosen),
            "unobserved_probability_mass": sum(a[op] for op in observed if not observed[op]),
            "pre_switch_latest_probability_mass": sum(a[op] for op, rows in observed.items()
                if rows and rows[-1]["end_step"] <= last_switch),
            "mode_at_start": replay.mode_at(history, start),
            "sources": credit_sources(blocks, index, chosen, selector.lr, history),
            "equivalent_label_scores_before": {op: selector.scores[op] for op in aliases},
            "realized_block_progress": block.get("progress_score"),
            "realized_block_objective_terms": (np.tanh(block["normalized_vector_gain"]) / 3).tolist()
                if block["complete"] else None,
            "switches_inside_chosen_block": switch_crossings(block, history),
            "adaptive_probabilities": a, "blind_probabilities": b,
            "effective_config": effective_config(selector, chosen)}


def work_counts(blocks):
    selector = CLASSES["base_adaptive"](3)
    # The first update at step 5000 precedes the first scored block.
    steps = Counter({selector.warmup_sequence[0]: 1})
    for block in blocks:
        steps[block["operator"]] += block["steps"]
    expanded = sum(n * selector.trainer_config(op)["critic_relabel_copies"]
                   for op, n in steps.items())
    return {"learner_updates": sum(steps.values()), "base_batch_rows": sum(steps.values()) * 256,
            "actor_and_critic_expanded_rows_each": expanded * 256,
            "bc_active_expanded_rows": sum(n * selector.trainer_config(op)["critic_relabel_copies"] * 256
                for op, n in steps.items() if selector.trainer_config(op)["behavior_clone_coef"] > 0)}


def run_feedback(raw):
    blocks = raw["adaptive_operator_final"]["block_history"]
    history = raw["ground_truth"]["mode_history"]
    records = [decision_record(blocks, i, history) for i in range(10, len(blocks))]
    ages = [r["feedback_age"] for r in records if r["feedback_age"] is not None]
    complete = [b for b in blocks if b["complete"]]
    terms = np.abs([np.tanh(b["normalized_vector_gain"]) / 3 for b in complete]).sum(axis=0)
    return records, {
        "postwarmup_choices_including_tail": len(records),
        "quota_choices": sum(r["route"] == "family_quota" for r in records),
        "adaptive_blind_tv_mean": mean(r["adaptive_blind_tv"] for r in records),
        "adaptive_blind_tv_max": max(r["adaptive_blind_tv"] for r in records),
        "chosen_feedback_age_mean": mean(ages), "chosen_feedback_age_max": max(ages),
        "unobserved_choices": sum(r["feedback_age"] is None for r in records),
        "unobserved_probability_mass_mean": mean(r["unobserved_probability_mass"] for r in records),
        "pre_switch_latest_probability_mass_mean": mean(r["pre_switch_latest_probability_mass"] for r in records),
        "switch_crossing_scored_blocks": sum(bool(switch_crossings(b, history)) for b in complete),
        "absolute_objective_score_mass_share": (terms / terms.sum()).tolist() if terms.sum() else [0.] * 3,
        "family_steps": raw["adaptive_operator_final"]["family_step_counts"],
        "work": work_counts(blocks),
    }


def diagnose_pair(raws, seed):
    adaptive, blind = [raws[c] for c in CONDITIONS]
    for key in ("evaluation_weights", "episode_seeds_per_mode", "ref_point"):
        assert adaptive[key] == blind[key], key
    assert adaptive["ground_truth"]["mode_history"] == blind["ground_truth"]["mode_history"]
    a, b = [raw["adaptive_operator_final"]["block_history"] for raw in (adaptive, blind)]
    differences = [i for i, (x, y) in enumerate(zip(a, b)) if x["operator"] != y["operator"]]
    first = differences[0] if differences else len(a)
    start = a[first]["end_step"] - a[first]["steps"] + 1 if differences else 75001
    assert a[:first] == b[:first], "Same-arm prefix is not reproducible"
    episodes = [[r for r in raw["episode_rewards"] if r["step"] < start] for raw in (adaptive, blind)]
    assert episodes[0] == episodes[1], "Episode records already differ before the first arm change"
    traces, summaries = {}, {}
    for condition, raw in raws.items():
        traces[condition], summaries[condition] = run_feedback(raw)
    choices = {c: traces[c][first - 10] for c in CONDITIONS} if differences else None
    if choices:
        for key in ("adaptive_probabilities", "blind_probabilities"):
            assert choices[CONDITIONS[0]][key] == choices[CONDITIONS[1]][key]
        for choice in choices.values():
            for key in ("adaptive_probabilities", "blind_probabilities"):
                choice.pop(key)
    selector = CLASSES["base_adaptive"](3)
    effective_differences = sum(effective_config(selector, a[i]["operator"]) !=
                                effective_config(selector, b[i]["operator"]) for i in differences)
    metrics = {c: experiment.metrics(raw) for c, raw in raws.items()}
    changed_config = ({k: {c: choices[c]["effective_config"][k] for c in CONDITIONS}
                       for k, value in choices[CONDITIONS[0]]["effective_config"].items()
                       if value != choices[CONDITIONS[1]]["effective_config"][k]} if choices else {})
    return {"seed": seed, "identical_prefix_complete_blocks": first,
            "identical_prefix_episode_records": len(episodes[0]),
            "first_difference_step": start if differences else None,
            "different_label_blocks": [i + 1 for i in differences],
            "different_effective_config_blocks": effective_differences,
            "first_changed_config_fields": changed_config,
            "first_choices": choices, "runs": summaries,
            "metrics": metrics, "adaptive_minus_blind": {
                k: metrics[CONDITIONS[0]][k] - metrics[CONDITIONS[1]][k] for k in experiment.METRICS}}


def diagnose(root=experiment.ROOT):
    pairs = []
    for seed in experiment.SEEDS:
        raws = {}
        for condition in CONDITIONS:
            folder = root / condition / f"seed{seed}"
            report = json.loads((folder / "report.json").read_text())
            assert (report["protocol"], report["condition"], report["seed"], report["complete"]) == (
                experiment.PROTOCOL, condition, seed, True)
            raw = json.loads((folder / f"{experiment.METHODS[condition]}_seed{seed}.json").read_text())
            validate_raw(raw, condition, seed)
            assert experiment.metrics(raw) == report["metrics"]
            raws[condition] = raw
        pairs.append(diagnose_pair(raws, seed))
    selector = CLASSES["base_adaptive"](3)
    groups = {}
    for op in selector.operators:
        key = json.dumps(effective_config(selector, op), sort_keys=True)
        groups.setdefault(key, []).append(op)
    return {"protocol": PROTOCOL, "source_root": str(root), "seeds": list(experiment.SEEDS),
            "pairs": pairs, "equivalent_configuration_groups": list(groups.values()),
            "new_training_interactions": 0, "new_evaluation_interactions": 0, "extra_updates": 0,
            "limitations": "Posthoc diagnosis, not independent confirmation. Prefix distributions do not "
                "recover RNG draws or counterfactual outcomes. Modes annotate feedback provenance only. "
                "Online block progress is not final deterministic preference utility. Work counts describe "
                "expanded batch rows, not wall-clock cost or unique transitions. Identical logged prefixes "
                "do not certify unsaved optimizer states. No fitted gate, retraining or algorithm adoption."}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=experiment.ROOT)
    parser.add_argument("--output", type=Path, default=ROOT / "readout.json")
    args = parser.parse_args()
    result = diagnose(args.root)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    print(json.dumps({"protocol": PROTOCOL, "pairs": len(result["pairs"]),
                      "output_bytes": args.output.stat().st_size,
                      "first_difference_steps": [p["first_difference_step"] for p in result["pairs"]]}))
