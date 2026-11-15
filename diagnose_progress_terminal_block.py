"""Read-only, prefix-only diagnosis of the completed v25 selector pairs."""

import argparse
import json
from pathlib import Path
from unittest.mock import patch

import numpy as np

import run_progress_factorial_completion as experiment


training = experiment.previous.pilot.training
SUMMARY = Path("results/iclr_progress_factorial_completion_20260921_summary.json")
FOCAL_OPERATORS = ("recovery:stratified_relabel_envelope", "extreme:edge_envelope_policy")


def prefix_selector(blocks, stop, blind=False, reward_dim=2):
    cls = (training.FixedProgressExtremeHierarchicalOperatorSelector if blind
           else training.ProgressExtremeHierarchicalOperatorSelector)
    selector = cls(reward_dim)
    for block in blocks[:stop]:
        assert block["complete"] and block["steps"] == selector.block_steps
        op = block["operator"]
        selector.counts[op] += 1
        selector.step_counts[op] += block["steps"]
        selector.scores[op] = block["score_after"]
    return selector


def selection_distribution(selector):
    calls = []

    def capture(names, p):
        calls.append((list(names), [float(x) for x in p]))
        return names[0]

    # Capture stochastic quota sampling: last_probs is one-hot in that branch.
    with patch.object(training.np.random, "choice", side_effect=capture):
        selector.select()
    if len(calls) == 1:
        names, probs = calls[0]
        distribution = dict.fromkeys(selector.operators, 0.)
        distribution.update(zip(names, probs))
        route = "family_quota"
    else:
        distribution = dict(selector.last_probs)
        route = "warmup" if not calls else "free_family_sampling"
    assert abs(sum(distribution.values()) - 1.) < 1e-12
    return {"route": route, "operator_probabilities": distribution,
            "family_probabilities": {f: sum(distribution[op] for op in ops)
                                     for f, ops in selector.families.items()}}


def mode_at(history, step):
    # Mode history records switches after that numbered transition's reward.
    return next(mode for start, mode in reversed(history) if start < step)


def block_diagnostic(data, index):
    blocks = data["adaptive_operator_final"]["block_history"]
    block = blocks[index]
    op, end = block["operator"], block["end_step"]
    start = end - block["steps"] + 1
    selector = prefix_selector(blocks, index)
    observed = data["adaptive_operator_final"]
    for key in ("operators", "families", "warmup_sequence", "epsilon", "temperature",
                "family_epsilon", "family_temperature", "min_family_fraction"):
        assert getattr(selector, key) == observed[key], key
    config = selector.trainer_config(op)
    assert config == observed["operator_trainer_configs"][op]
    past = [b for b in blocks[:index] if b["operator"] == op]
    history = data["ground_truth"]["mode_history"]
    return {"block": index + 1, "start_step": start, "end_step": end, "operator": op,
            "trainer_config": {**config, "preference_relabeling_rate": selector.relabel_rate(op)},
            "score_before_selection": selector.scores[op],
            "previous_observations": [{"end_step": b["end_step"], "progress_score": b["progress_score"],
                                       "mode_at_end": mode_at(history, b["end_step"])} for b in past],
            "last_feedback_age_steps": start - past[-1]["end_step"] if past else None,
            "adaptive_on_this_prefix": selection_distribution(selector),
            "blind_on_this_prefix": selection_distribution(prefix_selector(blocks, index, blind=True)),
            "mode_at_start": mode_at(history, start),
            "mode_at_end": mode_at(history, end),
            "switches_inside_block": [h for h in history if start <= h[0] < end],
            "post_block_progress_score": block["progress_score"],
            "post_block_objective_score_terms": (np.tanh(block["normalized_vector_gain"]) / 2).tolist(),
            "half_mean_rewards": block["half_mean_rewards"],
            "scale_before_block": block["scale_before_block"]}


def diagnose_pair(adaptive, blind, metrics):
    a = adaptive["adaptive_operator_final"]["block_history"]
    b = blind["adaptive_operator_final"]["block_history"]
    assert len(a) == len(b) == 30
    differences = [i for i, (x, y) in enumerate(zip(a, b)) if x["operator"] != y["operator"]]
    first = differences[0] if differences else None
    start = a[first]["end_step"] - a[first]["steps"] + 1 if first is not None else 50001
    prefix_a = [r for r in adaptive["episode_rewards"] if r["step"] < start]
    prefix_b = [r for r in blind["episode_rewards"] if r["step"] < start]
    report = {"different_blocks": [i + 1 for i in differences],
              "first_difference_step": start if first is not None else None,
              "preceding_complete_blocks_identical": a[:first] == b[:first] if first is not None else a == b,
              "preceding_episode_records_identical": prefix_a == prefix_b,
              "preceding_episode_records": len(prefix_a)}
    indices = sorted(set(range(25, 30)) | ({first} if first is not None else set()))
    for condition, data in (("adaptive", adaptive), ("blind", blind)):
        m = metrics[condition]
        history = data["ground_truth"]["mode_history"]
        report[condition] = {"mode_history": history,
            "inspected_blocks": [block_diagnostic(data, i) for i in indices],
            "last_six_episodes": data["episode_rewards"][-6:],
            "late_snapshot_hv": m["late_snapshot_hv"],
            "final_mode_hv": m["final_mode_hv"],
            "final_minus_47500_mode_hv": [curve[-1]["hv"] - curve[-2]["hv"]
                                           for curve in m["snapshot_mode_curves"]],
            "focal_operator_uses": [{"operator": block["operator"], "end_step": block["end_step"],
                                     "progress_score": block["progress_score"],
                                     "mode_at_end": mode_at(history, block["end_step"])}
                                    for block in data["adaptive_operator_final"]["block_history"]
                                    if block["operator"] in FOCAL_OPERATORS]}
    return report


def diagnose(root, reference_root, summary_path):
    summary = json.loads(summary_path.read_text())
    assert summary["protocol"] == experiment.PROTOCOL
    assert summary["seeds"] == list(experiment.SEEDS)
    assert summary["validated_new_runs"] == 5 and summary["reused_validated_runs"] == 15
    pairs = {}
    for seed in experiment.SEEDS:
        adaptive = json.loads((reference_root / f"{experiment.previous.METHODS['legacy']}_seed{seed}.json").read_text())
        blind = json.loads((root / f"{experiment.METHOD}_seed{seed}.json").read_text())
        assert adaptive["seed"] == blind["seed"] == seed
        metrics = {c: summary["conditions"][f"legacy_{c}"]["per_seed"][str(seed)] for c in ("adaptive", "blind")}
        pairs[seed] = diagnose_pair(adaptive, blind, metrics)
    return {"protocol": "progress_terminal_block_readout_v1", "seeds": list(experiment.SEEDS),
            "source_summary": str(summary_path), "per_seed": pairs,
            "new_training_interactions": 0, "new_evaluation_interactions": 0, "extra_updates": 0,
            "limits": "Prefix probabilities reuse the live selector but do not recover the historical random "
                      "draw or simulate counterfactual training. Same-prefix blind/adaptive probabilities "
                      "are not future returns. Ground-truth modes are offline annotations only. Episode "
                      "and block agreement does not verify every network/optimizer state. No per-update "
                      "losses, gradients or 48500-step resumable state were saved by this training path. "
                      "Snapshot HV uses each run's final encoder; operators change multiple components."}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=experiment.OUTPUT_ROOT)
    parser.add_argument("--reference-root", type=Path, default=experiment.previous.OUTPUT_ROOT)
    parser.add_argument("--summary", type=Path, default=SUMMARY)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = diagnose(args.root, args.reference_root, args.summary)
    args.output.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    print(json.dumps({"pairs": len(report["per_seed"]), "new_interactions": 0,
                      "first_difference_steps": {s: r["first_difference_step"] for s, r in report["per_seed"].items()}}))
