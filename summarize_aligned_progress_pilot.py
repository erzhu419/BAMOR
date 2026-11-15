"""Paired selector effects; final policies, late snapshots, and archives stay separate."""

import argparse
import json
from collections import Counter
from pathlib import Path
from statistics import mean

import numpy as np

import run_aligned_progress_pilot as pilot
from summarize_replay_context_refresh import metrics


def validate_selector(data, reward_dim=2):
    state = data["adaptive_operator_final"]
    expected = pilot.training.ProgressExtremeHierarchicalOperatorSelector(reward_dim).state_dict()
    for key in ("operators", "operator_trainer_configs", "min_family_fraction", "score_signal", "learning_starts",
                "warmup_sequence", "block_steps", "epsilon", "temperature", "family_epsilon", "family_temperature"):
        assert state[key] == expected[key], key
    blocks = state["block_history"]
    assert len(blocks) == 30
    assert [b["end_step"] for b in blocks] == list(range(6500, 50001, 1500))
    warmup = pilot.training.ProgressExtremeHierarchicalOperatorSelector(reward_dim).warmup_sequence
    assert [b["operator"] for b in blocks[:10]] == warmup
    counts = Counter(b["operator"] for b in blocks)
    assert state["counts"] == {op: counts[op] for op in state["operators"]}
    assert state["step_counts"] == {op: counts[op] * 1500 for op in state["operators"]}
    assert sum(state["family_step_counts"].values()) == 45000
    scores = dict.fromkeys(state["operators"], 0.)
    lr = pilot.training.ProgressExtremeHierarchicalOperatorSelector(reward_dim).lr
    for b in blocks:
        assert b["complete"] and b["steps"] == 1500 and b["half_counts"] == [750, 750]
        scale, rewards = np.asarray(b["scale_before_block"]), np.asarray(b["half_mean_rewards"])
        assert scale.shape == (reward_dim,) and np.all(scale >= 1e-6) and np.isfinite(scale).all()
        assert rewards.shape == (2, reward_dim)
        gain = (rewards[1] - rewards[0]) / scale
        np.testing.assert_allclose(gain, b["normalized_vector_gain"], rtol=1e-12, atol=1e-12)
        score = float(np.tanh(gain).mean())
        np.testing.assert_allclose(score, b["progress_score"], rtol=1e-12, atol=1e-12)
        op = b["operator"]
        scores[op] = (1 - lr) * scores[op] + lr * score
        np.testing.assert_allclose(scores[op], b["score_after"], rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(list(scores.values()), [state["scores"][op] for op in scores], rtol=1e-12, atol=1e-12)
    return {"operators": [b["operator"] for b in blocks], "scores": state["scores"],
            "family_steps": state["family_step_counts"]}


def validate_run(data, condition, seed, aligned):
    methods = pilot.METHODS if aligned else pilot.LEGACY_METHODS
    if aligned:
        assert data["critic_candidate_context_rule"] == "update_preference"
    else:
        # The six v25 files predate this log field; their named method uses candidate context.
        assert data.get("critic_candidate_context_rule", "candidate_preference") == "candidate_preference"
    assert data["critic_target_rule"] == "scalar_projection_of_soft_vector"
    assert data["bootstrap_terminal_rule"] == "terminated_only"
    assert not data.get("pareto_evals")
    assert "update_routing" not in data and "replay_context_refresh" not in data
    if aligned:
        assert data["aligned_progress_pilot"] == pilot.metadata(condition)
        assert data["training_interactions"] == 50000
        assert data["evaluation_interactions"] == 4 * 19 * 21 * 3 * 1000
        assert data["environment_initialization_probe_steps"] == 2
    result = metrics(data, seed, expected_method=methods[condition])
    result["late_snapshot_hv"] = result.pop("posthoc_late_snapshot_hv")
    result["late_mean_hv"] = result.pop("posthoc_late_mean_hv")
    result["late_min_hv"] = min(result["late_snapshot_hv"])
    result["selector"] = validate_selector(data)
    return result


def validate_pair(adaptive, blind):
    for key in ("evaluation_weights", "episode_seeds_per_mode", "ref_point", "policy_archive_snapshots"):
        assert adaptive[key] == blind[key], key
    assert adaptive["ground_truth"]["mode_history"] == blind["ground_truth"]["mode_history"]
    # Both algorithms are identical until the ten learning-only warmup blocks end.
    a, b = [r for r in adaptive["episode_rewards"] if r["step"] <= 20000], [r for r in blind["episode_rewards"] if r["step"] <= 20000]
    assert a and a == b, "Training differs before score-based selection is enabled"
    assert adaptive["adaptive_operator_final"]["block_history"][:10] == blind["adaptive_operator_final"]["block_history"][:10]
    x, y = adaptive["adaptive_operator_final"]["block_history"], blind["adaptive_operator_final"]["block_history"]
    return {"warmup_episodes_matched": len(a), "postwarmup_different_arms": sum(p["operator"] != q["operator"] for p, q in zip(x[10:], y[10:]))}


def aggregate(conditions):
    for condition in conditions.values():
        assert sorted(condition["per_seed"]) == list(pilot.SEEDS)
        for metric in ("final_hv", "archive_hv", "late_mean_hv", "late_min_hv"):
            for stat, fn in (("mean", mean), ("min", min)):
                condition[f"{metric}_{stat}"] = fn(r[metric] for r in condition["per_seed"].values())
    a, b = conditions["adaptive"], conditions["blind"]
    paired = {seed: {k: a["per_seed"][seed][k] - b["per_seed"][seed][k]
                     for k in ("final_hv", "archive_hv", "late_mean_hv", "late_min_hv")} for seed in pilot.SEEDS}
    checks = {"final_mean_improved": a["final_hv_mean"] > b["final_hv_mean"],
              "final_minimum_not_worse": a["final_hv_min"] >= b["final_hv_min"],
              "late_mean_not_worse": a["late_mean_hv_mean"] >= b["late_mean_hv_mean"]}
    return {"conditions": conditions, "paired_adaptive_minus_blind": paired,
            "final_wins": sum(p["final_hv"] > 0 for p in paired.values()),
            "development_checks": checks, "advance_to_fresh_seed_test": all(checks.values())}


def summarize(root, reference_root):
    results, validation, budget = {}, {}, {"training_interactions": 0, "evaluation_interactions": 0, "environment_initialization_probe_steps": 0}
    for backbone, methods, folder in (("aligned", pilot.METHODS, root), ("legacy", pilot.LEGACY_METHODS, reference_root)):
        conditions = {c: {"per_seed": {}} for c in methods}
        validation[backbone] = {}
        for seed in pilot.SEEDS:
            pair = {}
            for condition, method in methods.items():
                pair[condition] = data = json.loads((folder / f"{method}_seed{seed}.json").read_text())
                conditions[condition]["per_seed"][seed] = validate_run(data, condition, seed, backbone == "aligned")
                if backbone == "aligned":
                    for k in budget:
                        budget[k] += data[k]
            validation[backbone][seed] = validate_pair(pair["adaptive"], pair["blind"])
        results[backbone] = aggregate(conditions)
    for seed in pilot.SEEDS:
        assert results["aligned"]["conditions"]["adaptive"]["per_seed"][seed]["mode_history"] == results["legacy"]["conditions"]["adaptive"]["per_seed"][seed]["mode_history"]
    interaction = {seed: {k: results["aligned"]["paired_adaptive_minus_blind"][seed][k] - results["legacy"]["paired_adaptive_minus_blind"][seed][k]
                         for k in ("final_hv", "archive_hv", "late_mean_hv", "late_min_hv")} for seed in pilot.SEEDS}
    return {"protocol": pilot.PROTOCOL, "new_runs": 6, "reused_runs": 6, "validation": validation,
            "results": results, "selector_by_target_interaction": interaction,
            "completed_run_budget": budget, "late_snapshot_steps": list(pilot.LATE_STEPS),
            "scope": "Three reused development seeds. Final policy HV is primary; archive is secondary. "
                     "Late snapshots use final encoders. These are not independent confirmation or online recovery curves. "
                     "Completed-run budget excludes failed attempts; audit scheduler attempts before reporting total cost."}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=pilot.OUTPUT_ROOT)
    parser.add_argument("--reference-root", type=Path, default=pilot.REFERENCE_ROOT)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = summarize(args.root, args.reference_root)
    args.output.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    print(json.dumps(report["results"]["aligned"]["development_checks"]))
