"""Saved-data audit of native feedback, without fitting a new router."""

import argparse
import json
from pathlib import Path

import numpy as np

import diagnose_progress_terminal_block as replay
import run_scoped_arm_factorial as experiment


PROTOCOL = "scoped_arm_feedback_readout_v1"
CONDITIONS = ("legacy_adaptive", "envelope_only", "bc_only", "scoped_adaptive")
E, B = experiment.base.scoped.TARGETS
CONTRASTS = {
    "envelope_without_bc": ("envelope_only", "legacy_adaptive", E),
    "bc_without_envelope": ("bc_only", "legacy_adaptive", B),
    "envelope_with_bc": ("scoped_adaptive", "bc_only", E),
    "bc_with_envelope": ("scoped_adaptive", "envelope_only", B),
}
SIGNALS = ("terminal_score", "mean_target_progress", "first_target_progress")
METRICS = ("final_hv", "late_mean_hv", "archive_hv")


def total_variation(a, b):
    assert a.keys() == b.keys()
    return float(sum(abs(a[k] - b[k]) for k in a) / 2)


def native_feedback(blocks, history, dimension):
    records = []
    for i, block in enumerate(blocks[10:], 10):
        selector = replay.prefix_selector(blocks, i, reward_dim=dimension)
        adaptive = replay.selection_distribution(selector)
        blind = replay.selection_distribution(replay.prefix_selector(blocks, i, blind=True, reward_dim=dimension))
        start, end, op = block["end_step"] - block["steps"] + 1, block["end_step"], block["operator"]
        previous = [b for b in blocks[:i] if b["operator"] == op]
        probabilities = adaptive["operator_probabilities"]
        records.append(dict(start_step=start, end_step=end, operator=op,
            route=adaptive["route"], probabilities=probabilities,
            adaptive_blind_tv=total_variation(probabilities, blind["operator_probabilities"]),
            chosen_probability=probabilities[op],
            score_before=selector.scores[op], score_after=block["score_after"],
            progress_score=block["progress_score"],
            objective_score_terms=(np.tanh(block["normalized_vector_gain"]) / dimension).tolist(),
            last_feedback_age_steps=start - previous[-1]["end_step"] if previous else None,
            mode_at_start=replay.mode_at(history, start), mode_at_end=replay.mode_at(history, end),
            switches_inside_block=[h for h in history if start <= h[0] < end]))
    final_selector = replay.prefix_selector(blocks, len(blocks), reward_dim=dimension)
    arms = {}
    for op in (E, B):
        uses = [b for b in records if b["operator"] == op]
        assert uses, "Selected diagnostic cases must contain both target arms"
        arms[op] = dict(terminal_score=final_selector.scores[op],
            mean_target_progress=float(np.mean([b["progress_score"] for b in uses])),
            first_target_progress=uses[0]["progress_score"],
            target_blocks=len(uses), switch_blocks=sum(bool(b["switches_inside_block"]) for b in uses),
            last_observed_step=uses[-1]["end_step"],
            first_feedback_available_at=uses[0]["end_step"],
            last_progress=uses[-1]["progress_score"])
    return dict(blocks=records, arms=arms,
        extreme_family_score=final_selector._family_scores()["extreme"],
        quota_blocks=sum(b["route"] == "family_quota" for b in records),
        adaptive_blind_tv_mean=float(np.mean([b["adaptive_blind_tv"] for b in records])),
        adaptive_blind_tv_max=max(b["adaptive_blind_tv"] for b in records))


def case_comparison(feedback, metrics):
    first = feedback[CONDITIONS[0]]["blocks"]
    for condition in CONDITIONS[1:]:
        assert [(b["end_step"], b["operator"]) for b in feedback[condition]["blocks"]] == [
            (b["end_step"], b["operator"]) for b in first]
    contrasts = {}
    for name, (candidate, control, op) in CONTRASTS.items():
        a, b = feedback[candidate], feedback[control]
        deltas = {k: a["arms"][op][k] - b["arms"][op][k] for k in SIGNALS}
        outcomes = {k: metrics[candidate][k] - metrics[control][k] for k in METRICS}
        shifts = [total_variation(x["probabilities"], y["probabilities"])
                  for x, y in zip(a["blocks"], b["blocks"])]
        contrasts[name] = dict(candidate=candidate, control=control, operator=op,
            signal_deltas=deltas, outcome_deltas=outcomes,
            final_hv_relative_pct=100 * outcomes["final_hv"] / metrics[control]["final_hv"],
            direction_agreement={s: {m: None if deltas[s] == 0 or outcomes[m] == 0
                                    else bool(np.sign(deltas[s]) == np.sign(outcomes[m])) for m in METRICS}
                                 for s in SIGNALS},
            prefix_probability_tv_mean=float(np.mean(shifts)), prefix_probability_tv_max=max(shifts))
    interaction = {k: metrics["scoped_adaptive"][k] - metrics["envelope_only"][k]
                      - metrics["bc_only"][k] + metrics["legacy_adaptive"][k] for k in METRICS}
    interaction["extreme_family_score"] = (feedback["scoped_adaptive"]["extreme_family_score"]
        - feedback["envelope_only"]["extreme_family_score"] - feedback["bc_only"]["extreme_family_score"]
        + feedback["legacy_adaptive"]["extreme_family_score"])
    return dict(contrasts=contrasts, interaction=interaction)


def agreement_counts(cases):
    contrasts = [c for env in cases.values() for case in env.values() for c in case["contrasts"].values()]
    return {signal: {metric: dict(
                agree=sum(c["direction_agreement"][signal][metric] is True for c in contrasts),
                disagree=sum(c["direction_agreement"][signal][metric] is False for c in contrasts),
                ties=sum(c["direction_agreement"][signal][metric] is None for c in contrasts))
            for metric in METRICS} for signal in SIGNALS}


def diagnose(root=experiment.ROOT, reference_root=experiment.REFERENCE_ROOT):
    cases = {}
    for env_name, seeds in experiment.CASES.items():
        cases[env_name] = {}
        dimension = experiment.previous.ENVIRONMENTS[env_name]["reward_dim"]
        for seed, role in seeds.items():
            feedback, metrics, reference = {}, {}, None
            for condition in CONDITIONS:
                source = root if condition in experiment.TARGETS else reference_root
                folder = Path(source) / env_name.replace("-", "_") / condition
                report = json.loads((folder / f"seed{seed}.json").read_text())
                expected = experiment.PROTOCOL if condition in experiment.TARGETS else experiment.previous.PROTOCOL
                assert (report["protocol"], report["env"], report["condition"], report["seed"], report["complete"]) == (
                    expected, env_name, condition, seed, True)
                if reference is None:
                    reference = report
                experiment.validate_pair(report, reference)
                raw = json.loads((folder / f"{experiment.METHOD}_seed{seed}.json").read_text())
                assert (raw["env"], raw["seed"], raw["method"], raw["training_interactions"]) == (
                    env_name, seed, experiment.METHOD, 50000)
                # Recompute the native score and EMA before interpreting their direction.
                selector = experiment.base.common.validate_selector(raw, dimension)
                assert selector == report["selector"]
                blocks = raw["adaptive_operator_final"]["block_history"]
                history = raw["ground_truth"]["mode_history"]
                assert np.array_equal(history, report["metrics"]["mode_history"])
                assert blocks[:10] == report["warmup"]["blocks"]
                feedback[condition] = native_feedback(blocks, history, dimension)
                metrics[condition] = {k: report["metrics"][k] for k in METRICS}
            cases[env_name][seed] = dict(selected_role=role, feedback=feedback, metrics=metrics,
                                         **case_comparison(feedback, metrics))
    return dict(protocol=PROTOCOL, complete=True, cases=cases,
        descriptive_direction_counts=agreement_counts(cases),
        new_training_interactions=0, new_evaluation_interactions=0, extra_updates=0,
        scope="Four outcome-selected cases, 16 dependent conditional contrasts. Terminal native arm EMA is "
              "the primary feedback proxy; mean and first post-warmup target progress are fixed diagnostics. "
              "Counts are descriptive endpoint agreement, not online prediction accuracy or independent trials. "
              "Each choice distribution uses only completed prior blocks. Variant score differences require "
              "separate factual runs and are not observable counterfactual labels for a single online run. "
              "No future snapshots, evaluation returns, environment IDs or true modes enter the native feedback "
              "or probability replay. Modes only annotate diagnostics; prefix probability shifts do not replay "
              "historical random draws. No signal fitting, new router, training or adoption.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=experiment.ROOT)
    parser.add_argument("--reference-root", type=Path, default=experiment.REFERENCE_ROOT)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = diagnose(args.root, args.reference_root)
    args.output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    print(json.dumps(result["descriptive_direction_counts"]))
