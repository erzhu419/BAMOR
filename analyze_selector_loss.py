"""Finite-horizon allocation-loss feasibility, without training or result pulls.

For the actual next-block distribution p and stored scores s, let
C = max(s) - p @ s. If conditional mean block progress mu satisfies
max(abs(mu - s)) <= eta, then max(mu) - p @ mu is at most
min(2*(1-min(p)), C + 2*eta).
Indeed max(mu) <= max(s) + eta and p @ mu >= p @ s - eta.
For mu in [-1, 1], the sharp distribution-only bound is 2*(1-min(p)):
put +1 on a least-probable arm and -1 on the others to attain it.
Summing this inequality compares one-step choices at the realized histories,
not alternative training trajectories or final-policy utilities.

The examples are constructed selector states, not experimental measurements.
"""

import json
import math

import numpy as np

from diagnose_progress_terminal_block import prefix_selector, selection_distribution, training


def loss_decomposition(selector):
    law = selection_distribution(selector)
    scores = np.array([selector.scores[op] for op in selector.operators])
    probabilities = np.array([law["operator_probabilities"][op] for op in selector.operators])
    score_loss = float(scores.max() - probabilities @ scores)
    range_bound = float(2 * (1 - probabilities.min()))
    terms = dict(warmup=0.0, family_routing=0.0,
                 within_softmax=0.0, within_exploration=0.0)
    if law["route"] == "warmup":
        terms["warmup"] = score_loss
    else:
        for family, names in selector.families.items():
            mass = law["family_probabilities"][family]
            if mass == 0:
                continue
            values = np.array([selector.scores[op] for op in names])
            conditional = np.array([law["operator_probabilities"][op] / mass for op in names])
            softmax = (conditional - selector.epsilon / len(names)) / (1 - selector.epsilon)
            terms["family_routing"] += float(mass * (scores.max() - values.max()))
            terms["within_softmax"] += float(
                mass * (1 - selector.epsilon) * (values.max() - softmax @ values))
            terms["within_exploration"] += float(
                mass * selector.epsilon * (values.max() - values.mean()))
    return {"route": law["route"], "score_loss": score_loss, "components": terms,
            "bounded_progress_worst_case": range_bound,
            "calibration_error_required_below": (range_bound - score_loss) / 2,
            "operator_probabilities": law["operator_probabilities"]}


def coarse_free_bound(selector):
    # Gibbs sampling loses at most temperature * log(K) in stored score.
    # For scores in [-1, 1], max(score) - max(top-two family mean) <= 1.
    families = len(selector.families)
    largest = max(map(len, selector.families.values()))
    return {"top_two_surrogate": 1.0,
            "family_softmax": (1 - selector.family_epsilon) * selector.family_temperature * math.log(families),
            "family_exploration": 2 * selector.family_epsilon,
            "within_softmax": (1 - selector.epsilon) * selector.temperature * math.log(largest),
            "within_exploration": 2 * selector.epsilon}


def example_selector(route):
    template = training.ProgressExtremeHierarchicalOperatorSelector(2)
    operators = template.warmup_sequence + template.operators
    if route == "warmup":
        operators = template.warmup_sequence[:4]
    elif route == "family_quota":
        operators += [template.families["recovery"][0]] * 30
    blocks = [{"complete": True, "steps": template.block_steps,
               "operator": op, "score_after": 0.0} for op in operators]
    return prefix_selector(blocks, len(blocks))


def examples():
    zero = example_selector("free_family_sampling")
    one_arm = example_selector("free_family_sampling")
    one_arm.scores[one_arm.families["extreme"][0]] = 0.1
    one_family = example_selector("free_family_sampling")
    for op in one_family.families["extreme"]:
        one_family.scores[op] = 0.1
    quota = example_selector("family_quota")
    quota.scores[quota.families["recovery"][0]] = 0.1
    warmup = example_selector("warmup")
    warmup.scores[warmup.warmup_sequence[0]] = 0.1
    return dict(zero_scores=zero, one_good_arm=one_arm, one_good_family=one_family,
                quota_routes_away=quota, warmup_routes_away=warmup)


def analyze():
    states = examples()
    coarse = coarse_free_bound(states["zero_scores"])
    reports = {name: loss_decomposition(selector) for name, selector in states.items()}
    p = np.array(list(reports["zero_scores"]["operator_probabilities"].values()))
    hypothetical_mean = np.full(len(p), -1.0)
    hypothetical_mean[p.argmin()] = 1.0
    return {
        "evidence_kind": "deterministic_selector_rule_analysis_not_training",
        "bounded_progress_maximum_loss_per_block": 2.0,
        "coarse_free_sampling_bound": sum(coarse.values()),
        "coarse_components": coarse,
        "constructed_states": reports,
        "zero_scores_bounded_mean_example": {
            "score_loss": 0.0, "assumed_uniform_error": 1.0,
            "conditional_mean_loss": float(hypothetical_mean.max() - p @ hypothetical_mean),
            "transfer_bound": float(2 * (1 - p.min()))},
        "conclusion": "The coarse score bound is vacuous. Exact score loss is computable, "
                      "but this transfer improves the distribution-only bound only with "
                      "sufficient calibration, not supplied by bounded feedback. "
                      "The range-only bound is linear in the horizon. "
                      "No final-utility guarantee follows.",
        "new_environment_interactions": 0,
    }


if __name__ == "__main__":
    print(json.dumps(analyze(), indent=2, allow_nan=False))
