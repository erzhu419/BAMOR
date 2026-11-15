"""Align existing Ant ablations with the vector-reselection counterexample.

Reads compact local records only. Reconstructed control values come from saved
per-seed contrasts and are checked against independent records where available.
The development and independent-seed cohorts are never pooled.
"""

import argparse
import json
from math import isclose, isfinite
from pathlib import Path
from statistics import mean, median


SEEDS = list(range(1201, 1206))
METRICS = ("mode0_mixed_energy_utility", "all_mode_direct_utility", "final_hv")
SOURCES = {
    "actor": "iclr_ant_ownpref_actor_final75_20260927/readout.json",
    "adaptive": "iclr_ant_ownpref_adaptive_final75_20260927/readout.json",
    "critic": "iclr_ant_ownpref_critic_final75_20260927/readout.json",
    "selector": "iclr_ant_selector_final75_20260927/readout.json",
    "diagnostic": "iclr_ant_mixed_critic_20260927/readout.json",
    "transfer": "iclr_ant_mixed_critic_20260927/preference_transfer.json",
    "independent": "iclr_ant_pairwise_final75_20260927/readout.json",
}


def paired(values, controls):
    assert len(values) == len(controls) > 0
    assert all(isfinite(x) for x in (*values, *controls))
    deltas = [x - y for x, y in zip(values, controls)]
    return {"mean_delta": mean(deltas), "median_delta": median(deltas),
            "wins": sum(x > 0 for x in deltas),
            "losses": sum(x < 0 for x in deltas),
            "ties": sum(x == 0 for x in deltas), "paired_deltas": deltas}


def agree(actual, expected):
    assert isclose(actual, expected, rel_tol=1e-10, abs_tol=1e-6), (actual, expected)


def summarize(results=Path("results")):
    data = {name: json.loads((results / path).read_text())
            for name, path in SOURCES.items()}
    for name in ("actor", "adaptive", "critic", "selector", "diagnostic"):
        assert data[name]["seeds"] == SEEDS, name
    for name in ("actor", "adaptive", "critic", "selector"):
        assert [row["seed"] for row in data[name]["runs"]] == SEEDS, name
    assert data["independent"]["seeds"] == list(range(1301, 1306))
    assert [row["seed"] for row in data["independent"]["runs"]] == list(range(1301, 1306))
    assert all(data["actor"]["operator_schedule_matched"].values())

    rows = []
    for i, seed in enumerate(SEEDS):
        conditions = {
            "actor_off_blind": data["actor"]["runs"][i]["metrics"],
            "actor_off_adaptive": data["adaptive"]["runs"][i]["metrics"],
            "both_off_blind": data["critic"]["runs"][i]["metrics"],
            "original_blind": {}, "original_adaptive": {}, "capql": {},
        }
        for metric in METRICS:
            a = conditions["actor_off_blind"][metric]
            b = conditions["actor_off_adaptive"][metric]
            c = conditions["both_off_blind"][metric]
            delta = lambda name, comparison: data[name]["comparisons"][comparison][metric]["paired_deltas"][i]
            conditions["original_blind"][metric] = a - delta("actor", "new_minus_fixed")
            conditions["original_adaptive"][metric] = b - delta("adaptive", "new_minus_adaptive")
            conditions["capql"][metric] = a - delta("actor", "new_minus_capql")
            agree(c - a, delta("critic", "new_minus_actor_only"))
            agree(b - a, delta("adaptive", "new_minus_fixed_actor"))
            agree(c - delta("critic", "new_minus_capql"), conditions["capql"][metric])
            agree(b - delta("adaptive", "new_minus_capql"), conditions["capql"][metric])
        reference = data["selector"]["runs"][i]
        for name, ref in (("original_blind", "v25_fixed"),
                          ("original_adaptive", "v25_adaptive"), ("capql", "capql")):
            assert reference[ref]["complete"]
            agree(conditions[name]["final_hv"], reference[ref]["metrics"]["final_hv"])
            agree(conditions[name]["all_mode_direct_utility"],
                  reference[ref]["metrics"]["direct_utility"]["mean"])
        primary = METRICS[0]
        agree(conditions["original_adaptive"][primary] - conditions["capql"][primary],
              data["transfer"]["per_seed_mode0"][str(seed)]["bamor_own_minus_capql"])
        rows.append({"seed": seed, "conditions": conditions})

    def contrasts(records, names):
        return {f"{a}_minus_{b}": {metric: paired(
            [r["conditions"][a][metric] for r in records],
            [r["conditions"][b][metric] for r in records]) for metric in METRICS}
            for a, b in names}

    names = (("actor_off_blind", "original_blind"),
             ("both_off_blind", "actor_off_blind"),
             ("actor_off_adaptive", "original_adaptive"),
             ("actor_off_adaptive", "actor_off_blind"),
             ("both_off_blind", "capql"))
    development = contrasts(rows, names)
    # These four cells cross actor search with selection, not actor with critic search.
    interaction = {}
    for metric in METRICS:
        adaptive_effect = development["actor_off_adaptive_minus_original_adaptive"][metric]["paired_deltas"]
        blind_effect = development["actor_off_blind_minus_original_blind"][metric]["paired_deltas"]
        interaction[metric] = paired(adaptive_effect, blind_effect)

    independent = [{"seed": r["seed"], "conditions": r["metrics"]}
                   for r in data["independent"]["runs"]]
    independent_comparisons = contrasts(independent, (
        ("base_adaptive", "capql"), ("base_blind", "capql"),
        ("base_adaptive", "base_blind"), ("pairwise_adaptive", "base_adaptive"),
        ("pairwise_adaptive", "capql")))
    return {
        "protocol": "ant_reselection_evidence_alignment_v1",
        "sources": SOURCES,
        "new_budget": {"training_interactions": 0, "evaluation_interactions": 0,
                       "learner_updates": 0, "remote_bytes_downloaded": 0},
        "development": {"seeds": SEEDS, "runs": rows, "comparisons": development,
                        "actor_rule_by_selector_interaction": interaction,
                        "mixed_utility_means": {name: mean(r["conditions"][name][METRICS[0]]
                                                          for r in rows)
                                                for name in rows[0]["conditions"]}},
        "independent_both_searches_off": {"seeds": list(range(1301, 1306)),
                                          "comparisons": independent_comparisons},
        "frozen_first_action_diagnostic": data["diagnostic"]["primary"],
        "interpretation": {
            "supported": "Candidate-search ablations affect development-cohort utility.",
            "unidentified": "The vector-reselection amplification mechanism is not measured.",
            "missing_ant_factorial_cell": "Actor search on, critic search off, blind selector.",
            "diagnostic_scope": "Final online critics and deterministic first-action returns; not training-target perturbations.",
            "decision": "Do not launch another training matrix or adopt a stability repair from these results.",
        },
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path, default=Path("results"))
    parser.add_argument("--output", type=Path,
                        default=Path("results/iclr_ant_reselection_alignment_20260927/readout.json"))
    args = parser.parse_args()
    result = summarize(args.results)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    print(f"DONE {result['protocol']}: {args.output}; no new interactions")
