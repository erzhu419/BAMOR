"""Stored-return alignment and identifiability audit; no simulator or model calls."""

import argparse
import json
import math
from pathlib import Path


SOURCE = Path("results/iclr_history_return_gate_20261003")
PROTOCOL = "stored_history_gate_alignment_v1"
SOURCE_PROTOCOL = "historical_state_whole_policy_return_gate_v1"
ENVIRONMENTS = ("mo-halfcheetah-v5", "mo-hopper-v5")
SEEDS = (2901, 2902, 2903)


def specification():
    return dict(source_protocol=SOURCE_PROTOCOL, environments=list(ENVIRONMENTS), seeds=list(SEEDS),
        alignment="initial_mode_and_exact_sampled_preference_against_stored_paired_reset_mean_returns; "
                  "natural_trial_mode_schedule_remains_unlocked",
        design="bipartite_observed_state_preference_incidence; identical_shared_cells_deduplicated; "
               "rank_equals_vertices_minus_connected_components_for_additive_state_preference_effects",
        outputs="trial_and_matched_held_signs; full_mode_mean_change; endpoint_changes; episode_age_and_actual_lengths; "
                "physical_state_preference_support_and_residual_degrees_of_freedom",
        use="post_choice_diagnostic_only; no_new_selector_threshold_tuning_or_confidence_interval",
        new_interactions=0, new_updates=0)


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2) + "\n")


def scalar_margin(weight, old, new):
    assert len(weight) == len(old) == len(new)
    return math.fsum(w * (b - a) for w, a, b in zip(weight, old, new))


def sign(value):
    return (value > 0) - (value < 0)


def design(rows):
    cells = {}
    duplicates = 0
    for row in rows:
        key = (row["capture_step"], row["weight_index"])
        if key in cells:
            old = cells[key]
            assert old["episode_age"] == row["episode_age"]
            assert old["weight"] == row["weight"] and old["branches"] == row["branches"]
            duplicates += 1
        else:
            cells[key] = row
    adjacency = {}
    for state, preference in cells:
        a, b = ("state", state), ("preference", preference)
        adjacency.setdefault(a, set()).add(b)
        adjacency.setdefault(b, set()).add(a)
    remaining, components = set(adjacency), 0
    while remaining:
        components += 1
        stack = [remaining.pop()]
        while stack:
            for node in adjacency[stack.pop()] & remaining:
                remaining.remove(node)
                stack.append(node)
    states = sorted({k[0] for k in cells})
    preferences = sorted({k[1] for k in cells})
    rank = len(adjacency) - components
    return dict(states=states, preferences=preferences, observations=len(rows), unique_cells=len(cells),
        identical_shared_duplicates=duplicates, connected_components=components, additive_rank=rank,
        additive_residual_df=len(cells) - rank,
        missing_crossed_cells=len(states) * len(preferences) - len(cells),
        independent_same_cell_repeats=0)


def matched_margin(evaluations, mode, row):
    points = [evaluations[name]["pareto_per_mode"][mode][row["weight_index"]]
              for name in ("prefix", "proposal")]
    assert all(point["weight"] == row["weight"] for point in points)
    return scalar_margin(row["weight"], points[0]["return"], points[1]["return"])


def audit_record(record):
    assert record["protocol"] == SOURCE_PROTOCOL and record["complete"] and record["controls_passed"]
    evaluations, coverage = record["evaluations"], record["coverage"]
    old, new = [evaluations[name]["endpoint"] for name in ("prefix", "proposal")]
    assert evaluations["prefix"]["evaluation"] == evaluations["proposal"]["evaluation"]
    modes = dict(zip(coverage["pool_steps"], coverage["observed_mode_ids"]))
    endpoints = dict(utility=new["utility"] - old["utility"], hv=new["hv"] - old["hv"],
        per_mode_utility=[b - a for a, b in zip(old["per_mode_utility"], new["per_mode_utility"])])
    arms = {}
    for arm, trial in record["trials"].items():
        rows = []
        for index, row in enumerate(trial["rows"]):
            mode = modes[row["capture_step"]]
            a, b = [row["branches"][name] for name in ("prefix", "proposal")]
            margin = scalar_margin(row["weight"], a["return_vector"], b["return_vector"])
            assert math.isclose(margin, trial["decision"]["margins"][index], rel_tol=1e-10, abs_tol=1e-7)
            held = matched_margin(evaluations, mode, row)
            rows.append(dict(capture_step=row["capture_step"], episode_age=row["episode_age"],
                initial_mode_id=mode, weight_index=row["weight_index"], weight=row["weight"],
                trial_margin=margin, matched_held_margin=held,
                held_mode_mean_margin=endpoints["per_mode_utility"][mode],
                sign_agreement=sign(margin) == sign(held),
                negative_trial_but_positive_matched_held=margin < 0 < held,
                positive_trial_but_negative_matched_held=held < 0 < margin,
                steps={"prefix": a["steps"], "proposal": b["steps"]},
                terminated={"prefix": a["terminated"], "proposal": b["terminated"]},
                truncated={"prefix": a["truncated"], "proposal": b["truncated"]}))
        mean = math.fsum(r["trial_margin"] for r in rows) / len(rows)
        assert math.isclose(mean, trial["decision"]["score"], rel_tol=1e-10, abs_tol=1e-7)
        assert record["choices"][arm] == ("proposal" if mean > 0 else "prefix")
        arms[arm] = dict(chosen=record["choices"][arm], trial_mean_margin=mean, rows=rows,
            sign_agreements=sum(r["sign_agreement"] for r in rows),
            negative_trial_positive_held=sum(r["negative_trial_but_positive_matched_held"] for r in rows),
            positive_trial_negative_held=sum(r["positive_trial_but_negative_matched_held"] for r in rows),
            design=design(trial["rows"]))
    combined = design([row for t in record["trials"].values() for row in t["rows"]])
    return dict(env=record["env"], seed=record["seed"], arms=arms, combined_design=combined,
        endpoint_proposal_minus_prefix=endpoints,
        missing_initial_modes_in_pool=sorted(set(range(len(old["per_mode_utility"]))) - set(modes.values())))


def run(source=SOURCE, output=None):
    source = Path(source)
    output = Path(output) if output else source / "alignment"
    frozen = json.loads((output / "protocol.json").read_text())
    assert frozen == dict(protocol=PROTOCOL, experiment=specification())
    prior = json.loads((source / "compact_readout.json").read_text())
    assert prior["protocol"] == SOURCE_PROTOCOL and prior["complete"]
    assert prior["experiment"]["seeds"] == list(SEEDS)
    results = []
    for env in ENVIRONMENTS:
        for seed in SEEDS:
            record = json.loads((source / env.replace("-", "_") / f"seed{seed}.json").read_text())
            assert (record["env"], record["seed"]) == (env, seed)
            results.append(audit_record(record))
    summary = {}
    for arm in ("history", "current"):
        values = [r["arms"][arm] for r in results]
        summary[arm] = {k: sum(v[k] for v in values) for k in (
            "sign_agreements", "negative_trial_positive_held", "positive_trial_negative_held")}
        summary[arm]["pairs"] = sum(len(v["rows"]) for v in values)
    result = dict(protocol=PROTOCOL, experiment=specification(), complete=True,
        source=str(source), results=results, alignment=summary,
        combined_designs_have_no_residual_df=all(r["combined_design"]["additive_residual_df"] == 0 for r in results),
        new_interactions=0, new_updates=0,
        decision="crossed_independent_repeats_required_before_stratified_acceptance")
    write(output / "compact_readout.json", result)
    print("DONE", PROTOCOL, result["decision"], json.dumps(summary), flush=True)
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, default=SOURCE)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    run(args.source, args.output)
