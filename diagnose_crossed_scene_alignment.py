"""Decompose the frozen crossed gate using stored returns only."""

import argparse
import json
import math
from pathlib import Path

from diagnose_history_gate_alignment import scalar_margin, sign, write


SOURCE = Path("results/iclr_crossed_scene_return_20261003")
PROTOCOL = "stored_crossed_scene_alignment_v1"
SOURCE_PROTOCOL = "crossed_scene_mean_return_gate_v1"
ENVIRONMENTS = ("mo-halfcheetah-v5", "mo-hopper-v5")
SEEDS = (3001, 3002, 3003)


def specification():
    return dict(source_protocol=SOURCE_PROTOCOL, environments=list(ENVIRONMENTS), seeds=list(SEEDS),
        decomposition="trial_minus_uniform_full_grid = reset_measurement_gap + sampled_preference_gap + "
                      "selected_mode_multiplicity_coverage_gap",
        alignment="same_logged_mode_and_exact_preferences; held_returns_average_two_independent_evaluation_resets",
        coverage="post_choice_mode_labels; distinguish_not_captured_captured_not_selected_selected_not_detected",
        use="explanatory_only; held_sign_verdicts_are_not_deployable_gates_or_new_performance_estimates",
        new_interactions=0, new_updates=0)


def mean(values):
    return math.fsum(values) / len(values)


def accepts(values):
    return min(values) >= 0 and mean(values) > 0


def audit_record(record):
    assert record["protocol"] == SOURCE_PROTOCOL and record["complete"] and record["controls_passed"]
    evaluations = record["evaluations"]
    old, new = [evaluations[k] for k in ("prefix", "proposal")]
    assert old["evaluation"] == new["evaluation"]
    weights = old["evaluation"]["evaluation_weights"]
    selected = record["coverage"]["selected_mode_ids"]
    pool = record["coverage"]["observed_mode_ids"]
    assert len(selected) == 4
    fronts = [e["pareto_per_mode"] for e in (old, new)]
    assert len(fronts[0]) == len(fronts[1]) == 4
    held = []
    for mode in range(4):
        a, b = [front[mode] for front in fronts]
        assert len(a) == len(b) == len(weights)
        assert all(x["weight"] == y["weight"] == w for x, y, w in zip(a, b, weights))
        held.append([scalar_margin(w, x["return"], y["return"]) for w, x, y in zip(weights, a, b)])
    full = [mean(row) for row in held]
    utility_delta = new["endpoint"]["utility"] - old["endpoint"]["utility"]
    assert math.isclose(mean(full), utility_delta, rel_tol=1e-10, abs_tol=1e-7)
    assert all(math.isclose(v, b - a, rel_tol=1e-10, abs_tol=1e-7) for v, a, b in zip(
        full, old["endpoint"]["per_mode_utility"], new["endpoint"]["per_mode_utility"]))

    # Preserve the complete crossed design and compare the identical preference subset.
    cells = {}
    preference_indices = {}
    for row in record["trial"]["rows"]:
        cell = (row["scene"], row["preference"], row["repeat"])
        assert cell not in cells and row["weight"] == weights[row["weight_index"]]
        assert row["capture_step"] == record["coverage"]["selected_steps"][row["scene"]]
        assert preference_indices.setdefault(row["preference"], row["weight_index"]) == row["weight_index"]
        cells[cell] = scalar_margin(row["weight"], row["branches"]["prefix"]["return_vector"],
                                   row["branches"]["proposal"]["return_vector"])
    assert set(cells) == {(s, p, r) for s in range(4) for p in range(4) for r in range(2)}
    indices = [preference_indices[p] for p in range(4)]
    assert len(set(indices)) == 4
    sampled = [mean([row[i] for i in indices]) for row in held]
    scenes = []
    for scene, mode in enumerate(selected):
        trial = mean([cells[scene, p, r] for p in range(4) for r in range(2)])
        expected = record["trial"]["decisions"]["scene_means"][scene]
        assert math.isclose(trial, expected, rel_tol=1e-10, abs_tol=1e-7)
        scenes.append(dict(scene=scene, mode=mode, trial_mean=trial,
            trial_repeat_means=[mean([cells[scene, p, r] for p in range(4)]) for r in range(2)],
            held_sampled_mean=sampled[mode], held_full_mean=full[mode],
            reset_measurement_gap=trial - sampled[mode], sampled_preference_gap=sampled[mode] - full[mode],
            trial_sampled_sign_agreement=sign(trial) == sign(sampled[mode]),
            trial_full_sign_agreement=sign(trial) == sign(full[mode])))
    trial_means = [s["trial_mean"] for s in scenes]
    actual = accepts(trial_means)
    assert record["choices"]["scene"] == ("proposal" if actual else "prefix")
    assert math.isclose(mean(trial_means), record["trial"]["decisions"]["overall"], abs_tol=1e-7)
    decomposition = dict(reset_measurement_gap=mean([s["reset_measurement_gap"] for s in scenes]),
        sampled_preference_gap=mean([s["sampled_preference_gap"] for s in scenes]),
        mode_multiplicity_coverage_gap=mean([full[m] for m in selected]) - mean(full))
    gap = mean(trial_means) - mean(full)
    assert math.isclose(math.fsum(decomposition.values()), gap, rel_tol=1e-10, abs_tol=1e-7)
    decomposition.update(trial_overall=mean(trial_means), held_uniform_full_grid=mean(full), total_gap=gap)

    losses = []
    for mode, value in enumerate(full):
        if value >= 0:
            continue
        if mode not in pool:
            location = "not_captured"
        elif mode not in selected:
            location = "captured_not_selected"
        else:
            location = "selected_detected" if any(s["trial_mean"] < 0 for s in scenes if s["mode"] == mode) else "selected_not_detected"
        losses.append(dict(mode=mode, full_mean=value, sampled_mean=sampled[mode], location=location))
    return dict(env=record["env"], seed=record["seed"], actual_choice=record["choices"]["scene"],
        sampled_weight_indices=indices, selected_mode_ids=selected,
        missing_pool_modes=sorted(set(range(4)) - set(pool)),
        missing_selected_modes=sorted(set(range(4)) - set(selected)), scenes=scenes,
        held_sampled_mode_means=sampled, held_full_mode_means=full, loss_modes=losses,
        decomposition=decomposition,
        diagnostic_acceptance=dict(trial=actual,
            held_sampled_selected=accepts([sampled[m] for m in selected]),
            held_full_selected=accepts([full[m] for m in selected]),
            held_sampled_all_modes=accepts(sampled), held_full_all_modes=accepts(full)),
        endpoint_proposal_minus_prefix=dict(utility=utility_delta,
            hv=new["endpoint"]["hv"] - old["endpoint"]["hv"]),
        aggregate_gain_with_mode_loss=utility_delta > 0 and bool(losses))


def run(source=SOURCE, output=None):
    source = Path(source)
    output = Path(output) if output else source / "alignment"
    assert json.loads((output / "protocol.json").read_text()) == dict(protocol=PROTOCOL, experiment=specification())
    source_spec = json.loads((source / "protocol.json").read_text())
    assert source_spec["protocol"] == SOURCE_PROTOCOL
    results = []
    for env in ENVIRONMENTS:
        for seed in SEEDS:
            record = json.loads((source / env.replace("-", "_") / f"seed{seed}.json").read_text())
            assert (record["env"], record["seed"], record["experiment"]) == (env, seed, source_spec["experiment"])
            results.append(audit_record(record))
    scenes = [s for r in results for s in r["scenes"]]
    summary = dict(scene_count=len(scenes),
        trial_sampled_sign_agreements=sum(s["trial_sampled_sign_agreement"] for s in scenes),
        trial_full_sign_agreements=sum(s["trial_full_sign_agreement"] for s in scenes),
        positive_trial_negative_full=sum(s["trial_mean"] >= 0 > s["held_full_mean"] for s in scenes),
        negative_trial_nonnegative_full=sum(s["trial_mean"] < 0 <= s["held_full_mean"] for s in scenes),
        loss_locations={location: sum(l["location"] == location for r in results for l in r["loss_modes"])
                        for location in ("not_captured", "captured_not_selected", "selected_not_detected", "selected_detected")},
        aggregate_gain_with_mode_loss=sum(r["aggregate_gain_with_mode_loss"] for r in results),
        full_grid_all_mode_acceptances=sum(r["diagnostic_acceptance"]["held_full_all_modes"] for r in results))
    result = dict(protocol=PROTOCOL, experiment=specification(), complete=True, results=results, summary=summary,
        new_interactions=0, new_updates=0, decision="diagnostic_only_original_hold_unchanged")
    write(output / "compact_readout.json", result)
    print("DONE", PROTOCOL, json.dumps(summary), flush=True)
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, default=SOURCE)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    run(args.source, args.output)
