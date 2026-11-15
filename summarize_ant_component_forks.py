"""Read-only grid and timing analysis of the frozen Ant component forks."""

import argparse
import json
from pathlib import Path

import numpy as np

import run_ant_component_forks as experiment


def measurement(row, weights):
    returns = np.asarray(row["returns"], dtype=float)
    assert returns.shape == (3, 21, 3) and np.isfinite(returns).all()
    utility = (returns * weights).sum(axis=-1)
    masks = {"mixed_energy": (weights[:, 2] > 0) & (weights[:, 2] < 1),
             "zero_energy": weights[:, 2] == 0, "pure_energy": weights[:, 2] == 1}
    assert [int(mask.sum()) for mask in masks.values()] == [14, 6, 1]
    groups = {name: utility[:, mask].mean(axis=1).tolist() for name, mask in masks.items()}
    np.testing.assert_allclose(groups["mixed_energy"][0], row["mode0_mixed_energy_utility"], rtol=0, atol=1e-9)
    np.testing.assert_allclose(utility.mean(axis=1), row["direct_utility_per_mode"], rtol=0, atol=1e-9)
    np.testing.assert_allclose(returns.mean(axis=1), row["objective_means_per_mode"], rtol=0, atol=1e-9)
    np.testing.assert_allclose(np.mean(row["hv_per_mode"]), row["hv_mean_mode"], rtol=0, atol=1e-6)
    return {k: row[k] for k in ("step", "mode0_mixed_energy_utility", "direct_utility_per_mode",
                               "hv_per_mode", "hv_mean_mode", "evaluation_interactions")} | {
        "all_mode_direct_utility": float(utility.mean()), "preference_group_utility_per_mode": groups}


def difference(left, right):
    return {key: left[key] - right[key] for key in
            ("mode0_mixed_energy_utility", "all_mode_direct_utility", "hv_mean_mode")} | {
        "direct_utility_per_mode": (np.asarray(left["direct_utility_per_mode"]) - right["direct_utility_per_mode"]).tolist(),
        "preference_group_utility_per_mode": {name: (np.asarray(values) -
            right["preference_group_utility_per_mode"][name]).tolist()
            for name, values in left["preference_group_utility_per_mode"].items()}}


def summarize(report):
    assert (report["protocol"], report["seed"], report["env"], report["window"], report["end_step"]) == (
        experiment.PROTOCOL, experiment.SEED, experiment.ENV, [experiment.CUT, experiment.BLOCK_END], experiment.END)
    assert report["complete"] and report["controls_passed"] and "error" not in report
    assert tuple(report["branches"]) == experiment.BRANCHES
    weights = np.asarray(report["evaluation"]["evaluation_weights"])
    np.testing.assert_array_equal(weights, experiment.training.full_simplex_grid(3))
    branches, eval_steps = {}, 0
    original = report["branches"]["per_label"]
    full = report["branches"]["shared"]
    for name, branch in report["branches"].items():
        assert branch["complete"] and branch["mode_history"] == original["mode_history"]
        expected = [experiment.CUT, experiment.BLOCK_END, experiment.END] if name == "per_label" else [experiment.BLOCK_END, experiment.END]
        assert [row["step"] for row in branch["measurements"]] == expected
        routing = branch["routing"]
        assert routing["extra_updates"] == routing["extra_environment_steps"] == 0
        if name in ("per_label", "shared"):
            assert set(branch["reproduction"]) == set(map(str, expected))
            for check in branch["reproduction"].values():
                assert all(check[k] for k in ("passed", "episodes_matched", "blocks_matched", "schedule_matched"))
            assert branch["reproduction"][str(experiment.END)]["final_returns_matched"]
        else:
            assert routing["rng_matched_steps"] == experiment.END - experiment.CUT
        if name != "per_label":
            assert routing["intervention_updates"] == experiment.BLOCK_END - experiment.CUT
            assert routing["generated_rows_each"] == 1500 * 256 * 3
            assert routing["effective_loss_rows_each"] == 1500 * 256 * (1 if name == "no_extra_copy_loss" else 3)
            assert [(b["end_step"], b["operator"]) for b in branch["block_history"]] == [
                (b["end_step"], b["operator"]) for b in full["block_history"]]
            assert len(branch["selector_draws"]) == 7
        if name == "shared":
            assert all(row["drawn"] == row["executed"] for row in branch["selector_draws"])
        measurements = []
        for row in branch["measurements"]:
            assert all(row[k] == v for k, v in report["evaluation"].items())
            assert 0 < row["evaluation_interactions"] <= 189000
            eval_steps += row["evaluation_interactions"]
            measurements.append(measurement(row, weights))
        block = next(b for b in branch["block_history"] if b["end_step"] == experiment.BLOCK_END)
        branches[name] = dict(measurements=measurements, routing=routing,
            block_progress=block["progress_score"], block_operator=block["operator"],
            native_selector_draw_mismatches=sum(row["drawn"] != row["executed"] for row in branch["selector_draws"]))
    budget = report["completed_budget"]
    assert budget["evaluation_interactions"] == eval_steps
    for key, planned in experiment.planned_budget().items():
        assert (budget[key] <= planned if key == "evaluation_interactions" else budget[key] == planned), key
    prefix = branches["per_label"]["measurements"][0]
    contrasts, trajectories = {}, {}
    for name, branch in branches.items():
        rows = branch["measurements"][-2:]
        trajectories[name] = dict(since_cut=[dict(step=row["step"], **difference(row, prefix)) for row in rows],
                                  after_block=difference(rows[1], rows[0]))
        if name != "per_label":
            for baseline in (("per_label",) if name == "shared" else ("shared", "per_label")):
                contrasts[f"{name}_minus_{baseline}"] = [dict(step=a["step"], **difference(a, b))
                    for a, b in zip(rows, branches[baseline]["measurements"][-2:])]
    return dict(protocol=report["protocol"], seed=report["seed"], complete=True, validation_passed=True,
                completed_budget=budget, branches=branches, contrasts=contrasts, trajectories=trajectories,
                scope=report["scope"])


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=experiment.ROOT)
    args = parser.parse_args()
    result = summarize(json.loads((args.root / f"seed{experiment.SEED}.json").read_text()))
    output = args.root / "readout.json"
    experiment.write_report(output, result)
    print(f"DONE Ant component readout {output} bytes={output.stat().st_size}")
    print(json.dumps(result["completed_budget"]))
