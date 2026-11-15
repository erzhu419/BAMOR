"""Read-only imitation errors on the factual trajectories used by the guard."""

import argparse
import json
from pathlib import Path

import numpy as np
import torch

import run_return_labelled_guard as source


PROTOCOL = "return_guard_stored_context_imitation_v1"
ROOT = source.ROOT / "imitation_diagnostic"
MODELS = ("prefix", "proposal", "scored", "blind", "priced")
TEACHERS = {"old": "prefix", "new": "proposal"}
BATCH_SIZE, REPRODUCTION_ATOL = 1024, 1e-5


def specification():
    return dict(source_root=str(source.ROOT), source_protocol=source.PROTOCOL,
        environments=list(source.ENVIRONMENTS), seeds=list(source.SEEDS), models=list(MODELS), teachers=TEACHERS,
        batch_size=BATCH_SIZE, reproduction_atol=REPRODUCTION_ATOL,
        metric="full_trajectory_mean_squared_deterministic_action_error; equal_preference_means",
        baseline="100k_proposal_before_guard; compare_final_scored_blind_priced_on_identical_rows",
        strata="all; selected_old; selected_new; scored_blind_disagreement; use_original_choices",
        controls="teacher_reproduces_every_saved_action; lengths_and_weights_match_source_pair_order",
        scope="stored_factual_teacher_contexts_only; no_live_context_reconstruction_or_return_guarantee",
        decision="retain_source_HOLD; diagnose_average_and_per_preference_drift_without_retuning",
        new_work="zero_training_calibration_evaluation_interactions_probes_fits_or_optimizer_steps; forward_only")


@torch.no_grad()
def action_error(policy, data):
    squared, maximum, elements = 0., 0., data["actions"].numel()
    for start in range(0, len(data["states"]), BATCH_SIZE):
        part = slice(start, start + BATCH_SIZE)
        actual = policy.get_action(data["states"][part], data["contexts"][part], deterministic=True)
        delta = actual.double() - data["actions"][part].double()
        squared += float(delta.square().sum())
        maximum = max(maximum, float(delta.abs().max()))
    mse = squared / elements
    assert np.isfinite(mse) and np.isfinite(maximum)
    return mse, maximum


def load_models(path, data, hidden):
    saved = torch.load(path, map_location="cpu", weights_only=True)
    dims = [data[k].shape[1] for k in ("states", "actions", "contexts")]
    models = {}
    for name in MODELS:
        policy = source.training.PolicyNet(*dims, hidden=hidden)
        policy.load_state_dict(saved[name]["policy"])
        models[name] = policy.eval().requires_grad_(False)
    return models


def diagnose_pairs(models, pairs, rows):
    assert set(models) == set(MODELS) and len(pairs) == len(rows)
    records, forward_rows, reproduction_error = [], 0, 0.
    for pair, meta in zip(pairs, rows):
        torch.testing.assert_close(pair["old"]["states"][0], pair["new"]["states"][0], rtol=0, atol=0)
        row = dict(weight_index=meta["weight_index"], weight=meta["weight"],
            scored=meta["scored"], blind=meta["blind"], return_margin=meta["utility_difference"],
            trajectory_rows={}, mse={}, max_action_error={})
        for teacher, data in pair.items():
            n = len(data["states"])
            assert n == meta["steps"][teacher] and n > 0
            assert len(data["contexts"]) == len(data["actions"]) == n
            w = torch.tensor(meta["weight"], dtype=data["contexts"].dtype)
            torch.testing.assert_close(data["contexts"][:, :len(w)], w.expand(n, -1), rtol=0, atol=1e-7)
            row["trajectory_rows"][teacher] = n
            row["mse"][teacher], row["max_action_error"][teacher] = {}, {}
            for name, policy in models.items():
                mse, maximum = action_error(policy, data)
                row["mse"][teacher][name] = mse
                row["max_action_error"][teacher][name] = maximum
                forward_rows += n
            error = row["max_action_error"][teacher][TEACHERS[teacher]]
            reproduction_error = max(reproduction_error, error)
            assert error <= REPRODUCTION_ATOL, f"Saved {teacher} teacher mismatch: {error}"
        records.append(row)
    return records, forward_rows, reproduction_error


def target_metrics(records, condition):
    groups = dict(all=records,
        selected_old=[r for r in records if r[condition] == "old"],
        selected_new=[r for r in records if r[condition] == "new"],
        disagreement=[r for r in records if r["scored"] != r["blind"]])
    result = {}
    for name, rows in groups.items():
        value = dict(preferences=len(rows))
        if rows:
            errors = {m: np.asarray([r["mse"][r[condition]][m] for r in rows]) for m in MODELS}
            value["mean_mse"] = {m: float(v.mean()) for m, v in errors.items()}
            value["delta_from_proposal"] = {}
            for model in ("scored", "blind", "priced"):
                delta = errors[model] - errors["proposal"]
                value["delta_from_proposal"][model] = dict(mean=float(delta.mean()),
                    per_preference=delta.tolist(), increased=int((delta > 0).sum()),
                    decreased=int((delta < 0).sum()), unchanged=int((delta == 0).sum()))
        result[name] = value
    return result


def run(root=ROOT):
    root = Path(root)
    frozen = json.loads((root / "protocol.json").read_text())
    assert (frozen["protocol"], frozen["experiment"]) == (PROTOCOL, specification())
    result = dict(protocol=PROTOCOL, experiment=specification(), complete=False, environments={},
        decision="hold_return_labelled_tangent_guard",
        actual_work=dict(training_interactions=0, calibration_interactions=0, evaluation_interactions=0,
            initialization_probes=0, actor_steps=0, critic_steps=0, encoder_steps=0,
            fits=0, policy_forward_rows=0), source_acquisition_work={})
    for env in source.ENVIRONMENTS:
        reports = []
        for seed in source.SEEDS:
            path = source.ROOT / env.replace("-", "_") / f"seed{seed}.json"
            cached = json.loads(path.read_text())
            source.check(cached)
            assert (cached["env"], cached["seed"]) == (env, seed)
            pairs = torch.load(path.with_suffix(".calibration.pt"), map_location="cpu", weights_only=True)
            models = load_models(path.with_suffix(".actors.pt"), pairs[0]["old"],
                cached["experiment"]["training_kwargs"]["hidden"])
            records, forwards, error = diagnose_pairs(models, pairs, cached["calibration"]["rows"])
            assert sum(sum(r["trajectory_rows"].values()) for r in records) == cached["calibration"]["interactions"]
            assert forwards == len(MODELS) * cached["calibration"]["interactions"]
            targets = {c: target_metrics(records, c) for c in ("scored", "blind")}
            report = dict(seed=seed, source_report=str(path), teacher_reproduction_max_error=error,
                targets=targets, preferences=records,
                source_endpoints={m: cached["evaluations"][m]["endpoint"] for m in MODELS})
            reports.append(report)
            result["actual_work"]["policy_forward_rows"] += forwards
            for key, value in cached["actual_work"].items():
                result["source_acquisition_work"][key] = result["source_acquisition_work"].get(key, 0) + value
            print(f"IMITATION env={env} seed={seed} scored_target_MSE="
                  f"{targets['scored']['all']['mean_mse']} teacher_error={error:.3g}", flush=True)
        mean_mse = {c: {m: float(np.mean([r["targets"][c]["all"]["mean_mse"][m] for r in reports]))
            for m in MODELS} for c in ("scored", "blind")}
        delta = mean_mse["scored"]["scored"] - mean_mse["scored"]["proposal"]
        increased = sum(r["targets"]["scored"]["all"]["delta_from_proposal"]["scored"]["increased"] for r in reports)
        finding = "average_stored_imitation_drift" if delta > 0 else (
            "mixed_preference_imitation_retention" if increased else "stored_imitation_not_worsened")
        result["environments"][env] = dict(per_seed=reports, mean_target_mse=mean_mse,
            scored_target_delta_from_proposal=delta, scored_target_preferences_increased=increased,
            finding=finding, teacher_reproduction_max_error=max(r["teacher_reproduction_max_error"] for r in reports))
        source.write(root / "compact_readout.json", result)
    result["complete"] = True
    source.write(root / "compact_readout.json", result)
    print("DONE", PROTOCOL, json.dumps(result["actual_work"]), flush=True)
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=ROOT)
    args = parser.parse_args()
    run(args.root)
