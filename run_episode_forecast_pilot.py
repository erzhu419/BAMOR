"""Fresh matched-budget online ablation of episode forecast credit."""

import argparse
from collections import Counter
import json
from pathlib import Path

import numpy as np

import run_episode_allocation_pilot as previous
from episode_credit_blend import EpisodeCreditBlend


PROTOCOL = "episode_forecast_online_pilot_v1"
ROOT = Path("results/iclr_episode_forecast_online_20261001")
SEEDS = tuple(range(2201, 2206))
ENVIRONMENTS = previous.ENVIRONMENTS
METHODS = {name: f"bamor_common_target_episode_forecast_{name}"
           for name in ("fixed", "ema", "equal_mix", "past_mix")}
CREDIT_RULES = dict(fixed="ema", ema="ema", equal_mix="equal_mixture", past_mix="blend")
CONTRASTS = (("past_mix", "fixed"), ("ema", "fixed"), ("equal_mix", "fixed"),
             ("past_mix", "ema"), ("equal_mix", "ema"), ("past_mix", "equal_mix"))
training, controls = previous.training, previous.controls
COMMON_KWARGS, LEARNING_CONTRACT = previous.COMMON_KWARGS, previous.LEARNING_CONTRACT
SELECTOR_CONTRACT = dict(previous.SELECTOR_CONTRACT, score_signal="causal_episode_forecast_advantage",
                         baseline_initial_episodes=3, trend_window=8, blend_rule="clipped_past_pair_squared_error")
write_report = previous.write_report


def validate_forecasts(allocation, method):
    assert {key: allocation[key] for key in SELECTOR_CONTRACT} == SELECTOR_CONTRACT
    assert allocation["selection_uses_scores"] == (method != "fixed")
    assert allocation["credit_rule"] == CREDIT_RULES[method]
    forecaster = EpisodeCreditBlend(allocation["baseline_rate"])
    scores = dict.fromkeys(allocation["operators"], 0.)
    segment = dict.fromkeys(allocation["operators"], 0)
    for record in allocation["episode_history"]:
        before = forecaster.predict(record["start_step"])
        assert record["forecast_before"] == before
        weight = (before["trend_weight"] if method == "past_mix" else .5 if method == "equal_mix" else 0.)
        assert record["credit_trend_weight"] == weight and record["credit_rule"] == CREDIT_RULES[method]
        ready = before["scale"] is not None and before["trend_prediction"] is not None
        center = ((1 - weight) * before["ema_prediction"] + weight * before["trend_prediction"]) if ready else None
        assert (record["baseline_before"], record["scale_before"]) == (center, before["scale"])
        phase = ("warmup" if record["start_step"] <= allocation["learning_starts"] else
                 "forced" if sum(segment.values()) < len(segment) else "free")
        assert record["selection_phase"] == phase
        assert record["mixed_epoch"] == (record["epoch_start"] != record["epoch_end"])
        expected_credit = ready and record["start_step"] > allocation["learning_starts"] and not record["mixed_epoch"]
        assert record["credited"] == expected_credit
        if record["mixed_epoch"]:
            scores = dict.fromkeys(scores, 0.)
            segment = dict.fromkeys(segment, 0)
        elif expected_credit:
            score = float(np.tanh((record["utility"] - center) / before["scale"]))
            assert np.isclose(score, record["score"], rtol=0, atol=1e-12)
            operator = record["operator"]
            scores[operator] = (1 - allocation["score_rate"]) * scores[operator] + allocation["score_rate"] * score
            segment[operator] += 1
        assert record["score_after"] == scores[record["operator"]]
        forecaster.observe(record["start_step"], record["utility"], before, record["mixed_epoch"])
    last_end = allocation["episode_history"][-1]["end_step"] if allocation["episode_history"] else 0
    if allocation["reset_history"] and allocation["reset_history"][-1]["step"] > last_end:
        forecaster.reset()
        scores = dict.fromkeys(scores, 0.)
        segment = dict.fromkeys(segment, 0)
    assert allocation["scores"] == scores and allocation["segment_counts"] == segment
    assert allocation["forecast_evidence"] == dict(cross_product=forecaster.cross_product,
        disagreement_energy=forecaster.disagreement_energy, comparisons=forecaster.comparisons,
        scale_observations=forecaster.baseline.scale.count)


def validate(data, env, method, seed, model):
    report = previous.previous.validate(data, env, METHODS[method], seed, model)
    work = data["learning_work"]
    updates = previous.previous.TOTAL_STEPS - COMMON_KWARGS["learning_starts"] + 1
    assert all(work[key] == updates for key in (
        "critic_optimizer_steps", "actor_optimizer_steps", "encoder_optimizer_steps"))
    assert all(work[key] == updates * COMMON_KWARGS["batch_size"] for key in (
        "critic_rows", "target_candidate_rows", "actor_candidate_rows"))
    assert data["common_target_learning"] == LEARNING_CONTRACT
    assert data["critic_target_rule"] == "scalar_projection_of_soft_vector"
    assert data["critic_candidate_context_rule"] == "own_preference"
    assert data["bootstrap_terminal_rule"] == "terminated_only" and not data["reward_normalization"]
    allocation = data["adaptive_operator_final"]
    validate_forecasts(allocation, method)
    history = allocation["episode_history"]
    assert allocation["epoch"] == len(allocation["reset_history"])
    assert len(history) == len(data["episode_rewards"])
    for record, episode in zip(history, data["episode_rewards"]):
        assert (record["end_step"], record["length"], record["return"]) == (
            episode["step"], episode["length"], episode["return"])
        assert record["start_step"] == record["end_step"] - record["length"] + 1
        assert record["utility"] == float(np.mean(record["return"]))
    counted = Counter(record["operator"] for record in history if record["credited"])
    assert allocation["counts"] == {op: counted[op] for op in allocation["operators"]}
    assert sum(r["length"] for r in history) + allocation["partial_episode_length"] == previous.previous.TOTAL_STEPS
    assert sum(allocation["step_counts"].values()) == previous.previous.TOTAL_STEPS
    config = dict(preference_relabeling_strategy="uniform", critic_relabel_copies=1,
        envelope_target_candidates=1, envelope_policy_candidates=1, vector_critic_coef=0.0, behavior_clone_coef=0.0)
    assert allocation["operator_trainer_configs"] == {op: config for op in allocation["operators"]}
    if method == "fixed":
        assert allocation["last_probs"] == dict.fromkeys(allocation["operators"], 1 / 3)
    prefix = [{key: record[key] for key in ("start_step", "end_step", "length", "operator", "return")}
              for record in history if allocation["first_free_start_step"] is None
              or record["start_step"] < allocation["first_free_start_step"]]
    report.update(protocol=PROTOCOL, method=method, training_method=METHODS[method],
        learning_work=work, common_target_learning=data["common_target_learning"], allocation=allocation,
        paired_prefix=prefix, first_free_start_step=allocation["first_free_start_step"])
    return report


def run(env, method, seed, root=ROOT):
    assert env in ENVIRONMENTS and method in METHODS and seed in SEEDS
    folder = Path(root) / env.replace("-", "_") / method
    folder.mkdir(parents=True, exist_ok=True)
    output = folder / f"seed{seed}.json"
    raw = folder / f"{METHODS[method]}_seed{seed}.json"
    model = folder / f"{METHODS[method]}_seed{seed}_model.pt"
    if output.exists():
        saved = json.loads(output.read_text())
        assert (saved["protocol"], saved["env"], saved["method"], saved["seed"]) == (PROTOCOL, env, method, seed)
        if saved.get("complete"):
            return saved
        if not raw.exists():
            raise RuntimeError("Incomplete attempt preserved; inspect before a new-root retry")
    else:
        write_report(output, dict(protocol=PROTOCOL, env=env, method=method, seed=seed,
                                 complete=False, planned_budget_upper_bound=controls.budget(env)))
    if not raw.exists():
        training.train_method(env, METHODS[method], previous.previous.TOTAL_STEPS, 25000, seed,
            str(folder), archive_steps=(previous.previous.TOTAL_STEPS,), **COMMON_KWARGS)
    report = validate(json.loads(raw.read_text()), env, method, seed, model)
    write_report(output, report)
    print(f"DONE {PROTOCOL} {env} {method} seed={seed} "
          f"utility={report['metrics']['direct_utility']['mean']:.6f}", flush=True)
    return report


def allocation_profile(allocation):
    credited = [record for record in allocation["episode_history"] if record["credited"]]
    return dict(counts=allocation["counts"], step_counts=allocation["step_counts"],
        reset_count=len(allocation["reset_history"]), completed_episodes=len(allocation["episode_history"]),
        credited_episodes=len(credited), partial_episode_length=allocation["partial_episode_length"],
        credited_mean_length=float(np.mean([row["length"] for row in credited])) if credited else None,
        credited_mean_trend_weight=float(np.mean([row["credit_trend_weight"] for row in credited])) if credited else None,
        credited_positive_score_fraction=float(np.mean([row["score"] > 0 for row in credited])) if credited else None,
        statistical_baseline_fits=allocation["statistical_baseline_fits"])


def summarize(root=ROOT):
    result = dict(protocol=PROTOCOL, seeds=list(SEEDS), complete=True,
        primary="past_mix minus same-substrate fixed direct utility, separately by environment",
        adaptation_contrast="past_mix minus equal_mix; separate from allocation benefit",
        uncertainty="20000 paired-seed percentile bootstrap resamples, RNG0; descriptive95% intervals",
        environments={}, completed_budget=dict.fromkeys(controls.budget(ENVIRONMENTS[0]), 0))
    indices = np.random.default_rng(0).integers(0, len(SEEDS), (20000, len(SEEDS)))
    for env in ENVIRONMENTS:
        spec = previous.previous.extension.ENVIRONMENTS[env]
        rows = []
        for seed in SEEDS:
            pair = {method: json.loads((Path(root) / env.replace("-", "_") / method / f"seed{seed}.json").read_text())
                    for method in METHODS}
            evaluation = dict(evaluation_weights=training.full_simplex_grid(spec["reward_dim"]).tolist(),
                episode_seeds_per_mode=[[100000 + seed * 1000 + mode * 10 + i for i in range(3)]
                                        for mode in range(spec["num_modes"])],
                ref_point=training.reference_point_for_env(env, spec["reward_dim"]).tolist())
            for method, record in pair.items():
                assert (record["protocol"], record["env"], record["method"], record["seed"], record["complete"]) == (
                    PROTOCOL, env, method, seed, True)
                assert record["training_method"] == METHODS[method]
                assert record["evaluation"] == evaluation and record["mode_history"] == pair["fixed"]["mode_history"]
                assert record["common_target_learning"] == LEARNING_CONTRACT
                assert record["learning_work"] == pair["fixed"]["learning_work"]
                assert record["paired_prefix"] == pair["fixed"]["paired_prefix"]
                assert record["first_free_start_step"] == pair["fixed"]["first_free_start_step"]
                validate_forecasts(record["allocation"], method)
                for key, upper in controls.budget(env).items():
                    actual = record["completed_budget"][key]
                    assert (spec["num_modes"] * 21 * 3 <= actual <= upper
                            if key == "evaluation_interactions" else actual == upper)
                    result["completed_budget"][key] += actual
            rows.append(dict(seed=seed, **{method: dict(metrics=record["metrics"],
                completed_budget=record["completed_budget"], allocation_profile=allocation_profile(record["allocation"]),
                learning_work=record["learning_work"]) for method, record in pair.items()}))
        values = lambda method, metric: np.asarray([row[method]["metrics"][metric]["mean"]
            if metric == "direct_utility" else row[method]["metrics"][metric] for row in rows])
        contrasts = {}
        for a, b in CONTRASTS:
            contrast = {}
            for metric in ("direct_utility", "final_hv"):
                av, bv = values(a, metric), values(b, metric)
                assert np.isfinite(av).all() and np.isfinite(bv).all()
                contrast[metric] = dict(previous.previous.paired(av.tolist(), bv.tolist()),
                    paired_mean_ci95=np.quantile((av - bv)[indices].mean(axis=1), [.025, .975]).tolist())
            contrast["per_mode_utility_mean_delta"] = np.mean([
                np.asarray(row[a]["metrics"]["direct_utility"]["per_mode"]) -
                np.asarray(row[b]["metrics"]["direct_utility"]["per_mode"]) for row in rows], axis=0).tolist()
            contrasts[f"{a}_minus_{b}"] = contrast
        result["environments"][env] = dict(runs=rows, paired=contrasts,
            methods={method: dict(direct_utility_mean=float(values(method, "direct_utility").mean()),
                minimum_direct_utility=float(values(method, "direct_utility").min()),
                final_hv_mean=float(values(method, "final_hv").mean()),
                per_mode_utility_mean=np.mean([row[method]["metrics"]["direct_utility"]["per_mode"]
                                              for row in rows], axis=0).tolist(),
                training_seconds_mean=float(np.mean([row[method]["metrics"]["training_seconds"] for row in rows])))
                for method in METHODS})
    result["allocation_gate_passed"] = all(env["paired"]["past_mix_minus_fixed"]["direct_utility"]["mean_delta"] > 0
        for env in result["environments"].values())
    result["hopper_hv_guard_passed"] = (
        result["environments"]["mo-hopper-v5"]["paired"]["past_mix_minus_fixed"]["final_hv"]["mean_delta"] >= 0)
    result["weight_adaptation_gate_passed"] = all(
        env["paired"]["past_mix_minus_equal_mix"]["direct_utility"]["mean_delta"] > 0
        for env in result["environments"].values())
    result["decision"] = ("pilot_allocation_supported_confirmation_unproven"
        if result["allocation_gate_passed"] and result["hopper_hv_guard_passed"] else "hold_online_forecast_allocation")
    result["adaptive_weight_claim_supported"] = (
        result["allocation_gate_passed"] and result["hopper_hv_guard_passed"] and result["weight_adaptation_gate_passed"])
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--env", choices=ENVIRONMENTS)
    parser.add_argument("--method", choices=METHODS)
    parser.add_argument("--seed", type=int, choices=SEEDS)
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--summarize", action="store_true")
    args = parser.parse_args()
    if args.summarize:
        report = summarize(args.root)
        write_report(args.root / "compact_readout.json", report)
        print(json.dumps(dict(decision=report["decision"], adaptive_weight_claim_supported=report["adaptive_weight_claim_supported"])))
    else:
        assert args.env and args.method and args.seed is not None
        run(args.env, args.method, args.seed, args.root)
