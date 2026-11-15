"""Fresh paired pilot of complete-episode, change-aware collection feedback."""

import argparse
import json
from collections import Counter
from pathlib import Path

import numpy as np

import run_projected_allocation_pilot as projected


PROTOCOL = "projected_episode_allocation_v1"
ROOT = Path("results/iclr_episode_allocation_20260930")
SEEDS = tuple(range(2101, 2106))
ENVIRONMENTS = projected.ENVIRONMENTS
METHODS = dict(adaptive="bamor_common_target_episode_adaptive",
               fixed="bamor_common_target_episode_fixed", v25=projected.METHODS["v25"])
training, previous, controls = projected.training, projected.previous, projected.controls
write_report = projected.write_report
COMMON_KWARGS = projected.COMMON_KWARGS
LEARNING_CONTRACT = projected.LEARNING_CONTRACT
SELECTOR_CONTRACT = dict(operators=["uniform", "stratified", "mixed"], epsilon=0.15,
    temperature=0.35, score_rate=0.13, baseline_rate=0.13, learning_starts=5000,
    score_signal="causal_episode_return_advantage", credit_unit="complete_single_sampler_episode",
    utility_rule="equal_weight_undiscounted_raw_episode_return",
    detector_config=dict(ema_alpha=1 / 1500, warmup=2000, surprise_threshold=18.0,
        switch_cooldown=2000, cusum_required_consecutive=5, cusum_decay=0.5))


def validate(data, env, method, seed, model):
    if method == "v25":
        report = projected.validate(data, env, method, seed, model)
        report["protocol"] = PROTOCOL
        return report
    report = previous.validate(data, env, METHODS[method], seed, model)
    work = data["learning_work"]
    updates = previous.TOTAL_STEPS - COMMON_KWARGS["learning_starts"] + 1
    assert all(work[key] == updates for key in (
        "critic_optimizer_steps", "actor_optimizer_steps", "encoder_optimizer_steps"))
    assert all(work[key] == updates * COMMON_KWARGS["batch_size"] for key in (
        "critic_rows", "target_candidate_rows", "actor_candidate_rows"))
    assert data["common_target_learning"] == LEARNING_CONTRACT
    assert data["critic_target_rule"] == "scalar_projection_of_soft_vector"
    assert data["critic_candidate_context_rule"] == "own_preference"
    assert data["bootstrap_terminal_rule"] == "terminated_only" and not data["reward_normalization"]
    allocation = data["adaptive_operator_final"]
    assert {key: allocation[key] for key in SELECTOR_CONTRACT} == SELECTOR_CONTRACT
    assert allocation["selection_uses_scores"] == (method == "adaptive")
    assert allocation["epoch"] == len(allocation["reset_history"])
    history = allocation["episode_history"]
    assert len(history) == len(data["episode_rewards"])
    for record, episode in zip(history, data["episode_rewards"]):
        assert (record["end_step"], record["length"], record["return"]) == (
            episode["step"], episode["length"], episode["return"])
        assert record["start_step"] == record["end_step"] - record["length"] + 1
        assert record["utility"] == float(np.mean(record["return"]))
        if record["credited"]:
            assert record["start_step"] > COMMON_KWARGS["learning_starts"] and not record["mixed_epoch"]
            assert record["epoch_start"] == record["epoch_end"]
            score = np.tanh((record["utility"] - record["baseline_before"]) / record["scale_before"])
            assert np.isclose(score, record["score"], rtol=0, atol=1e-12)
    counted = Counter(record["operator"] for record in history if record["credited"])
    assert allocation["counts"] == {op: counted[op] for op in allocation["operators"]}
    assert sum(r["length"] for r in history) + allocation["partial_episode_length"] == previous.TOTAL_STEPS
    assert sum(allocation["step_counts"].values()) == previous.TOTAL_STEPS
    config = dict(preference_relabeling_strategy="uniform", critic_relabel_copies=1,
        envelope_target_candidates=1, envelope_policy_candidates=1, vector_critic_coef=0.0,
        behavior_clone_coef=0.0)
    assert allocation["operator_trainer_configs"] == {op: config for op in allocation["operators"]}
    if method == "fixed":
        assert all(probability == 1 / 3 for probability in allocation["last_probs"].values())
    report.update(protocol=PROTOCOL, method=method, training_method=METHODS[method],
                  learning_work=work, common_target_learning=data["common_target_learning"], allocation=allocation)
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
        kwargs = previous.pilot.TRAIN_KWARGS if method == "v25" else COMMON_KWARGS
        training.train_method(env, METHODS[method], previous.TOTAL_STEPS, 25000, seed,
                              str(folder), archive_steps=(previous.TOTAL_STEPS,), **kwargs)
    report = validate(json.loads(raw.read_text()), env, method, seed, model)
    write_report(output, report)
    print(f"DONE {PROTOCOL} {env} {method} seed={seed} "
          f"utility={report['metrics']['direct_utility']['mean']:.6f}", flush=True)
    return report


def summarize(root=ROOT):
    result = dict(protocol=PROTOCOL, seeds=list(SEEDS), complete=True,
        primary="episode adaptive minus same-substrate episode fixed direct utility, separately by environment",
        scope="Five-seed pilot; v25 is a bundled recovery reference, not a single-factor contrast.",
        uncertainty="20000 paired-seed percentile bootstrap resamples, RNG0; descriptive95% intervals",
        environments={}, completed_budget=dict.fromkeys(controls.budget(ENVIRONMENTS[0]), 0))
    indices = np.random.default_rng(0).integers(0, len(SEEDS), (20000, len(SEEDS)))
    for env in ENVIRONMENTS:
        spec = previous.extension.ENVIRONMENTS[env]
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
                for key, upper in controls.budget(env).items():
                    actual = record["completed_budget"][key]
                    assert (spec["num_modes"] * 21 * 3 <= actual <= upper
                            if key == "evaluation_interactions" else actual == upper)
                    result["completed_budget"][key] += actual
            assert pair["adaptive"]["learning_work"] == pair["fixed"]["learning_work"]
            for method in ("adaptive", "fixed"):
                assert pair[method]["common_target_learning"] == LEARNING_CONTRACT
                allocation = pair[method]["allocation"]
                assert {key: allocation[key] for key in SELECTOR_CONTRACT} == SELECTOR_CONTRACT
                assert allocation["selection_uses_scores"] == (method == "adaptive")
            rows.append(dict(seed=seed, **pair))
        contrasts = {}
        values = lambda method, metric: np.asarray([row[method]["metrics"][metric]["mean"]
            if metric == "direct_utility" else row[method]["metrics"][metric] for row in rows])
        for a, b in (("adaptive", "fixed"), ("adaptive", "v25"), ("fixed", "v25")):
            contrasts[f"{a}_minus_{b}"] = {}
            for metric in ("direct_utility", "final_hv"):
                av, bv = values(a, metric), values(b, metric)
                assert np.isfinite(av).all() and np.isfinite(bv).all()
                contrasts[f"{a}_minus_{b}"][metric] = dict(previous.paired(av.tolist(), bv.tolist()),
                    paired_mean_ci95=np.quantile((av - bv)[indices].mean(axis=1), [.025, .975]).tolist())
            contrasts[f"{a}_minus_{b}"]["per_mode_utility_mean_delta"] = np.mean([
                np.asarray(row[a]["metrics"]["direct_utility"]["per_mode"]) -
                np.asarray(row[b]["metrics"]["direct_utility"]["per_mode"]) for row in rows], axis=0).tolist()
        result["environments"][env] = dict(runs=rows, paired=contrasts,
            methods={method: dict(direct_utility_mean=float(values(method, "direct_utility").mean()),
                minimum_direct_utility=float(values(method, "direct_utility").min()),
                final_hv_mean=float(values(method, "final_hv").mean()),
                per_mode_utility_mean=np.mean([row[method]["metrics"]["direct_utility"]["per_mode"]
                                              for row in rows], axis=0).tolist(),
                training_seconds_mean=float(np.mean([row[method]["metrics"]["training_seconds"] for row in rows])),
                learning_work_total={key: sum(row[method]["learning_work"][key] for row in rows)
                    for key in rows[0][method]["learning_work"] if key != "network_parameters"},
                network_parameters=rows[0][method]["learning_work"]["network_parameters"]) for method in METHODS})
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
        write_report(args.root / "readout.json", summarize(args.root))
    else:
        assert args.env and args.method and args.seed is not None
        run(args.env, args.method, args.seed, args.root)
