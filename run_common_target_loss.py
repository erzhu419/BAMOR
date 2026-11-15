"""Fresh fixed-allocation critic-metric diagnostic with unchanged soft targets."""

import argparse
import json
from pathlib import Path

import numpy as np

import run_common_target_pilot as pilot


PROTOCOL = "common_target_critic_metric_v1"
ROOT = Path("results/iclr_common_target_loss_20260930")
SEEDS = tuple(range(1901, 1906))
ENVIRONMENTS = pilot.ENVIRONMENTS
METHODS = dict(vector="bamor_common_target_fixed",
               projected="bamor_common_target_projected_fixed", v25=pilot.METHODS["v25"])
LOSS_RULES = dict(vector="mean_twin_full_vector_mse",
                  projected="trace_normalized_vector_plus_projection_mse")
DIAGNOSTIC_START = 50000
training, previous, controls = pilot.training, pilot.previous, pilot.controls
write_report = pilot.write_report


def validate(data, env, method, seed, model):
    mapped = METHODS[method]
    report = previous.validate(data, env, "v25" if method == "v25" else mapped, seed, model)
    report.update(protocol=PROTOCOL, method=method, training_method=mapped,
                  learning_work=data["learning_work"])
    kwargs = previous.pilot.TRAIN_KWARGS if method == "v25" else pilot.COMMON_KWARGS
    updates = previous.TOTAL_STEPS - kwargs["learning_starts"] + 1
    work = data["learning_work"]
    assert all(work[k] == updates for k in (
        "critic_optimizer_steps", "actor_optimizer_steps", "encoder_optimizer_steps"))
    if method != "v25":
        assert all(work[k] == updates * kwargs["batch_size"] for k in (
            "critic_rows", "target_candidate_rows", "actor_candidate_rows"))
        assert data["critic_candidate_context_rule"] == "own_preference"
        assert data["critic_target_rule"] == "scalar_projection_of_soft_vector"
        assert data["bootstrap_terminal_rule"] == "terminated_only"
        assert not data["reward_normalization"]
        expected = dict(critic_loss_rule=LOSS_RULES[method], actor_update_rule="own_preference_soft_sac",
                        learning_preference_distribution="uniform_simplex", candidate_count=1,
                        behavior_clone_coef=0.0, critic_relabel_copies=1)
        assert data["common_target_learning"] == expected
        allocation = data["adaptive_operator_final"]
        assert not allocation["selection_uses_scores"]
        assert allocation["operators"] == ["uniform", "stratified", "mixed"]
        assert all(p == 1 / 3 for p in allocation["last_probs"].values())
        assert sum(allocation["counts"].values()) == (
            previous.TOTAL_STEPS - kwargs["learning_starts"]) // allocation["block_steps"]
        config = dict(preference_relabeling_strategy="uniform", critic_relabel_copies=1,
                      envelope_target_candidates=1, envelope_policy_candidates=1,
                      vector_critic_coef=0.0, behavior_clone_coef=0.0)
        assert set(allocation["operator_trainer_configs"]) == set(allocation["operators"])
        assert all(value == config for value in allocation["operator_trainer_configs"].values())
        report.update(common_target_learning=expected, allocation=allocation)
    grouped = {}
    for episode in data["episode_rewards"]:
        if episode["step"] >= DIAGNOSTIC_START:
            grouped.setdefault(episode["mode"], []).append(episode)
    report["completed_training_episode_profile"] = {
        mode: dict(count=len(rows), mean_length=float(np.mean([r["length"] for r in rows])),
                   mean_reward_per_step=np.mean(
                       [np.asarray(r["return"]) / r["length"] for r in rows], axis=0).tolist())
        for mode, rows in grouped.items()}
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
        kwargs = previous.pilot.TRAIN_KWARGS if method == "v25" else pilot.COMMON_KWARGS
        training.train_method(env, METHODS[method], previous.TOTAL_STEPS, 25000, seed,
                              str(folder), archive_steps=(previous.TOTAL_STEPS,), **kwargs)
    report = validate(json.loads(raw.read_text()), env, method, seed, model)
    write_report(output, report)
    print(f"DONE {PROTOCOL} {env} {method} seed={seed} "
          f"utility={report['metrics']['direct_utility']['mean']:.6f}", flush=True)
    return report


def summarize(root=ROOT):
    result = dict(protocol=PROTOCOL, seeds=list(SEEDS), complete=True,
                  primary="projected minus vector fixed-allocation direct utility, separately by environment",
                  uncertainty="20000 paired-seed percentile bootstrap resamples, RNG0; descriptive95% intervals",
                  environments={}, completed_budget=dict.fromkeys(controls.budget(ENVIRONMENTS[0]), 0))
    indices = np.random.default_rng(0).integers(0, len(SEEDS), (20000, len(SEEDS)))
    for env in ENVIRONMENTS:
        spec = previous.extension.ENVIRONMENTS[env]
        rows = []
        for seed in SEEDS:
            pair = {m: json.loads((Path(root) / env.replace("-", "_") / m / f"seed{seed}.json").read_text())
                    for m in METHODS}
            expected_eval = dict(evaluation_weights=training.full_simplex_grid(spec["reward_dim"]).tolist(),
                episode_seeds_per_mode=[[100000 + seed * 1000 + mode * 10 + i for i in range(3)]
                                        for mode in range(spec["num_modes"])],
                ref_point=training.reference_point_for_env(env, spec["reward_dim"]).tolist())
            for method, row in pair.items():
                assert (row["protocol"], row["env"], row["method"], row["seed"], row["complete"]) == (
                    PROTOCOL, env, method, seed, True)
                assert row["evaluation"] == expected_eval and row["mode_history"] == pair["vector"]["mode_history"]
                for key, upper in controls.budget(env).items():
                    actual = row["completed_budget"][key]
                    assert (spec["num_modes"] * 21 * 3 <= actual <= upper
                            if key == "evaluation_interactions" else actual == upper)
                    result["completed_budget"][key] += actual
            assert pair["vector"]["learning_work"] == pair["projected"]["learning_work"]
            for method in ("vector", "projected"):
                assert pair[method]["common_target_learning"]["critic_loss_rule"] == LOSS_RULES[method]
            contracts = [{k: v for k, v in pair[m]["common_target_learning"].items() if k != "critic_loss_rule"}
                         for m in ("vector", "projected")]
            assert contracts[0] == contracts[1]
            histories = [[{k: r[k] for k in ("end_step", "operator", "steps", "complete")}
                          for r in pair[m]["allocation"]["block_history"]] for m in ("vector", "projected")]
            assert histories[0] == histories[1]
            rows.append(dict(seed=seed, **pair))
        contrasts = {}
        for a, b in (("projected", "vector"), ("projected", "v25"), ("vector", "v25")):
            contrasts[f"{a}_minus_{b}"] = {}
            for metric in ("direct_utility", "final_hv"):
                values = lambda m: np.asarray([r[m]["metrics"][metric]["mean"] if metric == "direct_utility"
                                               else r[m]["metrics"][metric] for r in rows])
                av, bv = values(a), values(b)
                assert np.isfinite(av).all() and np.isfinite(bv).all()
                contrasts[f"{a}_minus_{b}"][metric] = dict(previous.paired(av.tolist(), bv.tolist()),
                    paired_mean_ci95=np.quantile((av - bv)[indices].mean(axis=1), [.025, .975]).tolist())
        result["environments"][env] = dict(runs=rows, paired=contrasts,
            methods={m: dict(direct_utility_mean=float(np.mean([r[m]["metrics"]["direct_utility"]["mean"] for r in rows])),
                minimum_direct_utility=min(r[m]["metrics"]["direct_utility"]["mean"] for r in rows),
                final_hv_mean=float(np.mean([r[m]["metrics"]["final_hv"] for r in rows])),
                per_mode_utility_mean=np.mean([r[m]["metrics"]["direct_utility"]["per_mode"] for r in rows], axis=0).tolist(),
                training_seconds_mean=float(np.mean([r[m]["metrics"]["training_seconds"] for r in rows])),
                network_parameters=rows[0][m]["learning_work"]["network_parameters"]) for m in METHODS})
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
        write_report(args.root / "readout.json", report)
        print(json.dumps({env: row["paired"] for env, row in report["environments"].items()}))
    else:
        assert args.env and args.method and args.seed is not None
        run(args.env, args.method, args.seed, args.root)
