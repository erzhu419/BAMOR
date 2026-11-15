"""Fresh paired pilot of a common learning objective and adaptive collection."""

import argparse
import json
from pathlib import Path

import numpy as np

import run_utility_confirmation as controls


PROTOCOL = "common_target_sampling_pilot_v1"
ROOT = Path("results/iclr_common_target_pilot_20260930")
SEEDS = tuple(range(1801, 1806))
ENVIRONMENTS = controls.ENVIRONMENTS
METHODS = dict(adaptive="bamor_common_target_adaptive", fixed="bamor_common_target_fixed",
               v25="bamor_v25_progress_archive_mix", capql="capql", capql_full_simplex="capql_full_simplex")
COMMON_KWARGS = dict(controls.previous.pilot.TRAIN_KWARGS, hidden=256)
CAPQL_LEARNING_STARTS = 5000
CAPQL_BATCH_SIZE = 256
previous = controls.previous
training = previous.training
write_report = controls.write_report


def validate(data, env, method, seed, model):
    mapped = METHODS[method]
    report = previous.validate(data, env, mapped if method in ("adaptive", "fixed") else method, seed, model)
    report.update(protocol=PROTOCOL, method=method, training_method=mapped, learning_work=data["learning_work"])
    kwargs = COMMON_KWARGS if method in ("adaptive", "fixed") else previous.pilot.TRAIN_KWARGS
    starts = kwargs["learning_starts"] if not training.is_capql_method(mapped) else CAPQL_LEARNING_STARTS
    batch = kwargs["batch_size"] if not training.is_capql_method(mapped) else CAPQL_BATCH_SIZE
    updates = max(0, previous.TOTAL_STEPS - starts + 1)
    work = data["learning_work"]
    assert work["critic_optimizer_steps"] == work["actor_optimizer_steps"] == updates
    assert work["encoder_optimizer_steps"] == (0 if training.is_capql_method(mapped) else updates)
    assert min(work[k] for k in ("critic_rows", "target_candidate_rows", "actor_candidate_rows")) >= updates * batch
    if method in ("adaptive", "fixed"):
        assert all(work[k] == updates * batch for k in ("critic_rows", "target_candidate_rows", "actor_candidate_rows"))
        assert data["critic_candidate_context_rule"] == "own_preference"
        assert data["common_target_learning"] == dict(critic_loss_rule="mean_twin_full_vector_mse",
            actor_update_rule="own_preference_soft_sac", learning_preference_distribution="uniform_simplex",
            candidate_count=1, behavior_clone_coef=0.0, critic_relabel_copies=1)
        selector = data["adaptive_operator_final"]
        assert selector["operators"] == ["uniform", "stratified", "mixed"]
        assert selector["selection_uses_scores"] == (method == "adaptive")
        assert sum(selector["counts"].values()) == (previous.TOTAL_STEPS - starts) // selector["block_steps"]
        assert len({tuple(sorted(config.items())) for config in selector["operator_trainer_configs"].values()}) == 1
        report.update(common_target_learning=data["common_target_learning"], allocation=selector)
    if training.is_capql_method(mapped):
        report["training_preference_sampler"] = data["training_preference_sampler"]
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
        if training.is_capql_method(METHODS[method]):
            training.train_capql(env, previous.TOTAL_STEPS, 25000, seed, str(folder),
                                 eval_freq=100000, device="cpu", method=METHODS[method])
        else:
            kwargs = COMMON_KWARGS if method in ("adaptive", "fixed") else previous.pilot.TRAIN_KWARGS
            training.train_method(env, METHODS[method], previous.TOTAL_STEPS, 25000, seed, str(folder),
                                  archive_steps=(previous.TOTAL_STEPS,), **kwargs)
    report = validate(json.loads(raw.read_text()), env, method, seed, model)
    write_report(output, report)
    print(f"DONE {PROTOCOL} {env} {method} seed={seed} "
          f"utility={report['metrics']['direct_utility']['mean']:.6f} "
          f"work={json.dumps(report['learning_work'])}", flush=True)
    return report


def summarize(root=ROOT):
    result = dict(protocol=PROTOCOL, seeds=list(SEEDS), complete=True,
        primary="common-target adaptive minus same-substrate fixed allocation direct utility, separately by environment",
        scope="Five-seed fresh pilot, not confirmation. Common-target versus v25 is a bundled "
              "substrate/width/sampling change; only adaptive versus fixed isolates allocation. "
              "Optimizer steps/rows/parameter counts are reported, not FLOP equivalence or neural convergence.",
        uncertainty="20000 paired-seed percentile bootstrap resamples, RNG0; descriptive95% intervals.",
        environments={}, completed_budget=dict.fromkeys(controls.budget(ENVIRONMENTS[0]), 0))
    indices = np.random.default_rng(0).integers(0, len(SEEDS), (20000, len(SEEDS)))
    for env in ENVIRONMENTS:
        rows = []
        spec = previous.extension.ENVIRONMENTS[env]
        for seed in SEEDS:
            pair = {m: json.loads((Path(root) / env.replace("-", "_") / m / f"seed{seed}.json").read_text()) for m in METHODS}
            expected_eval = dict(
                evaluation_weights=training.full_simplex_grid(spec["reward_dim"]).tolist(),
                episode_seeds_per_mode=[[100000 + seed * 1000 + mode * 10 + i for i in range(3)]
                                        for mode in range(spec["num_modes"])],
                ref_point=training.reference_point_for_env(env, spec["reward_dim"]).tolist())
            for method, row in pair.items():
                assert (row["protocol"], row["env"], row["method"], row["seed"], row["complete"]) == (
                    PROTOCOL, env, method, seed, True)
                assert row["evaluation"] == expected_eval and row["mode_history"] == pair["adaptive"]["mode_history"]
                for key, upper in controls.budget(env).items():
                    actual = row["completed_budget"][key]
                    assert (spec["num_modes"] * 21 * 3 <= actual <= upper
                            if key == "evaluation_interactions" else actual == upper)
                    result["completed_budget"][key] += actual
            assert pair["adaptive"]["learning_work"] == pair["fixed"]["learning_work"]
            assert pair["adaptive"]["common_target_learning"] == pair["fixed"]["common_target_learning"]
            rows.append(dict(seed=seed, **pair))
        contrasts = {}
        for a, b in (("adaptive", "fixed"), ("adaptive", "v25"), ("fixed", "v25"),
                     ("adaptive", "capql"), ("adaptive", "capql_full_simplex"),
                     ("fixed", "capql"), ("fixed", "capql_full_simplex")):
            comparisons = {}
            for metric in ("direct_utility", "final_hv"):
                values = lambda method: [r[method]["metrics"][metric]["mean"] if metric == "direct_utility"
                                         else r[method]["metrics"][metric] for r in rows]
                av, bv = np.asarray(values(a)), np.asarray(values(b))
                assert np.isfinite(av).all() and np.isfinite(bv).all()
                comparisons[metric] = dict(previous.paired(av.tolist(), bv.tolist()),
                    paired_mean_ci95=np.quantile((av - bv)[indices].mean(axis=1), [.025, .975]).tolist())
            contrasts[f"{a}_minus_{b}"] = comparisons
        result["environments"][env] = dict(runs=rows, paired=contrasts,
            methods={m: dict(direct_utility_mean=float(np.mean([r[m]["metrics"]["direct_utility"]["mean"] for r in rows])),
                minimum_direct_utility=float(np.min([r[m]["metrics"]["direct_utility"]["mean"] for r in rows])),
                final_hv_mean=float(np.mean([r[m]["metrics"]["final_hv"] for r in rows])),
                per_mode_utility_mean=np.mean([r[m]["metrics"]["direct_utility"]["per_mode"] for r in rows], axis=0).tolist(),
                training_seconds_mean=float(np.mean([r[m]["metrics"]["training_seconds"] for r in rows])),
                learning_work_total={key: sum(r[m]["learning_work"][key] for r in rows)
                                     for key in rows[0][m]["learning_work"] if key != "network_parameters"},
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
