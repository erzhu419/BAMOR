"""Independent 20-seed confirmation of v25 versus CAPQL direct utility."""

import argparse
import json
from pathlib import Path

import numpy as np

import run_final_policy_fresh as previous
from run_terminal_component_forks import write_report


PROTOCOL = "v25_capql_utility_confirmation_v1"
ROOT = Path("results/iclr_utility_confirmation_20260927")
SEEDS = tuple(range(1601, 1621))
ENVIRONMENTS = ("mo-halfcheetah-v5", "mo-hopper-v5")
METHODS = previous.METHODS
BOOTSTRAP_REPEATS = 20000


def budget(env):
    modes = previous.extension.ENVIRONMENTS[env]["num_modes"]
    return dict(training_interactions=previous.TOTAL_STEPS,
                evaluation_interactions=modes * 21 * 3 * 1000,
                environment_initialization_probe_steps=2)


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
                                 complete=False, planned_budget_upper_bound=budget(env)))
    if not raw.exists():
        if method == "v25":
            previous.training.train_method(env, METHODS[method], previous.TOTAL_STEPS, 25000,
                seed, str(folder), archive_steps=(previous.TOTAL_STEPS,), **previous.pilot.TRAIN_KWARGS)
        else:
            previous.training.train_capql(env, previous.TOTAL_STEPS, 25000, seed, str(folder),
                                           eval_freq=100000, device="cpu", method="capql")
    report = previous.validate(json.loads(raw.read_text()), env, method, seed, model)
    report["protocol"] = PROTOCOL
    write_report(output, report)
    print(f"DONE {PROTOCOL} {env} {method} seed={seed} "
          f"utility={report['metrics']['direct_utility']['mean']:.6f} "
          f"budget={json.dumps(report['completed_budget'])}", flush=True)
    return report


def paired_summary(values, controls, indices):
    a, b = np.asarray(values), np.asarray(controls)
    assert a.shape == b.shape == (len(SEEDS),) and np.isfinite(a).all() and np.isfinite(b).all()
    delta = a - b
    resampled = delta[indices].mean(axis=1)
    return dict(previous.paired(values, controls),
                paired_mean_ci95=np.quantile(resampled, [.025, .975]).tolist(),
                paired_mean_ci97_5=np.quantile(resampled, [.0125, .9875]).tolist())


def summarize(root=ROOT):
    result = dict(protocol=PROTOCOL, seeds=list(SEEDS), complete=True,
        primary="v25 minus CAPQL final-policy direct utility, separately in each environment",
        uncertainty="20000 paired-training-seed percentile bootstrap resamples, RNG0. "
                    "97.5% marginal intervals apply Bonferroni to the two primary environment claims; "
                    "coverage is approximate. Secondary metric intervals are descriptive.",
        environments={}, completed_budget=dict.fromkeys(budget(ENVIRONMENTS[0]), 0),
        scope="New seeds only; no pooling with1201-1205. No archive selection, learning curve, "
              "sample-efficiency, blind-selector attribution, or cross-environment pooling.")
    indices = np.random.default_rng(0).integers(0, len(SEEDS), (BOOTSTRAP_REPEATS, len(SEEDS)))
    for env in ENVIRONMENTS:
        spec = previous.extension.ENVIRONMENTS[env]
        rows = []
        for seed in SEEDS:
            pair = {m: json.loads((Path(root) / env.replace("-", "_") / m / f"seed{seed}.json").read_text())
                    for m in METHODS}
            expected_eval = dict(
                evaluation_weights=previous.training.full_simplex_grid(spec["reward_dim"]).tolist(),
                episode_seeds_per_mode=[[100000 + seed * 1000 + mode * 10 + i for i in range(3)]
                                        for mode in range(spec["num_modes"])],
                ref_point=previous.training.reference_point_for_env(env, spec["reward_dim"]).tolist())
            for method, row in pair.items():
                assert (row["protocol"], row["env"], row["method"], row["seed"], row["complete"]) == (
                    PROTOCOL, env, method, seed, True)
                assert row["evaluation"] == expected_eval
                assert row["mode_history"] == pair["v25"]["mode_history"]
                assert len(row["metrics"]["direct_utility"]["per_mode"]) == spec["num_modes"]
                assert len(row["metrics"]["mode_final_hv"]) == spec["num_modes"]
                for key, upper in budget(env).items():
                    actual = row["completed_budget"][key]
                    assert (spec["num_modes"] * 21 * 3 <= actual <= upper
                            if key == "evaluation_interactions" else actual == upper), (env, method, seed, key)
                    result["completed_budget"][key] += actual
            rows.append(dict(seed=seed, **pair))
        values = {method: dict(
            direct_utility=[r[method]["metrics"]["direct_utility"]["mean"] for r in rows],
            final_hv=[r[method]["metrics"]["final_hv"] for r in rows],
            training_seconds=[r[method]["metrics"]["training_seconds"] for r in rows],
            evaluation_seconds=[r[method]["metrics"]["evaluation_seconds"] for r in rows]) for method in METHODS}
        comparisons = {metric: paired_summary(values["v25"][metric], values["capql"][metric], indices)
                       for metric in ("direct_utility", "final_hv")}
        result["environments"][env] = dict(seeds=list(SEEDS), runs=rows, paired=comparisons,
            methods={method: {metric: dict(mean=float(np.mean(v)), minimum=float(np.min(v)),
                                          std=float(np.std(v, ddof=1)))
                              for metric, v in data.items()} for method, data in values.items()},
            per_mode_utility_mean_delta=np.mean([
                np.asarray(r["v25"]["metrics"]["direct_utility"]["per_mode"])
                - r["capql"]["metrics"]["direct_utility"]["per_mode"] for r in rows], axis=0).tolist(),
            per_mode_hv_mean_delta=np.mean([
                np.asarray(r["v25"]["metrics"]["mode_final_hv"])
                - r["capql"]["metrics"]["mode_final_hv"] for r in rows], axis=0).tolist())
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
        print(json.dumps({env: r["paired"]["direct_utility"] for env, r in report["environments"].items()}))
    else:
        assert args.env and args.method and args.seed is not None
        run(args.env, args.method, args.seed, args.root)
