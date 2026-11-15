"""Complete the inspected 20-seed cohort with full-simplex CAPQL controls."""

import argparse
import json
from pathlib import Path

import numpy as np

import run_utility_confirmation as controls


PROTOCOL = "capql_full_simplex_completion_v1"
ROOT = Path("results/iclr_capql_full_simplex_20260930")
METHOD = "capql_full_simplex"
SEEDS = controls.SEEDS
ENVIRONMENTS = controls.ENVIRONMENTS
previous = controls.previous
write_report = controls.write_report


def sampler_spec(env):
    dim = previous.extension.ENVIRONMENTS[env]["reward_dim"]
    return dict(distribution="Dirichlet", concentration=[1.0] * dim, support="full_simplex")


def validate(data, env, seed, model):
    assert data["training_preference_sampler"] == sampler_spec(env)
    assert data["capql_replica_index"] == 0 and data["eval_seed"] == seed + 1000
    report = previous.validate(data, env, METHOD, seed, model)
    report.update(protocol=PROTOCOL, training_preference_sampler=data["training_preference_sampler"])
    return report


def run(env, seed, root=ROOT):
    assert env in ENVIRONMENTS and seed in SEEDS
    folder = Path(root) / env.replace("-", "_") / METHOD
    folder.mkdir(parents=True, exist_ok=True)
    output = folder / f"seed{seed}.json"
    raw = folder / f"{METHOD}_seed{seed}.json"
    model = folder / f"{METHOD}_seed{seed}_model.pt"
    if output.exists():
        saved = json.loads(output.read_text())
        assert (saved["protocol"], saved["env"], saved["method"], saved["seed"]) == (PROTOCOL, env, METHOD, seed)
        if saved.get("complete"):
            return saved
        if not raw.exists():
            raise RuntimeError("Incomplete attempt preserved; inspect before a new-root retry")
    else:
        write_report(output, dict(protocol=PROTOCOL, env=env, method=METHOD, seed=seed,
            complete=False, planned_budget_upper_bound=controls.budget(env),
            training_preference_sampler=sampler_spec(env)))
    if not raw.exists():
        previous.training.train_capql(env, previous.TOTAL_STEPS, 25000, seed, str(folder),
                                      eval_freq=100000, device="cpu", method=METHOD)
    report = validate(json.loads(raw.read_text()), env, seed, model)
    write_report(output, report)
    print(f"DONE {PROTOCOL} {env} seed={seed} "
          f"utility={report['metrics']['direct_utility']['mean']:.6f} "
          f"budget={json.dumps(report['completed_budget'])}", flush=True)
    return report


def summarize(root=ROOT, control_root=controls.ROOT):
    result = dict(protocol=PROTOCOL, seeds=list(SEEDS), complete=True,
        primary="v25 minus full-simplex CAPQL final-policy direct utility, separately by environment",
        scope="Retrospective baseline completion on the already inspected1601-1620 cohort. "
              "Original v25 and native-angle CAPQL records are reused unchanged; no independent "
              "confirmation, archive selection, sample-efficiency or convergence claim.",
        uncertainty="20000 paired-seed percentile bootstrap resamples, RNG0; 97.5% marginal "
                    "intervals for two primary environments. Coverage is approximate; "
                    "secondary intervals are descriptive.",
        environments={}, new_completed_budget=dict.fromkeys(controls.budget(ENVIRONMENTS[0]), 0),
        reused_control_budget=dict.fromkeys(controls.budget(ENVIRONMENTS[0]), 0))
    indices = np.random.default_rng(0).integers(0, len(SEEDS), (controls.BOOTSTRAP_REPEATS, len(SEEDS)))
    for env in ENVIRONMENTS:
        rows = []
        spec = previous.extension.ENVIRONMENTS[env]
        expected_eval = dict(
            evaluation_weights=previous.training.full_simplex_grid(spec["reward_dim"]).tolist(),
            ref_point=previous.training.reference_point_for_env(env, spec["reward_dim"]).tolist())
        for seed in SEEDS:
            pair = {method: json.loads((Path(control_root) / env.replace("-", "_") /
                    method / f"seed{seed}.json").read_text()) for method in controls.METHODS}
            pair[METHOD] = json.loads((Path(root) / env.replace("-", "_") / METHOD / f"seed{seed}.json").read_text())
            expected_eval["episode_seeds_per_mode"] = [
                [100000 + seed * 1000 + mode * 10 + i for i in range(3)] for mode in range(spec["num_modes"])]
            for method, row in pair.items():
                protocol = PROTOCOL if method == METHOD else controls.PROTOCOL
                assert (row["protocol"], row["env"], row["method"], row["seed"], row["complete"]) == (
                    protocol, env, method, seed, True)
                assert row["evaluation"] == expected_eval
                assert row["mode_history"] == pair["v25"]["mode_history"]
                assert len(row["metrics"]["direct_utility"]["per_mode"]) == spec["num_modes"]
                assert len(row["metrics"]["mode_final_hv"]) == spec["num_modes"]
                if method == METHOD:
                    assert row["training_preference_sampler"] == sampler_spec(env)
                total = result["new_completed_budget" if method == METHOD else "reused_control_budget"]
                for key, upper in controls.budget(env).items():
                    actual = row["completed_budget"][key]
                    assert (spec["num_modes"] * 21 * 3 <= actual <= upper
                            if key == "evaluation_interactions" else actual == upper)
                    total[key] += actual
            rows.append(dict(seed=seed, **pair))
        values = {method: {
            "direct_utility": [r[method]["metrics"]["direct_utility"]["mean"] for r in rows],
            **{metric: [r[method]["metrics"][metric] for r in rows]
               for metric in ("final_hv", "training_seconds", "evaluation_seconds")}}
            for method in pair}
        contrasts = {name: {metric: controls.paired_summary(values[a][metric], values[b][metric], indices)
                            for metric in ("direct_utility", "final_hv")}
                     for name, a, b in (("v25_minus_full_simplex", "v25", METHOD),
                                        ("full_simplex_minus_native", METHOD, "capql"))}
        result["environments"][env] = dict(runs=rows, paired=contrasts,
            methods={m: {metric: dict(mean=float(np.mean(v)), minimum=float(np.min(v)),
                                      std=float(np.std(v, ddof=1))) for metric, v in data.items()}
                     for m, data in values.items()},
            per_mode_utility_mean_delta=np.mean([
                np.asarray(r["v25"]["metrics"]["direct_utility"]["per_mode"])
                - r[METHOD]["metrics"]["direct_utility"]["per_mode"] for r in rows], axis=0).tolist())
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--env", choices=ENVIRONMENTS)
    parser.add_argument("--seed", type=int, choices=SEEDS)
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--control-root", type=Path, default=controls.ROOT)
    parser.add_argument("--summarize", action="store_true")
    args = parser.parse_args()
    if args.summarize:
        report = summarize(args.root, args.control_root)
        write_report(args.root / "readout.json", report)
        print(json.dumps({env: r["paired"] for env, r in report["environments"].items()}))
    else:
        assert args.env and args.seed is not None
        run(args.env, args.seed, args.root)
