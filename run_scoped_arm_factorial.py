"""Two-arm attribution on four observed cases, reusing both existing endpoints."""

import argparse
import json
from pathlib import Path

import numpy as np

import run_scoped_extension as previous


PROTOCOL = "scoped_arm_factorial_v1"
ROOT = Path("results/iclr_scoped_arm_factorial_20260922")
REFERENCE_ROOT = previous.ROOT
CASES = {"mo-halfcheetah-v5": {1011: "negative", 1018: "positive"},
         "mo-hopper-v5": {1013: "negative", 1012: "positive"}}
base, training = previous.base, previous.training
TARGETS = {"envelope_only": (base.scoped.TARGETS[0],),
           "bc_only": (base.scoped.TARGETS[1],)}
METHOD = previous.METHODS["adaptive"]
SCOPE = ("Outcome-selected development cases, not fresh confirmation. Two operator-specific "
         "actor-selection changes form a factorial under unchanged live adaptive feedback. "
         "Operator choices and subsequent data can diverge. BC-only names the BC operator, "
         "not a BC-loss-only intervention. Snapshots use each branch's final encoder.")


def validate(data, env_name, condition, seed):
    assert seed in CASES[env_name] and condition in TARGETS
    spec = previous.ENVIRONMENTS[env_name]
    assert (data["ground_truth"]["reward_dim"], data["ground_truth"]["num_modes"]) == (
        spec["reward_dim"], spec["num_modes"])
    budget = previous.validate_budget(data, env_name)
    selector = base.validate_routing(data, "scoped_adaptive", spec["reward_dim"], TARGETS[condition])
    metrics = base.common.metrics(data, seed, expected_method=METHOD, env_name=env_name,
                                  reward_dim=spec["reward_dim"], num_modes=spec["num_modes"])
    for old, new in (("posthoc_late_snapshot_hv", "late_snapshot_hv"),
                     ("posthoc_late_mean_hv", "late_mean_hv"),
                     ("posthoc_late_mode_mean_hv", "late_mode_mean_hv")):
        metrics[new] = metrics.pop(old)
    metrics["terminal_returns"] = base.scoped.forks.terminal_returns(data).tolist()
    return dict(protocol=PROTOCOL, env=env_name, condition=condition, seed=seed, complete=True,
        completed_budget=budget, metrics=metrics, selector=selector, routing=data["update_routing"],
        evaluation={k: data[k] for k in ("evaluation_weights", "episode_seeds_per_mode", "ref_point",
                                        "policy_archive_snapshots")},
        warmup=dict(episodes=[r for r in data["episode_rewards"] if r["step"] <= base.scoped.CUT],
                    blocks=data["adaptive_operator_final"]["block_history"][:10]), scope=SCOPE)


def load_references(env_name, seed, root=REFERENCE_ROOT):
    reports = {}
    for condition in ("legacy_adaptive", "scoped_adaptive"):
        path = Path(root) / env_name.replace("-", "_") / condition / f"seed{seed}.json"
        report = json.loads(path.read_text())
        assert (report["protocol"], report["env"], report["condition"], report["seed"], report["complete"]) == (
            previous.PROTOCOL, env_name, condition, seed, True)
        reports[condition] = report
    validate_pair(reports["scoped_adaptive"], reports["legacy_adaptive"])
    return reports


def validate_pair(candidate, reference):
    assert candidate["env"] == reference["env"] and candidate["seed"] == reference["seed"]
    assert candidate["evaluation"] == reference["evaluation"]
    assert candidate["warmup"] == reference["warmup"], "Training differs before the intervention"
    assert np.array_equal(candidate["metrics"]["mode_history"], reference["metrics"]["mode_history"])


def run(env_name, condition, seed, root=ROOT, reference_root=REFERENCE_ROOT):
    assert seed in CASES[env_name] and condition in TARGETS
    references = load_references(env_name, seed, reference_root)
    folder = Path(root) / env_name.replace("-", "_") / condition
    folder.mkdir(parents=True, exist_ok=True)
    output, raw = folder / f"seed{seed}.json", folder / f"{METHOD}_seed{seed}.json"
    if output.exists():
        saved = json.loads(output.read_text())
        assert (saved["protocol"], saved["env"], saved["condition"], saved["seed"]) == (
            PROTOCOL, env_name, condition, seed)
        if saved["complete"]:
            validate_pair(saved, references["legacy_adaptive"])
            print(f"DONE {PROTOCOL} {env_name} {condition} seed={seed} reusing complete report", flush=True)
            return saved
        if not raw.exists():
            raise RuntimeError("Incomplete training attempt preserved; inspect its log before a new-root retry")
    if raw.exists():
        data = json.loads(raw.read_text())
    else:
        base.scoped.forks.write_report(output, dict(protocol=PROTOCOL, env=env_name, condition=condition,
            seed=seed, complete=False, planned_budget_upper_bound={**base.BUDGET,
                "evaluation_interactions": previous.evaluation_limit(env_name)}))
        router = base.IntegratedRouter("scoped_adaptive", target_operators=TARGETS[condition])
        try:
            data = training.train_method(env_name, METHOD, 50000, 25000, seed, str(folder),
                state_probe=router, update_router=router, **base.common.pilot.TRAIN_KWARGS)
        finally:
            router.close()
    report = validate(data, env_name, condition, seed)
    validate_pair(report, references["legacy_adaptive"])
    base.scoped.forks.write_report(output, report)
    print(f"DONE {PROTOCOL} {env_name} {condition} seed={seed} final_hv={report['metrics']['final_hv']:.6f} "
          f"evaluation_interactions={report['completed_budget']['evaluation_interactions']}", flush=True)
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--env", choices=CASES, required=True)
    parser.add_argument("--condition", choices=TARGETS, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--reference-root", type=Path, default=REFERENCE_ROOT)
    args = parser.parse_args()
    run(args.env, args.condition, args.seed, args.root, args.reference_root)
