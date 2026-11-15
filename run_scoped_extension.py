"""Frozen HalfCheetah extension and unchanged-rule three-objective transfer."""

import argparse
import json
from pathlib import Path

import run_scoped_adaptive_confirmation as base


PROTOCOL = "scoped_extension_v1"
ROOT = Path("results/iclr_scoped_extension_20260922")
ENVIRONMENTS = {
    "mo-halfcheetah-v5": dict(reward_dim=2, num_modes=4, seeds=tuple(range(1011, 1026))),
    "mo-hopper-v5": dict(reward_dim=3, num_modes=4, seeds=tuple(range(1011, 1016))),
    "mo-ant-v5": dict(reward_dim=3, num_modes=3, seeds=tuple(range(1011, 1016))),
}
CONDITIONS, METHODS, training = base.CONDITIONS, base.METHODS, base.training


def evaluation_limit(env_name):
    return ENVIRONMENTS[env_name]["num_modes"] * 19 * 21 * 3 * 1000


def validate_budget(data, env_name):
    assert data["training_interactions"] == 50000
    assert data["environment_initialization_probe_steps"] == 2
    actual, upper = data["evaluation_interactions"], evaluation_limit(env_name)
    assert isinstance(actual, int) and upper // 1000 <= actual <= upper
    if env_name == "mo-halfcheetah-v5":
        assert actual == upper
    return {**base.BUDGET, "evaluation_interactions": actual}


def validate(data, env_name, condition, seed):
    spec = ENVIRONMENTS[env_name]
    assert seed in spec["seeds"]
    assert (data["ground_truth"]["reward_dim"], data["ground_truth"]["num_modes"]) == (
        spec["reward_dim"], spec["num_modes"])
    budget = validate_budget(data, env_name)
    selector = base.validate_routing(data, condition, spec["reward_dim"])
    metrics = base.common.metrics(data, seed, expected_method=METHODS[base.factors(condition)[1]],
                                  env_name=env_name, reward_dim=spec["reward_dim"], num_modes=spec["num_modes"])
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
                    blocks=data["adaptive_operator_final"]["block_history"][:10]),
        scope="Frozen update rule and feedback, with no environment-specific tuning. New HalfCheetah "
              "seeds are the primary extension cohort; prior five are reported separately. Hopper/Ant "
              "are five-seed transfer checks. Actual evaluation interactions include early episode ends.")


def run(env_name, condition, seed, root=ROOT):
    assert env_name in ENVIRONMENTS and condition in CONDITIONS and seed in ENVIRONMENTS[env_name]["seeds"]
    folder = Path(root) / env_name.replace("-", "_") / condition
    folder.mkdir(parents=True, exist_ok=True)
    output = folder / f"seed{seed}.json"
    method = METHODS[base.factors(condition)[1]]
    raw = folder / f"{method}_seed{seed}.json"
    if output.exists():
        saved = json.loads(output.read_text())
        assert (saved["protocol"], saved["env"], saved["condition"], saved["seed"]) == (
            PROTOCOL, env_name, condition, seed)
        if saved["complete"]:
            print(f"DONE {PROTOCOL} {env_name} {condition} seed={seed} reusing complete report", flush=True)
            return saved
        if not raw.exists():
            raise RuntimeError("Incomplete training attempt preserved; inspect its log before a new-root retry")
    if raw.exists():
        data = json.loads(raw.read_text())
    else:
        base.scoped.forks.write_report(output, dict(protocol=PROTOCOL, env=env_name,
            condition=condition, seed=seed, complete=False,
            planned_budget_upper_bound={**base.BUDGET, "evaluation_interactions": evaluation_limit(env_name)}))
        router = base.IntegratedRouter(condition)
        try:
            data = training.train_method(env_name, method, 50000, 25000, seed, str(folder),
                state_probe=router, update_router=router, **base.common.pilot.TRAIN_KWARGS)
        finally:
            router.close()
    report = validate(data, env_name, condition, seed)
    base.scoped.forks.write_report(output, report)
    print(f"DONE {PROTOCOL} {env_name} {condition} seed={seed} final_hv={report['metrics']['final_hv']:.6f} "
          f"evaluation_interactions={report['completed_budget']['evaluation_interactions']}", flush=True)
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--env", choices=ENVIRONMENTS, required=True)
    parser.add_argument("--condition", choices=CONDITIONS, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--root", type=Path, default=ROOT)
    args = parser.parse_args()
    run(args.env, args.condition, args.seed, args.root)
