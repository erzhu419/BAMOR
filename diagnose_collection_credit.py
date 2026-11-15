"""Screen stored collection signals against paired learner gains, without fitting."""

import argparse
import json
from pathlib import Path

import numpy as np


PROTOCOL = "stored_collection_credit_screen_v1"
SOURCE_PROTOCOL = "paired_collection_value_probe_v1"
SOURCE_ROOT = Path("results/iclr_collection_value_20261001")
ROOT = Path("results/iclr_collection_credit_screen_20261002")
SEEDS = tuple(range(2301, 2306))
ENVIRONMENTS = ("mo-halfcheetah-v5", "mo-hopper-v5")
SIGNALS = ("reward_rate", "survival_rate", "mean_preference_balance")
PRIMARY = SIGNALS[0]
ARMS = ("uniform", "mixed")


def collection_signals(profile):
    steps = profile["interactions"]
    rewards = np.asarray(profile["total_raw_reward"], dtype=float)
    preference = np.asarray(profile["mean_preference"], dtype=float)
    return dict(reward_rate=float(rewards.mean() / steps),
        survival_rate=float(-profile["terminated"] / steps),
        mean_preference_balance=float(-np.sum((preference - 1 / len(preference)) ** 2)))


def choices_from_collection(collection):
    signals = {arm: collection_signals(collection[arm]) for arm in ARMS}
    choices = {name: "mixed" if signals["mixed"][name] > signals["uniform"][name]
               else "uniform" for name in SIGNALS}
    return signals, choices


def make_rows(source):
    rows = {}
    for env in ENVIRONMENTS:
        runs = source["environments"][env]["runs"]
        if [run["seed"] for run in runs] != list(SEEDS):
            raise ValueError("source seed roster differs from the frozen screen")
        group = []
        for run in runs:
            collection = run["collection"]
            for arm in (*ARMS, "stratified", "uniform_repeat"):
                if collection[arm]["interactions"] != 1500:
                    raise ValueError("source collection block differs from the frozen comparison")
            # Runtime is not a signal: only the recorded identity-control fields matter.
            for key in ("total_raw_reward", "terminated", "truncated", "mean_preference"):
                if collection["uniform"][key] != collection["uniform_repeat"][key]:
                    raise ValueError("recorded repeated-uniform collection control differs")
            signals, choices = choices_from_collection(collection)
            repeat_signals = collection_signals(collection["uniform_repeat"])
            if signals["uniform"] != repeat_signals:
                raise ValueError("identity control produces different collection credit")
            endpoints = {name: dict(utility=branch["direct_utility"]["mean"],
                per_mode_utility=branch["direct_utility"]["per_mode"], hv=branch["final_hv"])
                for name, branch in run["branches"].items()}
            group.append(dict(seed=run["seed"], signals=signals, choices=choices,
                stratified_signals=collection_signals(collection["stratified"]),
                endpoints=endpoints,
                mixed_minus_uniform_utility=endpoints["mixed"]["utility"] - endpoints["uniform"]["utility"],
                identity_control_passed=True))
        rows[env] = group
    return rows


def delta_stats(values, bootstrap_indices):
    values = np.asarray(values, dtype=float)
    return dict(mean=float(values.mean()), minimum=float(values.min()),
        paired_deltas=values.tolist(), wins=int(np.sum(values > 0)), losses=int(np.sum(values < 0)),
        ties=int(np.sum(values == 0)),
        paired_mean_ci95=np.quantile(values[bootstrap_indices].mean(axis=1), [.025, .975]).tolist())


def policy_metrics(rows, choices, bootstrap_indices):
    chosen = [row["endpoints"][arm] for row, arm in zip(rows, choices)]
    uniform = [row["endpoints"]["uniform"] for row in rows]
    oracle = [max(row["endpoints"][arm]["utility"] for arm in ARMS) for row in rows]
    utility_delta = [a["utility"] - b["utility"] for a, b in zip(chosen, uniform)]
    return dict(choices=choices, mixed_choices=choices.count("mixed"),
        utility_mean=float(np.mean([value["utility"] for value in chosen])),
        minimum_seed_utility=min(value["utility"] for value in chosen),
        utility_minus_uniform=delta_stats(utility_delta, bootstrap_indices),
        hv_minus_uniform=delta_stats([a["hv"] - b["hv"] for a, b in zip(chosen, uniform)], bootstrap_indices),
        per_mode_utility_mean_delta=np.mean([np.asarray(a["per_mode_utility"]) - b["per_mode_utility"]
            for a, b in zip(chosen, uniform)], axis=0).tolist(),
        oracle_utility_gap=float(np.mean([a - b["utility"] for a, b in zip(oracle, chosen)])))


def summarize(rows):
    indices = np.random.default_rng(0).integers(0, len(SEEDS), (20000, len(SEEDS)))
    environments = {}
    for env, group in rows.items():
        policies = {name: policy_metrics(group, [row["choices"][name] for row in group], indices)
                    for name in SIGNALS}
        for name in SIGNALS:
            policy = policies[name]
            # A tie chooses uniform; zero learner contrast is not a sign prediction win.
            informative = [row for row in group if row["mixed_minus_uniform_utility"] != 0]
            policy.update(sign_matches=sum((row["choices"][name] == "mixed") ==
                (row["mixed_minus_uniform_utility"] > 0) for row in informative),
                informative_pairs=len(informative),
                harmful_mixed_choices=sum(row["choices"][name] == "mixed" and
                    row["mixed_minus_uniform_utility"] < 0 for row in group),
                missed_positive_choices=sum(row["choices"][name] == "uniform" and
                    row["mixed_minus_uniform_utility"] > 0 for row in group))
        controls = {"always_uniform": policy_metrics(group, ["uniform"] * len(group), indices),
                    "always_mixed": policy_metrics(group, ["mixed"] * len(group), indices),
                    "hindsight_oracle": policy_metrics(group, ["mixed" if row["mixed_minus_uniform_utility"] > 0
                        else "uniform" for row in group], indices)}
        environments[env] = dict(runs=group, policies=policies, controls=controls)
    primary = {env: data["policies"][PRIMARY] for env, data in environments.items()}
    passed = (primary[ENVIRONMENTS[0]]["utility_minus_uniform"]["mean"] >= 0
        and primary[ENVIRONMENTS[1]]["utility_minus_uniform"]["mean"] > 0
        and primary[ENVIRONMENTS[1]]["hv_minus_uniform"]["mean"] >= 0)
    return dict(protocol=PROTOCOL, source_protocol=SOURCE_PROTOCOL, seeds=list(SEEDS),
        complete=True, primary_signal=PRIMARY, primary_screen_passed=passed,
        decision="development_reward_proxy_lead_fresh_validation_required" if passed else "hold_collection_reward_proxy",
        evidence_role="Reused inspected cohort; selection correspondence screen, not prospective validation",
        environments=environments, additional_environment_interactions=0, optimizer_updates=0,
        model_fits=0, v25_changed=False,
        uncertainty="Descriptive 95% paired-seed percentile bootstrap;20000 resamples,RNG0")


def run(source_root=SOURCE_ROOT, root=ROOT):
    root, source_root = Path(root), Path(source_root)
    frozen = json.loads((root / "protocol.json").read_text())
    expected = dict(protocol=PROTOCOL, source_protocol=SOURCE_PROTOCOL, seeds=list(SEEDS),
        environments=list(ENVIRONMENTS), primary_signal=PRIMARY, signals=list(SIGNALS),
        arms=list(ARMS), collection_steps=1500)
    if {key: frozen[key] for key in expected} != expected:
        raise ValueError("screen implementation differs from its frozen protocol")
    source = json.loads((source_root / "compact_readout.json").read_text())
    source_frozen = json.loads((source_root / "protocol.json").read_text())
    if (source["protocol"] != SOURCE_PROTOCOL or not source["complete"] or
            source["decision"] != "hold_collection_learning_value" or source["seeds"] != list(SEEDS) or
            sorted(source["environments"]) != sorted(ENVIRONMENTS) or
            source["experiment"] != source_frozen["experiment"]):
        raise ValueError("source differs from the completed frozen HOLD comparison")
    report = summarize(make_rows(source))
    report["source_completed_budget"] = source["completed_budget"]
    (root / "readout.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    compact = {env: {name: dict(mean_delta=policy["utility_minus_uniform"]["mean"],
        sign_matches=policy["sign_matches"], mixed_choices=policy["mixed_choices"])
        for name, policy in data["policies"].items()} for env, data in report["environments"].items()}
    print(json.dumps(dict(decision=report["decision"], environments=compact)))
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, default=SOURCE_ROOT)
    parser.add_argument("--root", type=Path, default=ROOT)
    args = parser.parse_args()
    run(args.source_root, args.root)
