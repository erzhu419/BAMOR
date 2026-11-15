"""Read stored block scores and completed episodes without environment queries."""

import argparse
import json
from pathlib import Path

import numpy as np


ROOT = Path("results/iclr_projected_allocation_20260930")
METHODS = {method: f"bamor_common_target_projected_{method}"
           for method in ("adaptive", "fixed")}


def episode_half(episode, start, middle, end):
    first = episode["step"] - episode["length"] + 1
    last = episode["step"]
    if start <= first <= last <= middle:
        return 0
    if middle < first <= last <= end:
        return 1
    return None


def diagnose_run(data):
    changes = [change["step"] for change in data["mode_switches"]
               if change["kind"] == "env_truth"]
    blocks = []
    for block in data["adaptive_operator_final"]["block_history"]:
        if not block["complete"]:
            continue
        start, end = block["end_step"] - block["steps"] + 1, block["end_step"]
        middle = start + block["half_counts"][0] - 1
        means = np.asarray(block["half_mean_rewards"])
        gain = (means[1] - means[0]) / np.asarray(block["scale_before_block"])
        rebuilt = float(np.tanh(gain).mean())
        if not np.isclose(rebuilt, block["progress_score"], rtol=0, atol=1e-12):
            raise ValueError(f"stored score differs from formula at step{end}")
        crossing = any(start < change <= end for change in changes)
        groups = [[], []]
        for episode in data["episode_rewards"]:
            half = episode_half(episode, start, middle, end)
            if half is not None:
                groups[half].append(episode)
        record = dict(end_step=end, operator=block["operator"], score=rebuilt,
                      mode_crossing=crossing,
                      raw_step_utility_delta=float((means[1] - means[0]).mean()),
                      completed_episodes_per_half=[len(group) for group in groups])
        if not crossing and all(groups):
            # Only full episodes within a half-window can be attributed from these logs.
            returns = [np.mean([np.mean(ep["return"]) for ep in group]) for group in groups]
            lengths = [np.mean([ep["length"] for ep in group]) for group in groups]
            record.update(episode_utility_delta=float(returns[1] - returns[0]),
                          episode_length_delta=float(lengths[1] - lengths[0]))
        blocks.append(record)
    return blocks


def summarize(blocks):
    crossing = [block for block in blocks if block["mode_crossing"]]
    stationary = [block for block in blocks if not block["mode_crossing"]]
    eligible = [block for block in stationary if "episode_utility_delta" in block]
    positive = [block for block in eligible if block["score"] > 0]
    scores = np.asarray([block["score"] for block in eligible])
    returns = np.asarray([block["episode_utility_delta"] for block in eligible])
    correlation = None
    if len(eligible) > 1 and scores.std() > 0 and returns.std() > 0:
        correlation = float(np.corrcoef(scores, returns)[0, 1])
    return dict(complete_blocks=len(blocks), mode_crossing_blocks=len(crossing),
        crossing_mean_abs_score=float(np.mean([abs(b["score"]) for b in crossing])) if crossing else None,
        stationary_mean_abs_score=float(np.mean([abs(b["score"]) for b in stationary])) if stationary else None,
        stationary_score_vs_raw_step_sign_disagreements=sum(
            b["score"] * b["raw_step_utility_delta"] < 0 for b in stationary),
        stationary_blocks_with_two_full_episode_halves=len(eligible),
        positive_score_eligible_blocks=len(positive),
        positive_score_episode_utility_declines=sum(b["episode_utility_delta"] < 0 for b in positive),
        positive_score_episode_length_declines=sum(b["episode_length_delta"] < 0 for b in positive),
        score_episode_utility_delta_correlation=correlation)


def run(root):
    root = Path(root)
    readout = json.loads((root / "compact_readout.json").read_text())
    result = dict(protocol="projected_allocation_score_diagnostic_v1",
        source_protocol=readout["protocol"], seeds=readout["seeds"],
        additional_environment_interactions=0, environments={},
        scope="Descriptive stored training data, not causal operator credit or final-policy evaluation.",
        limitations="Complete episodes must fit entirely inside750-step halves; long episodes are censored. "
                    "Equal-weight raw episode utility is a diagnostic proxy, not the matched preference-grid metric.")
    for env in readout["environments"]:
        all_blocks = {method: [] for method in METHODS}
        runs = []
        for seed in readout["seeds"]:
            for method, mapped in METHODS.items():
                path = root / env.replace("-", "_") / method / f"{mapped}_seed{seed}.json"
                data = json.loads(path.read_text())
                assert (data["env"], data["method"], data["seed"]) == (env, mapped, seed)
                blocks = diagnose_run(data)
                all_blocks[method].extend(blocks)
                runs.append(dict(seed=seed, method=method, summary=summarize(blocks),
                    crossing_blocks=[b for b in blocks if b["mode_crossing"]],
                    positive_score_return_declines=[b for b in blocks
                        if b["score"] > 0 and b.get("episode_utility_delta", 0) < 0]))
        result["environments"][env] = dict(
            methods={method: summarize(blocks) for method, blocks in all_blocks.items()}, runs=runs)
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT)
    args = parser.parse_args()
    result = run(args.root)
    path = args.root / "score_diagnostic.json"
    path.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    print(json.dumps({env: values["methods"] for env, values in result["environments"].items()}, indent=2))
