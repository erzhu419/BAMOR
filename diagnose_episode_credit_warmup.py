"""Re-score stored episode returns with a three-episode causal scale startup."""

import argparse
import json
import math
from pathlib import Path

import numpy as np

from diagnose_episode_allocation import METHODS, ROOT as SOURCE_ROOT, diagnose_run
from episode_credit_baseline import EpisodeCreditBaseline, INITIAL_EPISODES


ROOT = Path("results/iclr_episode_credit_warmup_20261001")
PROTOCOL = "stored_three_episode_credit_scale_v1"
SATURATION = .999999


def validate_legacy_baseline(records, rate, learning_starts):
    mean, variance = None, 0.
    for record in records:
        scale = max(math.sqrt(variance), 1e-6) if mean is not None else None
        for name, expected in (("baseline_before", mean), ("scale_before", scale)):
            actual = record[name]
            if ((actual is None) != (expected is None) or
                    (expected is not None and not np.isclose(actual, expected, rtol=1e-12, atol=1e-12))):
                raise ValueError(f"legacy {name} differs from reconstructed past episodes")
        eligible = (record["start_step"] > learning_starts and mean is not None
                    and not record["mixed_epoch"])
        if record["credited"] != eligible:
            raise ValueError("legacy credit eligibility differs from reconstructed baseline")
        if record["mixed_epoch"]:
            mean, variance = None, 0.
        elif mean is None:
            mean = record["utility"]
        else:
            delta = record["utility"] - mean
            mean += rate * delta
            variance = (1 - rate) * (variance + rate * delta ** 2)


def diagnose(data):
    allocation = data["adaptive_operator_final"]
    records = allocation["episode_history"]
    validate_legacy_baseline(records, allocation["baseline_rate"], allocation["learning_starts"])
    episodes, source_segments = diagnose_run(data)
    baseline = EpisodeCreditBaseline(allocation["baseline_rate"])
    states, readiness_steps = {}, {}
    for record, item in zip(records, episodes):
        before = baseline.snapshot()
        if before["mean"] != record["baseline_before"]:
            raise ValueError("candidate changed the legacy EMA return center")
        eligible = record["start_step"] > allocation["learning_starts"] and not record["mixed_epoch"]
        score = baseline.score(record["utility"], before) if eligible else None
        item.update(candidate_score=score, candidate_credited=score is not None,
                    candidate_scale_before=before["scale"], initialization_count_before=before["count"])
        if record["mixed_epoch"]:
            baseline.reset()
        else:
            baseline.observe(record["utility"])
            epoch = record["epoch_start"]
            states[epoch] = baseline.snapshot()
            if before["scale"] is None and baseline.variance is not None:
                readiness_steps[epoch] = record["end_step"]
    segments = []
    for source in source_segments:
        epoch = source["epoch"]
        group = [ep for ep in episodes if ep["epoch_start"] == epoch and ep["candidate_credited"]]
        state = states.get(epoch, dict(count=0, scale=None))
        segments.append(dict(source, candidate_credited_episodes=len(group),
            candidate_credited_source_free_episodes=sum(ep["phase"] == "free" for ep in group),
            completed_eligible_baseline_episodes=state["count"],
            scale_initialized=state["scale"] is not None,
            scale_initialized_at_end_step=readiness_steps.get(epoch),
            first_candidate_credited_start_step=group[0]["start_step"] if group else None))
    return episodes, segments


def score_summary(values):
    return dict(count=len(values), saturated=sum(abs(value) > SATURATION for value in values),
        saturated_fraction=float(np.mean([abs(value) > SATURATION for value in values])) if values else None,
        positive_fraction=float(np.mean([value > 0 for value in values])) if values else None,
        mean=float(np.mean(values)) if values else None)


def summarize(episodes, segments):
    source = [ep for ep in episodes if ep["credited"]]
    candidate = [ep for ep in episodes if ep["candidate_credited"]]
    common = [ep for ep in source if ep["candidate_credited"]]
    withheld = [ep for ep in source if not ep["candidate_credited"]]
    pure = [ep for ep in common if not ep["true_mixed"]]
    return dict(completed_episodes=len(episodes), source_credited_episodes=len(source),
        candidate_credited_episodes=len(candidate), common_credited_episodes=len(common),
        newly_credited_episodes=sum(not ep["credited"] for ep in candidate),
        withheld_source_credited_episodes=len(withheld),
        source_floor_credited_episodes=sum(ep["scale_before"] == 1e-6 for ep in source),
        withheld_source_floor_credited_episodes=sum(ep["scale_before"] == 1e-6 for ep in withheld),
        withheld_source_saturated_episodes=sum(abs(ep["score"]) > SATURATION for ep in withheld),
        source_all=score_summary([ep["score"] for ep in source]),
        candidate_all=score_summary([ep["candidate_score"] for ep in candidate]),
        common_source=score_summary([ep["score"] for ep in common]),
        common_candidate=score_summary([ep["candidate_score"] for ep in common]),
        common_source_saturated_candidate_not=sum(abs(ep["score"]) > SATURATION
            and abs(ep["candidate_score"]) <= SATURATION for ep in common),
        common_candidate_saturated_source_not=sum(abs(ep["candidate_score"]) > SATURATION
            and abs(ep["score"]) <= SATURATION for ep in common),
        common_true_pure_source=score_summary([ep["score"] for ep in pure]),
        common_true_pure_candidate=score_summary([ep["candidate_score"] for ep in pure]),
        postwarmup_detected_segments=len(segments),
        source_zero_credit_segments=sum(seg["credited_episodes"] == 0 for seg in segments),
        candidate_zero_credit_segments=sum(seg["candidate_credited_episodes"] == 0 for seg in segments),
        added_zero_credit_segments=sum(seg["credited_episodes"] > 0 and seg["candidate_credited_episodes"] == 0
                                      for seg in segments),
        segments_without_initialized_scale=sum(not seg["scale_initialized"] for seg in segments),
        minimum_candidate_credit_scale=min((ep["candidate_scale_before"] for ep in candidate), default=None),
        common_source_free_true_pure_by_operator={op: dict(count=len(group),
            source=score_summary([ep["score"] for ep in group]),
            candidate=score_summary([ep["candidate_score"] for ep in group]))
            for op in ("uniform", "stratified", "mixed")
            for group in [[ep for ep in pure if ep["phase"] == "free" and ep["operator"] == op]]})


def run(source_root, root):
    source_readout = json.loads((source_root / "compact_readout.json").read_text())
    protocol = json.loads((root / "protocol.json").read_text())
    if protocol["protocol"] != PROTOCOL or protocol["initial_episodes"] != INITIAL_EPISODES:
        raise ValueError("frozen protocol differs from credit baseline implementation")
    result = dict(protocol=PROTOCOL, source_protocol=source_readout["protocol"],
        initial_episodes=INITIAL_EPISODES, seeds=source_readout["seeds"],
        additional_environment_interactions=0, additional_probe_interactions=0,
        model_fits=0, optimizer_updates=0, behavior="Frozen recorded episodes and detector resets",
        baseline_center="Unchanged legacy EMA", source_baseline_round_trip_passed=True,
        decision="hold_online_allocation_evidence", environments={})
    for env in source_readout["environments"]:
        groups = {method: ([], []) for method in METHODS}
        runs = []
        for seed in source_readout["seeds"]:
            for method, mapped in METHODS.items():
                path = source_root / env.replace("-", "_") / method / f"{mapped}_seed{seed}.json"
                data = json.loads(path.read_text())
                if (data["env"], data["method"], data["seed"]) != (env, mapped, seed):
                    raise ValueError("source identity differs from episode allocation readout")
                episodes, segments = diagnose(data)
                groups[method][0].extend(episodes)
                groups[method][1].extend(segments)
                runs.append(dict(seed=seed, method=method, summary=summarize(episodes, segments), segments=segments))
        result["environments"][env] = dict(runs=runs,
            methods={method: summarize(*group) for method, group in groups.items()})
    result["source_runs_checked"] = sum(len(value["runs"]) for value in result["environments"].values())
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, default=SOURCE_ROOT)
    parser.add_argument("--root", type=Path, default=ROOT)
    args = parser.parse_args()
    result = run(args.source_root, args.root)
    path = args.root / "readout.json"
    path.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    print(json.dumps(dict(source_runs_checked=result["source_runs_checked"],
        environments={env: value["methods"] for env, value in result["environments"].items()}), indent=2))
