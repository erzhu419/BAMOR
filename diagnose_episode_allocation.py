"""Read episode-allocation feedback provenance without new interactions."""

import argparse
from bisect import bisect_right
from collections import Counter
import json
from pathlib import Path

import numpy as np


ROOT = Path("results/iclr_episode_allocation_20260930")
METHODS = {method: f"bamor_common_target_episode_{method}"
           for method in ("adaptive", "fixed")}


def episode_modes(history, start, end):
    # mode_history timestamps are after-transition boundaries, not new rewards.
    starts = [0] + [step + 1 for step, _ in history[1:]]
    first, last = bisect_right(starts, start) - 1, bisect_right(starts, end) - 1
    return [mode for _, mode in history[first:last + 1]]


def diagnose_run(data):
    allocation = data["adaptive_operator_final"]
    operators = allocation["operators"]
    rate = allocation["baseline_rate"]
    learning_starts = allocation["learning_starts"]
    mode_history = data["ground_truth"]["mode_history"]
    scores = dict.fromkeys(operators, 0.)
    credited_counts = Counter()
    provenance, epoch = {}, 0
    episodes = []
    for record in allocation["episode_history"]:
        if record["epoch_start"] != epoch:
            raise ValueError("episode epochs do not follow recorded resets")
        start, end = record["start_step"], record["end_step"]
        modes = episode_modes(mode_history, start, end)
        phase = ("warmup" if start <= learning_starts else
                 "forced" if sum(credited_counts.values()) < len(operators) else "free")
        if phase == "forced" and record["operator"] != operators[sum(credited_counts.values())]:
            raise ValueError("recorded operator differs from forced allocation")
        if (record["baseline_before"] is None) != (not provenance):
            raise ValueError("baseline availability differs from episode provenance")
        logits = np.asarray([scores[op] for op in operators]) / allocation["temperature"]
        probabilities = np.exp(logits - logits.max())
        probabilities /= probabilities.sum()
        probabilities = ((1 - allocation["epsilon"]) * probabilities
                         + allocation["epsilon"] / len(operators))
        if not allocation["selection_uses_scores"]:
            probabilities.fill(1 / len(operators))
        mixed = record["epoch_start"] != record["epoch_end"]
        if mixed != record["mixed_epoch"]:
            raise ValueError("mixed-epoch flag differs from recorded epochs")
        item = dict(start_step=start, end_step=end, length=record["length"],
            operator=record["operator"], phase=phase, epoch_start=epoch,
            credited=record["credited"], detected_mixed=mixed,
            true_mixed=len(modes) > 1, true_modes=modes, utility=record["utility"],
            score=record["score"], scale_before=record["scale_before"],
            maximum_choice_probability=float(probabilities.max()) if phase == "free" else None,
            baseline_nonmatching_or_mixed_mass=(sum(mass for mode, mass in provenance.items()
                                          if mode != str(modes[0]))
                                      if provenance and len(modes) == 1 else None))
        if mixed:
            provenance, credited_counts = {}, Counter()
            scores = dict.fromkeys(operators, 0.)
            epoch = record["epoch_end"]
            if record["credited"]:
                raise ValueError("mixed detected-epoch episode received credit")
        else:
            if record["credited"]:
                rebuilt = float(np.tanh((record["utility"] - record["baseline_before"])
                                        / record["scale_before"]))
                if not np.isclose(rebuilt, record["score"], rtol=0, atol=1e-12):
                    raise ValueError("score does not follow the recorded causal baseline")
                op = record["operator"]
                scores[op] = (1 - allocation["score_rate"]) * scores[op] + allocation["score_rate"] * rebuilt
                credited_counts[op] += 1
            label = str(modes[0]) if len(modes) == 1 else "mixed"
            if provenance:
                provenance = {mode: (1 - rate) * mass for mode, mass in provenance.items()}
                provenance[label] = provenance.get(label, 0.) + rate
            else:
                provenance = {label: 1.}
        if not np.isclose(scores[record["operator"]], record["score_after"], rtol=0, atol=1e-12):
            raise ValueError("arm-score reconstruction differs after feedback")
        episodes.append(item)

    boundaries = [1] + [reset["step"] for reset in allocation["reset_history"]]
    ends = [step - 1 for step in boundaries[1:]] + [data["total_steps"]]
    segments = []
    for index, (start, end) in enumerate(zip(boundaries, ends)):
        postwarmup_steps = max(0, end - max(start, learning_starts + 1) + 1)
        if postwarmup_steps == 0:
            continue
        credited = [ep for ep in episodes if ep["epoch_start"] == index and ep["credited"]]
        segments.append(dict(epoch=index, start_step=start, end_step=end,
            postwarmup_steps=postwarmup_steps, credited_episodes=len(credited),
            credited_by_operator=dict(Counter(ep["operator"] for ep in credited)),
            free_credited_episodes=sum(ep["phase"] == "free" for ep in credited)))
    return episodes, segments


def summarize(episodes, segments):
    postwarmup = [ep for ep in episodes if ep["phase"] != "warmup"]
    credited = [ep for ep in postwarmup if ep["credited"]]
    pure = [ep for ep in credited if not ep["true_mixed"]]
    free = [ep for ep in postwarmup if ep["phase"] == "free"]
    floor = [ep for ep in credited if ep["scale_before"] == 1e-6]
    mean = lambda values: float(np.mean(values)) if values else None
    return dict(completed_episodes=len(episodes),
        postwarmup_completed_episodes=len(postwarmup),
        postwarmup_completed_steps=sum(ep["length"] for ep in postwarmup),
        selection_episode_counts=dict(Counter(ep["phase"] for ep in postwarmup)),
        selection_step_counts=dict(Counter({phase: sum(ep["length"] for ep in postwarmup if ep["phase"] == phase)
                                           for phase in ("forced", "free")})),
        credited_episodes=len(credited), free_credited_episodes=sum(ep["credited"] for ep in free),
        detected_mixed_episodes=sum(ep["detected_mixed"] for ep in postwarmup),
        true_mixed_credited_episodes=sum(ep["true_mixed"] for ep in credited),
        pure_credited_baseline_nonmatching_or_mixed_mass_mean=mean([ep["baseline_nonmatching_or_mixed_mass"] for ep in pure]),
        scale_floor_credited_episodes=len(floor),
        scale_floor_saturated_scores=sum(abs(ep["score"]) > .999999 for ep in floor),
        free_selection_maximum_probability_mean=mean([ep["maximum_choice_probability"] for ep in free]),
        postwarmup_detected_segments=len(segments),
        segments_without_three_credited_episodes=sum(seg["credited_episodes"] < 3 for seg in segments),
        segments_without_free_credited_episodes=sum(seg["free_credited_episodes"] == 0 for seg in segments),
        pure_credited_positive_score_fraction=mean([ep["score"] > 0 for ep in pure]),
        free_credited_true_pure_by_operator={op: dict(count=len(group),
            mean_utility=mean([ep["utility"] for ep in group]),
            mean_length=mean([ep["length"] for ep in group]),
            mean_score=mean([ep["score"] for ep in group]))
            for op in ("uniform", "stratified", "mixed")
            for group in [[ep for ep in pure if ep["phase"] == "free" and ep["operator"] == op]]})


def run(root):
    root = Path(root)
    readout = json.loads((root / "compact_readout.json").read_text())
    result = dict(protocol="episode_feedback_provenance_diagnostic_v1",
        source_protocol=readout["protocol"], seeds=readout["seeds"],
        additional_environment_interactions=0, environments={},
        scope="Stored training feedback provenance, not counterfactual operator benefit.",
        limitations="Baseline mass is the EMA coefficient weight of past episode utilities, "
                    "not their share of the current reward. True-mixed episodes use one unknown bucket. "
                    "Completed-step summaries omit warmup-starting and final partial episodes. "
                    "Conditional state/action predictions cannot be evaluated from these saved logs.")
    for env in readout["environments"]:
        groups = {method: ([], []) for method in METHODS}
        runs = []
        for seed in readout["seeds"]:
            for method, mapped in METHODS.items():
                path = root / env.replace("-", "_") / method / f"{mapped}_seed{seed}.json"
                data = json.loads(path.read_text())
                assert (data["env"], data["method"], data["seed"]) == (env, mapped, seed)
                episodes, segments = diagnose_run(data)
                groups[method][0].extend(episodes)
                groups[method][1].extend(segments)
                runs.append(dict(seed=seed, method=method, summary=summarize(episodes, segments),
                    true_mode_history=data["ground_truth"]["mode_history"],
                    reset_steps=[reset["step"] for reset in data["adaptive_operator_final"]["reset_history"]],
                    segments=segments,
                    true_mixed_credited_examples=[ep for ep in episodes if ep["credited"] and ep["true_mixed"]][:3],
                    scale_floor_credited_examples=[ep for ep in episodes if ep["credited"] and ep["scale_before"] == 1e-6][:3]))
        result["environments"][env] = dict(runs=runs,
            methods={method: summarize(*group) for method, group in groups.items()})
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT)
    args = parser.parse_args()
    result = run(args.root)
    path = args.root / "feedback_diagnostic.json"
    path.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    print(json.dumps({env: values["methods"] for env, values in result["environments"].items()}, indent=2))
