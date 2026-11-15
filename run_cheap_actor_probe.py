"""One-episode archive selection versus equal-interaction v25 continuation."""

import argparse
import copy
import json
import time
from pathlib import Path
from statistics import mean

import numpy as np
import torch

from diagnose_mean_action_gradient import fork_env
import run_archive_actor_probe as previous
import run_aligned_progress_pilot as pilot
import run_scoped_extension as extension
import train_bamor_mujoco as training


PROTOCOL = "cheap_actor_probe_vs_training_v1"
ROOT = Path("results/iclr_cheap_actor_probe_20260926")
SEEDS = (1101, 1102, 1103)
BASE_STEP = 50000
STEPS = previous.STEPS
METHOD = previous.METHOD
PROBE_DWELL = 5000


def clone_training_state(state):
    env, eval_env = fork_env(state["env"]), fork_env(state["eval_env"])
    return copy.deepcopy(state, {id(state["env"]): env, id(state["eval_env"]): eval_env})


class CaptureState:
    def __init__(self, step):
        self.step = step
        self.state = None

    def __call__(self, state):
        self.state = clone_training_state(state)
        return True


def choose_step(cells):
    return max(range(len(cells)), key=lambda index: cells[index]["score_per_step"])


def probe_actors(env_name, seed, weights, policy, encoder, archive):
    cells, interactions = [], 0
    actor = previous.ActorProbe(policy, len(weights[0]))
    for index, weight in enumerate(weights):
        candidates = []
        for step in STEPS:
            policy.load_state_dict(archive[step])
            policy.eval()
            env = training.make_piecewise_mo_env(
                env_name, switch_interval=PROBE_DWELL,
                seed=6000000 + 1000 * seed + index)
            try:
                before = env._total_steps
                point = training.evaluate_pareto_v2(
                    env, actor, METHOD, encoder, [weight], num_episodes=1,
                    device="cpu", episode_seeds=[7000000 + 10000 * seed + index])[0]
                used = env._total_steps - before
                assert 0 < used <= 1000
                candidates.append({"actor_step": step, "interactions": used,
                                   "score_per_step": float(np.dot(weight, point["return"]) / used),
                                   "mode_switches_audit_only": len(env.get_ground_truth()["mode_history"]) - 1})
                interactions += used
            finally:
                env.close()
        selected = choose_step(candidates)
        cells.append({"weight": weight.tolist(), "selected_step": STEPS[selected],
                      "candidates": candidates})
    return cells, interactions


def evaluate_three(env_name, seed, weights, selected_steps, archive, base_policy,
                   base_encoder, continued_policy, continued_encoder):
    env = training.make_piecewise_mo_env(env_name, switch_interval=25000, seed=seed + 1000)
    fronts = {name: [] for name in ("selected", "final", "continued")}
    base_actor = previous.ActorProbe(base_policy, env.reward_dim)
    continued_actor = previous.ActorProbe(continued_policy, env.reward_dim)
    try:
        env._lock_mode = True
        for mode in range(env.num_modes):
            env._apply_mode(mode)
            mode_fronts = {name: [] for name in fronts}
            reset_seeds = [100000 + seed * 1000 + mode * 10 + episode for episode in range(3)]
            for weight, selected_step in zip(weights, selected_steps):
                base_policy.load_state_dict(archive[BASE_STEP])
                final = training.evaluate_pareto_v2(
                    env, base_actor, METHOD, base_encoder, [weight], num_episodes=3,
                    device="cpu", episode_seeds=reset_seeds)[0]["return"]
                mode_fronts["final"].append(final)
                if selected_step == BASE_STEP:
                    selected = final
                else:
                    base_policy.load_state_dict(archive[selected_step])
                    selected = training.evaluate_pareto_v2(
                        env, base_actor, METHOD, base_encoder, [weight], num_episodes=3,
                        device="cpu", episode_seeds=reset_seeds)[0]["return"]
                mode_fronts["selected"].append(selected)
                continued = training.evaluate_pareto_v2(
                    env, continued_actor, METHOD, continued_encoder, [weight], num_episodes=3,
                    device="cpu", episode_seeds=reset_seeds)[0]["return"]
                mode_fronts["continued"].append(continued)
            for name in fronts:
                fronts[name].append(np.asarray(mode_fronts[name], dtype=float))
        ref = training.reference_point_for_env(env_name, env.reward_dim)
        metrics = {}
        for name, mode_fronts in fronts.items():
            utilities = [mean(float(np.dot(weight, value))
                              for weight, value in zip(weights, front)) for front in mode_fronts]
            metrics[name] = {
                "mean_same_preference_utility": mean(utilities),
                "per_mode_same_preference_utility": utilities,
                "subset_mode_mean_hv": mean(training.compute_hv_nd(front, ref)
                                            for front in mode_fronts),
            }
        return metrics, env._total_steps
    finally:
        env.close()


def run(env_name, seed, root=ROOT):
    assert env_name in extension.ENVIRONMENTS and seed in SEEDS
    output = Path(root) / env_name.replace("-", "_") / f"seed{seed}.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.exists():
        saved = json.loads(output.read_text())
        if saved.get("complete"):
            assert (saved["protocol"], saved["env"], saved["seed"]) == (PROTOCOL, env_name, seed)
            return saved
        raise RuntimeError(f"Incomplete attempt retained at {output}; use a new root for a rerun")
    output.write_text(json.dumps({"protocol": PROTOCOL, "env": env_name,
                                  "seed": seed, "complete": False}) + "\n")
    torch.set_num_threads(2)
    capture = CaptureState(BASE_STEP)
    start = time.perf_counter()
    training.train_method(env_name, METHOD, BASE_STEP, 25000, seed, str(output.parent / "unused"),
                          state_probe=capture, archive_steps=STEPS, **pilot.TRAIN_KWARGS)
    state = capture.state
    assert state is not None and state["env"]._total_steps == BASE_STEP
    assert [snap["step"] for snap in state["policy_archive"]] == list(STEPS)
    archive = {snap["step"]: snap["policy"] for snap in state["policy_archive"]}
    for key, value in state["trainer"].policy.state_dict().items():
        assert torch.equal(value.cpu(), archive[BASE_STEP][key])
    base_seconds = time.perf_counter() - start
    weights = previous.weights_for_dim(state["env"].reward_dim)
    base_policy = copy.deepcopy(state["trainer"].policy).cpu().eval()
    base_encoder = copy.deepcopy(state["encoder"]).cpu().eval()
    start = time.perf_counter()
    cells, probe_steps = probe_actors(env_name, seed, weights, base_policy, base_encoder, archive)
    probe_seconds = time.perf_counter() - start
    selected_steps = [cell["selected_step"] for cell in cells]
    assert probe_steps == sum(candidate["interactions"] for cell in cells for candidate in cell["candidates"])
    end_step = BASE_STEP + probe_steps
    state["policy_archive"] = []  # It is evaluation-only; the continuation still uses the captured learner state.
    endpoint = CaptureState(end_step)
    start = time.perf_counter()
    training.train_method(env_name, METHOD, end_step, 25000, seed, str(output.parent / "unused"),
                          resume_state=state, state_probe=endpoint, archive_steps={end_step},
                          **pilot.TRAIN_KWARGS)
    continuation_seconds = time.perf_counter() - start
    continued = endpoint.state
    assert continued is not None and continued["env"]._total_steps == end_step
    continued_policy = copy.deepcopy(continued["trainer"].policy).cpu().eval()
    continued_encoder = copy.deepcopy(continued["encoder"]).cpu().eval()
    start = time.perf_counter()
    test, test_steps = evaluate_three(env_name, seed, weights, selected_steps, archive,
                                      base_policy, base_encoder, continued_policy, continued_encoder)
    test_seconds = time.perf_counter() - start
    for captured in (state, continued):
        captured["env"].close()
        captured["eval_env"].close()
    report = {"protocol": PROTOCOL, "env": env_name, "seed": seed, "complete": True,
              "training_interactions": end_step, "base_training_interactions": BASE_STEP,
              "continuation_interactions": probe_steps, "probe_evaluation_interactions": probe_steps,
              "test_evaluation_interactions": test_steps,
              "initialization_probe_steps": {"base_training": 2, "continuation": 2,
                                             "actor_probe": len(STEPS) * len(weights), "test": 1},
              "candidate_steps": list(STEPS), "probe_weights": weights.tolist(),
              "selected_steps": selected_steps, "probe_cells": cells, "test": test,
              "seconds": {"base_training": base_seconds, "probe": probe_seconds,
                          "continuation": continuation_seconds, "test": test_seconds}}
    output.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    delta = test["selected"]["mean_same_preference_utility"] - test["continued"]["mean_same_preference_utility"]
    print(f"DONE {PROTOCOL} {env_name} seed={seed} probe_steps={probe_steps} "
          f"selected_minus_continued={delta:.6f}", flush=True)
    return report


def summarize(root=ROOT):
    result = {"protocol": PROTOCOL, "seeds": list(SEEDS), "environments": {}}
    for env_name in extension.ENVIRONMENTS:
        rows = [json.loads((Path(root) / env_name.replace("-", "_") /
                            f"seed{seed}.json").read_text()) for seed in SEEDS]
        assert all((row["protocol"], row["env"], row["seed"], row["complete"]) ==
                   (PROTOCOL, env_name, seed, True) for row, seed in zip(rows, SEEDS))
        compact = []
        for row in rows:
            assert row["continuation_interactions"] == row["probe_evaluation_interactions"]
            values = {name: row["test"][name]["mean_same_preference_utility"]
                      for name in ("selected", "final", "continued")}
            compact.append({"seed": row["seed"], "probe_steps": row["probe_evaluation_interactions"],
                            "selected_steps": row["selected_steps"], "utility": values,
                            "subset_hv": {name: row["test"][name]["subset_mode_mean_hv"]
                                          for name in values},
                            "per_mode_utility": {name: row["test"][name]["per_mode_same_preference_utility"]
                                                 for name in values}})
        deltas = [row["utility"]["selected"] - row["utility"]["continued"] for row in compact]
        result["environments"][env_name] = {
            "runs": compact, "selected_minus_continued_mean": mean(deltas),
            "selected_minus_continued_wins": sum(delta > 0 for delta in deltas),
            "selected_minus_continued_losses": sum(delta < 0 for delta in deltas),
            "selected_minus_continued_ties": sum(delta == 0 for delta in deltas),
            "mean_utility": {name: mean(row["utility"][name] for row in compact)
                             for name in compact[0]["utility"]},
            "total_probe_interactions": sum(row["probe_steps"] for row in compact),
        }
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--env", choices=extension.ENVIRONMENTS)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--summarize", action="store_true")
    args = parser.parse_args()
    if args.summarize:
        output = args.root / "readout.json"
        output.write_text(json.dumps(summarize(args.root), indent=2, allow_nan=False) + "\n")
        print(f"DONE {PROTOCOL} summary {output}")
    else:
        assert args.env and args.seed is not None
        run(args.env, args.seed, args.root)
