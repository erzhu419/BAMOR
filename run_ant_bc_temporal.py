"""Replay the frozen Ant BC pairs with intermediate evaluations and native terms."""

import argparse
import copy
import json
from pathlib import Path
from unittest.mock import patch

import numpy as np
import torch

import run_ant_component_replication as previous


base, training = previous.base, previous.base.training
PROTOCOL = "ant_bc_temporal_v1"
ROOT = Path("results/iclr_ant_bc_temporal_20260927")
SEEDS, KWARGS = previous.SEEDS, previous.KWARGS
CUT, BLOCK_END, END = 65000, 66500, 75000
MIDDLE = (68000, 69500, 71000, 72500, 74000)
BRANCHES = ("full", "no_bc")
TERMS = ("negative_unweighted_control_cost", "survival", "contact")


def times():
    return (BLOCK_END, *MIDDLE, END)


def planned_budget():
    training_steps = 2 * END - CUT
    return dict(training_interactions=training_steps,
                learner_updates=training_steps - KWARGS["learning_starts"] + 1,
                evaluation_interactions=(4 * 3 + 2 * len(MIDDLE)) * 21 * 3 * 1000,
                environment_initialization_probe_steps=4)


class NativeTerms:
    """Observe existing eval steps; preserve returned rewards and actions exactly."""

    def __init__(self, env):
        self.env, self.episodes = env, []
        self.original_reset, self.original_step = env.reset, env.step

    def reset(self, *args, **kwargs):
        result = self.original_reset(*args, **kwargs)
        self.episodes.append(dict(seed=kwargs["seed"], mode=self.env.current_mode_idx,
                                  steps=0, native=np.zeros(3), roundoff=0.))
        return result

    def step(self, action):
        result = self.original_step(action)
        _, reward, _, _, info = result
        row = self.episodes[-1]
        assert self.env._lock_mode and info["mode_idx"] == row["mode"]
        terms = np.array([info["reward_ctrl"] / self.env.unwrapped._ctrl_cost_weight,
                          info["reward_survive"], info["reward_contact"]], dtype=float)
        scale = self.env.mode_configs[row["mode"]].reward_scale[2]
        row["native"] += terms
        row["roundoff"] += float(reward[2]) - float(terms.sum() * scale)
        row["steps"] += 1
        return result

    def attach(self, result, modes, metadata):
        assert len(self.episodes) == len(modes) * 21 * 3
        for i, row in enumerate(self.episodes):
            mode, episode = modes[i // 63], i % 3
            assert (row["mode"], row["seed"]) == (mode, metadata["episode_seeds_per_mode"][mode][episode])
        shape = (len(modes), 21, 3)
        native = np.array([r["native"] for r in self.episodes]).reshape(*shape, 3)
        roundoff = np.array([r["roundoff"] for r in self.episodes]).reshape(shape)
        scale = np.array([self.env.mode_configs[m].reward_scale[2] for m in modes])
        reconstructed = native.mean(axis=2).sum(axis=-1) * scale[:, None] + roundoff.mean(axis=2)
        np.testing.assert_allclose(reconstructed, np.asarray(result["returns"])[:, :, 2], rtol=1e-12, atol=1e-8)
        result.update(modes=list(modes), native_term_names=list(TERMS),
            native_episode_returns=native.tolist(), objective2_scales=scale.tolist(),
            episode_roundoff=roundoff.tolist(),
            episode_steps=np.array([r["steps"] for r in self.episodes]).reshape(shape).tolist())


def measure(state, seed, metadata):
    env, step = state["eval_env"], state["step"]
    recorder = NativeTerms(env)
    numpy_rng, torch_rng = np.random.get_state(), torch.get_rng_state()
    mode, locked, before = env.current_mode_idx, env._lock_mode, env._total_steps
    try:
        with patch.object(env, "reset", recorder.reset), patch.object(env, "step", recorder.step):
            if step in (BLOCK_END, END):
                result = base.measure(state, seed)
                modes = [0, 1, 2]
                assert all(result[k] == value for k, value in metadata.items())
            else:
                modes = [0]
                env._lock_mode = True
                env._apply_mode(0)
                front = training.evaluate_pareto_v2(env, state["trainer"], base.METHOD, state["encoder"],
                    metadata["evaluation_weights"], episode_seeds=metadata["episode_seeds_per_mode"][0])
                result = dict(step=step, returns=[[p["return"] for p in front]],
                              evaluation_interactions=env._total_steps - before)
        recorder.attach(result, modes, metadata)
        assert sum(r["steps"] for r in recorder.episodes) == result["evaluation_interactions"]
        return result
    finally:
        env._apply_mode(mode)
        env._lock_mode = locked
        np.random.set_state(numpy_rng)
        torch.set_rng_state(torch_rng)


def load_reference(seed):
    raw = json.loads((previous.ROOT / f"seed{seed}.json").read_text())
    assert (raw["protocol"], raw["seed"], raw["complete"]) == (previous.PROTOCOL, seed, True)
    return raw


class Probe:
    def __init__(self, name, seed, router, report, output, reference):
        self.name, self.seed, self.router = name, seed, router
        self.report, self.output, self.reference = report, output, reference
        self.step = min(KWARGS["learning_starts"], CUT) if name == "full" else BLOCK_END
        self.state = None

    def __call__(self, state):
        step, selector = state["step"], state["adaptive_selector"]
        row = self.report["branches"][self.name]
        reference = self.reference["branches"][self.name]
        if step == END:
            counts = selector.half_counts.copy()
            assert np.all(counts > 0) and selector.block_count < selector.block_steps
            row["partial_tail"] = dict(start=selector.block_start_step, end=step,
                half_counts=counts.tolist(), half_mean_rewards=(selector.half_sums / counts[:, None]).tolist(),
                scale_before_block=selector.block_scale.tolist(), used_for_feedback=False)
            selector.maybe_finish_block(step, force=True)
        expected = [b for b in reference["block_history"] if b["end_step"] <= step]
        assert selector.block_history == expected, (self.seed, self.name, step, "training block reproduction")
        assert state["env"].get_ground_truth()["mode_history"] == [
            tuple(m) for m in reference["mode_history"] if m[0] <= step], (step, "mode history")
        if step == CUT:
            assert selector.current_operator is None
            self.state = base.clone_state(state)
            np.random.set_state(state["numpy_rng"])
            torch.set_rng_state(state["torch_rng"])
        if step in times():
            before = state["eval_env"]._total_steps
            try:
                measured = measure(state, self.seed, self.report["evaluation"])
            finally:
                self.report["completed_budget"]["evaluation_interactions"] += state["eval_env"]._total_steps - before
            row["measurements"].append(measured)
            if step in (BLOCK_END, END):
                old = next(m for m in reference["measurements"] if m["step"] == step)
                matched = bool(np.array_equal(measured["returns"], old["returns"]))
                row["reproduced_grids"][str(step)] = matched
                base.write_report(self.output, self.report)
                assert matched, (self.seed, self.name, step, "held-out grid reproduction")
        row["routing"] = self.router.state_dict()
        row["training_reproduced_through"] = step
        if step == END:
            assert "progress_score" not in selector.block_history[-1]
            row.update(complete=True, block_history=copy.deepcopy(selector.block_history),
                       mode_history=reference["mode_history"], selector_draws=copy.deepcopy(selector.draws))
        base.write_report(self.output, self.report)
        print(f"TEMPORAL seed={self.seed} {self.name} step={step} budget={self.report['completed_budget']}", flush=True)
        if step == END:
            return None
        self.step = min([t for t in (CUT, *times()) if t > step] + [(step // 5000 + 1) * 5000])
        return False


def contrast(full, no_bc, weights):
    assert full["step"] == no_bc["step"] and full["modes"] == no_bc["modes"]
    mixed = (weights[:, 2] > 0) & (weights[:, 2] < 1)
    pure = weights[:, 2] == 1
    values = [(np.asarray(r["returns"])[0] * weights).sum(axis=-1) for r in (full, no_bc)]
    delta = values[1] - values[0]
    scaled_terms = [np.asarray(r["native_episode_returns"])[0].mean(axis=1) * r["objective2_scales"][0]
                    for r in (full, no_bc)]
    native = ((scaled_terms[1] - scaled_terms[0]) * weights[:, 2, None])[mixed].mean(axis=0)
    eps = [np.asarray(r["episode_roundoff"])[0].mean(axis=1) for r in (full, no_bc)]
    rounding = float(((eps[1] - eps[0]) * weights[:, 2])[mixed].mean())
    objectives = ((np.asarray(no_bc["returns"])[0] - full["returns"][0]) * weights)[mixed].mean(axis=0)
    np.testing.assert_allclose(native.sum() + rounding, objectives[2], rtol=1e-10, atol=1e-8)
    return dict(step=full["step"], full_primary=float(values[0][mixed].mean()),
        no_bc_primary=float(values[1][mixed].mean()), delta=float(delta[mixed].mean()),
        pure_objective2_delta=float(delta[pure][0]),
        mixed_positive_weights=int((delta[mixed] > 0).sum()), mixed_negative_weights=int((delta[mixed] < 0).sum()),
        weighted_objective_contributions=objectives.tolist(),
        weighted_native_objective2_contributions=dict(zip(TERMS, native.tolist())),
        objective2_roundoff_contribution=rounding, mode0_delta_by_weight=delta.tolist())


def summarize_seed(report):
    assert report["protocol"] == PROTOCOL and tuple(report["branches"]) == BRANCHES
    weights = np.asarray(report["evaluation"]["evaluation_weights"])
    evaluation_steps = 0
    for name, row in report["branches"].items():
        assert row["complete"] and row["training_reproduced_through"] == END
        assert row["reproduced_grids"] == {str(t): True for t in (BLOCK_END, END)}
        assert [r["step"] for r in row["measurements"]] == list(times())
        tail = row["partial_tail"]
        assert tail["end"] == END and not tail["used_for_feedback"]
        assert sum(tail["half_counts"]) == END - tail["start"] + 1
        assert tail["half_counts"] == row["block_history"][-1]["half_counts"]
        routing = row["routing"]
        assert routing["intervention_updates"] == BLOCK_END - CUT
        assert routing["generated_rows_each"] == routing["effective_loss_rows_each"] == 3 * KWARGS["batch_size"] * (BLOCK_END - CUT)
        if name == "no_bc":
            assert routing["rng_matched_steps"] == END - CUT
        for measured in row["measurements"]:
            assert measured["modes"] == ([0, 1, 2] if measured["step"] in (BLOCK_END, END) else [0])
            lengths = np.asarray(measured["episode_steps"])
            assert lengths.shape == (len(measured["modes"]), 21, 3) and np.all((lengths > 0) & (lengths <= 1000))
            assert int(lengths.sum()) == measured["evaluation_interactions"]
            evaluation_steps += measured["evaluation_interactions"]
    budget = report["completed_budget"]
    assert budget["evaluation_interactions"] == evaluation_steps
    for key, bound in planned_budget().items():
        assert (budget[key] <= bound if key == "evaluation_interactions" else budget[key] == bound), key
    timeline = [contrast(a, b, weights) for a, b in zip(
        report["branches"]["full"]["measurements"], report["branches"]["no_bc"]["measurements"])]
    first = next((i for i, row in enumerate(timeline) if row["delta"] < 0), None)
    return dict(protocol=PROTOCOL, seed=report["seed"], complete=True, validation_passed=True,
        completed_budget=budget, timeline=timeline,
        first_negative_interval=[timeline[first - 1]["step"], timeline[first]["step"]] if first is not None and first > 0 else None,
        negative_at_first_measurement=first == 0,
        mode_history=report["branches"]["full"]["mode_history"],
        partial_tail={name: row["partial_tail"] for name, row in report["branches"].items()},
        reproduced_grids={name: row["reproduced_grids"] for name, row in report["branches"].items()}, scope=report["scope"])


def run(seed, root=ROOT):
    assert seed in SEEDS
    root = Path(root)
    output = root / f"seed{seed}.json"
    if output.exists():
        saved = json.loads(output.read_text())
        assert (saved["protocol"], saved["seed"]) == (PROTOCOL, seed)
        if saved.get("complete"):
            base.write_report(root / f"seed{seed}_readout.json", summarize_seed(saved))
            print(f"DONE {PROTOCOL} seed={seed} reusing complete report", flush=True)
            return saved
        raise RuntimeError("Incomplete attempt preserved; inspect before a new-root retry")
    reference = load_reference(seed)
    root.mkdir(parents=True, exist_ok=True)
    report = dict(protocol=PROTOCOL, seed=seed, complete=False, evaluation=reference["evaluation"],
        window=[CUT, BLOCK_END], times=list(times()), planned_budget_upper_bound=planned_budget(),
        completed_budget=dict.fromkeys(planned_budget(), 0),
        branches={name: dict(complete=False, measurements=[], reproduced_grids={}) for name in BRANCHES},
        scope="Retrospective temporal diagnosis of the frozen full/no_bc pairs, not a new policy or independent "
              "confirmation. Full training and no_bc replay must reproduce historical blocks and four endpoint "
              "grids. Intermediate mode0 evaluations and native reward sums never enter training or selection.")
    base.write_report(output, report)
    state, trace = None, []
    try:
        for name in BRANCHES:
            branch = None if name == "full" else previous.replay_state(
                base.clone_state(state), reference["branches"]["full"]["block_history"])
            router = base.ComponentRouter("shared" if name == "full" else name, trace, report["completed_budget"],
                cut=CUT, block_end=BLOCK_END, learning_starts=KWARGS["learning_starts"])
            probe = Probe(name, seed, router, report, output, reference)
            report["completed_budget"]["environment_initialization_probe_steps"] += 2
            try:
                with patch.object(training, "OwnPreferenceBothProgressExtremeOperatorSelector", previous.WindowSelector):
                    training.train_method(base.ENV, base.METHOD, END, 25000, seed, str(root / f"seed{seed}" / name),
                        resume_state=branch, state_probe=probe, update_router=router, archive_steps=[END], **KWARGS)
            finally:
                router.close()
                if probe.state is not None:
                    state = probe.state
                if branch is not None:
                    branch["env"].close()
                    branch["eval_env"].close()
                base.write_report(output, report)
            assert state is not None and len(trace) == END - CUT
            print(f"BRANCH_DONE {PROTOCOL} seed={seed} {name}", flush=True)
            del branch, router, probe
        result = summarize_seed(report)
        report["complete"] = True
        base.write_report(output, report)
        base.write_report(root / f"seed{seed}_readout.json", result)
        print(f"DONE {PROTOCOL} seed={seed} {json.dumps(report['completed_budget'])}", flush=True)
        return report
    except Exception as exc:
        report["error"] = f"{type(exc).__name__}: {exc}"
        base.write_report(output, report)
        raise
    finally:
        if state is not None:
            state["env"].close()
            state["eval_env"].close()


def summarize(root=ROOT):
    rows = [json.loads((root / f"seed{s}_readout.json").read_text()) for s in SEEDS]
    for seed, row in zip(SEEDS, rows):
        assert (row["protocol"], row["seed"], row["complete"], row["validation_passed"]) == (PROTOCOL, seed, True, True)
    return dict(protocol=PROTOCOL, seeds=list(SEEDS), complete=True, runs=rows,
        completed_budget={k: sum(r["completed_budget"][k] for r in rows) for k in planned_budget()},
        mean_delta_by_step=[dict(step=step, mean=float(np.mean([r["timeline"][i]["delta"] for r in rows])))
                           for i, step in enumerate(times())])


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, choices=SEEDS)
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--summarize", action="store_true")
    args = parser.parse_args()
    if args.summarize:
        base.write_report(args.root / "readout.json", summarize(args.root))
    else:
        assert args.seed is not None
        run(args.seed, args.root)
