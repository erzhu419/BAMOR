"""One BC-arm pulse followed by live, identical-rule adaptive continuation."""

import argparse
import copy
import json
from pathlib import Path

import numpy as np
import torch

import run_scoped_arm_factorial as previous


PROTOCOL = "scoped_bc_credit_v1"
ROOT = Path("results/iclr_scoped_bc_credit_20260922")
ENV, END, WARMUP, BLOCK = "mo-hopper-v5", 50000, 20000, 1500
CASES = {1013: ("envelope_only", 32000), 1012: ("legacy_adaptive", 24500)}
base, training = previous.base, previous.training
forks, earlier = base.scoped.forks, base.scoped.earlier
METHOD, E, B = previous.METHOD, *base.scoped.TARGETS
KWARGS = base.common.pilot.TRAIN_KWARGS


def measurement_steps(cut):
    return (cut, cut + BLOCK, cut + 2 * BLOCK, 40000, END)


def planned_budget(cut):
    return dict(training_interactions=2 * END - cut,
                learner_updates=2 * END - cut - KWARGS["learning_starts"] + 1,
                evaluation_interactions=9 * 4 * 21 * 3 * 1000,
                environment_initialization_probe_steps=4)


class PulseRouter(base.IntegratedRouter):
    def __init__(self, background, pulse, window, selector=None, warmup=WARMUP, end=END):
        assert background in ("envelope_only", "legacy_adaptive")
        super().__init__("scoped_adaptive", cut=warmup, end=end, target_operators=())
        self.background, self.pulse, self.window = background, pulse, window
        self.selector = selector

    def configure(self, step, trainer):
        inside = self.window[0] < step <= self.window[1]
        if inside:
            assert self.selector.current_operator == B, "Frozen intervention is not a BC block"
        self.target_operators = ((E,) if self.background == "envelope_only" else ()) + (
            (B,) if self.pulse and inside else ())
        super().configure(step, trainer)

    def state_dict(self):
        result = super().state_dict()
        result.update(protocol=PROTOCOL, background=self.background, pulse=self.pulse,
                      window=list(self.window),
                      pulse_updates=sum(r["steps"] for r in self.records
                                        if r["operator"] == B and r["own_selected"]))
        result.pop("targeted_updates")
        result.pop("target_operators")
        return result


def measure(state, seed):
    env = state["eval_env"]
    before = env._total_steps
    numpy_rng, torch_rng = np.random.get_state(), torch.get_rng_state()
    try:
        result = training.evaluate_locked_modes(env, lambda weights, seeds:
            training.evaluate_pareto_v2(env, state["trainer"], METHOD, state["encoder"],
                                       weights, episode_seeds=seeds),
            seed, training.reference_point_for_env(ENV, env.reward_dim))
    finally:
        np.random.set_state(numpy_rng)
        torch.set_rng_state(torch_rng)
    returns = np.asarray([[point["return"] for point in front]
                          for front in result.pop("pareto_per_mode")])
    weights = np.asarray(result["evaluation_weights"])
    result.update(step=state["step"], returns=returns.tolist(),
                  matched_utility_per_mode=(returns * weights).sum(axis=-1).mean(axis=-1).tolist(),
                  objective_means_per_mode=returns.mean(axis=1).tolist(),
                  evaluation_interactions=env._total_steps - before,
                  encoder="contemporaneous", policy="current_only")
    return result


class TrajectoryProbe:
    def __init__(self, router, report, output, branch, cut, times, reference=None,
                 warmup=WARMUP, learning_starts=KWARGS["learning_starts"], end=END):
        self.router, self.report, self.output = router, report, output
        self.branch, self.cut, self.reference = branch, cut, reference
        self.learning_starts, self.end = learning_starts, end
        self.times = tuple(times)
        self.step = warmup if branch == "native" else times[1]
        self.state = None

    def __call__(self, state):
        step, budget = state["step"], self.report["completed_budget"]
        spent = step if self.branch == "native" else self.end + step - self.cut
        budget.update(training_interactions=spent,
                      learner_updates=spent - self.learning_starts + 1)
        self.router.selector = state["adaptive_selector"]
        if self.branch == "native" and self.reference is not None:
            log = {**state["log_data"], "ground_truth": state["env"].get_ground_truth()}
            check = earlier.history_check(log, self.router.selector.block_history, self.reference, step)
            self.report["control_reproduction"][str(step)] = check
            forks.write_report(self.output, self.report)
            assert check["passed"], "Factual trajectory differs from the selected reference"
        if step == self.cut and self.branch == "native":
            self.state = forks.clone_state(state)
            # The clone gets its own router; never retain an instance-bound loss patch.
            self.state["trainer"].__dict__.pop("envelope_actor_loss", None)
            np.random.set_state(state["numpy_rng"])
            torch.set_rng_state(state["torch_rng"])
        if step in self.times:
            row = measure(state, self.report["seed"])
            budget["evaluation_interactions"] += row["evaluation_interactions"]
            row["feedback"] = copy.deepcopy(self.router.selector.block_history)
            row["scores"] = dict(self.router.selector.scores)
            self.report["branches"][self.branch]["measurements"].append(row)
            if step == self.end and self.reference is not None:
                matched = bool(np.array_equal(row["returns"], forks.terminal_returns(self.reference)))
                self.report["control_reproduction"][str(step)]["final_returns_matched"] = matched
                forks.write_report(self.output, self.report)
                assert matched, "Current-policy evaluation differs from the reference endpoint"
        branch = self.report["branches"][self.branch]
        branch.update(routing=self.router.state_dict(),
                      mode_history=state["env"].get_ground_truth()["mode_history"])
        if step == self.end:
            branch["complete"] = True
        forks.write_report(self.output, self.report)
        print(f"CREDIT seed={self.report['seed']} branch={self.branch} step={step} budget={budget}", flush=True)
        if step == self.end:
            return None
        self.step = next(t for t in self.times if t > step)
        return False


def summarize(report):
    cut, end = report["window"]
    native, pulse = (report["branches"][name] for name in ("native", "pulse"))
    assert native["complete"] and pulse["complete"]
    assert np.array_equal(native["mode_history"], pulse["mode_history"])
    assert native["routing"]["pulse_updates"] == 0
    assert pulse["routing"]["pulse_updates"] == end - cut
    expected = list(measurement_steps(cut))
    assert [r["step"] for r in native["measurements"]] == expected
    assert [r["step"] for r in pulse["measurements"]] == expected[1:]
    contrasts = []
    for a, b in zip(pulse["measurements"], native["measurements"][1:]):
        for key in ("evaluation_weights", "episode_seeds_per_mode", "ref_point"):
            assert a[key] == b[key], key
        contrasts.append(dict(step=a["step"], hv_delta=a["hv_mean_mode"] - b["hv_mean_mode"],
            hv_relative_pct=100 * (a["hv_mean_mode"] / b["hv_mean_mode"] - 1)
                if b["hv_mean_mode"] else None,
            hv_delta_per_mode=(np.asarray(a["hv_per_mode"]) - b["hv_per_mode"]).tolist(),
            matched_utility_delta_per_mode=(np.asarray(a["matched_utility_per_mode"])
                                            - b["matched_utility_per_mode"]).tolist(),
            objective_mean_delta_per_mode=(np.asarray(a["objective_means_per_mode"])
                                           - b["objective_means_per_mode"]).tolist()))
    a, b = pulse["measurements"][0], native["measurements"][1]
    first = [next(r for r in row["feedback"] if r["end_step"] == end) for row in (a, b)]
    assert all(r["operator"] == B and r["complete"] for r in first)
    native_ops = [(r["end_step"], r["operator"]) for r in native["measurements"][-1]["feedback"]
                  if r["end_step"] > end]
    pulse_ops = [(r["end_step"], r["operator"]) for r in pulse["measurements"][-1]["feedback"]
                 if r["end_step"] > end]
    assert len(native_ops) == len(pulse_ops)
    mismatch = [x[0] for x, y in zip(native_ops, pulse_ops) if x != y]
    actual, planned = report["completed_budget"], report["planned_budget_upper_bound"]
    assert actual["evaluation_interactions"] == sum(row["evaluation_interactions"]
        for branch in (native, pulse) for row in branch["measurements"])
    for key in planned:
        if key == "evaluation_interactions":
            assert actual[key] <= planned[key]
        else:
            assert actual[key] == planned[key]
    return dict(contrasts=contrasts, block_progress_delta=first[0]["progress_score"] - first[1]["progress_score"],
                bc_score_delta_at_block_end=a["scores"][B] - b["scores"][B],
                subsequent_operator_mismatches=len(mismatch),
                first_subsequent_operator_mismatch=mismatch[0] if mismatch else None)


def load_reference(seed):
    background, cut = CASES[seed]
    root = previous.ROOT if background == "envelope_only" else previous.REFERENCE_ROOT
    path = root / ENV.replace("-", "_") / background / f"{METHOD}_seed{seed}.json"
    reference = json.loads(path.read_text())
    assert (reference["env"], reference["seed"], reference["method"], reference["total_steps"]) == (
        ENV, seed, METHOD, END)
    first = next(b for b in reference["adaptive_operator_final"]["block_history"]
                 if b["end_step"] > WARMUP and b["operator"] == B)
    assert first["end_step"] == cut + BLOCK and first["steps"] == BLOCK and first["complete"]
    return reference, path


def run(seed, root=ROOT):
    background, cut = CASES[seed]
    output = Path(root) / f"seed{seed}.json"
    if output.exists():
        saved = json.loads(output.read_text())
        assert (saved["protocol"], saved["seed"]) == (PROTOCOL, seed)
        if all(saved["branches"][name]["complete"] for name in ("native", "pulse")):
            saved.update(summary=summarize(saved), complete=True)
            forks.write_report(output, saved)
            print(f"DONE {PROTOCOL} seed={seed} reusing completed branches", flush=True)
            return saved
        raise RuntimeError("Incomplete attempt preserved; inspect its log before a new-root retry")
    reference, reference_path = load_reference(seed)
    output.parent.mkdir(parents=True, exist_ok=True)
    report = dict(protocol=PROTOCOL, seed=seed, env=ENV, background=background, window=[cut, cut + BLOCK],
        complete=False, reference=str(reference_path), measurement_steps=list(measurement_steps(cut)),
        planned_budget_upper_bound=planned_budget(cut),
        completed_budget=dict.fromkeys(planned_budget(cut), 0), control_reproduction={},
        branches={name: dict(complete=False, measurements=[]) for name in ("native", "pulse")},
        scope="Two outcome-selected development cases; one BC-arm own-candidate pulse, not a BC-loss-only "
              "change. Identical live adaptive continuation rule, not forced identical arms. Evaluation "
              "uses contemporaneous policy and encoder and never enters training or selection. Native "
              "feedback precedes the last update of its block. No archive/recovery or confirmation claim.")
    forks.write_report(output, report)
    router = PulseRouter(background, False, (cut, cut + BLOCK))
    probe = TrajectoryProbe(router, report, output, "native", cut, measurement_steps(cut), reference)
    try:
        report["completed_budget"]["environment_initialization_probe_steps"] += 2
        training.train_method(ENV, METHOD, END, 25000, seed, str(output.parent),
            state_probe=probe, update_router=router, archive_steps=[END], **KWARGS)
    finally:
        router.close()
    state = probe.state
    assert state is not None and state["step"] == cut
    router = PulseRouter(background, True, (cut, cut + BLOCK), selector=state["adaptive_selector"])
    pulse_probe = TrajectoryProbe(router, report, output, "pulse", cut, measurement_steps(cut))
    try:
        report["completed_budget"]["environment_initialization_probe_steps"] += 2
        training.train_method(ENV, METHOD, END, 25000, seed, str(output.parent),
            state_probe=pulse_probe, update_router=router, resume_state=state,
            archive_steps=[END], **KWARGS)
    finally:
        router.close()
        state["env"].close()
        state["eval_env"].close()
    report.update(summary=summarize(report), complete=True)
    forks.write_report(output, report)
    print(f"DONE {PROTOCOL} seed={seed} {json.dumps(report['summary'])}", flush=True)
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, choices=CASES, required=True)
    parser.add_argument("--root", type=Path, default=ROOT)
    args = parser.parse_args()
    run(args.seed, args.root)
