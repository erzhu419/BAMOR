"""Four terminal continuations from one reproduced, full v25 training state."""

import argparse
import copy
import json
from pathlib import Path

import numpy as np
import torch

import run_progress_factorial_completion as previous
from diagnose_mean_action_gradient import fork_env


training = previous.previous.pilot.training
PROTOCOL = "terminal_component_forks_v1"
SEED, CUT, END = 1004, 48500, 50000
EXTREME = "extreme:edge_envelope_policy"
RECOVERY = "recovery:stratified_relabel_envelope"
BRANCHES = ("factual", "recovery", "actor_one", "relabel_one")
OVERRIDES = {"factual": {}, "recovery": {},
             "actor_one": {"envelope_policy_candidates": 1},
             "relabel_one": {"preference_relabeling_rate": 1.0}}
ROOT = Path("results/iclr_terminal_component_forks_20260921")
KWARGS = previous.previous.pilot.TRAIN_KWARGS


def clone_state(state):
    env, eval_env = fork_env(state["env"]), fork_env(state["eval_env"])
    return copy.deepcopy(state, {id(state["env"]): env, id(state["eval_env"]): eval_env})


class CaptureState:
    def __init__(self, step):
        self.step, self.state = step, None

    def __call__(self, state):
        self.state = clone_state(state)


def adaptive_state(state):
    selector = training.ProgressExtremeHierarchicalOperatorSelector(2)
    selector.__dict__.update(state["adaptive_selector"].__dict__)
    state["adaptive_selector"] = selector
    return state


class TerminalIntervention:
    def __init__(self, branch, start=CUT + 1, end=END, expected_operator=None):
        self.branch, self.start, self.end = branch, start, end
        self.expected_operator = expected_operator
        self.records, self.handles, self.config = [], [], None
        self.step, self.pending = 0, {}

    def configure(self, step, trainer):
        self.step = step
        for key, value in OVERRIDES[self.branch].items():
            setattr(trainer, key, value)
        if self.config is None:
            keys = ("preference_relabeling_rate", "preference_relabeling_strategy", "critic_relabel_copies",
                    "envelope_target_candidates", "vector_critic_coef", "envelope_policy_candidates", "behavior_clone_coef")
            self.config = {k: getattr(trainer, k) for k in keys}
            for name, optimizer in (("critic", trainer.q_opt), ("actor", trainer.policy_opt)):
                params = [p for group in optimizer.param_groups for p in group["params"]]
                initial = [p.detach().clone() for p in params]

                def record(opt, args, kwargs, name=name, params=params, initial=initial):
                    if self.step in (self.start, self.end) or self.step % 100 == 0:
                        self.pending[name] = {
                            "gradient_norm": float(torch.sqrt(sum(p.grad.detach().square().sum() for p in params if p.grad is not None))),
                            "parameter_distance_from_fork": float(torch.sqrt(sum((p.detach() - q).square().sum() for p, q in zip(params, initial))))}

                # Critic gradients must be read before actor backward adds to them.
                self.handles.append(optimizer.register_step_post_hook(record))
        assert self.config == {k: getattr(trainer, k) for k in self.config}

    def after_step(self, step, trainer, encoder, env, operator):
        if self.expected_operator is not None:
            assert operator == self.expected_operator, (step, operator)
        if self.pending:
            self.records.append({"step": step, "operator": operator, **self.pending})
            self.pending = {}

    def interaction_step(self, step):
        return step

    def state_dict(self):
        return {"protocol": PROTOCOL, "branch": self.branch, "overrides": OVERRIDES[self.branch],
                "config": self.config, "gradient_records": self.records,
                "extra_environment_steps": 0, "extra_updates": 0}

    def close(self):
        for handle in self.handles:
            handle.remove()


def terminal_returns(data):
    return np.asarray([[p["return"] for p in front if p["archive_step"] == END]
                       for front in data["pareto_per_mode"]])


def reproduction(actual, reference):
    for key in ("evaluation_weights", "episode_seeds_per_mode", "ref_point"):
        assert actual[key] == reference[key], key
    schedule = bool(np.array_equal(actual["ground_truth"]["mode_history"], reference["ground_truth"]["mode_history"]))
    a, b = actual["episode_rewards"], reference["episode_rewards"]
    assert [r["step"] for r in a] == [r["step"] for r in b]
    episode_error = float(np.max(np.abs(np.asarray([r["return"] for r in a]) - np.asarray([r["return"] for r in b]))))
    returns = terminal_returns(actual)
    expected = terminal_returns(reference)
    assert returns.shape == expected.shape == (4, 21, 2)
    return_error = float(np.max(np.abs(returns - expected)))
    blocks = actual["adaptive_operator_final"]["block_history"] == reference["adaptive_operator_final"]["block_history"]
    return {"schedule_matched": schedule, "block_records_matched": blocks,
            "max_episode_return_error": episode_error, "max_terminal_return_error": return_error,
            "passed": schedule and blocks and episode_error <= 1e-6 and return_error <= 1e-6}


def prefix_check(state, references):
    episodes = state["log_data"]["episode_rewards"]
    blocks = state["adaptive_selector"].block_history
    for reference in references.values():
        assert episodes == [r for r in reference["episode_rewards"] if r["step"] <= CUT]
        assert blocks == reference["adaptive_operator_final"]["block_history"][:29]
        assert np.array_equal(state["env"].get_ground_truth()["mode_history"],
                              [h for h in reference["ground_truth"]["mode_history"] if h[0] <= CUT])
    assert state["trainer"].buffer.size == CUT and state["episode_steps"] == 500
    return {"matched": True, "complete_blocks": len(blocks), "complete_episodes": len(episodes),
            "partial_episode_steps": state["episode_steps"], "replay_size": state["trainer"].buffer.size}


def write_report(path, report):
    path.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")


def run(root=ROOT):
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    output = root / f"seed{SEED}.json"
    if output.exists():
        saved = json.loads(output.read_text())
        if saved.get("complete"):
            assert saved["protocol"] == PROTOCOL and saved["controls_passed"]
            print("DONE terminal_component_forks reusing complete report", flush=True)
            return saved
        raise RuntimeError("Incomplete attempt preserved; review it and use a new --root for a retry")
    references = {
        "factual": json.loads((previous.OUTPUT_ROOT / f"{previous.METHOD}_seed{SEED}.json").read_text()),
        "recovery": json.loads((previous.previous.OUTPUT_ROOT / f"{previous.previous.METHODS['legacy']}_seed{SEED}.json").read_text())}
    report = {"protocol": PROTOCOL, "seed": SEED, "fork_step": CUT, "end_step": END,
              "complete": False, "controls_passed": False, "branches": {},
              "evaluation": {k: references["factual"][k] for k in
                             ("evaluation_weights", "episode_seeds_per_mode", "ref_point")},
              "completed_budget": {"training_interactions": 0, "evaluation_interactions": 0,
                                   "environment_initialization_probe_steps": 0, "learner_updates": 0}}
    write_report(output, report)
    capture = CaptureState(CUT)
    training.train_method("mo-halfcheetah-v5", previous.METHOD, END, 25000, SEED,
                          str(root / "prefix"), state_probe=capture, **KWARGS)
    state = capture.state
    budget = report["completed_budget"]
    budget.update(training_interactions=CUT, environment_initialization_probe_steps=2,
                  learner_updates=CUT - KWARGS["learning_starts"] + 1)
    write_report(output, report)
    try:
        report["prefix_reproduction"] = prefix_check(state, references)
        write_report(output, report)
        for name in BRANCHES:
            branch = clone_state(state)
            branch["policy_archive"] = []
            if name == "recovery":
                adaptive_state(branch)
            method = previous.previous.METHODS["legacy"] if name == "recovery" else previous.METHOD
            router = TerminalIntervention(name, expected_operator=RECOVERY if name == "recovery" else EXTREME)
            try:
                result = training.train_method("mo-halfcheetah-v5", method, END, 25000, SEED,
                    str(root / name), resume_state=branch, update_router=router, **KWARGS)
            finally:
                router.close()
            assert result["continuation_from_step"] == CUT
            assert result["training_interactions_this_call"] == END - CUT
            assert result["evaluation_interactions"] == 4 * 21 * 3 * 1000
            assert [s["step"] for s in result["policy_archive_snapshots"]] == [END]
            for key, value in report["evaluation"].items():
                assert result[key] == value, key
            budget["training_interactions"] += END - CUT
            budget["evaluation_interactions"] += result["evaluation_interactions"]
            budget["environment_initialization_probe_steps"] += result["environment_initialization_probe_steps"]
            budget["learner_updates"] += END - CUT
            row = {"final_hv": result["final_hv_mean_mode"], "final_mode_hv": result["final_hv_per_mode"],
                   "last_episodes": result["episode_rewards"][-2:], "update_routing": result["update_routing"],
                   "terminal_returns": terminal_returns(result).tolist()}
            report["branches"][name] = row
            if name in references:
                row["reproduction"] = reproduction(result, references[name])
                write_report(output, report)
                if not row["reproduction"]["passed"]:
                    raise RuntimeError(f"{name} control did not reproduce; component branches are not interpretable")
            report["controls_passed"] = all(c in report["branches"] and report["branches"][c]["reproduction"]["passed"] for c in references)
            write_report(output, report)
            print(f"BRANCH_DONE {name} final_hv={row['final_hv']:.6f}", flush=True)
        report["paired_minus_factual"] = {name: {
            "final_hv": row["final_hv"] - report["branches"]["factual"]["final_hv"],
            "final_mode_hv": (np.asarray(row["final_mode_hv"]) - report["branches"]["factual"]["final_mode_hv"]).tolist()}
            for name, row in report["branches"].items() if name != "factual"}
        report["complete"] = True
        report["scope"] = "One observed seed; local component diagnosis, not a new algorithm or independent confirmation. " \
                          "Equal gradient-step counts, not equal FLOPs. No archive selection. Failed/restarted attempts " \
                          "must be counted separately from this completed-attempt budget. Partial attempts remain " \
                          "in their original output root; an explicit retry reruns the prefix in a new root."
        write_report(output, report)
        print(f"DONE terminal_component_forks {json.dumps(budget)}", flush=True)
        return report
    finally:
        state["env"].close()
        state["eval_env"].close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=ROOT)
    args = parser.parse_args()
    run(args.root)
