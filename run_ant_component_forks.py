"""Matched-draw, fixed-schedule component diagnosis of Ant seed1405 block41."""

import argparse
import copy
import json
from pathlib import Path
from types import MethodType

import numpy as np
import torch

import diagnose_ant_shared_block as diagnosis
import run_ant_shared_credit as previous
from run_nonterminal_rng_contrasts import history_check
from run_terminal_component_forks import clone_state, write_report


training = previous.training
PROTOCOL = "ant_component_forks_v1"
ROOT = Path("results/iclr_ant_component_forks_20260927")
SEED, CUT, BLOCK_END, END = 1405, 65000, 66500, 75000
ENV, METHOD = previous.previous.ENV, previous.previous.METHODS["base_adaptive"]
KWARGS = previous.previous.pilot.TRAIN_KWARGS
BRANCHES = ("per_label", "shared", "no_first_relabel", "no_extra_copy_loss", "no_bc")
B = "extreme:edge_relabel_envelope_policy_bc"


def planned_budget():
    return dict(training_interactions=115000, learner_updates=110001,
                evaluation_interactions=11 * 3 * 21 * 3 * 1000,
                environment_initialization_probe_steps=10)


class FixedSharedSchedule(previous.SharedCreditOwnPreferenceSelector):
    def select(self):
        drawn = super().select()
        index = len(self.block_history)
        expected = self.schedule[index]
        self.draws.append(dict(block=index + 1, drawn=drawn, executed=expected))
        if self.require_native_draws:
            assert drawn == expected, (index, drawn, expected)
        # Consume the ordinary draw before applying the diagnostic intervention.
        return self._force_choice(expected)


def shared_state(state, reference, require_native_draws):
    blocks = reference["adaptive_operator_final"]["block_history"]
    count = len(state["adaptive_selector"].block_history)
    prefix = diagnosis.prefix_selector(blocks, count, shared=True)
    selector = FixedSharedSchedule(3)
    selector.__dict__.update(copy.deepcopy(state["adaptive_selector"].__dict__))
    assert selector.counts == prefix.counts and selector.step_counts == prefix.step_counts
    selector.scores = prefix.scores
    selector.block_history = copy.deepcopy(blocks[:count])
    selector.schedule = [row["operator"] for row in blocks]
    selector.require_native_draws, selector.draws = require_native_draws, []
    state["adaptive_selector"] = selector
    return state


class ComponentRouter:
    def __init__(self, name, trace, budget, cut=CUT, block_end=BLOCK_END, learning_starts=5000):
        self.name, self.trace, self.budget = name, trace, budget
        self.cut, self.block_end, self.learning_starts = cut, block_end, learning_starts
        self.patches, self.trainer = {}, None
        self.matched_steps = self.intervention_updates = 0
        self.generated_rows = self.effective_loss_rows = 0

    def install(self, trainer, name, function):
        assert name not in trainer.__dict__
        original = getattr(trainer, name)
        self.patches[name] = original
        setattr(trainer, name, MethodType(function(original), trainer))

    def configure(self, step, trainer):
        inside = self.cut < step <= self.block_end
        if inside and self.name not in ("per_label",):
            assert (trainer.critic_relabel_copies, trainer.preference_relabeling_rate,
                    trainer.behavior_clone_coef, trainer.envelope_policy_candidates,
                    trainer.envelope_target_candidates) == (3, 1., .004, 1, 1)
            assert not trainer.reward_normalization and not trainer.dual_branch_update
            self.intervention_updates += 1
            self.generated_rows += 3 * trainer.batch_size
            first_copy_only = self.name in ("no_extra_copy_loss", "no_extra_copy_loss_no_bc")
            self.effective_loss_rows += trainer.batch_size * (1 if first_copy_only else 3)
            if self.name in ("no_bc", "no_extra_copy_loss_no_bc"):
                trainer.behavior_clone_coef = 0.
            if self.trainer is None:
                self.trainer = trainer
                if self.name == "no_first_relabel":
                    def discard(original):
                        def relabel(_self, w, c, c_next):
                            original(w, c, c_next)
                            return w, c, c_next
                        return relabel
                    self.install(trainer, "relabel_preferences", discard)
                elif first_copy_only:
                    for name in ("critic_objective", "actor_objective", "behavior_clone_loss"):
                        self.install(trainer, name, lambda original:
                            lambda _self, *args: original(*(x[:_self.batch_size] for x in args)))
        elif self.trainer is not None:
            self.close()

    def after_step(self, step, trainer, encoder, env, operator):
        self.budget["training_interactions"] += 1
        self.budget["learner_updates"] += int(step >= self.learning_starts)
        if step <= self.cut or self.name == "per_label":
            return
        if step <= self.block_end:
            assert operator == B, (step, operator)
        actual = (np.random.get_state(), torch.get_rng_state())
        if self.name == "shared":
            self.trace.append(actual)
        else:
            expected = self.trace[step - self.cut - 1]
            assert actual[0][0] == expected[0][0] and actual[0][2:] == expected[0][2:], step
            assert np.array_equal(actual[0][1], expected[0][1]), (step, "numpy")
            assert torch.equal(actual[1], expected[1]), (step, "torch")
            self.matched_steps += 1

    def interaction_step(self, step):
        return step

    def state_dict(self):
        return dict(intervention_updates=self.intervention_updates,
                    generated_rows_each=self.generated_rows, effective_loss_rows_each=self.effective_loss_rows,
                    rng_matched_steps=self.matched_steps, extra_updates=0, extra_environment_steps=0)

    def close(self):
        for name in self.patches:
            self.trainer.__dict__.pop(name)
        self.patches.clear()
        self.trainer = None


def measure(state, seed=SEED):
    env = state["eval_env"]
    before, numpy_rng, torch_rng = env._total_steps, np.random.get_state(), torch.get_rng_state()
    try:
        result = training.evaluate_locked_modes(env, lambda weights, seeds:
            training.evaluate_pareto_v2(env, state["trainer"], METHOD, state["encoder"],
                                       weights, episode_seeds=seeds),
            seed, training.reference_point_for_env(ENV, env.reward_dim))
    finally:
        np.random.set_state(numpy_rng)
        torch.set_rng_state(torch_rng)
    mixed_utility = previous.previous.actor.mixed_energy_utility(result)
    returns = np.asarray([[point["return"] for point in front] for front in result.pop("pareto_per_mode")])
    weights = np.asarray(result["evaluation_weights"])
    utility = (returns * weights).sum(axis=-1)
    result.update(step=state["step"], returns=returns.tolist(),
                  direct_utility_per_mode=utility.mean(axis=1).tolist(),
                  mode0_mixed_energy_utility=mixed_utility,
                  objective_means_per_mode=returns.mean(axis=1).tolist(),
                  evaluation_interactions=env._total_steps - before)
    return result


class Probe:
    def __init__(self, name, router, report, output, reference=None):
        self.name, self.router, self.report, self.output, self.reference = name, router, report, output, reference
        self.times = [CUT, BLOCK_END, END] if name == "per_label" else [BLOCK_END, END]
        self.step, self.state = (5000 if name == "per_label" else BLOCK_END), None

    def __call__(self, state):
        step, selector = state["step"], state["adaptive_selector"]
        if step == END:
            selector.maybe_finish_block(step, force=True)
        row = self.report["branches"][self.name]
        if step in self.times:
            if self.reference is not None:
                check = history_check({**state["log_data"], "ground_truth": state["env"].get_ground_truth()},
                                      selector.block_history, self.reference, step)
                row.setdefault("reproduction", {})[str(step)] = check
                write_report(self.output, self.report)
                assert check["passed"], (self.name, step, check)
            if step == CUT:
                self.state = clone_state(state)
                np.random.set_state(state["numpy_rng"])
                torch.set_rng_state(state["torch_rng"])
            before = state["eval_env"]._total_steps
            try:
                measurement = measure(state)
            finally:
                self.report["completed_budget"]["evaluation_interactions"] += state["eval_env"]._total_steps - before
            row["measurements"].append(measurement)
            for key, value in self.report["evaluation"].items():
                assert measurement[key] == value, key
            if step == END and self.reference is not None:
                expected = [[p["return"] for p in front] for front in self.reference["pareto_per_mode"]]
                matched = bool(np.array_equal(measurement["returns"], expected))
                row["reproduction"][str(step)]["final_returns_matched"] = matched
                assert matched, (self.name, "endpoint returns")
            row.update(block_history=copy.deepcopy(selector.block_history),
                       mode_history=state["env"].get_ground_truth()["mode_history"],
                       selector_draws=getattr(selector, "draws", []))
            if step == END:
                assert np.array_equal(row["mode_history"],
                                      self.report["branches"]["per_label"]["mode_history"])
        row["routing"] = self.router.state_dict()
        if step == END:
            row["complete"] = True
        write_report(self.output, self.report)
        print(f"FORK {self.name} step={step} budget={self.report['completed_budget']}", flush=True)
        if step == END:
            return None
        self.step = min([t for t in self.times if t > step] + [(step // 5000 + 1) * 5000])
        return False


def contrasts(report):
    output = {}
    for name in BRANCHES[1:]:
        baseline = "per_label" if name == "shared" else "shared"
        pairs = zip(report["branches"][name]["measurements"],
                    report["branches"][baseline]["measurements"][-2:])
        output[f"{name}_minus_{baseline}"] = [dict(step=a["step"],
            mode0_mixed_energy_utility=a["mode0_mixed_energy_utility"] - b["mode0_mixed_energy_utility"],
            direct_utility_per_mode=(np.asarray(a["direct_utility_per_mode"]) - b["direct_utility_per_mode"]).tolist(),
            hv_per_mode=(np.asarray(a["hv_per_mode"]) - b["hv_per_mode"]).tolist(),
            objective_means_per_mode=(np.asarray(a["objective_means_per_mode"]) - b["objective_means_per_mode"]).tolist())
            for a, b in pairs]
    return output


def run(root=ROOT):
    output = Path(root) / f"seed{SEED}.json"
    if output.exists():
        saved = json.loads(output.read_text())
        assert saved["protocol"] == PROTOCOL
        if saved.get("complete"):
            print(f"DONE {PROTOCOL} reusing complete report", flush=True)
            return saved
        raise RuntimeError("Incomplete attempt preserved; inspect before a new-root retry")
    references = {name: json.loads((previous.ROOT / name / f"seed{SEED}" /
                  f"{METHOD}_seed{SEED}.json").read_text()) for name in BRANCHES[:2]}
    diagnosis.common_prefix(references, diagnosis.INDEX)
    output.parent.mkdir(parents=True, exist_ok=True)
    report = dict(protocol=PROTOCOL, seed=SEED, env=ENV, window=[CUT, BLOCK_END], end_step=END,
        complete=False, controls_passed=False, planned_budget_upper_bound=planned_budget(),
        completed_budget=dict.fromkeys(planned_budget(), 0),
        evaluation={k: references["per_label"][k] for k in ("evaluation_weights", "episode_seeds_per_mode", "ref_point")},
        branches={name: dict(complete=False, measurements=[]) for name in BRANCHES},
        scope="One outcome-selected seed, fixed historical shared schedule after the common65k state. "
              "Three leave-one-component-out interventions in block41, not a factorial or deployed selector. "
              "No-first-relabel retains the two extra relabeled copies; no-extra-copy-loss retains expanded "
              "forwards and draws but only first-copy gradients. Shared and ablations match learner RNG, "
              "not environment trajectories. Current policy/encoder; evaluation never enters selection.")
    write_report(output, report)
    state, trace = None, []
    try:
        for name in BRANCHES:
            branch = None if name == "per_label" else clone_state(state)
            if branch is not None:
                shared_state(branch, references["shared"], require_native_draws=name == "shared")
            router = ComponentRouter(name, trace, report["completed_budget"])
            probe = Probe(name, router, report, output, references.get(name))
            report["completed_budget"]["environment_initialization_probe_steps"] += 2
            try:
                training.train_method(ENV, METHOD, END, 25000, SEED, str(output.parent / name),
                    state_probe=probe, update_router=router, resume_state=branch, archive_steps=[END], **KWARGS)
            finally:
                router.close()
                if branch is not None:
                    branch["env"].close()
                    branch["eval_env"].close()
                if probe.state is not None:
                    state = probe.state
                write_report(output, report)
            if name == "shared":
                assert len(trace) == END - CUT
                report["controls_passed"] = True
            elif name not in BRANCHES[:2]:
                assert router.matched_steps == END - CUT
            print(f"BRANCH_DONE {PROTOCOL} {name}", flush=True)
            del branch, probe, router
        for key, bound in planned_budget().items():
            actual = report["completed_budget"][key]
            assert (actual <= bound if key == "evaluation_interactions" else actual == bound), (key, actual)
        report.update(complete=True, contrasts=contrasts(report))
        write_report(output, report)
        print(f"DONE {PROTOCOL} {json.dumps(report['completed_budget'])}", flush=True)
        return report
    except Exception as exc:
        report["error"] = f"{type(exc).__name__}: {exc}"
        write_report(output, report)
        raise
    finally:
        if state is not None:
            state["env"].close()
            state["eval_env"].close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT)
    run(parser.parse_args().root)
