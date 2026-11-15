"""Fixed-teacher preference retention during one BC-arm intervention."""

import argparse
import copy
import json
from pathlib import Path
import time

import numpy as np
import torch

import run_scoped_bc_credit as credit
from summarize_scoped_bc_credit import preference_decomposition


PROTOCOL = "preference_retention_v1"
ROOT = Path("results/iclr_preference_retention_20260922")
CONDITIONS, COEF = ("coverage", "current"), 1.0
training, forks = credit.training, credit.forks


class RetentionRouter(credit.PulseRouter):
    def __init__(self, background, window, condition, selector=None, coefficient=COEF,
                 warmup=credit.WARMUP, end=credit.END):
        assert condition in CONDITIONS
        super().__init__(background, True, window, selector, warmup, end)
        self.retention_condition, self.coefficient = condition, coefficient
        self.teacher, self.native_objective = None, None
        self.loss_updates, self.action_pairs, self.mse_sum, self.forward_seconds = 0, 0, 0., 0.

    def configure(self, step, trainer):
        super().configure(step, trainer)
        inside = self.window[0] < step <= self.window[1]
        if inside and self.teacher is None:
            assert step == self.window[0] + 1
            assert "actor_objective" not in trainer.__dict__
            self.teacher = copy.deepcopy(trainer.policy).eval().requires_grad_(False)
            self.templates = torch.as_tensor(training.preference_expert_templates(trainer.reward_dim),
                                             device=trainer.device)
            idx = np.linspace(0, trainer.buffer.size - 1, min(64, trainer.buffer.size), dtype=int)
            self.audit_states = torch.as_tensor(trainer.buffer.states[idx].copy(), device=trainer.device)
            self.audit_contexts = torch.as_tensor(trainer.buffer.contexts[idx].copy(), device=trainer.device)
            self.native_objective = trainer.actor_objective
            trainer.actor_objective = self.objective
        elif not inside and self.native_objective is not None:
            del trainer.actor_objective
            self.native_objective = None

    def anchor_batch(self, s, c, coverage):
        count = len(self.templates)
        states, contexts = s.repeat_interleave(count, dim=0), c.repeat_interleave(count, dim=0)
        if coverage:
            prefs = self.templates.repeat(len(s), 1)
            contexts = self.trainer.replace_context_preferences(contexts, prefs)
        return states, contexts

    def objective(self, s, w, c, q_scalar, log_prob):
        base_loss = self.native_objective(s, w, c, q_scalar, log_prob)
        start = time.perf_counter()
        # Critic relabel copies repeat states; retain each original replay row once.
        states, contexts = self.anchor_batch(s[:self.trainer.batch_size], c[:self.trainer.batch_size],
                                             self.retention_condition == "coverage")
        with torch.no_grad():
            target = self.teacher.get_action(states, contexts, deterministic=True)
        action = self.trainer.policy.get_action(states, contexts, deterministic=True)
        mse = (action - target).square().mean()
        self.loss_updates += 1
        self.action_pairs += len(states)
        self.mse_sum += float(mse.detach())
        self.forward_seconds += time.perf_counter() - start
        return base_loss + self.coefficient * mse

    @torch.no_grad()
    def audit(self):
        states, contexts = self.anchor_batch(self.audit_states, self.audit_contexts, True)
        target = self.teacher.get_action(states, contexts, deterministic=True)
        actual = self.trainer.policy.get_action(states, contexts, deterministic=True)
        return dict(fixed_context_action_mse_per_template=(actual - target).square().mean(-1)
                    .reshape(len(self.audit_states), len(self.templates)).mean(0).tolist(),
                    replay_states=len(self.audit_states))

    def state_dict(self):
        result = super().state_dict()
        result.update(protocol=PROTOCOL, retention=dict(condition=self.retention_condition, coefficient=self.coefficient,
            templates=self.templates.tolist() if self.teacher is not None else None,
            loss_updates=self.loss_updates, action_pairs=self.action_pairs,
            mean_action_mse=self.mse_sum / max(1, self.loss_updates),
            auxiliary_forward_seconds=self.forward_seconds))
        return result

    def close(self):
        if self.native_objective is not None:
            del self.trainer.actor_objective
            self.native_objective = None
        super().close()


def planned_budget(cut):
    return {**credit.planned_budget(cut), "environment_initialization_probe_steps": 6,
            "retention_loss_evaluations": 2 * credit.BLOCK,
            "retention_action_pairs": 2 * credit.BLOCK * credit.KWARGS["batch_size"] * 7}


class Probe:
    def __init__(self, router, report, output, branch, times, start=0, offset=0,
                 reference=None, expected_prefix=None, learning_starts=5000):
        self.router, self.report, self.output, self.branch = router, report, output, branch
        self.times, self.step = tuple(times), times[0]
        self.start, self.offset, self.learning_starts = start, offset, learning_starts
        self.reference, self.expected_prefix, self.state = reference, expected_prefix, None

    def __call__(self, state):
        step, budget = state["step"], self.report["completed_budget"]
        spent = self.offset + step - self.start
        budget.update(training_interactions=spent, learner_updates=spent - self.learning_starts + 1)
        self.router.selector = state["adaptive_selector"]
        if self.branch == "prefix":
            log = {**state["log_data"], "ground_truth": state["env"].get_ground_truth()}
            check = credit.earlier.history_check(log, self.router.selector.block_history, self.reference, step)
            self.report["prefix_checks"][str(step)] = check
            forks.write_report(self.output, self.report)
            assert check["passed"], "Prefix reproduction failed"
            if step == self.times[-1]:
                self.state = forks.clone_state(state)
                self.state["trainer"].__dict__.pop("envelope_actor_loss", None)
                np.random.set_state(state["numpy_rng"])
                torch.set_rng_state(state["torch_rng"])
            else:
                self.step = self.times[1]
                return False
        row = credit.measure(state, self.report["seed"])
        budget["evaluation_interactions"] += row["evaluation_interactions"]
        if self.branch == "prefix":
            self.report["prefix_measurement"] = row
            matched = np.array_equal(row["returns"], self.expected_prefix["returns"])
            self.report["prefix_checks"][str(step)]["returns_matched"] = bool(matched)
            forks.write_report(self.output, self.report)
            assert matched, "Prefix evaluation differs from the completed credit study"
            self.report["prefix_complete"] = True
        else:
            row.update(feedback=copy.deepcopy(self.router.selector.block_history),
                       scores=dict(self.router.selector.scores), retention_audit=self.router.audit())
            branch = self.report["branches"][self.branch]
            branch["measurements"].append(row)
            branch.update(routing=self.router.state_dict(),
                          mode_history=state["env"].get_ground_truth()["mode_history"],
                          complete=step == self.times[-1])
            for key, field in (("retention_loss_evaluations", "loss_updates"),
                               ("retention_action_pairs", "action_pairs")):
                budget[key] = sum(b.get("routing", {}).get("retention", {}).get(field, 0)
                                  for b in self.report["branches"].values())
        forks.write_report(self.output, self.report)
        print(f"RETENTION seed={self.report['seed']} branch={self.branch} step={step} budget={budget}", flush=True)
        if step == self.times[-1]:
            return None
        self.step = next(t for t in self.times if t > step)
        return False


def load_source(seed):
    path = credit.ROOT / f"seed{seed}.json"
    source = json.loads(path.read_text())
    assert (source["protocol"], source["seed"], source["complete"]) == (credit.PROTOCOL, seed, True)
    assert source["summary"] == credit.summarize(source)
    return source, path


def contrast(a, b):
    for key in ("step", "evaluation_weights", "episode_seeds_per_mode", "ref_point", "encoder", "policy"):
        assert a[key] == b[key], key
    return dict(step=a["step"], hv_delta=a["hv_mean_mode"] - b["hv_mean_mode"],
        hv_relative_pct=100 * (a["hv_mean_mode"] / b["hv_mean_mode"] - 1) if b["hv_mean_mode"] else None,
        hv_delta_per_mode=(np.asarray(a["hv_per_mode"]) - b["hv_per_mode"]).tolist(),
        **preference_decomposition(a["returns"], b["returns"], a["evaluation_weights"]))


def finalize(report, source):
    assert report["prefix_complete"] and all(b["complete"] for b in report["branches"].values())
    evaluated = report["prefix_measurement"]["evaluation_interactions"] + sum(
        row["evaluation_interactions"] for branch in report["branches"].values() for row in branch["measurements"])
    assert evaluated == report["completed_budget"]["evaluation_interactions"]
    for key, value in report["planned_budget_upper_bound"].items():
        actual = report["completed_budget"][key]
        if key == "evaluation_interactions":
            assert actual <= value, key
        else:
            assert actual == value, key
    summaries = {}
    for condition, branch in report["branches"].items():
        route = branch["routing"]
        assert route["pulse_updates"] == route["retention"]["loss_updates"] == credit.BLOCK
        assert route["retention"]["coefficient"] == COEF
        assert route["retention"]["condition"] == condition
        assert route["retention"]["action_pairs"] == credit.BLOCK * credit.KWARGS["batch_size"] * 7
        assert np.array_equal(branch["mode_history"], source["branches"]["pulse"]["mode_history"])
        assert [r["step"] for r in branch["measurements"]] == list(credit.measurement_steps(report["window"][0])[1:])
        ops = [(b["end_step"], b["operator"]) for b in branch["measurements"][-1]["feedback"]]
        refs = [(b["end_step"], b["operator"]) for b in source["branches"]["pulse"]["measurements"][-1]["feedback"]]
        assert len(ops) == len(refs)
        summaries[condition] = dict(operator_mismatches_vs_pulse=sum(a != b for a, b in zip(ops, refs)),
            versus_pulse=[contrast(a, b) for a, b in zip(branch["measurements"], source["branches"]["pulse"]["measurements"])],
            versus_native=[contrast(a, b) for a, b in zip(branch["measurements"], source["branches"]["native"]["measurements"][1:])])
    summaries["coverage_vs_current"] = [contrast(a, b) for a, b in zip(
        report["branches"]["coverage"]["measurements"], report["branches"]["current"]["measurements"])]
    return summaries


def run(seed, root=ROOT):
    background, cut = credit.CASES[seed]
    output = Path(root) / f"seed{seed}.json"
    source, source_path = load_source(seed)
    if output.exists():
        saved = json.loads(output.read_text())
        assert (saved["protocol"], saved["seed"]) == (PROTOCOL, seed)
        if saved["prefix_complete"] and all(b["complete"] for b in saved["branches"].values()):
            saved.update(summary=finalize(saved, source), complete=True)
            forks.write_report(output, saved)
            print(f"DONE {PROTOCOL} seed={seed} reusing completed branches", flush=True)
            return saved
        raise RuntimeError("Incomplete attempt preserved; inspect its log before a new-root retry")
    reference, _ = credit.load_reference(seed)
    output.parent.mkdir(parents=True, exist_ok=True)
    report = dict(protocol=PROTOCOL, seed=seed, env=credit.ENV, background=background,
        window=[cut, cut + credit.BLOCK], complete=False, prefix_complete=False, prefix_checks={},
        source=str(source_path), planned_budget_upper_bound=planned_budget(cut),
        completed_budget=dict.fromkeys(planned_budget(cut), 0), reused_source_budget=source["completed_budget"],
        branches={name: dict(complete=False, measurements=[]) for name in CONDITIONS},
        scope="Selected two-case development test, not confirmation. Both branches use one own-candidate "
              "BC pulse plus coefficient-1 frozen-prefix mean-action retention only inside that block. "
              "Seven existing simplex templates versus seven repeated current preferences, same replay "
              "rows and forward batch size. No true modes, future data, rollout returns or critic ranks "
              "enter retention. Live selector rule is unchanged; choices may diverge. Frozen-context "
              "actor retention is not a full-return or encoder-retention guarantee. Native and pulse "
              "controls are reused, not retrained. HV stays primary; EUM is a diagnostic.")
    forks.write_report(output, report)
    router = credit.PulseRouter(background, False, (cut, cut + credit.BLOCK))
    prefix = Probe(router, report, output, "prefix", (credit.WARMUP, cut), reference=reference,
                   expected_prefix=source["branches"]["native"]["measurements"][0])
    try:
        report["completed_budget"]["environment_initialization_probe_steps"] += 2
        training.train_method(credit.ENV, credit.METHOD, credit.END, 25000, seed, str(output.parent),
            state_probe=prefix, update_router=router, archive_steps=[credit.END], **credit.KWARGS)
    finally:
        router.close()
    state = prefix.state
    try:
        for i, condition in enumerate(CONDITIONS):
            branch = forks.clone_state(state) if i == 0 else state
            router = RetentionRouter(background, (cut, cut + credit.BLOCK), condition,
                                     selector=branch["adaptive_selector"])
            probe = Probe(router, report, output, condition, credit.measurement_steps(cut)[1:],
                          start=cut, offset=cut + i * (credit.END - cut))
            try:
                report["completed_budget"]["environment_initialization_probe_steps"] += 2
                training.train_method(credit.ENV, credit.METHOD, credit.END, 25000, seed, str(output.parent),
                    state_probe=probe, update_router=router, resume_state=branch,
                    archive_steps=[credit.END], **credit.KWARGS)
            finally:
                router.close()
                branch["env"].close()
                branch["eval_env"].close()
    finally:
        state["env"].close()
        state["eval_env"].close()
    report.update(summary=finalize(report, source), complete=True)
    forks.write_report(output, report)
    print(f"DONE {PROTOCOL} seed={seed} final_coverage_vs_current="
          f"{report['summary']['coverage_vs_current'][-1]['hv_relative_pct']:.6f}%", flush=True)
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, choices=credit.CASES, required=True)
    parser.add_argument("--root", type=Path, default=ROOT)
    args = parser.parse_args()
    run(args.seed, args.root)
