"""Isolate terminal actor selection while retaining the factual random draws."""

import argparse
import json
from pathlib import Path
from types import MethodType

import numpy as np
import torch

import run_terminal_component_forks as previous


PROTOCOL = "terminal_rng_matched_v1"
ROOT = Path("results/iclr_terminal_rng_matched_20260921")
REFERENCE_ROOT = Path("results/iclr_terminal_component_forks_20260921_resetfix")
BRANCHES = ("factual", "actor_one", "own_matched")
SEED, CUT, END = previous.SEED, previous.CUT, previous.END
METHOD, training = previous.previous.METHOD, previous.training


def own_candidate_loss(self, s, w, c):
    # Preserve the legacy ordering: all preference draws, then all action draws.
    contexts = [c]
    for _ in range(self.envelope_policy_candidates - 1):
        prefs = self.sample_preference_batch(w.shape[0], self.preference_relabeling_strategy)
        contexts.append(self.replace_context_preferences(c, prefs))
    actions, log_probs, scalars = [], [], []
    for context in contexts:
        action, log_prob = self.policy.get_action(s, context)
        q = torch.min(self.q1(s, action, c), self.q2(s, action, c))
        actions.append(action)
        log_probs.append(log_prob)
        scalars.append((w * q).sum(dim=-1, keepdim=True))
    indices = torch.zeros(w.shape[0], dtype=torch.long, device=self.device)
    batch = torch.arange(w.shape[0], device=self.device)
    action = torch.stack(actions)[indices, batch]
    log_prob = torch.stack(log_probs)[indices, batch]
    scalar = torch.stack(scalars)[indices, batch]
    return action, scalar, self.actor_objective(s, w, c, scalar, log_prob)


class MatchedIntervention(previous.TerminalIntervention):
    protocol = PROTOCOL

    def __init__(self, name, rng_trace, start=CUT + 1, end=END,
                 expected_operator=previous.EXTREME):
        assert name in BRANCHES
        super().__init__("factual" if name == "own_matched" else name,
                         start=start, end=end, expected_operator=expected_operator)
        selector = training.FixedProgressExtremeHierarchicalOperatorSelector(2)
        self.expected_config = {**selector.trainer_config(expected_operator),
                                "preference_relabeling_rate": selector.relabel_rate(expected_operator),
                                **previous.OVERRIDES[self.branch]}
        self.name, self.rng_trace = name, rng_trace
        self.trainer = None
        self.checked, self.mismatches, self.first_mismatch = 0, 0, None

    def configure(self, step, trainer):
        if self.name == "own_matched" and self.trainer is None:
            assert "envelope_actor_loss" not in trainer.__dict__
            self.trainer = trainer
            trainer.envelope_actor_loss = MethodType(own_candidate_loss, trainer)
        super().configure(step, trainer)
        assert self.config == self.expected_config

    def after_step(self, step, trainer, encoder, env, operator):
        super().after_step(step, trainer, encoder, env, operator)
        if self.name == "actor_one":
            return
        numpy_rng, torch_rng = np.random.get_state(), torch.get_rng_state()
        if self.name == "factual":
            self.rng_trace.append((numpy_rng, torch_rng))
            return
        ref_numpy, ref_torch = self.rng_trace[step - self.start]
        matched = (numpy_rng[0] == ref_numpy[0]
                   and np.array_equal(numpy_rng[1], ref_numpy[1])
                   and numpy_rng[2:] == ref_numpy[2:]
                   and torch.equal(torch_rng, ref_torch))
        self.checked += 1
        if not matched:
            self.mismatches += 1
            if self.first_mismatch is None:
                self.first_mismatch = step

    def state_dict(self):
        out = super().state_dict()
        out.update(protocol=self.protocol, branch=self.name,
                   actor_selection="own_candidate_zero" if self.name == "own_matched" else "unchanged")
        if self.name == "own_matched":
            out["rng_alignment"] = dict(checked_steps=self.checked, mismatched_steps=self.mismatches,
                first_mismatch_step=self.first_mismatch,
                passed=self.checked == self.end - self.start + 1 and self.mismatches == 0)
        return out

    def close(self):
        super().close()
        if self.trainer is not None:
            del self.trainer.envelope_actor_loss
            self.trainer = None


def paired_difference(row, baseline, weights):
    delta = np.asarray(row["terminal_returns"]) - np.asarray(baseline["terminal_returns"])
    utility = (delta * np.asarray(weights)[None, :, :]).sum(axis=-1)
    return dict(
        final_hv=row["final_hv"] - baseline["final_hv"],
        final_mode_hv=(np.asarray(row["final_mode_hv"]) - baseline["final_mode_hv"]).tolist(),
        terminal_return_differences=delta.tolist(), matched_weight_utility_differences=utility.tolist(),
        utility_mean_per_mode=utility.mean(axis=1).tolist(),
        utility_wins_per_mode=(utility > 0).sum(axis=1).tolist())


def paired_readout(branches, weights):
    return {f"{name}_minus_{control}": paired_difference(branches[name], branches[control], weights)
            for name, control in (("actor_one", "factual"), ("own_matched", "factual"),
                                  ("own_matched", "actor_one"))}


def run(root=ROOT):
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    output = root / f"seed{SEED}.json"
    if output.exists():
        saved = json.loads(output.read_text())
        if saved.get("complete"):
            assert saved["protocol"] == PROTOCOL and saved["controls_passed"] and saved["rng_alignment_passed"]
            print("DONE terminal_rng_matched reusing complete report", flush=True)
            return saved
        raise RuntimeError("Incomplete attempt preserved; review it and use a new --root for a retry")
    source = json.loads((REFERENCE_ROOT / f"seed{SEED}.json").read_text())
    assert source["complete"] and source["controls_passed"] and source["protocol"] == previous.PROTOCOL
    references = {name: json.loads((REFERENCE_ROOT / name / f"{METHOD}_seed{SEED}.json").read_text())
                  for name in BRANCHES[:2]}
    report = dict(protocol=PROTOCOL, seed=SEED, fork_step=CUT, end_step=END,
                  reference_root=str(REFERENCE_ROOT), complete=False, controls_passed=False,
                  rng_alignment_passed=False, evaluation=source["evaluation"], branches={},
                  completed_budget=dict(training_interactions=0, learner_updates=0,
                      evaluation_interactions=0, environment_initialization_probe_steps=0))
    previous.write_report(output, report)
    capture = previous.CaptureState(CUT)
    training.train_method("mo-halfcheetah-v5", METHOD, END, 25000, SEED, str(root / "prefix"),
                          state_probe=capture, **previous.KWARGS)
    state = capture.state
    budget = report["completed_budget"]
    budget.update(training_interactions=CUT, learner_updates=CUT - previous.KWARGS["learning_starts"] + 1,
                  environment_initialization_probe_steps=2)
    previous.write_report(output, report)
    rng_trace = []
    try:
        report["prefix_reproduction"] = previous.prefix_check(state, references)
        previous.write_report(output, report)
        for name in BRANCHES:
            branch = previous.clone_state(state)
            branch["policy_archive"] = []
            router = MatchedIntervention(name, rng_trace)
            try:
                data = training.train_method("mo-halfcheetah-v5", METHOD, END, 25000, SEED,
                    str(root / name), resume_state=branch, update_router=router, **previous.KWARGS)
            finally:
                router.close()
            assert data["continuation_from_step"] == CUT and data["training_interactions_this_call"] == END - CUT
            assert data["evaluation_interactions"] == 252000
            assert [s["step"] for s in data["policy_archive_snapshots"]] == [END]
            for key, value in report["evaluation"].items():
                assert data[key] == value, key
            budget["training_interactions"] += END - CUT
            budget["learner_updates"] += END - CUT
            budget["evaluation_interactions"] += data["evaluation_interactions"]
            budget["environment_initialization_probe_steps"] += data["environment_initialization_probe_steps"]
            row = dict(final_hv=data["final_hv_mean_mode"], final_mode_hv=data["final_hv_per_mode"],
                       last_episodes=data["episode_rewards"][-2:], update_routing=data["update_routing"],
                       terminal_returns=previous.terminal_returns(data).tolist())
            report["branches"][name] = row
            if name in references:
                row["reproduction"] = previous.reproduction(data, references[name])
                previous.write_report(output, report)
                if not row["reproduction"]["passed"]:
                    raise RuntimeError(f"{name} control did not reproduce; new branch not interpretable")
            else:
                report["rng_alignment_passed"] = row["update_routing"]["rng_alignment"]["passed"]
                previous.write_report(output, report)
                if not report["rng_alignment_passed"]:
                    raise RuntimeError("Random states did not align; matched-draw interpretation invalid")
            report["controls_passed"] = all(n in report["branches"] and
                report["branches"][n]["reproduction"]["passed"] for n in references)
            previous.write_report(output, report)
            print(f"BRANCH_DONE {name} final_hv={row['final_hv']:.6f}", flush=True)
        assert len(rng_trace) == END - CUT
        report["paired_comparisons"] = paired_readout(report["branches"], report["evaluation"]["evaluation_weights"])
        report["complete"] = True
        report["scope"] = ("Selected seed1004 terminal block; candidate selection changes under the factual "
            "draw ordering. Not identical state/action trajectories or independent confirmation. "
            "No archive selection or adaptive deployment. All attempt and verification costs remain separate.")
        previous.write_report(output, report)
        print(f"DONE terminal_rng_matched {json.dumps(budget)}", flush=True)
        return report
    finally:
        state["env"].close()
        state["eval_env"].close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=ROOT)
    run(parser.parse_args().root)
