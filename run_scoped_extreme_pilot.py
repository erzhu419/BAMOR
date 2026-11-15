"""Full-training pilot: own-candidate actor updates on two post-warmup arms."""

import argparse
import json
from pathlib import Path
from types import MethodType

import numpy as np
import torch

import run_nonterminal_rng_contrasts as earlier
import summarize_progress_factorial_completion as baseline_summary


PROTOCOL = "scoped_extreme_pilot_v1"
ROOT = Path("results/iclr_scoped_extreme_pilot_20260921")
SEEDS, CUT, END = (1003, 1004, 1005), 20000, 50000
TARGETS = (earlier.forks.EXTREME, earlier.terminal.OPERATOR)
training, forks, matched = earlier.training, earlier.forks, earlier.matched
METHOD, KWARGS = earlier.METHOD, earlier.KWARGS
REFERENCE_ROOT = earlier.terminal.REFERENCE_ROOT
ARCHIVE_STEPS = tuple(range(5000, 50001, 2500))
CONFIG_KEYS = tuple(earlier.terminal.CONFIG)
NEW_BUDGET = dict(training_interactions=80000, learner_updates=75001,
                  evaluation_interactions=4788000, environment_initialization_probe_steps=4)


class ScopedRouter:
    def __init__(self, name, trace, cut=CUT, end=END, selector=None, target_operators=TARGETS):
        assert name in ("factual", "own_matched")
        self.name, self.trace, self.cut, self.end, self.selector = name, trace, cut, end, selector
        self.target_operators = tuple(target_operators)
        self.records, self.trainer, self.patched, self.operator = [], None, False, None
        self.checked, self.mismatches, self.first_mismatch = 0, 0, None

    def configure(self, step, trainer):
        if step <= self.cut:
            return
        assert step <= self.end and self.selector is not None
        self.operator = self.selector.current_operator
        expected = {**self.selector.trainer_config(self.operator),
                    "preference_relabeling_rate": self.selector.relabel_rate(self.operator)}
        config = {k: getattr(trainer, k) for k in CONFIG_KEYS}
        assert config == expected
        active = self.name == "own_matched" and self.operator in self.target_operators
        if self.trainer is None:
            assert "envelope_actor_loss" not in trainer.__dict__
            self.trainer = trainer
        assert trainer is self.trainer
        if active and not self.patched:
            trainer.envelope_actor_loss = MethodType(matched.own_candidate_loss, trainer)
            self.patched = True
        elif self.patched and not active:
            del trainer.envelope_actor_loss
            self.patched = False
        start = self.selector.block_start_step
        if not self.records or self.records[-1]["start_step"] != start:
            self.records.append(dict(start_step=start, end_step=step, operator=self.operator,
                                     own_selected=active, steps=0, config=config))
        record = self.records[-1]
        assert record["operator"] == self.operator and record["config"] == config
        record["end_step"], record["steps"] = step, record["steps"] + 1

    def after_step(self, step, trainer, encoder, env, operator):
        if step <= self.cut:
            return
        assert operator == self.operator
        n, t = np.random.get_state(), torch.get_rng_state()
        if self.name == "factual":
            self.trace.append((n, t))
            return
        a, b = self.trace[step - self.cut - 1]
        equal = (n[0] == a[0] and np.array_equal(n[1], a[1])
                 and n[2:] == a[2:] and torch.equal(t, b))
        self.checked += 1
        if not equal:
            self.mismatches += 1
            if self.first_mismatch is None:
                self.first_mismatch = step

    def interaction_step(self, step):
        return step

    def state_dict(self):
        result = dict(protocol=PROTOCOL, branch=self.name, cut=self.cut, end=self.end,
                      target_operators=list(self.target_operators), blocks=self.records,
                      targeted_updates=sum(r["steps"] for r in self.records if r["operator"] in self.target_operators),
                      own_selected_updates=sum(r["steps"] for r in self.records if r["own_selected"]),
                      extra_environment_steps=0, extra_updates=0)
        if self.name == "own_matched":
            result["rng_alignment"] = dict(checked_steps=self.checked, mismatched_steps=self.mismatches,
                first_mismatch_step=self.first_mismatch,
                passed=self.checked == self.end - self.cut and self.mismatches == 0)
        return result

    def close(self):
        if self.patched:
            del self.trainer.envelope_actor_loss
            self.patched = False


def weights_match(actual, expected):
    assert actual.keys() == expected.keys()
    for key in actual:
        torch.testing.assert_close(actual[key], expected[key], atol=0, rtol=0)
    return len(actual)


def state_check(state, reference, model, endpoint=False):
    step = state["step"]
    log = {**state["log_data"], "ground_truth": state["env"].get_ground_truth()}
    history = earlier.history_check(log, state["adaptive_selector"].block_history, reference, step)
    assert history["passed"]
    snapshots = [s for s in model["policy_archive"] if s["step"] <= step]
    assert [s["step"] for s in state["policy_archive"]] == [s["step"] for s in snapshots]
    tensors = sum(weights_match(a["policy"], b["policy"])
                  for a, b in zip(state["policy_archive"], snapshots))
    nets = {}
    if endpoint:
        trainer = state["trainer"]
        for name, net in (("policy", trainer.policy), ("q1", trainer.q1),
                          ("q2", trainer.q2), ("encoder", state["encoder"])):
            nets[name] = weights_match(net.state_dict(), model[name])
    return dict(history=history, archive_snapshots_matched=len(snapshots),
                archive_tensors_matched=tensors, endpoint_network_tensors_matched=nets, passed=True)


class ReplayCapture:
    def __init__(self, router, reference, model, output, report, cut=CUT, end=END,
                 learning_starts=KWARGS["learning_starts"]):
        self.router, self.reference, self.model = router, reference, model
        self.output, self.report = output, report
        self.cut, self.end, self.learning_starts = cut, end, learning_starts
        self.step, self.state = cut, None

    def __call__(self, state):
        budget = self.report["completed_budget"]
        budget.update(training_interactions=state["step"],
                      learner_updates=state["step"] - self.learning_starts + 1,
                      environment_initialization_probe_steps=2)
        forks.write_report(self.output, self.report)
        assert state["eval_env"]._total_steps == 0
        if state["step"] == self.cut:
            self.report["prefix_reproduction"] = state_check(state, self.reference, self.model)
            self.state = forks.clone_state(state)
            np.random.set_state(state["numpy_rng"])
            torch.set_rng_state(state["torch_rng"])
            self.router.selector = state["adaptive_selector"]
            assert self.state["trainer"].buffer.size == self.cut
            self.step = self.end
            forks.write_report(self.output, self.report)
            return False
        self.report["control_reproduction"] = state_check(state, self.reference, self.model, endpoint=True)
        assert state["step"] == self.end and len(self.router.trace) == self.end - self.cut
        self.report["control_routing"] = self.router.state_dict()
        self.report["controls_passed"] = True
        forks.write_report(self.output, self.report)
        return None


def compact_metrics(data, seed):
    result = baseline_summary.previous.common.metrics(data, seed, expected_method=METHOD)
    result["late_snapshot_hv"] = result.pop("posthoc_late_snapshot_hv")
    result["late_mean_hv"] = result.pop("posthoc_late_mean_hv")
    result["late_mode_mean_hv"] = result.pop("posthoc_late_mode_mean_hv")
    result["terminal_returns"] = forks.terminal_returns(data).tolist()
    return result


def validate_candidate(data, reference, seed):
    assert data["training_interactions"] == END
    assert data["continuation_from_step"] == CUT and data["training_interactions_this_call"] == END - CUT
    assert data["evaluation_interactions"] == NEW_BUDGET["evaluation_interactions"]
    assert data["environment_initialization_probe_steps"] == 2
    assert (data["critic_target_rule"], data["bootstrap_terminal_rule"], data["critic_candidate_context_rule"]) == (
        "scalar_projection_of_soft_vector", "terminated_only", "candidate_preference")
    baseline_summary.previous.common.validate_selector(data)
    for key in ("evaluation_weights", "episode_seeds_per_mode", "ref_point"):
        assert data[key] == reference[key], key
    assert np.array_equal(data["ground_truth"]["mode_history"], reference["ground_truth"]["mode_history"])
    assert [r for r in data["episode_rewards"] if r["step"] <= CUT] == [
        r for r in reference["episode_rewards"] if r["step"] <= CUT]
    a, b = data["adaptive_operator_final"]["block_history"], reference["adaptive_operator_final"]["block_history"]
    assert a[:10] == b[:10]
    assert [(r["operator"], r["end_step"], r["steps"]) for r in a] == [
        (r["operator"], r["end_step"], r["steps"]) for r in b]
    route = data["update_routing"]
    assert route["protocol"] == PROTOCOL and route["branch"] == "own_matched"
    assert route["rng_alignment"]["passed"]
    assert route["rng_alignment"]["checked_steps"] == END - CUT
    assert len(route["blocks"]) == 20
    expected_config = training.FixedProgressExtremeHierarchicalOperatorSelector(2)
    for actual, original in zip(route["blocks"], b[10:]):
        op = original["operator"]
        assert (actual["start_step"], actual["end_step"], actual["steps"], actual["operator"]) == (
            original["end_step"] - 1499, original["end_step"], 1500, op)
        assert actual["own_selected"] == (op in TARGETS)
        assert actual["config"] == {**expected_config.trainer_config(op),
                                    "preference_relabeling_rate": expected_config.relabel_rate(op)}
    count = sum(r["steps"] for r in b[10:] if r["operator"] in TARGETS)
    assert route["own_selected_updates"] == route["targeted_updates"] == count
    assert route["extra_updates"] == route["extra_environment_steps"] == 0
    return compact_metrics(data, seed)


def complete_report(report, data, reference):
    report["branches"]["own_matched"] = validate_candidate(data, reference, report["seed"])
    a, b = report["branches"]["own_matched"], report["branches"]["factual"]
    report["paired_comparisons"] = {"own_matched_minus_factual": {
        **matched.paired_difference(a, b, report["evaluation"]["evaluation_weights"]),
        "archive_hv": a["archive_hv"] - b["archive_hv"],
        "late_mean_hv": a["late_mean_hv"] - b["late_mean_hv"]}}
    report["grid_diagnosis"] = earlier.grid_diagnosis(report["branches"], report["evaluation"]["evaluation_weights"])
    assert report["completed_budget"] == NEW_BUDGET
    report["complete"] = True
    report["scope"] = ("Three observed development seeds; scoped update-rule pilot, not adaptive-selector evidence. "
        "Candidate retains the20k factual warmup and all non-target actor updates. Baseline evaluation is reused "
        "only after exact endpoint network/archive and historical-record reproduction. Historical optimizer "
        "states were not saved. Snapshots use each branch's final encoder, not an online recovery curve. "
        "Report all preference losses; finite-grid maxima are diagnostic only. Failed attempts/tests stay separate.")
    return report


def recover_saved(seed, root, output_root):
    """Finish validation only, preserving the interrupted report and all raw files."""
    assert seed in SEEDS
    root, output_root = Path(root), Path(output_root)
    assert root.resolve() != output_root.resolve()
    source = root / f"seed{seed}.json"
    report = json.loads(source.read_text())
    assert (report["protocol"], report["seed"], report["warmup_steps"], report["end_step"]) == (
        PROTOCOL, seed, CUT, END)
    assert report["controls_passed"] and report["rng_alignment_passed"]
    assert report["completed_budget"] == NEW_BUDGET
    assert report["target_operators"] == list(TARGETS)
    for key, count in (("prefix_reproduction", 7), ("control_reproduction", 19)):
        record = report[key]
        assert record["passed"] and record["history"]["passed"]
        assert all(record["history"][k] for k in ("episodes_matched", "blocks_matched", "schedule_matched"))
        assert record["archive_snapshots_matched"] == count
    assert set(report["control_reproduction"]["endpoint_network_tensors_matched"]) == {
        "policy", "q1", "q2", "encoder"}
    reference = json.loads(Path(report["reference_path"]).read_text())
    baseline_summary.validate_new_run(reference, seed)
    assert report["branches"]["factual"] == compact_metrics(reference, seed)
    for key, value in report["evaluation"].items():
        assert reference[key] == value
    raw_path = root / f"seed{seed}" / "own_matched" / f"{METHOD}_seed{seed}.json"
    data = json.loads(raw_path.read_text())
    assert data["update_routing"] == report["candidate_routing"]
    complete_report(report, data, reference)
    report["postprocessing_recovery"] = dict(source_report=str(source), source_candidate=str(raw_path),
        reason="Live tuple mode histories compared directly with JSON lists; numeric schedule unchanged.",
        new_training_interactions=0, new_evaluation_interactions=0, new_learner_updates=0)
    output_root.mkdir(parents=True, exist_ok=True)
    output = output_root / f"seed{seed}.json"
    if output.exists():
        assert json.loads(output.read_text()) == report
    else:
        forks.write_report(output, report)
    print(f"RECOVERED scoped_extreme_pilot seed={seed} from saved results; no new interactions", flush=True)
    return report


def run(seed, root=ROOT):
    assert seed in SEEDS
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    output = root / f"seed{seed}.json"
    if output.exists():
        saved = json.loads(output.read_text())
        if saved.get("complete"):
            assert saved["protocol"] == PROTOCOL and saved["seed"] == seed
            assert saved["controls_passed"] and saved["rng_alignment_passed"]
            print(f"DONE scoped_extreme_pilot seed={seed} reusing complete report", flush=True)
            return saved
        raise RuntimeError("Incomplete attempt preserved; review it and use a new --root for a retry")
    reference_path = REFERENCE_ROOT / f"{METHOD}_seed{seed}.json"
    model_path = REFERENCE_ROOT / f"{METHOD}_seed{seed}_model.pt"
    reference = json.loads(reference_path.read_text())
    baseline_summary.validate_new_run(reference, seed)
    model = torch.load(model_path, map_location="cpu", weights_only=False)
    report = dict(protocol=PROTOCOL, seed=seed, warmup_steps=CUT, end_step=END,
                  target_operators=list(TARGETS), complete=False, controls_passed=False,
                  rng_alignment_passed=False, reference_path=str(reference_path), reference_model_path=str(model_path),
                  evaluation={k: reference[k] for k in ("evaluation_weights", "episode_seeds_per_mode", "ref_point")},
                  branches={"factual": compact_metrics(reference, seed)},
                  completed_budget=dict.fromkeys(NEW_BUDGET, 0))
    forks.write_report(output, report)
    trace = []
    control = ScopedRouter("factual", trace)
    capture = ReplayCapture(control, reference, model, output, report)
    try:
        try:
            training.train_method("mo-halfcheetah-v5", METHOD, END, 25000, seed,
                str(root / f"seed{seed}" / "control_replay"), state_probe=capture,
                update_router=control, **KWARGS)
        finally:
            control.close()
        assert report["controls_passed"]
        print(f"CONTROL_DONE seed={seed} exact networks and19 snapshots; reusing reference evaluation", flush=True)
        branch = forks.clone_state(capture.state)
        router = ScopedRouter("own_matched", trace, selector=branch["adaptive_selector"])
        try:
            data = training.train_method("mo-halfcheetah-v5", METHOD, END, 25000, seed,
                str(root / f"seed{seed}" / "own_matched"), resume_state=branch, update_router=router, **KWARGS)
        finally:
            router.close()
        budget = report["completed_budget"]
        budget["training_interactions"] += data["training_interactions_this_call"]
        budget["learner_updates"] += END - CUT
        budget["evaluation_interactions"] += data["evaluation_interactions"]
        budget["environment_initialization_probe_steps"] += data["environment_initialization_probe_steps"]
        report["candidate_routing"] = data["update_routing"]
        report["rng_alignment_passed"] = data["update_routing"]["rng_alignment"]["passed"]
        forks.write_report(output, report)
        complete_report(report, data, reference)
        forks.write_report(output, report)
        print(f"DONE scoped_extreme_pilot seed={seed} {json.dumps(budget)}", flush=True)
        return report
    finally:
        if capture.state is not None:
            capture.state["env"].close()
            capture.state["eval_env"].close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, choices=SEEDS, required=True)
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--recover-to", type=Path, help="Validate saved completed work into a separate report directory; no training")
    args = parser.parse_args()
    if args.recover_to is not None:
        recover_saved(args.seed, args.root, args.recover_to)
    else:
        run(args.seed, args.root)
