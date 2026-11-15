"""Complete the copy-loss x BC diagnostic with one matched-draw joint branch."""

import argparse
import copy
import json
from pathlib import Path

import numpy as np
import torch

import run_ant_component_forks as base
import summarize_ant_component_forks as readout


PROTOCOL = "ant_joint_component_v1"
ROOT = Path("results/iclr_ant_joint_component_20260927")
SEED, CUT, BLOCK_END, END = base.SEED, base.CUT, base.BLOCK_END, base.END
KWARGS = base.KWARGS
JOINT = "no_extra_copy_loss_no_bc"


def planned_budget():
    return dict(training_interactions=2 * END - CUT,
                learner_updates=2 * END - CUT - KWARGS["learning_starts"] + 1,
                evaluation_interactions=5 * 3 * 21 * 3 * 1000,
                environment_initialization_probe_steps=4)


def load_references():
    history = json.loads((base.ROOT / f"seed{SEED}.json").read_text())
    assert (history["protocol"], history["seed"], history["complete"], history["controls_passed"]) == (
        base.PROTOCOL, SEED, True, True)
    reference = json.loads((base.previous.ROOT / "shared" / f"seed{SEED}" /
                           f"{base.METHOD}_seed{SEED}.json").read_text())
    assert (reference["env"], reference["method"], reference["seed"], reference["total_steps"]) == (
        base.ENV, base.METHOD, SEED, END)
    return history, reference


class JointProbe:
    def __init__(self, name, router, report, output, history, reference):
        self.name, self.router, self.report, self.output = name, router, report, output
        self.history, self.reference = history, reference
        self.times = [CUT, BLOCK_END, END] if name == "shared" else [BLOCK_END, END]
        self.step = min(KWARGS["learning_starts"], CUT) if name == "shared" else BLOCK_END
        self.state = None

    def __call__(self, state):
        step, selector = state["step"], state["adaptive_selector"]
        row, budget = self.report["branches"][self.name], self.report["completed_budget"]
        if step == END:
            selector.maybe_finish_block(step, force=True)
        if step in self.times:
            if self.name == "shared":
                check = base.history_check({**state["log_data"], "ground_truth": state["env"].get_ground_truth()},
                                           selector.block_history, self.reference, step)
                row.setdefault("reproduction", {})[str(step)] = check
                base.write_report(self.output, self.report)
                assert check["passed"], (step, check)
                if step == CUT:
                    self.state = base.clone_state(state)
                    np.random.set_state(state["numpy_rng"])
                    torch.set_rng_state(state["torch_rng"])
            before = state["eval_env"]._total_steps
            try:
                measurement = base.measure(state, SEED)
            finally:
                budget["evaluation_interactions"] += state["eval_env"]._total_steps - before
            row["measurements"].append(measurement)
            assert all(measurement[k] == v for k, v in self.report["evaluation"].items())
            if self.name == "shared":
                old_name = "per_label" if step == CUT else "shared"
                old = next(m for m in self.history["branches"][old_name]["measurements"] if m["step"] == step)
                matched = bool(np.array_equal(measurement["returns"], old["returns"]))
                row["reproduction"][str(step)]["grid_returns_matched"] = matched
                assert matched, (step, "control grid returns")
            row.update(block_history=copy.deepcopy(selector.block_history),
                       mode_history=state["env"].get_ground_truth()["mode_history"],
                       selector_draws=getattr(selector, "draws", []))
        row["routing"] = self.router.state_dict()
        if step == END:
            assert np.array_equal(row["mode_history"], self.reference["ground_truth"]["mode_history"])
            if self.name == JOINT:
                assert [(b["end_step"], b["operator"]) for b in row["block_history"]] == [
                    (b["end_step"], b["operator"]) for b in self.reference["adaptive_operator_final"]["block_history"]]
            row["complete"] = True
        base.write_report(self.output, self.report)
        print(f"JOINT {self.name} step={step} budget={budget}", flush=True)
        if step == END:
            return None
        self.step = min([t for t in self.times if t > step] + [(step // 5000 + 1) * 5000])
        return False


def interaction(joint, copy_only, bc_only, full):
    return readout.difference(readout.difference(joint, copy_only), readout.difference(bc_only, full))


def summarize(report, history):
    assert report["controls_passed"] and all(b["complete"] for b in report["branches"].values())
    weights = np.asarray(report["evaluation"]["evaluation_weights"])
    rows = {name: [readout.measurement(m, weights) for m in branch["measurements"]]
            for name, branch in report["branches"].items()}
    assert [m["step"] for m in rows["shared"]] == [CUT, BLOCK_END, END]
    assert [m["step"] for m in rows[JOINT]] == [BLOCK_END, END]
    for check in report["branches"]["shared"]["reproduction"].values():
        assert check["passed"] and check["grid_returns_matched"]
    routing = report["branches"][JOINT]["routing"]
    assert routing["rng_matched_steps"] == END - CUT
    assert routing["intervention_updates"] == BLOCK_END - CUT
    assert routing["effective_loss_rows_each"] == (BLOCK_END - CUT) * KWARGS["batch_size"]
    assert routing["generated_rows_each"] == 3 * routing["effective_loss_rows_each"]
    budget = report["completed_budget"]
    assert budget["evaluation_interactions"] == sum(m["evaluation_interactions"] for branch in rows.values() for m in branch)
    for key, limit in planned_budget().items():
        assert (budget[key] <= limit if key == "evaluation_interactions" else budget[key] == limit), key
    previous = {name: [readout.measurement(m, weights) for m in history["branches"][name]["measurements"][-2:]]
                for name in ("per_label", "no_extra_copy_loss", "no_bc")}
    contrasts = {f"joint_minus_{name}": [dict(step=a["step"], **readout.difference(a, b))
        for a, b in zip(rows[JOINT], measurements)]
        for name, measurements in {"shared": rows["shared"][-2:], **previous}.items()}
    contrasts["interaction"] = [dict(step=joint["step"], **interaction(joint, c, b, full))
        for joint, c, b, full in zip(rows[JOINT], previous["no_extra_copy_loss"], previous["no_bc"], rows["shared"][-2:])]
    return dict(protocol=PROTOCOL, seed=SEED, validation_passed=True,
                completed_budget=budget, historical_budget=history["completed_budget"],
                measurements=rows, contrasts=contrasts,
                joint_natural_draw_mismatches=sum(d["drawn"] != d["executed"] for d in report["branches"][JOINT]["selector_draws"]),
                scope=report["scope"])


def run(root=ROOT):
    output = Path(root) / f"seed{SEED}.json"
    if output.exists():
        saved = json.loads(output.read_text())
        assert saved["protocol"] == PROTOCOL
        if saved.get("complete"):
            print(f"DONE {PROTOCOL} reusing completed report", flush=True)
            return saved
        raise RuntimeError("Incomplete attempt preserved; inspect before a new-root retry")
    history, reference = load_references()
    output.parent.mkdir(parents=True, exist_ok=True)
    report = dict(protocol=PROTOCOL, seed=SEED, env=base.ENV, window=[CUT, BLOCK_END], end_step=END,
        complete=False, controls_passed=False, planned_budget_upper_bound=planned_budget(),
        completed_budget=dict.fromkeys(planned_budget(), 0), evaluation=history["evaluation"],
        historical_component_report=str(base.ROOT / f"seed{SEED}.json"),
        branches={name: dict(complete=False, measurements=[]) for name in ("shared", JOINT)},
        scope="Outcome-selected seed1405; completes copy-loss x BC cells using historical single ablations. "
              "Shared control retrained with live scores and matched to old grids at all three times. "
              "Joint branch uses the recorded shared schedule and matched learner draws; retains first-copy "
              "relabel and expanded forwards, drops extra-copy losses and BC only in block41. "
              "Evaluation uses current actor/encoder and cannot enter training or selection. Not independent confirmation.")
    base.write_report(output, report)
    state, trace = None, []
    try:
        for name in ("shared", JOINT):
            if name == JOINT:
                base.shared_state(state, reference, require_native_draws=False)
            router = base.ComponentRouter(name, trace, report["completed_budget"],
                cut=CUT, block_end=BLOCK_END, learning_starts=KWARGS["learning_starts"])
            probe = JointProbe(name, router, report, output, history, reference)
            report["completed_budget"]["environment_initialization_probe_steps"] += 2
            try:
                with base.previous.selector_patch("shared"):
                    base.training.train_method(base.ENV, base.METHOD, END, 25000, SEED,
                        str(output.parent / name), resume_state=state if name == JOINT else None,
                        state_probe=probe, update_router=router, archive_steps=[END], **KWARGS)
            finally:
                router.close()
                if probe.state is not None:
                    state = probe.state
                base.write_report(output, report)
            if name == "shared":
                assert state is not None and len(trace) == END - CUT
                report["controls_passed"] = True
            print(f"BRANCH_DONE {PROTOCOL} {name}", flush=True)
        result = summarize(report, history)
        report["complete"] = True
        base.write_report(output, report)
        base.write_report(output.parent / "readout.json", result)
        print(f"DONE {PROTOCOL} {json.dumps(report['completed_budget'])}", flush=True)
        return report
    except Exception as exc:
        report["error"] = f"{type(exc).__name__}: {exc}"
        base.write_report(output, report)
        raise
    finally:
        if state is not None:
            state["env"].close()
            state["eval_env"].close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT)
    run(parser.parse_args().root)
