"""Fresh-seed, fixed-window copy-loss x BC experiment; no default policy change."""

import argparse
import copy
import json
from math import comb
from pathlib import Path
from unittest.mock import patch

import numpy as np
import torch

import run_ant_component_forks as base
import summarize_ant_component_forks as readout
from summarize_ant_mechanism_alignment import paired


PROTOCOL = "ant_component_replication_v1"
ROOT = Path("results/iclr_ant_component_replication_20260927")
SEEDS = tuple(range(1501, 1506))
CUT, BLOCK_END, END = 65000, 66500, 75000
KWARGS = base.KWARGS
BRANCHES = ("full", "no_extra_copy_loss", "no_bc", "no_extra_copy_loss_no_bc")
PRIMARY = "mode0_mixed_energy_utility"
NativeSelector = base.training.OwnPreferenceBothProgressExtremeOperatorSelector


def planned_budget():
    training = END + 3 * (END - CUT)
    return dict(training_interactions=training,
                learner_updates=training - KWARGS["learning_starts"] + 1,
                evaluation_interactions=9 * 3 * 21 * 3 * 1000,
                environment_initialization_probe_steps=8)


class WindowSelector(NativeSelector):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.draws = []

    def select(self):
        drawn = super().select()
        start = self.learning_starts + len(self.block_history) * self.block_steps + 1
        executed = base.B if CUT < start <= BLOCK_END else drawn
        if start > CUT:
            self.draws.append(dict(start=start, drawn=drawn, executed=executed))
        return self._force_choice(executed) if CUT < start <= BLOCK_END else drawn


class ReplaySelector(NativeSelector):
    def select(self):
        drawn = super().select()
        index = len(self.block_history)
        executed = self.schedule[index]
        start = self.learning_starts + index * self.block_steps + 1
        self.draws.append(dict(start=start, drawn=drawn, executed=executed))
        return self._force_choice(executed)


def replay_state(state, blocks):
    selector = ReplaySelector(3)
    selector.__dict__.update(copy.deepcopy(state["adaptive_selector"].__dict__))
    selector.schedule = [b["operator"] for b in blocks]
    selector.draws = []
    state["adaptive_selector"] = selector
    return state


class Probe:
    def __init__(self, name, seed, router, report, output):
        self.name, self.seed, self.router, self.report, self.output = name, seed, router, report, output
        self.times = [CUT, BLOCK_END, END] if name == "full" else [BLOCK_END, END]
        self.step = min(KWARGS["learning_starts"], CUT) if name == "full" else BLOCK_END
        self.state = None

    def __call__(self, state):
        step, selector = state["step"], state["adaptive_selector"]
        row, budget = self.report["branches"][self.name], self.report["completed_budget"]
        if step == END:
            selector.maybe_finish_block(step, force=True)
        if step == CUT:
            assert selector.current_operator is None
            assert selector.learning_starts + len(selector.block_history) * selector.block_steps == CUT
            assert BLOCK_END - CUT == selector.block_steps
            self.state = base.clone_state(state)
            np.random.set_state(state["numpy_rng"])
            torch.set_rng_state(state["torch_rng"])
        if step in self.times:
            before = state["eval_env"]._total_steps
            try:
                measured = base.measure(state, self.seed)
            finally:
                budget["evaluation_interactions"] += state["eval_env"]._total_steps - before
            row["measurements"].append(measured)
            if step == CUT:
                self.report["evaluation"] = {k: measured[k] for k in
                    ("evaluation_weights", "episode_seeds_per_mode", "ref_point")}
            assert all(measured[k] == v for k, v in self.report["evaluation"].items())
            row.update(block_history=copy.deepcopy(selector.block_history),
                       mode_history=state["env"].get_ground_truth()["mode_history"],
                       selector_draws=copy.deepcopy(selector.draws))
        row["routing"] = self.router.state_dict()
        if step == END:
            row["complete"] = True
        base.write_report(self.output, self.report)
        print(f"FACTORIAL seed={self.seed} {self.name} step={step} budget={budget}", flush=True)
        if step == END:
            return None
        self.step = min([t for t in self.times if t > step] + [(step // 5000 + 1) * 5000])
        return False


def summarize_seed(report):
    assert (report["protocol"], report["env"], report["window"], report["end_step"]) == (
        PROTOCOL, base.ENV, [CUT, BLOCK_END], END)
    assert report["seed"] in SEEDS and "error" not in report
    assert tuple(report["branches"]) == BRANCHES
    weights = np.asarray(report["evaluation"]["evaluation_weights"])
    np.testing.assert_array_equal(weights, base.training.full_simplex_grid(3))
    full, rows, eval_steps = report["branches"]["full"], {}, 0
    schedule = [(b["end_step"], b["operator"]) for b in full["block_history"]]
    for name, branch in report["branches"].items():
        assert branch["complete"] and branch["mode_history"] == full["mode_history"]
        assert [(b["end_step"], b["operator"]) for b in branch["block_history"]] == schedule
        intervention = next(b for b in branch["block_history"] if b["end_step"] == BLOCK_END)
        assert intervention["operator"] == base.B and intervention["steps"] == BLOCK_END - CUT
        times = [CUT, BLOCK_END, END] if name == "full" else [BLOCK_END, END]
        assert [m["step"] for m in branch["measurements"]] == times
        routing = branch["routing"]
        assert routing["intervention_updates"] == BLOCK_END - CUT
        generated = 3 * (BLOCK_END - CUT) * KWARGS["batch_size"]
        assert routing["generated_rows_each"] == generated
        assert routing["effective_loss_rows_each"] == generated // (3 if "no_extra_copy_loss" in name else 1)
        assert routing["extra_updates"] == routing["extra_environment_steps"] == 0
        if name != "full":
            assert routing["rng_matched_steps"] == END - CUT
        assert [d["start"] for d in branch["selector_draws"]] == list(range(CUT + 1, END + 1, BLOCK_END - CUT))
        for measured in branch["measurements"]:
            assert all(measured[k] == v for k, v in report["evaluation"].items())
            assert 0 < measured["evaluation_interactions"] <= 189000
            eval_steps += measured["evaluation_interactions"]
        rows[name] = [readout.measurement(m, weights) for m in branch["measurements"]]
    budget = report["completed_budget"]
    assert budget["evaluation_interactions"] == eval_steps
    for key, limit in planned_budget().items():
        assert (budget[key] <= limit if key == "evaluation_interactions" else budget[key] == limit), key
    contrasts = {key: [] for key in ("copy_loss_removal", "bc_removal_full_copies",
                                    "bc_removal_first_copy", "joint_removal", "interaction")}
    for i, step in enumerate((BLOCK_END, END)):
        f, c, b, j = [rows[name][-2:][i] for name in BRANCHES]
        bc_full, bc_first = readout.difference(b, f), readout.difference(j, c)
        values = (readout.difference(c, f), bc_full, bc_first,
                  readout.difference(j, f), readout.difference(bc_first, bc_full))
        for key, value in zip(contrasts, values):
            contrasts[key].append(dict(step=step, **value))
    return dict(protocol=PROTOCOL, seed=report["seed"], complete=True, validation_passed=True,
        completed_budget=budget, evaluation=report["evaluation"], measurements=rows, contrasts=contrasts,
        natural_draw_mismatches={name: sum(d["drawn"] != d["executed"] for d in b["selector_draws"])
                                for name, b in report["branches"].items()}, scope=report["scope"])


def summarize(root=ROOT):
    runs = [json.loads((Path(root) / f"seed{seed}_readout.json").read_text()) for seed in SEEDS]
    for seed, row in zip(SEEDS, runs):
        assert (row["protocol"], row["seed"], row["complete"], row["validation_passed"]) == (PROTOCOL, seed, True, True)
    metrics = (PRIMARY, "all_mode_direct_utility", "hv_mean_mode")
    contrasts = {key: [dict(step=step, **{metric: paired(
        [r["contrasts"][key][i][metric] for r in runs], [0.] * len(runs)) for metric in metrics})
        for i, step in enumerate((BLOCK_END, END))] for key in runs[0]["contrasts"]}
    primary = contrasts["interaction"][0][PRIMARY]
    n = primary["wins"] + primary["losses"]
    p_negative = sum(comb(n, k) for k in range(primary["losses"], n + 1)) / 2 ** n
    return dict(protocol=PROTOCOL, seeds=list(SEEDS), complete=True, contrasts=contrasts,
        primary=dict(step=BLOCK_END, metric=PRIMARY, expected_direction="negative",
                     one_sided_sign_p=p_negative, negative_direction_replicated=p_negative <= .05),
        completed_budget={k: sum(r["completed_budget"][k] for r in runs) for k in planned_budget()},
        runs=runs, scope=runs[0]["scope"])


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
    root.mkdir(parents=True, exist_ok=True)
    report = dict(protocol=PROTOCOL, seed=seed, env=base.ENV, window=[CUT, BLOCK_END], end_step=END,
        complete=False, planned_budget_upper_bound=planned_budget(),
        completed_budget=dict.fromkeys(planned_budget(), 0),
        branches={name: dict(complete=False, measurements=[]) for name in BRANCHES},
        scope="Independent seeds, one prespecified window on a native per-label prefix. Force B in all four "
              "cells during the window. Afterwards replay the full-cell training-only schedule, consuming "
              "ordinary selector draws. First-copy relabel and all expanded forwards/draws remain intact. "
              "Matched learner RNG, not identical environment trajectories. Current actor/encoder evaluation "
              "never enters training. Conditional component replication, not a live-selector or CAPQL comparison.")
    base.write_report(output, report)
    state, trace = None, []
    try:
        for name in BRANCHES:
            branch = None if name == "full" else replay_state(
                base.clone_state(state), report["branches"]["full"]["block_history"])
            # The legacy router's 'shared' name only records the reference draw stream.
            router = base.ComponentRouter("shared" if name == "full" else name, trace,
                report["completed_budget"], cut=CUT, block_end=BLOCK_END, learning_starts=KWARGS["learning_starts"])
            probe = Probe(name, seed, router, report, output)
            report["completed_budget"]["environment_initialization_probe_steps"] += 2
            try:
                with patch.object(base.training, "OwnPreferenceBothProgressExtremeOperatorSelector", WindowSelector):
                    base.training.train_method(base.ENV, base.METHOD, END, 25000, seed,
                        str(root / f"seed{seed}" / name), resume_state=branch,
                        state_probe=probe, update_router=router, archive_steps=[END], **KWARGS)
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


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, choices=SEEDS)
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--summarize", action="store_true")
    args = parser.parse_args()
    if args.summarize:
        result = summarize(args.root)
        base.write_report(args.root / "readout.json", result)
        print(json.dumps(result["primary"]))
    else:
        assert args.seed is not None
        run(args.seed, args.root)
