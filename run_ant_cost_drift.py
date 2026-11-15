"""Read-only fixed-input control-cost signal on frozen Ant BC trajectories."""

import argparse
import json
from pathlib import Path
from unittest.mock import patch

import numpy as np
from scipy.stats import spearmanr
import torch

import run_ant_bc_temporal as temporal


previous, base, training = temporal.previous, temporal.base, temporal.training
PROTOCOL = "ant_fixed_input_cost_drift_v1"
ROOT = Path("results/iclr_ant_cost_drift_20260927")
SEEDS, KWARGS = temporal.SEEDS, temporal.KWARGS
CUT, BLOCK_END, END = temporal.CUT, temporal.BLOCK_END, temporal.END
TIMES = temporal.times()
WIDTH, SAMPLES = 5000, 128
BRANCHES = ("full", "no_bc")


def planned_budget():
    steps = 2 * END - CUT
    return dict(training_interactions=steps, learner_updates=steps - KWARGS["learning_starts"] + 1,
                evaluation_interactions=2 * 3 * 21 * 3 * 1000,
                environment_initialization_probe_steps=4)


def capture_banks(trainer):
    buffer = trainer.buffer
    assert buffer.size == buffer.ptr == CUT and WIDTH <= CUT
    banks = {}
    for name, start, end in (("early", 0, WIDTH), ("recent", CUT - WIDTH, CUT)):
        indices = np.linspace(start, end - 1, min(SAMPLES, end - start), dtype=int)
        np.testing.assert_array_equal(buffer.timesteps[indices, 0], indices + 1)
        banks[name] = dict(timesteps=(indices + 1).tolist(),
                           states=buffer.states[indices].copy(), contexts=buffer.contexts[indices].copy())
    return banks


@torch.no_grad()
def measure_cost(trainer, banks):
    """No labels, critic, encoder refresh, rollout, or return input is accepted."""
    assert not trainer.critic_select_policy
    weights = training.full_simplex_grid(3)
    mixed = (weights[:, 2] > 0) & (weights[:, 2] < 1)
    numpy_rng, torch_rng = np.random.get_state(), torch.get_rng_state()
    result = {}
    try:
        for name, bank in banks.items():
            states = torch.as_tensor(bank["states"], device=trainer.device)
            contexts = torch.as_tensor(bank["contexts"], device=trainer.device)
            negative_cost = []
            for weight in weights:
                prefs = torch.as_tensor(weight, device=trainer.device).expand(len(states), -1)
                ctx = trainer.replace_context_preferences(contexts, prefs)
                actions = trainer.select_action(states, ctx, deterministic=True).cpu().numpy().astype(float)
                negative_cost.append(float(-np.square(actions).sum(axis=-1).mean()))
            negative_cost = np.asarray(negative_cost)
            assert np.isfinite(negative_cost).all()
            result[name] = dict(negative_cost_by_weight=negative_cost.tolist(),
                mixed_score=float((negative_cost * weights[:, 2])[mixed].mean()))
    finally:
        np.random.set_state(numpy_rng)
        torch.set_rng_state(torch_rng)
    return result


def tracking_stats(signal, target):
    x, y = np.asarray(signal), np.asarray(target)
    assert x.shape == y.shape and x.ndim == 1 and np.isfinite(x).all() and np.isfinite(y).all()
    rho = None if np.ptp(x) == 0 or np.ptp(y) == 0 else float(spearmanr(x, y).statistic)
    directional = (x != 0) & (y != 0)
    return dict(n=len(x), spearman=rho, sign_agreements=int(((x * y) > 0).sum()),
                sign_disagreements=int(((x * y) < 0).sum()), ties=int((~directional).sum()))


def compare_history(report, reference):
    weights = training.full_simplex_grid(3)
    mixed = (weights[:, 2] > 0) & (weights[:, 2] < 1)
    records, gaps, tracking = {}, [], {}
    for name in BRANCHES:
        old = reference["branches"][name]["measurements"]
        observed = report["branches"][name]["observations"]
        assert [m["step"] for m in old] == list(TIMES)
        assert [o["step"] for o in observed] == [CUT, *TIMES]
        records[name] = [dict(step=m["step"],
            utility=float((np.asarray(m["returns"])[0] * weights).sum(axis=-1)[mixed].mean()),
            **{bank: o["banks"][bank]["mixed_score"] for bank in ("early", "recent")})
            for m, o in zip(old, observed[1:])]
    intervals = [dict(branch=name, start=a["step"], end=b["step"],
                     **{key: b[key] - a[key] for key in ("utility", "early", "recent")})
                 for name in BRANCHES for a, b in zip(records[name], records[name][1:])]
    for a, b in zip(records["full"], records["no_bc"]):
        gaps.append(dict(step=a["step"], **{k: b[k] - a[k] for k in ("utility", "early", "recent")}))
    for bank in ("early", "recent"):
        tracking[bank] = tracking_stats([r[bank] for r in intervals], [r["utility"] for r in intervals])
    return dict(records=records, intervals=intervals, bc_effect_gaps=gaps, tracking=tracking)


def load_reference(seed):
    ref = json.loads((temporal.ROOT / f"seed{seed}.json").read_text())
    assert (ref["protocol"], ref["seed"], ref["complete"]) == (temporal.PROTOCOL, seed, True)
    assert "error" not in ref
    return ref


class Probe:
    def __init__(self, name, seed, router, report, output, reference, banks):
        self.name, self.seed, self.router = name, seed, router
        self.report, self.output, self.reference, self.banks = report, output, reference, banks
        self.state = None
        self.step = min(KWARGS["learning_starts"], CUT) if name == "full" else BLOCK_END

    def __call__(self, state):
        step, selector = state["step"], state["adaptive_selector"]
        row, old = self.report["branches"][self.name], self.reference["branches"][self.name]
        if step == END:
            selector.maybe_finish_block(step, force=True)
        assert selector.block_history == [b for b in old["block_history"] if b["end_step"] <= step]
        assert state["env"].get_ground_truth()["mode_history"] == [tuple(m) for m in old["mode_history"] if m[0] <= step]
        if step == CUT:
            assert selector.current_operator is None
            self.banks.update(capture_banks(state["trainer"]))
            self.report["banks"] = {k: dict(timesteps=v["timesteps"], context="frozen stored pre-transition")
                                    for k, v in self.banks.items()}
            self.state = base.clone_state(state)
            np.random.set_state(state["numpy_rng"])
            torch.set_rng_state(state["torch_rng"])
        if step in (CUT, *TIMES):
            row["observations"].append(dict(step=step, banks=measure_cost(state["trainer"], self.banks)))
        if step == END:
            before = state["eval_env"]._total_steps
            try:
                final = base.measure(state, self.seed)
            finally:
                self.report["completed_budget"]["evaluation_interactions"] += state["eval_env"]._total_steps - before
            row["final_grid"] = final
            row["final_grid_reproduced"] = bool(np.array_equal(final["returns"], old["measurements"][-1]["returns"]))
            assert row["final_grid_reproduced"], (self.seed, self.name, "endpoint mismatch")
            row["complete"] = True
        row.update(routing=self.router.state_dict(), training_reproduced_through=step)
        base.write_report(self.output, self.report)
        print(f"COST_DRIFT seed={self.seed} {self.name} step={step} budget={self.report['completed_budget']}", flush=True)
        if step == END:
            return None
        self.step = min([t for t in (CUT, *TIMES) if t > step] + [(step // 5000 + 1) * 5000])
        return False


def summarize_seed(report, reference):
    assert report["protocol"] == PROTOCOL and tuple(report["branches"]) == BRANCHES
    assert report["seed"] == reference["seed"] and "error" not in report
    evaluation = 0
    for name, row in report["branches"].items():
        assert row["complete"] and row["training_reproduced_through"] == END and row["final_grid_reproduced"]
        assert np.array_equal(row["final_grid"]["returns"], reference["branches"][name]["measurements"][-1]["returns"])
        routing = row["routing"]
        assert routing["intervention_updates"] == BLOCK_END - CUT
        assert routing["generated_rows_each"] == routing["effective_loss_rows_each"] == 3 * KWARGS["batch_size"] * (BLOCK_END - CUT)
        assert routing["extra_updates"] == routing["extra_environment_steps"] == 0
        if name == "no_bc":
            assert routing["rng_matched_steps"] == END - CUT
        evaluation += row["final_grid"]["evaluation_interactions"]
    assert report["completed_budget"]["evaluation_interactions"] == evaluation
    for key, bound in planned_budget().items():
        value = report["completed_budget"][key]
        assert (0 < value <= bound if key == "evaluation_interactions" else value == bound), key
    return dict(protocol=PROTOCOL, seed=report["seed"], complete=True, validation_passed=True,
        completed_budget=report["completed_budget"], banks=report["banks"],
        **compare_history(report, reference), scope=report["scope"])


def run(seed, root=ROOT):
    assert seed in SEEDS
    root, banks, state, trace = Path(root), {}, None, []
    output = root / f"seed{seed}.json"
    reference = load_reference(seed)
    if output.exists():
        saved = json.loads(output.read_text())
        assert (saved["protocol"], saved["seed"]) == (PROTOCOL, seed)
        if not saved.get("complete"):
            raise RuntimeError("Incomplete attempt preserved; inspect before a new-root retry")
        base.write_report(root / f"seed{seed}_readout.json", summarize_seed(saved, reference))
        return saved
    root.mkdir(parents=True, exist_ok=True)
    report = dict(protocol=PROTOCOL, seed=seed, complete=False,
        planned_budget_upper_bound=planned_budget(), completed_budget=dict.fromkeys(planned_budget(), 0),
        branches={name: dict(complete=False, observations=[]) for name in BRANCHES},
        scope="Retrospective fixed-input signal, not a new algorithm or independent confirmation. "
              "Early/recent banks use training transitions only and freeze all context except preference. "
              "Historical grid returns are reused only for readout; new final grids verify reproduction. "
              "Primary: within-branch interval utility changes versus early-bank score changes, per seed. "
              "Recent bank and between-branch gaps are secondary. No threshold fit or selection feedback.")
    base.write_report(output, report)
    try:
        for name in BRANCHES:
            branch = None if name == "full" else previous.replay_state(
                base.clone_state(state), reference["branches"]["full"]["block_history"])
            if branch is not None:
                observation = dict(step=CUT, banks=measure_cost(branch["trainer"], banks))
                assert observation == report["branches"]["full"]["observations"][0]
                report["branches"][name]["observations"].append(observation)
            router = base.ComponentRouter("shared" if name == "full" else name, trace, report["completed_budget"],
                cut=CUT, block_end=BLOCK_END, learning_starts=KWARGS["learning_starts"])
            probe = Probe(name, seed, router, report, output, reference, banks)
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
        result = summarize_seed(report, reference)
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
    runs = [json.loads((Path(root) / f"seed{s}_readout.json").read_text()) for s in SEEDS]
    for seed, row in zip(SEEDS, runs):
        assert (row["protocol"], row["seed"], row["complete"], row["validation_passed"]) == (PROTOCOL, seed, True, True)
    return dict(protocol=PROTOCOL, seeds=list(SEEDS), complete=True, runs=runs,
        completed_budget={k: sum(r["completed_budget"][k] for r in runs) for k in planned_budget()},
        per_seed_tracking=[dict(seed=r["seed"], **r["tracking"]) for r in runs],
        scope="Report seed-level evidence; the 12 correlated intervals per seed are not independent replicates.")


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
