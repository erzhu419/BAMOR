"""Prespecified earlier-block actor-selection contrasts, not adaptive deployment."""

import argparse
import json
from pathlib import Path

import numpy as np
import torch

import run_terminal_rng_contrasts as terminal


PROTOCOL = "nonterminal_rng_contrasts_v1"
ROOT = Path("results/iclr_nonterminal_rng_contrasts_20260921")
WINDOWS = {1003: (23000, 24500), 1004: (29000, 30500), 1005: (38000, 39500)}
matched, forks, training = terminal.matched, terminal.forks, terminal.training
METHOD, KWARGS = terminal.METHOD, forks.KWARGS


def select_window(reference, seed):
    assert (reference["method"], reference["env"], reference["seed"],
            reference["total_steps"], reference["switch_interval"]) == (
                METHOD, "mo-halfcheetah-v5", seed, 50000, 25000)
    assert reference["progress_factorial_completion"]["protocol"] == forks.previous.PROTOCOL
    blocks = reference["adaptive_operator_final"]["block_history"]
    operator = forks.EXTREME if seed == 1004 else terminal.OPERATOR
    assert blocks[-1]["operator"] == operator
    eligible = [b for b in blocks if b["operator"] == operator and b["complete"]
                and b["steps"] == 1500 and 20000 < b["end_step"] < 48500]
    chosen = min(eligible, key=lambda b: b["end_step"])
    cut, end = chosen["end_step"] - chosen["steps"], chosen["end_step"]
    assert (cut, end) == WINDOWS[seed]
    assert reference["evaluation_protocol"] == "locked_full_simplex_v1"
    np.testing.assert_array_equal(reference["evaluation_weights"], training.full_simplex_grid(2))
    assert reference["ref_point"] == [-50., -3000.]
    assert reference["episode_seeds_per_mode"] == [
        [100000 + 1000 * seed + 10 * m + i for i in range(3)] for m in range(4)]
    return cut, end, operator


def history_check(log, blocks, reference, step):
    episodes = [r for r in reference["episode_rewards"] if r["step"] <= step]
    expected_blocks = [b for b in reference["adaptive_operator_final"]["block_history"]
                       if b["end_step"] <= step]
    modes = [h for h in reference["ground_truth"]["mode_history"] if h[0] <= step]
    checks = dict(episodes_matched=log["episode_rewards"] == episodes,
                  blocks_matched=blocks == expected_blocks,
                  schedule_matched=np.array_equal(log["ground_truth"]["mode_history"], modes))
    return {**checks, "passed": bool(all(checks.values())),
            "complete_episodes": len(episodes), "complete_blocks": len(expected_blocks)}


class CaptureAndContinue(forks.CaptureState):
    def __init__(self, step, on_capture=None):
        super().__init__(step)
        self.on_capture = on_capture

    def __call__(self, state):
        super().__call__(state)
        # Cloning must not perturb the continuing factual random stream.
        np.random.set_state(state["numpy_rng"])
        torch.set_rng_state(state["torch_rng"])
        if self.on_capture is not None:
            self.on_capture(self.state)
        return False


class BlockIntervention(matched.MatchedIntervention):
    protocol = PROTOCOL

    def __init__(self, name, trace, cut, end, operator):
        assert name in ("factual", "own_matched")
        super().__init__(name, trace, start=cut + 1, end=end, expected_operator=operator)
        config = dict(terminal.CONFIG)
        if operator == forks.EXTREME:
            config.update(preference_relabeling_rate=0., critic_relabel_copies=1, behavior_clone_coef=0.)
        assert self.expected_config == config

    def configure(self, step, trainer):
        if step >= self.start:
            assert step <= self.end
            super().configure(step, trainer)

    def after_step(self, step, trainer, encoder, env, operator):
        if step >= self.start:
            super().after_step(step, trainer, encoder, env, operator)


def grid_diagnosis(branches, weights):
    weights, rows = np.asarray(weights), {}
    for name, branch in branches.items():
        returns = np.asarray(branch["terminal_returns"])
        actual = np.einsum("wi,mwi->mw", weights, returns)
        best = np.einsum("wi,mji->mwj", weights, returns).max(axis=-1)
        rows[name] = dict(matched_utility=actual, grid_best_utility=best, mapping_gap=best - actual)
    delta = {k: rows["own_matched"][k] - rows["factual"][k] for k in rows["factual"]}
    np.testing.assert_allclose(delta["matched_utility"], delta["grid_best_utility"] - delta["mapping_gap"],
                               atol=1e-9, rtol=0)
    return dict(per_weight_differences={k: v.tolist() for k, v in delta.items()},
                mean_differences_per_mode={k: v.mean(axis=1).tolist() for k, v in delta.items()},
                grid_best_wins_per_mode=(delta["grid_best_utility"] > 0).sum(axis=1).tolist(),
                scope="Finite saved-grid diagnostic using evaluation outcomes; never a deployed selector.")


def endpoint_row(data, report, end, expected_config):
    assert data["total_steps"] == end and data["training_interactions"] == end
    assert data["evaluation_interactions"] == 252000
    assert [s["step"] for s in data["policy_archive_snapshots"]] == [end]
    assert data["update_routing"]["config"] == expected_config
    for key, value in report["evaluation"].items():
        assert data[key] == value, key
    returns = np.asarray([[p["return"] for p in front if p["archive_step"] == end]
                          for front in data["pareto_per_mode"]])
    assert returns.shape == (4, 21, 2)
    return dict(final_hv=data["final_hv_mean_mode"], final_mode_hv=data["final_hv_per_mode"],
                terminal_returns=returns.tolist(), last_episodes=data["episode_rewards"][-2:],
                update_routing=data["update_routing"], evaluation_step=end)


def run(seed, root=ROOT):
    assert seed in WINDOWS
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    output = root / f"seed{seed}.json"
    if output.exists():
        saved = json.loads(output.read_text())
        if saved.get("complete"):
            assert (saved["protocol"], saved["seed"], saved["fork_step"], saved["end_step"]) == (
                PROTOCOL, seed, *WINDOWS[seed])
            assert saved["controls_passed"] and saved["rng_alignment_passed"]
            print(f"DONE nonterminal_rng_contrasts seed={seed} reusing complete report", flush=True)
            return saved
        raise RuntimeError("Incomplete attempt preserved; review it and use a new --root for a retry")
    reference_path = terminal.REFERENCE_ROOT / f"{METHOD}_seed{seed}.json"
    reference = json.loads(reference_path.read_text())
    cut, end, operator = select_window(reference, seed)
    report = dict(protocol=PROTOCOL, seed=seed, fork_step=cut, end_step=end, operator=operator,
                  reference_path=str(reference_path), complete=False, controls_passed=False,
                  rng_alignment_passed=False, branches={},
                  evaluation={k: reference[k] for k in ("evaluation_weights", "episode_seeds_per_mode", "ref_point")},
                  completed_budget=dict(training_interactions=0, learner_updates=0,
                      evaluation_interactions=0, environment_initialization_probe_steps=0))
    budget = report["completed_budget"]
    forks.write_report(output, report)

    def captured(state):
        log = {**state["log_data"], "ground_truth": state["env"].get_ground_truth()}
        report["prefix_reproduction"] = history_check(log, state["adaptive_selector"].block_history, reference, cut)
        budget.update(training_interactions=cut, learner_updates=cut - KWARGS["learning_starts"] + 1,
                      environment_initialization_probe_steps=2)
        forks.write_report(output, report)
        assert report["prefix_reproduction"]["passed"]
        assert state["trainer"].buffer.size == cut and state["episode_steps"] == 0

    capture, trace = CaptureAndContinue(cut, captured), []
    try:
        for name in ("factual", "own_matched"):
            router = BlockIntervention(name, trace, cut, end, operator)
            kwargs = dict(KWARGS, archive_steps=[end], update_router=router)
            if name == "factual":
                kwargs["state_probe"] = capture
            else:
                kwargs["resume_state"] = forks.clone_state(capture.state)
            try:
                data = training.train_method("mo-halfcheetah-v5", METHOD, end, 25000, seed,
                                             str(root / f"seed{seed}" / name), **kwargs)
            finally:
                router.close()
            row = endpoint_row(data, report, end, router.expected_config)
            if name == "factual":
                row["reproduction"] = history_check(data, data["adaptive_operator_final"]["block_history"], reference, end)
                report["controls_passed"] = row["reproduction"]["passed"]
                assert len(trace) == end - cut
            else:
                assert data["continuation_from_step"] == cut and data["training_interactions_this_call"] == end - cut
                report["rng_alignment_passed"] = row["update_routing"]["rng_alignment"]["passed"]
                budget["environment_initialization_probe_steps"] += data["environment_initialization_probe_steps"]
            budget["training_interactions"] += end - cut
            budget["learner_updates"] += end - cut
            budget["evaluation_interactions"] += data["evaluation_interactions"]
            report["branches"][name] = row
            forks.write_report(output, report)
            if not report["controls_passed"]:
                raise RuntimeError("Uninterrupted factual history did not reproduce")
            if name == "own_matched" and not report["rng_alignment_passed"]:
                raise RuntimeError("Random states did not align")
            print(f"BRANCH_DONE seed={seed} {name} endpoint={end} hv={row['final_hv']:.6f}", flush=True)
        report["paired_comparisons"] = {"own_matched_minus_factual": matched.paired_difference(
            report["branches"]["own_matched"], report["branches"]["factual"], report["evaluation"]["evaluation_weights"])}
        report["grid_diagnosis"] = grid_diagnosis(report["branches"], report["evaluation"]["evaluation_weights"])
        assert budget == dict(training_interactions=end + end - cut,
                              learner_updates=end + end - cut - KWARGS["learning_starts"] + 1,
                              evaluation_interactions=504000, environment_initialization_probe_steps=4)
        report["complete"] = True
        report["scope"] = ("Observed seeds, metadata-selected earlier blocks. Endpoint policies use their own encoder; "
            "historical grids used a later encoder and are not endpoint controls. The factual run is uninterrupted; "
            "historical episode/block/schedule records must match. Stage and encountered context can both differ "
            "from terminal forks. No adaptive deployment or fresh confirmation. Failed attempts/tests stay separate.")
        forks.write_report(output, report)
        print(f"DONE nonterminal_rng_contrasts seed={seed} {json.dumps(budget)}", flush=True)
        return report
    finally:
        if capture.state is not None:
            capture.state["env"].close()
            capture.state["eval_env"].close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, choices=WINDOWS, required=True)
    parser.add_argument("--root", type=Path, default=ROOT)
    args = parser.parse_args()
    run(args.seed, args.root)
