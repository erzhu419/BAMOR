"""Two observed-seed contrasts for RNG-matched actor selection with native BC."""

import argparse
import json
from pathlib import Path

import numpy as np

import run_terminal_rng_matched as matched


PROTOCOL = "terminal_rng_contrasts_v1"
SEEDS = (1003, 1005)
BRANCHES = ("factual", "own_matched")
OPERATOR = "extreme:edge_relabel_envelope_policy_bc"
CONFIG = dict(preference_relabeling_rate=1.0, preference_relabeling_strategy="edge",
              critic_relabel_copies=3, envelope_target_candidates=5, vector_critic_coef=.06,
              envelope_policy_candidates=5, behavior_clone_coef=.004)
ROOT = Path("results/iclr_terminal_rng_contrasts_20260921")
forks, training = matched.previous, matched.training
REFERENCE_ROOT = forks.previous.OUTPUT_ROOT
CUT, END, METHOD = matched.CUT, matched.END, matched.METHOD


class ContrastIntervention(matched.MatchedIntervention):
    protocol = PROTOCOL

    def __init__(self, name, rng_trace, start=CUT + 1, end=END):
        assert name in BRANCHES
        super().__init__(name, rng_trace, start=start, end=end, expected_operator=OPERATOR)
        assert self.expected_config == CONFIG


def validate_reference(data, seed):
    assert (data["method"], data["env"], data["seed"], data["total_steps"], data["switch_interval"]) == (
        METHOD, "mo-halfcheetah-v5", seed, END, 25000)
    assert data["progress_factorial_completion"]["protocol"] == forks.previous.PROTOCOL
    final = data["adaptive_operator_final"]["block_history"][-1]
    assert (final["operator"], final["end_step"], final["steps"], final["complete"]) == (
        OPERATOR, END, END - CUT, True)
    assert data["evaluation_protocol"] == "locked_full_simplex_v1"
    np.testing.assert_array_equal(data["evaluation_weights"], training.full_simplex_grid(2))
    assert data["ref_point"] == [-50., -3000.]
    assert data["episode_seeds_per_mode"] == [
        [100000 + 1000 * seed + 10 * mode + i for i in range(3)] for mode in range(4)]


def run(seed, root=ROOT):
    assert seed in SEEDS
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    output = root / f"seed{seed}.json"
    if output.exists():
        saved = json.loads(output.read_text())
        if saved.get("complete"):
            assert (saved["protocol"], saved["seed"]) == (PROTOCOL, seed)
            assert saved["controls_passed"] and saved["rng_alignment_passed"]
            print(f"DONE terminal_rng_contrasts seed={seed} reusing complete report", flush=True)
            return saved
        raise RuntimeError("Incomplete attempt preserved; review it and use a new --root for a retry")
    reference_path = REFERENCE_ROOT / f"{METHOD}_seed{seed}.json"
    reference = json.loads(reference_path.read_text())
    validate_reference(reference, seed)
    report = dict(protocol=PROTOCOL, seed=seed, fork_step=CUT, end_step=END,
                  reference_path=str(reference_path), complete=False, controls_passed=False,
                  rng_alignment_passed=False, branches={},
                  evaluation={k: reference[k] for k in
                              ("evaluation_weights", "episode_seeds_per_mode", "ref_point")},
                  completed_budget=dict(training_interactions=0, learner_updates=0,
                      evaluation_interactions=0, environment_initialization_probe_steps=0))
    forks.write_report(output, report)
    run_root = root / f"seed{seed}"
    capture = forks.CaptureState(CUT)
    training.train_method("mo-halfcheetah-v5", METHOD, END, 25000, seed, str(run_root / "prefix"),
                          state_probe=capture, **forks.KWARGS)
    state = capture.state
    budget = report["completed_budget"]
    budget.update(training_interactions=CUT, learner_updates=CUT - forks.KWARGS["learning_starts"] + 1,
                  environment_initialization_probe_steps=2)
    forks.write_report(output, report)
    trace = []
    try:
        report["prefix_reproduction"] = forks.prefix_check(state, {"factual": reference})
        forks.write_report(output, report)
        for name in BRANCHES:
            branch = forks.clone_state(state)
            branch["policy_archive"] = []
            router = ContrastIntervention(name, trace)
            try:
                data = training.train_method("mo-halfcheetah-v5", METHOD, END, 25000, seed,
                    str(run_root / name), resume_state=branch, update_router=router, **forks.KWARGS)
            finally:
                router.close()
            assert data["continuation_from_step"] == CUT and data["training_interactions_this_call"] == END - CUT
            assert data["evaluation_interactions"] == 252000
            assert [s["step"] for s in data["policy_archive_snapshots"]] == [END]
            assert data["update_routing"]["config"] == CONFIG
            for key, value in report["evaluation"].items():
                assert data[key] == value, key
            budget["training_interactions"] += END - CUT
            budget["learner_updates"] += END - CUT
            budget["evaluation_interactions"] += data["evaluation_interactions"]
            budget["environment_initialization_probe_steps"] += data["environment_initialization_probe_steps"]
            row = dict(final_hv=data["final_hv_mean_mode"], final_mode_hv=data["final_hv_per_mode"],
                       last_episodes=data["episode_rewards"][-2:], update_routing=data["update_routing"],
                       terminal_returns=forks.terminal_returns(data).tolist())
            report["branches"][name] = row
            if name == "factual":
                row["reproduction"] = forks.reproduction(data, reference)
                report["controls_passed"] = row["reproduction"]["passed"]
                forks.write_report(output, report)
                if not report["controls_passed"]:
                    raise RuntimeError("Factual control did not reproduce; new branch not interpretable")
            else:
                report["rng_alignment_passed"] = row["update_routing"]["rng_alignment"]["passed"]
                forks.write_report(output, report)
                if not report["rng_alignment_passed"]:
                    raise RuntimeError("Random states did not align; matched-draw interpretation invalid")
            print(f"BRANCH_DONE seed={seed} {name} final_hv={row['final_hv']:.6f}", flush=True)
        assert len(trace) == END - CUT
        report["paired_comparisons"] = {"own_matched_minus_factual": matched.paired_difference(
            report["branches"]["own_matched"], report["branches"]["factual"],
            report["evaluation"]["evaluation_weights"])}
        report["complete"] = True
        report["scope"] = ("Previously observed seed contrast, not fresh confirmation. Native relabeling "
            "and BC retained; candidate selection also changes the action used by BC. Policies, replay "
            "values and trajectories can diverge under matching random draws. No adaptive deployment; "
            "failed attempts and verification costs stay separate.")
        forks.write_report(output, report)
        print(f"DONE terminal_rng_contrasts seed={seed} {json.dumps(budget)}", flush=True)
        return report
    finally:
        state["env"].close()
        state["eval_env"].close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, choices=SEEDS, required=True)
    parser.add_argument("--root", type=Path, default=ROOT)
    args = parser.parse_args()
    run(args.seed, args.root)
