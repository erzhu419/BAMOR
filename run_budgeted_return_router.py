"""Budget-matched, simulator-assisted actor-update routing pilot."""

import argparse
import json
from pathlib import Path

import numpy as np

import diagnose_update_forks as forks
from diagnose_raw_return_forks import raw_scores
import train_bamor_mujoco as training


CONDITIONS = ("route", "forced_on", "forced_off", "train_on", "train_off")
DECISION_STEPS = (10000, 20000, 30000, 40000)
MAIN_STEPS = 50000
HORIZONS = (32, 128, 512)
REPEATS = 3
UPDATES = 128
TRIAL_STEPS = 2 * len(forks.WEIGHTS) * REPEATS * max(HORIZONS)
TOTAL_BUDGET = MAIN_STEPS + len(DECISION_STEPS) * TRIAL_STEPS
ARCHIVE_MAIN_STEPS = tuple(range(5000, MAIN_STEPS + 1, 2500))
ARCHIVE_BUDGET_STEPS = tuple(
    s + sum(d <= s for d in DECISION_STEPS) * TRIAL_STEPS for s in ARCHIVE_MAIN_STEPS)


def choose_branch(advantages, condition, previous_on, distinguishable=True):
    if condition == "forced_on":
        return True
    if condition == "forced_off":
        return False
    assert condition == "route"
    # A single-candidate arm provides no evidence to change the persistent gate.
    if not distinguishable:
        return previous_on
    return bool(np.mean(advantages) > 0.)


def commit_branch(trainer, branch):
    # Transfer the networks and their bound optimizers together, retaining live replay.
    live_replay = trainer.buffer
    trainer.__dict__.update(branch.__dict__)
    trainer.buffer = live_replay


class BudgetedReturnRouter:
    def __init__(self, condition, seed, trace_path, decision_steps=DECISION_STEPS,
                 updates=UPDATES, repeats=REPEATS, horizons=HORIZONS):
        assert condition in CONDITIONS[:3]
        self.condition, self.seed, self.trace_path = condition, seed, Path(trace_path)
        self.decision_steps, self.updates = tuple(decision_steps), updates
        self.repeats, self.horizons = repeats, tuple(horizons)
        self.enabled = condition != "forced_off"
        self.on_candidates = None
        self.extra_steps = 0
        self.cases = []

    def configure(self, step, trainer):
        # The normal portfolio refreshes its arm config before every call.
        self.on_candidates = trainer.envelope_policy_candidates
        if not self.enabled:
            trainer.envelope_policy_candidates = 1

    def interaction_step(self, main_step):
        return int(main_step + self.extra_steps)

    def after_step(self, step, trainer, encoder, env, operator):
        if step not in self.decision_steps:
            return
        assert self.on_candidates is not None
        fork_seed = 30000000 + self.seed * 100000 + step * 100
        resets = [fork_seed + 20000 + r for r in range(self.repeats)]
        old_enabled = self.enabled
        with forks.isolated_rng(fork_seed):
            indices = np.random.permutation(trainer.buffer.size)
            excluded = min(1024, trainer.buffer.size // 4)
            replay = forks.ReplaySubset(trainer.buffer, indices[excluded:])
            branches, returns = {}, {}
            for label in ("on", "off"):
                branch = forks.clone_trainer(trainer, replay)
                branch.envelope_policy_candidates = self.on_candidates if label == "on" else 1
                for update in range(self.updates):
                    with forks.isolated_rng(fork_seed + 1000 + update):
                        branch.update()
                returns[label] = forks.rollout_returns(branch, encoder, env, resets, self.horizons)
                branches[label] = branch
            scores = raw_scores(returns)
            advantages = np.asarray(scores["mean_scalar_on_minus_off"])[:, -1]
            if self.on_candidates == 1:
                np.testing.assert_array_equal(returns["on"], returns["off"])
            self.enabled = choose_branch(advantages, self.condition, old_enabled,
                                         distinguishable=self.on_candidates > 1)
            chosen = "on" if self.enabled else "off"
            commit_branch(trainer, branches[chosen])
        extra = 2 * len(forks.WEIGHTS) * self.repeats * max(self.horizons)
        self.extra_steps += extra
        self.cases.append({
            "main_step": step, "interaction_step": self.interaction_step(step),
            "operator": operator, "on_candidates": self.on_candidates,
            "previous_on": old_enabled, "chosen": chosen,
            "preference_advantages": advantages.tolist(),
            "mean_utility_advantage": float(advantages.mean()),
            "score_reset_seeds": resets, "score_returns": returns,
            "extra_environment_steps": extra, "fork_updates": 2 * self.updates,
            "committed_updates": self.updates,
        })
        self.trace_path.parent.mkdir(parents=True, exist_ok=True)
        self.trace_path.write_text(json.dumps(self.state_dict(), indent=2, allow_nan=False) + "\n")
        print(f"ROUTE step={step} budget_step={self.interaction_step(step)} "
              f"condition={self.condition} chosen={chosen} utility_delta={advantages.mean():.6f}",
              flush=True)

    def state_dict(self):
        return {"condition": self.condition, "decision_steps": list(self.decision_steps),
                "horizons": list(self.horizons), "repeats": self.repeats,
                "updates_per_branch": self.updates, "enabled": self.enabled,
                "extra_environment_steps": self.extra_steps, "cases": self.cases}


def run(condition, seed, output_root):
    assert condition in CONDITIONS and seed in (26, 27, 28)
    output = Path(output_root) / condition
    uses_trials = condition in CONDITIONS[:3]
    method = ("bamor_v27_aligned_target_ownpref_actor_fixed_archive_mix"
              if condition == "train_off" else forks.METHOD)
    router = (BudgetedReturnRouter(condition, seed, output / f"seed{seed}_routing.json")
              if uses_trials else None)
    main_steps = MAIN_STEPS if uses_trials else TOTAL_BUDGET
    archives = ARCHIVE_MAIN_STEPS if uses_trials else ARCHIVE_BUDGET_STEPS
    result = training.train_method(
        "mo-halfcheetah-v5", method, main_steps, 25000, seed, str(output),
        hidden=64, eval_freq=100000, device="cpu", update_router=router, archive_steps=archives)
    observed = [s["interaction_step"] for s in result["policy_archive_snapshots"]]
    assert observed == list(ARCHIVE_BUDGET_STEPS), "Archive interaction budgets are not matched"
    extra = router.extra_steps if router is not None else 0
    assert main_steps + extra == TOTAL_BUDGET
    if router is not None:
        assert [c["main_step"] for c in router.cases] == list(DECISION_STEPS)
    result["budgeted_routing_experiment"] = {
        "protocol": "budgeted_raw_return_router_v1", "condition": condition, "complete": True,
        "main_environment_steps": main_steps, "trial_environment_steps": extra,
        "total_training_interactions": main_steps + extra,
        "normal_updates": main_steps - 5000 + 1,
        "fork_updates": len(DECISION_STEPS) * 2 * UPDATES if uses_trials else 0,
        "committed_fork_updates": len(DECISION_STEPS) * UPDATES if uses_trials else 0,
        "archive_interaction_steps": observed,
        "scope": "Simulator-assisted development pilot. Route/forced controls share main "
                 "schedule, trial cost and branch-update budget. Train-only controls spend "
                 "all interactions on normal training and may see additional mode exposure. "
                 "Equal interactions do not imply equal compute. Trial data are not added to "
                 "replay; all trial steps are charged. Three-preference average utility, not "
                 "a Pareto dominance guarantee. Locked final evaluation never feeds routing.",
    }
    path = output / f"{method}_seed{seed}.json"
    path.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    print(f"DONE budgeted_router condition={condition} seed={seed} interactions={TOTAL_BUDGET}",
          flush=True)
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--condition", choices=CONDITIONS, required=True)
    parser.add_argument("--seed", type=int, choices=(26, 27, 28), required=True)
    parser.add_argument("--output-root", type=Path,
                        default=Path("results/iclr_budgeted_router_20260920"))
    args = parser.parse_args()
    run(args.condition, args.seed, args.output_root)
