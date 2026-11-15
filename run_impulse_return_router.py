"""Apply raw-return branch selection only to the evaluated block of updates."""

import argparse
import json
from pathlib import Path

import run_budgeted_return_router as pilot


CONDITIONS = pilot.CONDITIONS[:3]
PROTOCOL = "impulse_raw_return_router_v1"


class ImpulseReturnRouter(pilot.BudgetedReturnRouter):
    def configure(self, step, trainer):
        # train_method has just installed the default portfolio's current arm.
        self.enabled = True
        super().configure(step, trainer)

    def state_dict(self):
        state = super().state_dict()
        del state["enabled"]
        return {**state, "protocol": PROTOCOL, "selection_scope": "fork_updates_only",
                "normal_training_gate": "default_portfolio"}


def run(condition, seed, output_root):
    assert condition in CONDITIONS and seed in (26, 27, 28)
    output = Path(output_root) / condition
    router = ImpulseReturnRouter(condition, seed, output / f"seed{seed}_routing.json")
    result = pilot.training.train_method(
        "mo-halfcheetah-v5", pilot.forks.METHOD, pilot.MAIN_STEPS, 25000, seed, str(output),
        hidden=64, eval_freq=100000, device="cpu", update_router=router,
        archive_steps=pilot.ARCHIVE_MAIN_STEPS)
    observed = [s["interaction_step"] for s in result["policy_archive_snapshots"]]
    assert observed == list(pilot.ARCHIVE_BUDGET_STEPS)
    assert pilot.MAIN_STEPS + router.extra_steps == pilot.TOTAL_BUDGET
    assert [c["main_step"] for c in router.cases] == list(pilot.DECISION_STEPS)
    assert all(c["previous_on"] for c in router.cases)
    result["budgeted_routing_experiment"] = {
        "protocol": PROTOCOL, "condition": condition, "complete": True,
        "selection_scope": "fork_updates_only", "normal_training_gate": "default_portfolio",
        "main_environment_steps": pilot.MAIN_STEPS,
        "trial_environment_steps": router.extra_steps,
        "total_training_interactions": pilot.MAIN_STEPS + router.extra_steps,
        "normal_updates": pilot.MAIN_STEPS - 5000 + 1,
        "fork_updates": len(pilot.DECISION_STEPS) * 2 * pilot.UPDATES,
        "committed_fork_updates": len(pilot.DECISION_STEPS) * pilot.UPDATES,
        "archive_interaction_steps": observed,
        "scope": "Simulator-assisted development ablation. All three conditions use the "
                 "same default portfolio for every normal update, including before the first "
                 "trial. Only the scored fork updates are selected and committed. Learned "
                 "networks and optimizer moments persist; the actor-search gate does not. "
                 "All trial interactions, including identity comparisons, are charged. "
                 "Locked evaluation never feeds selection. Final-encoder archive evaluation.",
    }
    path = output / f"{pilot.forks.METHOD}_seed{seed}.json"
    path.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    print(f"DONE impulse_router condition={condition} seed={seed} "
          f"interactions={pilot.TOTAL_BUDGET}", flush=True)
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--condition", choices=CONDITIONS, required=True)
    parser.add_argument("--seed", type=int, choices=(26, 27, 28), required=True)
    parser.add_argument("--output-root", type=Path,
                        default=Path("results/iclr_impulse_router_20260920"))
    args = parser.parse_args()
    run(args.condition, args.seed, args.output_root)
