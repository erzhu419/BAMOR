"""Single-factor v27 ablation: periodically refresh only replay latent contexts."""

import argparse
import json
from pathlib import Path
import time

import numpy as np

import diagnose_context_drift as drift


training = drift.training
METHOD = drift.forks.METHOD
PROTOCOL = "periodic_causal_replay_context_refresh_v1"
MAIN_STEPS = 50000
LEARNING_STARTS = 5000
INTERVAL = 1000
REFRESH_STEPS = tuple(range(LEARNING_STARTS, MAIN_STEPS, INTERVAL))


class ReplayContextRefresh:
    def __init__(self, output, steps=REFRESH_STEPS):
        self.output, self.steps = Path(output), tuple(steps)
        self.cases = []

    def __call__(self, step, trainer, encoder, env, operator):
        if step not in self.steps:
            return False
        started = time.monotonic()
        buffer, width = trainer.buffer, encoder.latent_dim
        assert buffer.size == step and trainer.context_dim == 2 + 3 + width
        latent = drift.reencode_history(buffer, encoder)
        rms = {}
        for field, values in (("contexts", latent[:-1]), ("next_contexts", latent[1:])):
            stored = getattr(buffer, field)[:step, -width:]
            difference = values.astype(np.float64) - stored
            rms[field] = float(np.sqrt(np.mean(difference ** 2)))
            stored[:] = values
        self.cases.append({"step": step, "transitions_reencoded": step,
                           "latent_change_rms": rms, "seconds": time.monotonic() - started})
        self.save()
        print(f"REPLAY_REFRESH step={step} transitions={step} "
              f"context_rms={rms['contexts']:.6f} seconds={self.cases[-1]['seconds']:.2f}", flush=True)
        return False

    def state_dict(self, complete=False):
        return {
            "protocol": PROTOCOL, "complete": complete, "scheduled_steps": list(self.steps),
            "cases": self.cases,
            "transitions_reencoded": sum(c["transitions_reencoded"] for c in self.cases),
            "refresh_seconds": sum(c["seconds"] for c in self.cases),
            "extra_environment_steps": 0, "extra_updates": 0,
            "timing": "After normal learner/encoder updates; first used by the next update. "
                      "No refresh after the final training update.",
            "scope": "Replay latent suffixes only. Current encoder reconstructs the full observed "
                     "prefix and continuous EMA across episodes. Online tracker, replay sampling, "
                     "nonlearned features, preferences and optimizer states are unchanged by refresh. "
                     "No ground-truth modes or evaluation returns enter training. This is periodic "
                     "replay refresh, not exact per-update or online-history synchronization.",
        }

    def save(self, complete=False):
        self.output.parent.mkdir(parents=True, exist_ok=True)
        self.output.write_text(json.dumps(self.state_dict(complete), indent=2, allow_nan=False) + "\n")


def paired_protocol_check(result, reference):
    keys = ("method", "env", "seed", "total_steps", "switch_interval", "context_dim",
            "critic_target_rule", "bootstrap_terminal_rule", "critic_candidate_context_rule",
            "v2_context", "evaluation_protocol", "evaluation_weights", "episode_seeds_per_mode",
            "ref_point", "archive_context", "policy_archive_snapshots")
    checks = {key: result[key] == reference[key] for key in keys}
    checks["mode_schedule"] = np.array_equal(
        result["ground_truth"]["mode_history"], reference["ground_truth"]["mode_history"])
    prefix = lambda d: [e for e in d["episode_rewards"] if e["step"] <= LEARNING_STARTS]
    checks["pre_intervention_episode_returns"] = prefix(result) == prefix(reference)
    assert all(checks.values()), f"Paired protocol differs: {checks}"
    return checks


def run(seed, output_root):
    assert seed in (26, 27, 28)
    output = Path(output_root)
    reference_path = drift.forks.REFERENCE_ROOT / f"{METHOD}_seed{seed}.json"
    reference = json.loads(reference_path.read_text())
    assert reference["total_steps"] == MAIN_STEPS
    assert reference["evaluation_protocol"] == "locked_full_simplex_v1"
    refresh = ReplayContextRefresh(output / f"seed{seed}_refresh.json")
    refresh.save()
    result = training.train_method(
        "mo-halfcheetah-v5", METHOD, MAIN_STEPS, 25000, seed, str(output),
        hidden=64, learning_starts=LEARNING_STARTS, eval_freq=100000, device="cpu",
        training_probe=refresh)
    assert [c["step"] for c in refresh.cases] == list(REFRESH_STEPS)
    assert [s["step"] for s in result["policy_archive_snapshots"]] == list(range(5000, 50001, 2500))
    checks = paired_protocol_check(result, reference)
    result["replay_context_refresh"] = {
        **refresh.state_dict(complete=True), "reference": str(reference_path),
        "paired_protocol_checks": checks, "main_environment_steps": MAIN_STEPS,
        "total_training_interactions": MAIN_STEPS, "normal_updates": MAIN_STEPS - LEARNING_STARTS + 1,
    }
    path = output / f"{METHOD}_seed{seed}.json"
    path.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    refresh.save(complete=True)
    print(f"DONE replay_context_refresh seed={seed} interactions={MAIN_STEPS} "
          f"refreshes={len(refresh.cases)} seconds={refresh.state_dict()['refresh_seconds']:.2f}", flush=True)
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, choices=(26, 27, 28), required=True)
    parser.add_argument("--output-root", type=Path,
                        default=Path("results/iclr_replay_context_refresh_20260920"))
    args = parser.parse_args()
    run(args.seed, args.output_root)
