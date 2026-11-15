"""Independent Monte Carlo scores for frozen update forks, without Q or entropy."""

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch

import diagnose_update_forks as forks
import train_bamor_mujoco as training


CACHED_ROOT = Path("results/iclr_update_forks_20260919")
SCORE_REPEATS = 3
SCORE_SEED_OFFSET = 20000


def raw_scores(returns):
    on, off = (np.asarray(returns[k]) for k in ("on", "off"))
    assert on.shape == off.shape and on.shape[0] == len(forks.WEIGHTS)
    assert np.isfinite(on).all() and np.isfinite(off).all()
    scalar = np.einsum("wrhd,wd->wrh", on - off, forks.WEIGHTS)
    mean = scalar.mean(1)
    return {"paired_scalar_on_minus_off": scalar.tolist(),
            "mean_scalar_on_minus_off": mean.tolist(), "choose_on": (mean > 0).tolist()}


def run_probe(trainer, encoder, env, seed, cached, horizons):
    config = {k: getattr(trainer, k) for k in forks.CONFIG}
    assert config == cached["config"]
    reset_seeds = [seed + SCORE_SEED_OFFSET + r for r in range(SCORE_REPEATS)]
    assert set(reset_seeds).isdisjoint(cached["reset_seeds"])
    with forks.isolated_rng(seed):
        indices = np.random.permutation(trainer.buffer.size)
        audit_size = min(1024, trainer.buffer.size // 4)
        audit = forks.batch_at(trainer.buffer, indices[:min(audit_size, trainer.batch_size)])
        replay = forks.ReplaySubset(trainer.buffer, indices[audit_size:])
        assert (audit_size, len(audit[0]), replay.size) == (
            cached["audit_pool_size"], cached["audit_batch_size"], cached["replay_pool_size"])
        # The old loss only checks branch reconstruction; it never enters the score.
        np.testing.assert_array_equal(forks.audit_losses(trainer, trainer, audit, seed + 200),
                                      cached["own_soft_loss_with_bc_frozen_critic"]["base"])
        returns = {}
        for label in ("on", "off"):
            branch = forks.clone_trainer(trainer, replay)
            if label == "off":
                branch.envelope_policy_candidates = 1
            for update in range(cached["updates_per_branch"]):
                with forks.isolated_rng(seed + 1000 + update):
                    branch.update()
            np.testing.assert_array_equal(
                forks.audit_losses(branch, trainer, audit, seed + 200),
                cached["own_soft_loss_with_bc_frozen_critic"][label],
                err_msg=f"Reconstructed {label} branch differs from cached outcome policy")
            returns[label] = forks.rollout_returns(branch, encoder, env, reset_seeds, horizons)
            del branch
        scores = raw_scores(returns)
        if config["envelope_policy_candidates"] == 1:
            np.testing.assert_array_equal(returns["on"], returns["off"])
        return {"config": config, "matched_original_losses": True,
                "score_reset_seeds": reset_seeds, "score_returns": returns, "scores": scores,
                "heldout_reset_seeds": cached["reset_seeds"],
                "cached_return_effects": cached["effects"],
                "extra_environment_steps": 2 * len(forks.WEIGHTS) * SCORE_REPEATS * max(horizons),
                "diagnostic_resets": 2 * len(forks.WEIGHTS) * SCORE_REPEATS,
                "fork_updates": 2 * cached["updates_per_branch"]}


class RawReturnProbe:
    def __init__(self, seed, output, cached):
        if (cached["protocol"] != "live_paired_update_forks_v1"
                or cached["method"] != forks.METHOD or cached["seed"] != seed
                or not cached["complete"] or not cached["base_reproduction"]["matched"]):
            raise ValueError("Need completed, reproduced fork outcomes from the same seed")
        assert cached["steps"] == [c["step"] for c in cached["cases"]]
        self.seed, self.output = seed, Path(output)
        self.cases = {c["step"]: c for c in cached["cases"]}
        self.report = {
            "protocol": "raw_return_fork_scores_v1", "method": forks.METHOD, "seed": seed,
            "steps": cached["steps"], "weights": cached["weights"],
            "horizons": cached["horizons"], "primary_horizon": max(cached["horizons"]),
            "score_repeats": SCORE_REPEATS, "score_seed_offset": SCORE_SEED_OFFSET,
            "torch_threads": torch.get_num_threads(), "complete": False, "cases": [],
            "scope": "Direct raw Monte Carlo evaluator, not a learned critic. Deterministic "
                     "branch continuation, current training schedule, no mode-label input, "
                     "entropy bonus or value bootstrap. Three fresh paired score resets are "
                     "disjoint from cached outcome resets. No feedback enters main training. "
                     "Requires resettable simulator copies; not a deployed online selector or "
                     "independent-seed confirmation. Old-mode retention is not measured.",
        }

    def __call__(self, step, trainer, encoder, env, operator):
        if step not in self.cases:
            return False
        cached = self.cases[step]
        assert operator == cached["operator"]
        started = time.monotonic()
        case = run_probe(trainer, encoder, env,
                         10000000 + self.seed * 100000 + step * 100,
                         cached, self.report["horizons"])
        case.update({"step": step, "operator": operator,
                     "seconds": time.monotonic() - started})
        self.report["cases"].append(case)
        self.output.parent.mkdir(parents=True, exist_ok=True)
        self.output.write_text(json.dumps(self.report, indent=2, allow_nan=False) + "\n")
        print(f"RAW_RETURN step={step} matched_losses=True "
              f"extra_steps={case['extra_environment_steps']} seconds={case['seconds']:.1f}",
              flush=True)
        return step == self.report["steps"][-1]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    cached_path = CACHED_ROOT / f"seed{args.seed}.json"
    cached = json.loads(cached_path.read_text())
    assert cached["steps"] == list(forks.STEPS) and cached["horizons"] == [32, 128, 512]
    assert all(c["updates_per_branch"] == 128 for c in cached["cases"])
    reference = json.loads((forks.REFERENCE_ROOT / f"{forks.METHOD}_seed{args.seed}.json").read_text())
    probe = RawReturnProbe(args.seed, args.output, cached)
    probe.report["cached_source"] = str(cached_path)
    result = training.train_method(
        "mo-halfcheetah-v5", forks.METHOD, 50000, 25000, args.seed,
        str(args.output.parent), hidden=64, eval_freq=100000, device="cpu", training_probe=probe)
    probe.report["base_reproduction"] = forks.reproduction_check(result, reference)
    probe.report["complete"] = probe.report["base_reproduction"]["matched"]
    probe.report["cost"] = {
        "main_environment_steps": 50000,
        "extra_environment_steps": sum(c["extra_environment_steps"] for c in probe.report["cases"]),
        "fork_updates": sum(c["fork_updates"] for c in probe.report["cases"]),
        "diagnostic_resets": sum(c["diagnostic_resets"] for c in probe.report["cases"]),
    }
    args.output.write_text(json.dumps(probe.report, indent=2, allow_nan=False) + "\n")
    if not probe.report["complete"]:
        raise RuntimeError("Main trajectory did not reproduce the existing v27 control")
    print(f"RAW_RETURN_COMPLETE seed={args.seed} cost={probe.report['cost']}", flush=True)


if __name__ == "__main__":
    main()
