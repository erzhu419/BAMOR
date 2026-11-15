"""Audit two fixed critic-only gates using existing paired branch returns."""

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch

import diagnose_candidate_calibration as probe


OUTPUT = Path("results/iclr_candidate_gates_20260919")
GATES = ("original", "agreement", "separated")


def gate_masks(critic_q, original_q):
    """Filter the original winners; rejection falls back to their own action."""
    eligible = original_q[1:] > original_q[0]
    agreement = eligible & np.all(critic_q[1:] > critic_q[0], axis=1)
    separated = eligible & (critic_q[1:].min(axis=1) > critic_q[0].max())
    return {"original": eligible, "agreement": agreement, "separated": separated}


@torch.no_grad()
def critic_values(model, obs, actions, context, weight):
    states = torch.as_tensor(obs, dtype=torch.float32).unsqueeze(0).expand(3, -1)
    actions = torch.as_tensor(actions)
    contexts = context.expand(3, -1)
    values = [q(states, actions, contexts) for q in (model.trainer.q1, model.trainer.q2)]
    w = torch.as_tensor(weight)
    separate = torch.stack([(q * w).sum(-1) for q in values], dim=1)
    original = (torch.minimum(*values) * w).sum(-1)
    return separate.numpy(), original.numpy()


def reconstruct_case(env, model, reference, case, prefix_steps):
    weight = np.asarray(case["weight"], dtype=np.float32)
    history = probe.collect_prefix(env, reference, case["mode"], case["reset_seed"],
                                   weight, prefix_steps)
    trackers = model.trackers(history)
    actions, q, metadata = probe.probe_actions(
        model, history[-1][3], weight, trackers, case["random_seed"])
    for key in ("candidate_preferences", "cross_index", "own_resample_index"):
        assert metadata[key] == case[key], f"Reconstruction mismatch: {key}"
    for key in ("candidate_q", "own_resample_q"):
        np.testing.assert_allclose(metadata[key], case[key], rtol=1e-6, atol=1e-5)
    np.testing.assert_allclose(q, case["q_values"], rtol=1e-6, atol=1e-5)
    separate, original = critic_values(
        model, history[-1][3], actions, model.context(weight, trackers), weight)
    np.testing.assert_allclose(original, q, rtol=1e-6, atol=1e-5)
    masks = gate_masks(separate, np.asarray(case["q_values"]))
    return {key: case[key] for key in ("mode", "weight", "reset_seed")} | {
        "critic_q": separate.tolist(), "gates": {k: v.tolist() for k, v in masks.items()},
        "reconstruction_max_q_difference": float(np.abs(q - case["q_values"]).max()),
        "effects": {rule: {str(h): case["effects"][rule][str(h)]["mean_gain"]
                           for h in probe.HORIZONS} for rule in probe.CONTINUATIONS}}


def gate_statistics(gains, eligible, accepted):
    gated = np.where(accepted, gains, 0.0)
    positive = eligible & (gains > 0)
    negative = eligible & (gains < 0)
    rejected = eligible & ~accepted
    return {"states": len(gains), "eligible": int(eligible.sum()),
            "accepted": int(accepted.sum()), "rejected": int(rejected.sum()),
            "mean_gain": float(gated.mean()),
            "change_from_original": float((gated - np.where(eligible, gains, 0.)).mean()),
            "accepted_mean_gain": float(gains[accepted].mean()) if accepted.any() else None,
            "positive_retained": int((accepted & positive).sum()),
            "positive_rejected": int((rejected & positive).sum()),
            "negative_retained": int((accepted & negative).sum()),
            "negative_rejected": int((rejected & negative).sum()),
            "positive_mass_retained_fraction": (
                float(gains[accepted & positive].sum() / gains[positive].sum())
                if positive.any() else None),
            "negative_mass_removed_fraction": (
                float(gains[rejected & negative].sum() / gains[negative].sum())
                if negative.any() else None)}


def summarize(cases, horizons=probe.HORIZONS):
    eligible = np.array([c["gates"]["original"] for c in cases], dtype=bool)
    out = {}
    for rule in probe.CONTINUATIONS:
        out[rule] = {}
        for h in horizons:
            gains = np.array([c["effects"][rule][str(h)] for c in cases])
            out[rule][str(h)] = {}
            for gate in GATES:
                accepted = np.array([c["gates"][gate] for c in cases], dtype=bool)
                out[rule][str(h)][gate] = {
                    label: {metric: gate_statistics(gains[:, i, j], eligible[:, i], accepted[:, i])
                            for j, metric in enumerate(("raw", "bonus_adjusted"))}
                    for i, label in enumerate(probe.LABELS[1:])}
    return out


def run(args):
    torch.set_num_threads(1)
    start = time.monotonic()
    source = args.probes / f"{args.method}_seed{args.seed}.json"
    data = json.loads(source.read_text())
    assert data["protocol"] == "paired_candidate_calibration_v1" and not data["smoke"]
    assert (data["method"], data["seed"]) == (args.method, args.seed)
    assert data["prefix_reference"] == probe.METHODS[0] and data["prefix_steps"] == 128
    assert data["horizons"] == list(probe.HORIZONS) and data["repeats"] == 4
    assert data["candidate_count"] == 5 and data["candidate_strategy"] == "mixed"
    assert data["gamma"] == .99 and data["alpha"] == .2
    assert len(data["cases"]) == 24
    model = probe.load_model(args.models, args.method, args.seed)
    reference = (model if args.method == probe.METHODS[0] else
                 probe.load_model(args.models, probe.METHODS[0], args.seed))
    env = probe.training.make_piecewise_mo_env("mo-halfcheetah-v5", seed=args.seed)
    try:
        cases = [reconstruct_case(env, model, reference, c, data["prefix_steps"])
                 for c in data["cases"]]
    finally:
        env.close()
    result = {"protocol": "fixed_candidate_gates_v1", "source": str(source),
              "method": args.method, "seed": args.seed, "cases": cases,
              "summary": summarize(cases),
              "balanced": summarize([c for c in cases if c["weight"] == [.5, .5]]),
              "seconds": time.monotonic() - start,
              "limitations": data["limitations"] + [
                  "Gates filter original selected actions only; no reranking or policy updates.",
                  "Critic disagreement is not a statistical confidence bound.",
                  "Previously observed development returns; no independent confirmation."]}
    args.output.mkdir(parents=True, exist_ok=True)
    path = args.output / f"{args.method}_seed{args.seed}.json"
    path.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    print(json.dumps({"output": str(path), "states": len(cases), "seconds": result["seconds"],
                      "accepted": {g: np.array([c["gates"][g] for c in cases]).sum(0).tolist()
                                   for g in GATES}}), flush=True)
    print("DONE", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--models", type=Path, default=probe.ROOT)
    parser.add_argument("--probes", type=Path, default=probe.OUTPUT)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    parser.add_argument("--method", choices=probe.METHODS, required=True)
    parser.add_argument("--seed", type=int, choices=(26, 27, 28), required=True)
    run(parser.parse_args())
