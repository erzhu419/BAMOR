"""Fixed-critic action-radius diagnosis, reusing the recorded 0.05 control."""

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from torch import nn

import diagnose_return_critic as pilot
import summarize_return_critic as validation


g = pilot.gradient
training = pilot.training
PROTOCOL = "fixed_critic_action_radius_v1"
RADII = (.05, .01, .002)
SOURCE_ROOT = pilot.OUTPUT_ROOT
OUTPUT_ROOT = Path("results/iclr_action_radius_20260921")


def load_critics(path):
    state = torch.load(path, map_location="cpu", weights_only=True)
    # Allocate shapes only; load_state_dict restores every train-fitted buffer.
    template = {key: torch.zeros(2, state[f"0.{key}_mean"].numel())
                for key in ("obs", "actions", "contexts")}
    template["rewards"] = torch.zeros(2, state["0.reward_scale"].numel())
    with torch.random.fork_rng(devices=[]):
        critics = nn.ModuleList([pilot.RawQ(template, hidden=state[f"{i}.net.net.0.weight"].shape[0])
                                 for i in range(2)])
    critics.load_state_dict(state)
    return critics.eval().requires_grad_(False)


def load_inputs(condition, seed):
    source = SOURCE_ROOT / condition / f"seed{seed}.json"
    reference = json.loads(source.read_text())
    validation.validate_run(reference, condition, seed)
    networks, policy_path = g.load_models(condition, seed)
    critic_path = source.with_name(f"seed{seed}_raw_critic.pt")
    critics = load_critics(critic_path)
    return networks, critics, reference, {
        "diagnostic": str(source), "policy": str(policy_path), "raw_critics": str(critic_path)}


def check_replay(current, reference):
    assert current["reset_seed"] == reference["reset_seed"]
    assert current["weight"] == reference["weight"]
    assert current["identity_matched"]
    assert current["branches"]["base"] == reference["branches"]["base"], "Historical base replay changed"
    for name in ("soft", "raw"):
        a, b = current["diagnostics"][name], reference["diagnostics"][name]
        for key in ("mean_output", "gradient", "gradient_norm"):
            np.testing.assert_allclose(a[key], b[key], rtol=1e-6, atol=1e-6,
                                       err_msg=f"Changed {name} anchor {key}")
        np.testing.assert_allclose(a["q_scalar"]["base"], b["q_scalar"]["base"], rtol=1e-6, atol=1e-6)


def summarize_cases(cases):
    by_radius = {}
    for radius in RADII:
        result = pilot.summarize_cases([c["by_radius"][str(radius)]["case"] for c in cases])
        for cell in result["cells"]:
            for name in ("soft", "raw"):
                row = cell[name]
                p, d = row["plus_base"]["mean"], row["plus_minus"]["mean"]
                row.update(minus_base_mean=p - d, centered_return_slope=d / (2 * radius),
                           centered_return_curvature=(2 * p - d) / radius ** 2,
                           both_sides_below_base=bool(p < 0 and p - d < 0))
        by_radius[str(radius)] = result
    diagnoses = []
    for large, middle, small in zip(*(by_radius[str(r)]["cells"] for r in RADII)):
        row = {"mode": large["mode"], "weight_index": large["weight_index"]}
        for name in ("soft", "raw"):
            finer = (middle[name], small[name])
            row[name] = {
                "consistent_with_oversized_step": bool(large[name]["plus_base"]["mean"] < 0 and all(
                    x["plus_base"]["mean"] > 0 and x["centered_return_slope"] > 0 for x in finer)),
                "direction_mismatch_at_both_small_radii": bool(all(
                    x["centered_return_slope"] < 0 and x["q_plus_minus_mean"] > 0 for x in finer))}
        diagnoses.append(row)
    return {"primary_horizon": max(g.HORIZONS), "by_radius": by_radius,
            "cell_diagnoses": diagnoses,
            "diagnosis_counts": {name: {key: sum(c[name][key] for c in diagnoses)
                for key in ("consistent_with_oversized_step", "direction_mismatch_at_both_small_radii")}
                for name in ("soft", "raw")}}


def run(condition, seed, output_root=OUTPUT_ROOT):
    networks, critics, reference, sources = load_inputs(condition, seed)
    method = g.METHODS[condition]
    output = Path(output_root) / condition
    output.mkdir(parents=True, exist_ok=True)
    path = output / f"seed{seed}.json"
    report = {"protocol": PROTOCOL, "condition": condition, "seed": seed, "method": method,
              "sources": sources, "complete": False, "radii": list(RADII),
              "reused_radius": RADII[0], "new_radii": list(RADII[1:]),
              "weights": g.WEIGHTS.tolist(), "horizons": list(g.HORIZONS),
              "warmup": g.WARMUP, "gamma": g.GAMMA, "cases": [],
              "training_interactions": 0, "critic_updates": 0, "policy_updates": 0,
              "evaluation_interactions": 0, "environment_initialization_probe_steps": 0,
              "scope": "Same observed models and reset states, not fresh confirmation. "
                       "Reused radius costs are historical; every new rollout is charged. "
                       "Finite-radius slopes/curvatures are descriptive, not exact derivatives. "
                       "No deployment radius selection or revision of the failed critic gate."}

    def save():
        path.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")

    save()
    env = training.make_piecewise_mo_env("mo-halfcheetah-v5", switch_interval=25000, seed=seed + 2000)
    report["environment_initialization_probe_steps"] += 1
    env._lock_mode = True
    try:
        for old in reference["cases"]:
            env._apply_mode(old["mode"])
            identifiers = {k: old[k] for k in ("mode", "weight_index", "repeat", "reset_seed", "weight")}
            case = {**identifiers, "by_radius": {str(RADII[0]): {"reused": True, "case": old}},
                    "evaluation_interactions": 0}
            for radius in RADII[1:]:
                current = pilot.evaluate_case(env, networks, critics, method,
                    np.asarray(old["weight"], dtype=np.float32), old["reset_seed"], epsilon=radius)
                check_replay(current, old)
                current.update(identifiers)
                case["by_radius"][str(radius)] = {"reused": False, "case": current}
                case["evaluation_interactions"] += current["evaluation_interactions"]
            case["historical_replay_matched"] = True
            report["cases"].append(case)
            report["evaluation_interactions"] += case["evaluation_interactions"]
            save()
            print(f"RADIUS_CASE condition={condition} seed={seed} cases={len(report['cases'])}/"
                  f"{len(reference['cases'])} evaluation={report['evaluation_interactions']}", flush=True)
        assert env._total_steps == len(report["cases"]) * len(RADII[1:]) * (g.WARMUP + max(g.HORIZONS))
    finally:
        env.close()
    report["summary"] = summarize_cases(report["cases"])
    report["complete"] = True
    save()
    print(f"DONE action_radius condition={condition} seed={seed} "
          f"evaluation={report['evaluation_interactions']}", flush=True)
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--condition", choices=g.METHODS, required=True)
    parser.add_argument("--seed", type=int, choices=g.SEEDS, required=True)
    parser.add_argument("--output-root", type=Path, default=OUTPUT_ROOT)
    args = parser.parse_args()
    run(args.condition, args.seed, args.output_root)
