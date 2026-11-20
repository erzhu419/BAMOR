"""Reduce completed locked-mode evaluations on the host that stores them."""

import argparse
from collections import Counter
import json
from pathlib import Path

import numpy as np

MAIN = ("mo-halfcheetah-v5", "mo-hopper-v5")
ANT = "mo-ant-v5"
SEEDS = range(1601, 1621)
ANT_SEEDS = range(1201, 1206)
NATIVE = "iclr_utility_confirmation_20260927"
FULL = "iclr_capql_full_simplex_20260930"
PILOT = "iclr_75k_final_fresh_20260926"
NEUTRAL = "nca_score_ablation_20261120"
ADAPTIVE_METHOD = "bamor_v25_progress_archive_mix"
NEUTRAL_METHOD = "bamor_v25_progress_fixed_archive_mix"


def load(root, env, study, arm, method, seed):
    path = root / study / env.replace("-", "_") / arm
    raw = json.loads((path / f"{method}_seed{seed}.json").read_text())
    receipt = json.loads((path / f"seed{seed}.json").read_text())
    assert (raw["env"], raw["method"], raw["seed"], raw["evaluation_protocol"],
            raw["training_interactions"]) == (env, method, seed, "locked_full_simplex_v1", 75000)
    assert receipt["complete"]
    if method in (ADAPTIVE_METHOD, NEUTRAL_METHOD):
        assert [s["step"] for s in raw["policy_archive_snapshots"]] == [75000]
    else:
        assert raw["policy_archive_snapshots"] is None
    return raw, receipt.get("metrics", receipt)["direct_utility"]["per_mode"]


def grid_metrics(weights, returns):
    """Diagonal command utility and same-actor maximum over all commands."""
    score = np.asarray(returns) @ np.asarray(weights).T
    assert score.shape == (len(weights), len(weights)) and np.isfinite(score).all()
    direct, best = score.diagonal(), score.max(axis=0)
    return direct, best


def geometry(weights):
    support = (weights > 0).sum(axis=1)
    return np.select([support == 1, support == weights.shape[1]],
                     ["vertex", "interior"], default="boundary")


def reduce_front(raw, expected_u):
    weights = np.asarray(raw["evaluation_weights"])
    assert weights.shape[0] == 21 and np.all(weights >= 0)
    assert np.allclose(weights.sum(axis=1), 1) and len(expected_u) == len(raw["pareto_per_mode"])
    regions = geometry(weights)
    energy = weights[:, -1]
    modes = []
    for h, front in enumerate(raw["pareto_per_mode"]):
        assert np.allclose([p["weight"] for p in front], weights, atol=1e-7)
        direct, best = grid_metrics(weights, [p["return"] for p in front])
        assert np.isclose(direct.mean(), expected_u[h], rtol=1e-8)
        assert np.isclose(best.mean(), raw["eum_per_mode"][h], rtol=1e-8)
        subsets = {"all": np.ones(len(weights), dtype=bool)}
        subsets.update({label: regions == label for label in sorted(set(regions))})
        row = {label: [float(direct[mask].mean()), float((best-direct)[mask].mean())]
               for label, mask in subsets.items()}
        if raw["env"] == ANT and h == 0:
            rerouted = weights @ np.asarray(front[np.flatnonzero(energy == 1).item()]["return"])
            row["mixed_energy"] = {}
            for label in ("boundary", "interior"):
                mask = (regions == label) & (energy > 0) & (energy < 1)
                row["mixed_energy"][label] = {"weights": int(mask.sum()),
                    "u": float(direct[mask].mean()),
                    "vertex_wins": int(np.count_nonzero((rerouted-direct)[mask] > 0))}
        modes.append(row)
    return modes


def assert_matched(reference, other):
    for key in ("evaluation_weights", "episode_seeds_per_mode", "ref_point"):
        assert reference[key] == other[key], key
    assert reference["ground_truth"]["mode_history"] == other["ground_truth"]["mode_history"]


def front_readout(root):
    environments = {}
    for env in (*MAIN, ANT):
        environments[env] = rows = []
        study, seeds = (PILOT, ANT_SEEDS) if env == ANT else (NATIVE, SEEDS)
        for seed in seeds:
            on, expected = load(root, env, study, "v25", ADAPTIVE_METHOD, seed)
            arms = {"v25": reduce_front(on, expected)}
            controls = [("capql", study)]
            if env != ANT:
                controls.append(("capql_full_simplex", FULL))
            for arm, source in controls:
                raw, expected = load(root, env, source, arm, arm, seed)
                assert_matched(on, raw)
                arms[arm] = reduce_front(raw, expected)
            rows.append({"seed": seed, "mode_history": on["ground_truth"]["mode_history"],
                         "arms": arms})
    return {"protocol": "nca_locked_grid_reanalysis_v1", "environments": environments}


def allocation(raw):
    state = raw["adaptive_operator_final"]
    slots = state["operators"]
    hist = state["block_history"]
    full = [h["operator"] for h in hist if h["complete"]]
    nw = len(state["warmup_sequence"])
    assert len(slots) == 18 and nw == 10
    assert full[:nw] == state["warmup_sequence"]
    assert len(full) == 46 and len(hist) == 47 and hist[-1]["steps"] == 1000
    assert dict(Counter(full)) == {key: n for key, n in state["counts"].items() if n}
    return slots, {"full": [slots.index(o) for o in full],
                   "partial": slots.index(hist[-1]["operator"])}


def routing_readout(root):
    environments = {}
    for env in MAIN:
        rows = []
        for seed in SEEDS:
            on, u_on = load(root, env, NATIVE, "v25", ADAPTIVE_METHOD, seed)
            neutral, u_neutral = load(root, env, NEUTRAL, "score_neutral", NEUTRAL_METHOD, seed)
            assert_matched(on, neutral)
            for key in ("context_dim", "critic_target_rule", "bootstrap_terminal_rule",
                        "critic_candidate_context_rule", "switch_interval"):
                assert on[key] == neutral[key], key
            for key in ("families", "operator_trainer_configs", "warmup_sequence",
                        "block_steps", "min_family_fraction", "epsilon", "temperature",
                        "family_epsilon", "family_temperature", "score_signal", "learning_starts"):
                assert on["adaptive_operator_final"][key] == neutral["adaptive_operator_final"][key], key
            slots, trace_on = allocation(on)
            neutral_slots, trace_neutral = allocation(neutral)
            assert slots == neutral_slots
            rows.append({"seed": seed, "u_on": u_on, "u_neutral": u_neutral,
                         "trace_on": trace_on, "trace_neutral": trace_neutral,
                         "evaluation_interactions": neutral["evaluation_interactions"],
                         "training_interactions": neutral["training_interactions"],
                         "environment_initialization_probe_steps":
                             neutral["environment_initialization_probe_steps"]})
        environments[env] = {"slots": slots, "runs": rows}
    return {"protocol": "nca_75k_score_ablation_v1", "environments": environments}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path, default=Path("results"))
    parser.add_argument("--routing", action="store_true")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = (routing_readout if args.routing else front_readout)(args.results)
    args.output.write_text(json.dumps(result, separators=(",", ":"), allow_nan=False) + "\n")
    print(f"Wrote {args.output} ({args.output.stat().st_size} bytes)")
