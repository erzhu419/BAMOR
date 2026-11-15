"""Recompute the frozen radius diagnosis without selecting a deployment radius."""

import argparse
import json
from itertools import product
from pathlib import Path

import numpy as np

import diagnose_action_radius as pilot


GROUPS = tuple(product(("v27", "v28"), (31, 33)))
BUDGET_KEYS = ("training_interactions", "evaluation_interactions",
               "environment_initialization_probe_steps", "critic_updates", "policy_updates")


def validate_run(data, reference, condition, seed):
    expected = {"protocol": pilot.PROTOCOL, "complete": True, "condition": condition,
                "seed": seed, "method": pilot.g.METHODS[condition],
                "radii": [.05, .01, .002], "reused_radius": .05, "new_radii": [.01, .002],
                "weights": pilot.g.WEIGHTS.tolist(), "horizons": [32, 128, 512],
                "warmup": 128, "gamma": .99, "training_interactions": 0,
                "evaluation_interactions": 230400, "environment_initialization_probe_steps": 1,
                "critic_updates": 0, "policy_updates": 0}
    for key, value in expected.items():
        assert data[key] == value, f"Mismatched {condition}/{seed} field: {key}"
    assert reference["complete"] and reference["protocol"] == pilot.pilot.PROTOCOL
    assert (reference["condition"], reference["seed"]) == (condition, seed)
    identifiers = ("mode", "weight_index", "repeat", "reset_seed", "weight")
    identity = lambda c: tuple(c[k] for k in identifiers[:3])
    old = {identity(c): c for c in reference["cases"]}
    grid = set(product(range(4), range(3), range(3)))
    assert len(old) == 36 and set(old) == grid
    assert len(data["cases"]) == 36 and {identity(c) for c in data["cases"]} == grid
    for case in data["cases"]:
        source = old[identity(case)]
        assert case["historical_replay_matched"]
        assert set(case["by_radius"]) == {"0.05", "0.01", "0.002"}
        assert case["evaluation_interactions"] == 6400
        assert case["reset_seed"] == pilot.pilot.reset_seed("test", seed, case["mode"], case["repeat"])
        assert case["weight"] == pilot.g.WEIGHTS[case["weight_index"]].tolist()
        new_steps = 0
        for radius in pilot.RADII:
            item = case["by_radius"][str(radius)]
            current = item["case"]
            assert item["reused"] == (radius == .05)
            assert all(current[k] == case[k] == source[k] for k in identifiers)
            if item["reused"]:
                assert current == source, "Reused radius differs from its source artifact"
            else:
                pilot.check_replay(current, source)
                new_steps += current["evaluation_interactions"]
            assert current["evaluation_interactions"] == 3200
            assert current["identity_matched"] and current["branches"]["base"] == current["branches"]["repeat"]
            assert set(current["branches"]) == set(pilot.pilot.BRANCHES)
            for branch in current["branches"].values():
                assert branch["steps"] == 512
                for key in ("discounted_returns", "raw_returns"):
                    assert set(branch[key]) == {"32", "128", "512"}
                    values = np.asarray(list(branch[key].values()))
                    assert values.shape == (3, 2) and np.isfinite(values).all()
            for name in ("soft", "raw"):
                stats = current["diagnostics"][name]
                z, gradient = np.asarray(stats["mean_output"]), np.asarray(stats["gradient"])
                norm = np.linalg.norm(gradient)
                direction = gradient / norm if norm > 0 else np.zeros_like(gradient)
                for label, sign in (("base", 0), ("plus", 1), ("minus", -1)):
                    # Float32 torch tanh and NumPy reconstruction can differ by one ULP.
                    np.testing.assert_allclose(stats["actions"][label], np.tanh(z + sign * radius * direction),
                                               rtol=0, atol=2e-7, equal_nan=False)
                scores = current["predictions"][name]
                assert set(scores) == set(pilot.pilot.BRANCHES) - {"repeat"}
                assert np.isfinite(list(scores.values())).all()
        assert new_steps == case["evaluation_interactions"]
    assert sum(c["evaluation_interactions"] for c in data["cases"]) == data["evaluation_interactions"]
    result = pilot.summarize_cases(data["cases"])
    assert result == data["summary"], "Saved radius summary differs from recomputation"
    return result


def summarize(root, reference_root):
    runs, budget = [], dict.fromkeys(BUDGET_KEYS, 0)
    for condition, seed in GROUPS:
        path = root / condition / f"seed{seed}.json"
        source = reference_root / condition / f"seed{seed}.json"
        data, reference = [json.loads(p.read_text()) for p in (path, source)]
        result = validate_run(data, reference, condition, seed)
        runs.append({"condition": condition, "seed": seed, "source": str(path), "summary": result})
        for key in budget:
            budget[key] += data[key]
    counts = {name: {key: sum(r["summary"]["diagnosis_counts"][name][key] for r in runs)
                    for key in ("consistent_with_oversized_step", "direction_mismatch_at_both_small_radii")}
              for name in ("soft", "raw")}
    return {"protocol": pilot.PROTOCOL, "validated_runs": len(runs), "anchor_cases": 144,
            "new_radius_cases": 288, "historical_base_matches": 288, "new_identity_matches": 288,
            "new_budget": budget, "radii": list(pilot.RADII), "runs": runs,
            "diagnosis_counts": counts, "total_cells_per_scorer": 48,
            "scope": "Same observed models/reset states; three resets per cell. "
                     "All radii reported; 0.05 is reused, not charged again. "
                     "Finite-radius derivatives and sign counts are descriptive. "
                     "No deployment selection or revision of the failed shadow-critic gate."}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=pilot.OUTPUT_ROOT)
    parser.add_argument("--reference-root", type=Path, default=pilot.SOURCE_ROOT)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = summarize(args.root, args.reference_root)
    args.output.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    print(json.dumps({k: v for k, v in report.items() if k != "runs"}))
