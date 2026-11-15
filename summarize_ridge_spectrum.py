"""Compact spectral diagnostics; no new performance gate or selector."""

import argparse
import json
from itertools import product
from pathlib import Path

import numpy as np

import diagnose_ridge_spectrum as pilot


def validate_run(data, condition, seed):
    assert data["complete"] and data["protocol"] == pilot.PROTOCOL
    assert (data["condition"], data["seed"], data["source_split"], data["horizon"]) == (condition, seed, "train_cases", 512)
    assert all(data[k] == 0 for k in ("new_environment_interactions", "optimizer_updates", "new_selector_variants"))
    assert data["svd_decompositions"] == data["ridge_reconstructions"] == 4
    assert len(data["folds"]) == 4 and {f["fold"] for f in data["folds"]} == set(range(4))
    for f in data["folds"]:
        expected = {"fit_anchors": 144, "fit_nonbase_rows": 576, "parameters": 243, "penalty": 1.,
            "svd_decompositions": 1, "ridge_reconstructions": 1, "horizon": 512,
            "target_dtype": "float32", "solve_dtype": "float64", "rank_rule": pilot.RANK_RULE}
        assert all(f[k] == v for k, v in expected.items())
        for name, held in (("fit", False), ("held", True)):
            ids = sorted(pilot.p.reset_seed("train", seed, mode, repeat)
                         for mode, repeat in product(range(4), range(16)) if (repeat % 4 == f["fold"]) == held)
            assert f[name + "_reset_ids"] == ids
        assert 0 <= f["rank"] <= 243 and 0 <= f["effective_df"] <= 243
        assert min(f["ridge_mse"], f["projection_mse"], f["shrinkage_mse"]) >= 0
        np.testing.assert_allclose(f["projection_mse"] + f["shrinkage_mse"], f["ridge_mse"], rtol=1e-10, atol=1e-12)
        np.testing.assert_allclose(f["reconstructed_train_rmse"], f["original_train_rmse"], rtol=1e-8, atol=1e-8)
    result = pilot.summarize_folds(data["folds"])
    assert result == data["summary"]
    return result


def summarize(root):
    runs = []
    for condition, seed in product(("v27", "v28"), (31, 33)):
        path = root / condition / f"seed{seed}.json"
        data = json.loads(path.read_text())
        result = validate_run(data, condition, seed)
        fits = [{k: v for k, v in f.items() if k not in ("fit_reset_ids", "held_reset_ids")} for f in data["folds"]]
        runs.append({"condition": condition, "seed": seed, "source": str(path), "summary": result, "folds": fits})
    return {"protocol": pilot.PROTOCOL, "validated_runs": 4, "svd_decompositions": 16,
        "ridge_reconstructions": 16, "new_environment_interactions": 0, "optimizer_updates": 0,
        "new_selector_variants": 0, "runs": runs,
        "scope": "Descriptive fitting-fold error decomposition; no changed selectors, gates or held-out performance claims."}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=pilot.OUTPUT_ROOT)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = summarize(args.root)
    args.output.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    print(json.dumps({k: v for k, v in report.items() if k != "runs"}))
