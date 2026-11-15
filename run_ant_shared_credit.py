"""Fresh Ant shared-credit pilot with per-label and blind controls."""

import argparse
from contextlib import nullcontext
import json
from pathlib import Path
from unittest.mock import patch

import numpy as np

from ant_shared_credit import SharedCreditOwnPreferenceSelector
from diagnose_ant_allocation import work_counts
import run_ant_pairwise_final75 as previous
from summarize_ant_mechanism_alignment import paired


PROTOCOL = "ant_shared_credit_final75_v1"
ROOT = Path("results/iclr_ant_shared_credit_20260927")
SEEDS = tuple(range(1401, 1406))
BASE_CONDITION = {"per_label": "base_adaptive", "shared": "base_adaptive", "blind": "base_blind"}
training = previous.training


def selector_patch(condition):
    return (patch.object(training, "OwnPreferenceBothProgressExtremeOperatorSelector",
                         SharedCreditOwnPreferenceSelector) if condition == "shared" else nullcontext())


def credit_check(raw, condition):
    state = raw["adaptive_operator_final"]
    selector = SharedCreditOwnPreferenceSelector(3)
    if condition == "shared":
        assert state["score_credit_rule"] == "effective_configuration_shared_ema"
        assert state["credit_groups"] == selector.credit_groups
    else:
        assert "score_credit_rule" not in state
    scores = dict.fromkeys(selector.operators, 0.)
    blocks = state["block_history"]
    assert [b["end_step"] for b in blocks] == list(range(6500, 74001, 1500)) + [75000]
    for block in blocks:
        op = block["operator"]
        if block["complete"]:
            means, scale = np.asarray(block["half_mean_rewards"]), np.asarray(block["scale_before_block"])
            score = float(np.tanh((means[1] - means[0]) / scale).mean())
            np.testing.assert_allclose(score, block["progress_score"], atol=1e-12)
            scores[op] = (1 - selector.lr) * scores[op] + selector.lr * score
            if condition == "shared":
                for alias in selector.credit_aliases[op]:
                    scores[alias] = scores[op]
        else:
            assert block is blocks[-1] and block["steps"] == 1000
            assert "progress_score" not in block
        np.testing.assert_allclose(scores[op], block["score_after"], atol=1e-12)
    np.testing.assert_allclose(list(scores.values()), [state["scores"][op] for op in scores], atol=1e-12)
    return {"rule": state.get("score_credit_rule", "per_label_ema"), "reconstructed": True}


def warmup_record(raw):
    keys = ("end_step", "operator", "steps", "half_counts", "half_mean_rewards",
            "scale_before_block", "normalized_vector_gain", "progress_score")
    return {"blocks": [{k: b[k] for k in keys} for b in raw["adaptive_operator_final"]["block_history"][:10]],
            "episodes": [r for r in raw["episode_rewards"] if r["step"] <= 20000]}


def validate(raw, condition, seed, model):
    report = previous.validate(raw, BASE_CONDITION[condition], seed, model)
    report.update(protocol=PROTOCOL, condition=condition, credit=credit_check(raw, condition),
                  work=work_counts(raw["adaptive_operator_final"]["block_history"]),
                  warmup=warmup_record(raw))
    return report


def run(condition, seed, root=ROOT):
    assert condition in BASE_CONDITION and seed in SEEDS
    folder = root / condition / f"seed{seed}"
    folder.mkdir(parents=True, exist_ok=True)
    output = folder / "report.json"
    method = previous.METHODS[BASE_CONDITION[condition]]
    raw_path = folder / f"{method}_seed{seed}.json"
    model = folder / f"{method}_seed{seed}_model.pt"
    if output.exists():
        saved = json.loads(output.read_text())
        assert (saved["protocol"], saved["condition"], saved["seed"]) == (PROTOCOL, condition, seed)
        if saved.get("complete"):
            print(f"DONE {PROTOCOL} {condition} seed={seed} reusing report", flush=True)
            return saved
        if not raw_path.exists():
            raise RuntimeError(f"Incomplete attempt preserved at {folder}; inspect before a new-root retry")
    else:
        output.write_text(json.dumps(dict(protocol=PROTOCOL, condition=condition, seed=seed, complete=False)) + "\n")
    if not raw_path.exists():
        with selector_patch(condition):
            training.train_method(previous.ENV, method, previous.STEPS, 25000, seed, str(folder),
                                  archive_steps=(previous.STEPS,), **previous.pilot.TRAIN_KWARGS)
    report = validate(json.loads(raw_path.read_text()), condition, seed, model)
    output.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    print(f"DONE {PROTOCOL} {condition} seed={seed} mixed={report['metrics'][previous.METRICS[0]]:.6f}", flush=True)
    return report


def summarize(root=ROOT):
    rows = []
    budget = dict(training_interactions=0, evaluation_interactions=0, environment_initialization_probe_steps=0)
    for seed in SEEDS:
        row = {c: json.loads((root / c / f"seed{seed}" / "report.json").read_text()) for c in BASE_CONDITION}
        reference = row["per_label"]
        for condition, report in row.items():
            assert (report["protocol"], report["condition"], report["seed"], report["complete"]) == (
                PROTOCOL, condition, seed, True)
            for key in ("evaluation", "mode_history", "operator_configs", "warmup"):
                assert report[key] == reference[key], (seed, condition, key)
            for key in budget:
                budget[key] += report["completed_budget"][key]
        rows.append(row)
    comparisons = {f"{a}_minus_{b}": {m: paired([r[a]["metrics"][m] for r in rows],
                                                [r[b]["metrics"][m] for r in rows])
                                            for m in previous.METRICS}
                   for a, b in (("shared", "per_label"), ("shared", "blind"), ("per_label", "blind"))}
    return {"protocol": PROTOCOL, "seeds": list(SEEDS), "completed_budget": budget,
            "comparisons": comparisons,
            "runs": [{"seed": seed, "metrics": {c: r[c]["metrics"] for c in BASE_CONDITION},
                      "work": {c: r[c]["work"] for c in BASE_CONDITION}} for seed, r in zip(SEEDS, rows)]}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--condition", choices=BASE_CONDITION)
    parser.add_argument("--seed", type=int, choices=SEEDS)
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--summarize", action="store_true")
    args = parser.parse_args()
    if args.summarize:
        result = summarize(args.root)
        path = args.root / "readout.json"
        path.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
        print(f"DONE {PROTOCOL} summary {path}")
    else:
        assert args.condition is not None and args.seed is not None
        run(args.condition, args.seed, args.root)
