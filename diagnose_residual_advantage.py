"""Fold-local critic calibration and state-dependent residual return correction."""

import argparse
import json
from pathlib import Path

import numpy as np
from scipy.optimize import nnls

import diagnose_advantage_cv as cv


p = cv.pilot
PROTOCOL = "critic_residual_grouped_cv_v1"
OUTPUT_ROOT = Path("results/iclr_residual_advantage_20260921")
PENALTY = 1.
SELECTORS = ("residual", "calibrated_prior", *cv.SELECTORS)
CONTROLS = ("calibrated_prior", "ridge", "mlp", "fixed_preference", "soft", "raw")


def case_key(case):
    return tuple(case[k] for k in ("mode", "weight_index", "repeat"))


def model_inputs(cases):
    states = np.stack([cv.features(c)[0, :-6].numpy() for c in cases]).astype(np.float64)
    contrasts = np.asarray([[[c["predictions"][name][label] - c["predictions"][name]["base"]
                            for name in ("soft", "raw")] for label in p.CANDIDATES] for c in cases])
    assert np.isfinite(states).all() and np.isfinite(contrasts).all()
    assert (contrasts[:, 0] == 0.).all()
    return states, contrasts


class CriticResidual:
    def __init__(self, state, contrasts, targets):
        self.state_mean = state.mean(0)
        self.state_scale = np.maximum(state.std(0), 1e-3)
        self.q_scale = np.maximum(np.sqrt(np.mean(contrasts ** 2, axis=0)), 1e-3)
        self.target_scale = max(float(targets.std()), 1e-3)
        z, y = contrasts / self.q_scale, targets / self.target_scale
        n = len(y)
        self.prior, _ = nnls(np.vstack((z / np.sqrt(n), np.sqrt(PENALTY) * np.eye(2))),
                            np.concatenate((y / np.sqrt(n), np.zeros(2))))
        phi = self.basis(state, contrasts)
        self.feature_scale = np.maximum(np.sqrt(np.mean(phi ** 2, axis=0)), 1e-3)
        phi = phi / self.feature_scale
        residual = y - z @ self.prior
        self.beta = np.linalg.solve(phi.T @ phi / n + PENALTY * np.eye(phi.shape[1]),
                                    phi.T @ residual / n)
        assert np.isfinite(self.prior).all() and np.isfinite(self.beta).all()

    def basis(self, state, contrasts):
        s = (state - self.state_mean) / self.state_scale
        z = contrasts / self.q_scale
        i, j = np.triu_indices(2)
        return np.concatenate((z, (s[:, :, None] * z[:, None, :]).reshape(len(s), -1),
                               z[:, i] * z[:, j]), axis=1)

    def predict(self, state, contrasts):
        prior = contrasts / self.q_scale @ self.prior
        correction = self.basis(state, contrasts) / self.feature_scale @ self.beta
        return {"calibrated_prior": prior * self.target_scale,
                "residual": (prior + correction) * self.target_scale}


def fit_predict(fit_cases, held_inputs):
    state, contrasts = model_inputs(fit_cases)
    targets = np.asarray([p.realized_gains(c)[label] for c in fit_cases for label in p.CANDIDATES[1:]])
    state, contrasts = np.repeat(state, 4, axis=0), contrasts[:, 1:].reshape(-1, 2)
    model = CriticResidual(state, contrasts, targets)
    fitted = model.predict(state, contrasts)
    held_state, held_q = held_inputs
    predictions = {name: values.reshape(-1, 5) for name, values in
                   model.predict(np.repeat(held_state, 5, axis=0), held_q.reshape(-1, 2)).items()}
    assert all(np.isfinite(v).all() and (v[:, 0] == 0.).all() for v in predictions.values())
    return predictions, {"fit_rows": len(targets), "prior_coefficients": model.prior.tolist(),
        "residual_coefficients": len(model.beta), "penalty": PENALTY, "nnls_fits": 1, "ridge_solves": 1,
        "train_rmse": {name: float(np.sqrt(np.mean((values - targets) ** 2))) for name, values in fitted.items()}}


def join_row(case, previous, predictions, selectors=SELECTORS):
    assert all(case[k] == previous[k] for k in ("mode", "weight_index", "repeat", "reset_seed"))
    assert previous["fold"] == case["repeat"] % 4
    scores = {**previous["scores"], **{name: dict(zip(p.CANDIDATES, map(float, values)))
                                      for name, values in predictions.items()}}
    choices = {name: p.choose(scores[name]) for name in selectors}
    gains = p.realized_gains(case)
    assert gains == previous["candidate_gains"], "Cached controls must match the current labels"
    return {**previous, "scores": scores, "choices": choices,
            "selected_gains": {name: gains[choices[name]] for name in selectors}}


def summarize_rows(rows):
    return cv.summarize_rows(rows, selectors=SELECTORS, controls=CONTROLS, target="residual")


def run(condition, seed, source_root=p.OUTPUT_ROOT, reference_root=cv.OUTPUT_ROOT, output_root=OUTPUT_ROOT):
    source = Path(source_root) / condition / f"seed{seed}.json"
    reference = Path(reference_root) / condition / f"seed{seed}.json"
    cases = cv.training_cases(json.loads(source.read_text()), condition, seed)
    old = json.loads(reference.read_text())
    assert old["complete"] and old["protocol"] == cv.PROTOCOL and old["source_split"] == "train_cases"
    assert (old["condition"], old["seed"], old["fold_rule"]) == (condition, seed, "repeat % 4")
    cached = {case_key(r): r for r in old["rows"]}
    assert len(old["rows"]) == len(cached) == len(cases) and set(cached) == {case_key(c) for c in cases}
    path = Path(output_root) / condition / f"seed{seed}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    report = {"protocol": PROTOCOL, "condition": condition, "seed": seed, "complete": False,
        "source": str(source), "reference": str(reference), "source_split": "train_cases",
        "new_environment_interactions": 0, "mlp_updates": 0, "policy_updates": 0, "critic_updates": 0,
        "nnls_fits": 0, "ridge_solves": 0, "reused_anchors": 192, "reused_nonbase_labels": 768,
        "folds": [], "rows": [], "scope": "Training-only development CV; cached OOF controls. "
        "Critic contrasts precede outcomes. No simulator, checkpoints or old validation inputs."}

    def save():
        path.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")

    save()
    for fold in range(4):
        fit, held = cv.split_fold(cases, fold)
        predictions, meta = fit_predict(fit, model_inputs(held))
        # The fitted model never receives held-out labels or cached choices.
        report["rows"].extend(join_row(c, cached[case_key(c)], {name: v[i] for name, v in predictions.items()})
                              for i, c in enumerate(held))
        report["folds"].append({"fold": fold, "fit_reset_ids": sorted({c["reset_seed"] for c in fit}),
            "held_reset_ids": sorted({c["reset_seed"] for c in held}), **meta})
        report["nnls_fits"] += meta["nnls_fits"]
        report["ridge_solves"] += meta["ridge_solves"]
        save()
        print(f"RESIDUAL_CV condition={condition} seed={seed} folds={fold + 1}/4 interactions=0", flush=True)
    report["summary"] = summarize_rows(report["rows"])
    report["complete"] = True
    save()
    print(f"DONE residual_cv condition={condition} seed={seed} passes={report['summary']['group_passes']}", flush=True)
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--condition", choices=p.g.METHODS, required=True)
    parser.add_argument("--seed", type=int, choices=p.g.SEEDS, required=True)
    parser.add_argument("--source-root", type=Path, default=p.OUTPUT_ROOT)
    parser.add_argument("--reference-root", type=Path, default=cv.OUTPUT_ROOT)
    parser.add_argument("--output-root", type=Path, default=OUTPUT_ROOT)
    args = parser.parse_args()
    run(args.condition, args.seed, args.source_root, args.reference_root, args.output_root)
