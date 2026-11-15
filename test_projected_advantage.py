import copy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import torch

import diagnose_projected_advantage as d
import summarize_projected_advantage as summary
from test_vector_advantage import fixture as vector_fixture


def fixture():
    source, labels, cached, legacy = vector_fixture()
    cases = {d.reference.case_key(c): c for c in source["train_cases"] + labels["train_cases"]}
    rows = []
    for row in cached["rows"]:
        case = cases[d.reference.case_key(row)]
        r = d.previous.join_row(row, np.array([[0., 0.], [1., 1.], [0., 0.], [.5, .5], [0., 0.]]), case["weight"])
        r["vector_gains"] = d.previous.vector_gains(case).tolist()
        rows.append(r)
    aggregate = legacy["summary"]["aggregate"]
    return source, labels, {**cached, "protocol": d.previous.PROTOCOL, "rows": rows,
        "summary": d.previous.summarize_rows(rows, aggregate)}, legacy


class ProjectedAdvantageTests(unittest.TestCase):
    def test_projection_and_normal_equations_preserve_phi(self):
        rng = torch.Generator().manual_seed(17)
        x = torch.randn(60, 8, generator=rng, dtype=torch.float64)
        w = torch.tensor([[0., 1.], [.5, .5], [1., 0.]], dtype=torch.float64).repeat(20, 1)
        y = torch.randn(60, generator=rng, dtype=torch.float64) * 7.
        model = d.ProjectedRidge({"inputs": x, "weights": w, "targets": y, "action_dim": 2})
        base = d.cv.QuadraticFeatures(x, 2)
        for attr in ("state_mean", "state_scale", "delta_scale", "feature_scale"):
            torch.testing.assert_close(getattr(model, attr), getattr(base, attr), rtol=0, atol=0)
        phi = base.transform(x)
        psi = torch.cat([phi * w[:, i:i+1] for i in range(2)], dim=1)
        torch.testing.assert_close(model.project(x, w), psi, rtol=0, atol=0)
        raw_beta = torch.linalg.solve(psi.T @ psi / len(x) + torch.eye(psi.shape[1]), psi.T @ y / len(x))
        torch.testing.assert_close(model.beta * model.target_scale, raw_beta, rtol=1e-10, atol=1e-10)
        latent = phi @ model.beta.reshape(2, -1).T * model.target_scale
        torch.testing.assert_close(model(x, w), (latent * w).sum(-1), rtol=1e-10, atol=1e-10)

    def test_fixed_preference_effective_penalty_and_zero_base(self):
        rng = torch.Generator().manual_seed(29)
        x = torch.randn(60, 8, generator=rng, dtype=torch.float64)
        y = torch.randn(60, generator=rng, dtype=torch.float64)
        for weight in ([1., 0.], [0., 1.], [.5, .5], [.3, .7]):
            w = torch.tensor(weight, dtype=torch.float64).expand(len(x), -1)
            data = {"inputs": x, "targets": y, "weights": w, "action_dim": 2}
            model = d.ProjectedRidge(data)
            scalar = d.cv.QuadraticRidge(data, penalty=1. / float(w[0].square().sum()))
            torch.testing.assert_close(model(x, w), scalar(x), rtol=1e-10, atol=1e-10)
            base = x.clone()
            base[:, -2:] = 0.
            self.assertTrue((model(base, w) == 0.).all())

    def test_training_scalar_target_and_context_are_unchanged(self):
        source, _, _, _ = fixture()
        case = copy.deepcopy(source["train_cases"][0])
        case["weight"] = [0., 1.]
        case["anchor_context"][:2] = case["weight"]
        case["branches"]["soft_plus"]["discounted_returns"]["512"] = [17., .25]
        data = d.training_data([case])
        torch.testing.assert_close(data["inputs"], d.cv.features(case)[1:], rtol=0, atol=0)
        self.assertEqual(data["inputs"][0, 17:19].tolist(), [0., 1.])
        self.assertEqual(data["targets"][0].item(), .25)
        self.assertEqual(data["weights"].tolist(), [[0., 1.]] * 4)

    def test_fit_uses_no_held_labels_or_critic_scores(self):
        source, labels, _, _ = fixture()
        fits, held = d.cross.split_batches(source["train_cases"], labels["train_cases"], 0)
        cases = held["old"] + held["new"]
        weights = torch.tensor([c["weight"] for c in cases], dtype=torch.float64)[:, None, :].expand(-1, 5, -1)
        before = copy.deepcopy(fits)
        rng = torch.get_rng_state().clone()
        scores, meta = d.fit_predict(fits["combined96"], torch.stack([d.cv.features(c) for c in cases]), weights)
        self.assertEqual(before, fits)
        for c in cases:
            c["branches"] = c["predictions"] = None
        for c in fits["combined96"]:
            c["predictions"] = None
            c["branches"]["base"]["discounted_returns"]["128"] = None
        repeated, repeated_meta = d.fit_predict(fits["combined96"], torch.stack([d.cv.features(c) for c in cases]), weights)
        np.testing.assert_array_equal(scores, repeated)
        self.assertEqual(meta, repeated_meta)
        self.assertEqual(scores.shape, (96, 5))
        torch.testing.assert_close(torch.get_rng_state(), rng, rtol=0, atol=0)

    def test_gate_retains_vector_huber_and_legacy_controls(self):
        _, _, cached, _ = fixture()
        row = d.join_row(cached["rows"][0], np.zeros(5))
        self.assertEqual(row["choices"][d.TARGET], "base")
        rows = [d.join_row(r, [0., 1., 0., .5, 0.]) for r in cached["rows"]]
        harder = copy.deepcopy(cached["legacy_aggregate"])
        harder[d.expansion.CONTROLS[-1]]["mean_gain"] = 1e5
        self.assertFalse(d.summarize_rows(rows, harder)["old_primary_checks"]["mean_not_worse_than_controls"])
        for control in (d.previous.TARGET, d.gates.TARGET):
            changed = copy.deepcopy(rows)
            for r in changed:
                r["selected_gains"][d.TARGET] = 1.
                if r["batch"] == "new":
                    r["selected_gains"][control] = 2.
            result = d.summarize_rows(changed, cached["legacy_aggregate"])
            self.assertFalse(result["batch_checks"]["new"]["mean_not_worse_than_controls"])
            self.assertFalse(result["group_passes"])

    def test_runner_preserves_controls_and_validates_protocol(self):
        source, labels, cached, legacy = fixture()
        with (tempfile.TemporaryDirectory() as directory,
              patch.object(d.previous, "fit_predict", side_effect=AssertionError("No vector refits")),
              patch.object(d.expansion, "collect_labels", side_effect=AssertionError("No collection")),
              patch.object(d.p.training, "make_piecewise_mo_env", side_effect=AssertionError("No environments")),
              patch.object(torch, "load", side_effect=AssertionError("No checkpoints"))):
            root = Path(directory)
            for name, data in (("source/v27/seed31.json", source), ("expansion/v27/seed31_labels.json", labels),
                               ("expansion/v27/seed31.json", legacy), ("reference/v27/seed31.json", cached)):
                path = root / name
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text(json.dumps(data))
            result = d.run("v27", 31, root / "source", root / "expansion", root / "reference", root / "out")
            compact = summary.validate_run(result, cached, legacy, source, labels, "v27", 31)
            self.assertEqual(compact["summary"]["group_passes"], result["summary"]["group_passes"])
            for kind in ("fold", "scale", "prediction", "control", "budget", "grid", "gate"):
                bad = copy.deepcopy(result)
                if kind == "fold":
                    bad["folds"][0]["fit_reset_ids"] = bad["folds"][0]["held_reset_ids"]["old"]
                elif kind == "scale":
                    bad["folds"][0]["target_scale"] *= 2.
                elif kind == "prediction":
                    bad["rows"][0]["scores"][d.TARGET]["base"] = .1
                elif kind == "control":
                    bad["rows"][0]["scores"]["vector_ridge"]["soft_plus"] += .001
                elif kind == "budget":
                    bad["linear_solves"] += 1
                elif kind == "grid":
                    bad["rows"][0] = copy.deepcopy(bad["rows"][1])
                else:
                    bad["summary"]["group_passes"] = not bad["summary"]["group_passes"]
                with self.subTest(kind=kind), self.assertRaises(AssertionError):
                    summary.validate_run(bad, cached, legacy, source, labels, "v27", 31)


if __name__ == "__main__":
    unittest.main()
