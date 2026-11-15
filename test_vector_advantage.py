import copy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import torch

import diagnose_vector_advantage as d
import summarize_vector_advantage as summary
from test_huber_advantage import fixture as huber_fixture


def fixture():
    source, labels, cached, legacy = huber_fixture()
    rows = [d.previous.join_row(r, np.array([0., 1., 0., .5, 0.])) for r in cached["rows"]]
    aggregate = legacy["summary"]["aggregate"]
    return source, labels, {**cached, "protocol": d.previous.PROTOCOL, "rows": rows,
        "legacy_aggregate": aggregate, "summary": d.previous.summarize_rows(rows, aggregate)}, legacy


class VectorAdvantageTests(unittest.TestCase):
    def test_fixed_preference_matches_scalar_ridge_and_objective_units(self):
        rng = torch.Generator().manual_seed(13)
        x = torch.randn(80, 8, generator=rng, dtype=torch.float64)
        y = torch.randn(80, 2, generator=rng, dtype=torch.float64) * torch.tensor([10., .1])
        data = {"inputs": x, "targets": y, "action_dim": 2}
        vector = d.VectorRidge(data)
        for weight in ([1., 0.], [0., 1.], [.3, .7]):
            w = torch.tensor(weight, dtype=torch.float64)
            scalar = d.cv.QuadraticRidge({**data, "targets": y @ w})
            torch.testing.assert_close(vector(x) @ w, scalar(x), rtol=1e-10, atol=1e-10)
        multiplier = torch.tensor([7., .25])
        scaled = d.VectorRidge({**data, "targets": y * multiplier})
        torch.testing.assert_close(scaled(x), vector(x) * multiplier, rtol=1e-10, atol=1e-10)
        base = x.clone()
        base[:, -2:] = 0.
        self.assertTrue((vector(base) == 0.).all())

    def test_training_keeps_context_and_both_returns_even_at_endpoint(self):
        source, _, _, _ = fixture()
        case = copy.deepcopy(source["train_cases"][0])
        case["weight"] = [0., 1.]
        case["anchor_context"][:2] = case["weight"]
        case["branches"]["soft_plus"]["discounted_returns"]["512"] = [17., .25]
        data = d.training_data([case])
        torch.testing.assert_close(data["inputs"], d.cv.features(case)[1:], rtol=0, atol=0)
        self.assertEqual(data["inputs"][0, 17:19].tolist(), [0., 1.])
        self.assertEqual(data["targets"][0].tolist(), [17., .25])
        np.testing.assert_allclose(d.vector_gains(case) @ np.asarray(case["weight"]),
                                  list(d.p.realized_gains(case).values()), rtol=0, atol=0)

    def test_fit_uses_no_held_labels_or_critic_scores(self):
        source, labels, _, _ = fixture()
        fits, held = d.cross.split_batches(source["train_cases"], labels["train_cases"], 0)
        cases = held["old"] + held["new"]
        before = copy.deepcopy(fits)
        rng = torch.get_rng_state().clone()
        scores, meta = d.fit_predict(fits["combined96"], torch.stack([d.cv.features(c) for c in cases]))
        self.assertEqual(before, fits)
        for c in cases:
            c["branches"] = c["predictions"] = None
        for c in fits["combined96"]:
            c["predictions"] = None
            c["branches"]["base"]["discounted_returns"]["128"] = None
        repeated, repeated_meta = d.fit_predict(fits["combined96"], torch.stack([d.cv.features(c) for c in cases]))
        np.testing.assert_array_equal(scores, repeated)
        self.assertEqual(meta, repeated_meta)
        self.assertEqual(scores.shape, (96, 5, 2))
        torch.testing.assert_close(torch.get_rng_state(), rng, rtol=0, atol=0)

    def test_scalarization_and_gate_do_not_choose_the_best_objective(self):
        source, labels, cached, _ = fixture()
        vectors = np.array([[0., 0.], [2., -2.], [-2., 2.], [0., 0.], [0., 0.]])
        self.assertEqual(d.join_row(cached["rows"][0], vectors, [.5, .5])["choices"][d.TARGET], "base")
        self.assertEqual(d.join_row(cached["rows"][0], vectors, [1., 0.])["choices"][d.TARGET], "soft_plus")
        cases = {d.reference.case_key(c): c for c in source["train_cases"] + labels["train_cases"]}
        rows = []
        for row in cached["rows"]:
            case = cases[d.reference.case_key(row)]
            r = d.join_row(row, vectors, case["weight"])
            r["vector_gains"] = d.vector_gains(case).tolist()
            rows.append(r)
        harder = copy.deepcopy(cached["legacy_aggregate"])
        harder[d.expansion.CONTROLS[-1]]["mean_gain"] = 1e5
        self.assertFalse(d.summarize_rows(rows, harder)["old_primary_checks"]["mean_not_worse_than_controls"])
        for r in rows:
            if r["batch"] == "new":
                r["selected_gains"][d.TARGET] = -1.
        result = d.summarize_rows(rows, cached["legacy_aggregate"])
        self.assertFalse(result["batch_checks"]["new"]["positive_mean_gain"])
        self.assertFalse(result["group_passes"])

    def test_runner_preserves_cached_huber_and_validates_head_outputs(self):
        source, labels, cached, legacy = fixture()
        with (tempfile.TemporaryDirectory() as directory,
              patch.object(d.previous, "fit_predict", side_effect=AssertionError("No Huber refits")),
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
            for kind in ("fold", "scale", "vector", "weight", "control", "budget", "grid", "gate"):
                bad = copy.deepcopy(result)
                if kind == "fold":
                    bad["folds"][0]["fit_reset_ids"] = bad["folds"][0]["held_reset_ids"]["old"]
                elif kind == "scale":
                    bad["folds"][0]["target_scales"][0] *= 2.
                elif kind == "vector":
                    bad["rows"][0]["vector_gains"][1][0] += .1
                elif kind == "weight":
                    bad["rows"][0]["weight"] = [.5, .5]
                elif kind == "control":
                    bad["rows"][0]["scores"]["huber"]["soft_plus"] += .001
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
