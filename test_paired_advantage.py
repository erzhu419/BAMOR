import copy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import torch
from torch import nn

import diagnose_paired_advantage as diagnostic
import test_mean_action_gradient as fixtures


def toy_data():
    rng = torch.Generator().manual_seed(6)
    x = torch.randn(128, 8, generator=rng)
    return {"inputs": x, "targets": .7 * x[:, -2] - .3 * x[:, -1], "action_dim": 2}


class NegativeQuadratic(nn.Module):
    def forward(self, inputs):
        return -inputs[:, -1:].square()


class PairedAdvantageTests(unittest.TestCase):
    def test_zero_reference_allows_both_directions_to_lose_and_ties_keep_base(self):
        data = {"inputs": torch.tensor([[-1., -1.], [1., 1.]]),
                "targets": torch.tensor([-1., 1.]), "action_dim": 1}
        model = diagnostic.PairedAdvantage(data)
        model.net = NegativeQuadratic()
        values = model(torch.tensor([[0., 0.], [0., .2], [0., -.2]]))
        torch.testing.assert_close(values, torch.tensor([0., -.04, -.04]))
        scores = dict.fromkeys(diagnostic.CANDIDATES, -.04)
        scores["base"] = 0.
        self.assertEqual(diagnostic.choose(scores), "base")
        self.assertEqual(diagnostic.choose(dict.fromkeys(diagnostic.CANDIDATES, 0.)), "base")

    def test_fit_is_train_only_deterministic_and_preserves_data_and_rng(self):
        data = toy_data()
        before = copy.deepcopy(data)
        rng = torch.get_rng_state().clone()
        initial, _ = diagnostic.fit_evaluator(data, 31, updates=0)
        model, meta = diagnostic.fit_evaluator(data, 31, updates=400)
        repeated, _ = diagnostic.fit_evaluator(data, 31, updates=400)
        error = lambda m: float((m(data["inputs"]) - data["targets"]).square().mean())
        self.assertLess(error(model), .15 * error(initial))
        torch.testing.assert_close(torch.get_rng_state(), rng, rtol=0, atol=0)
        for key in ("inputs", "targets"):
            torch.testing.assert_close(data[key], before[key], rtol=0, atol=0)
        for key, value in model.state_dict().items():
            torch.testing.assert_close(value, repeated.state_dict()[key], rtol=0, atol=0)
        torch.testing.assert_close(model.input_mean, data["inputs"].mean(0))
        self.assertEqual(meta["updates"], 400)
        self.assertTrue(all(not p.requires_grad and p.grad is None for p in model.parameters()))

    def test_real_runner_fits_before_validation_and_scores_before_branch_outcomes(self):
        d, p, g, t = diagnostic, diagnostic.p, diagnostic.g, diagnostic.training
        nets = fixtures.networks()
        stats = {k: torch.randn(8, n) for k, n in (("obs", 17), ("actions", 6), ("contexts", 13), ("rewards", 2))}
        critics = nn.ModuleList([p.RawQ(stats), p.RawQ(stats)]).eval().requires_grad_(False)
        originals = {k: copy.deepcopy(n.state_dict()) for k, n in nets.items()}
        raw_before = copy.deepcopy(critics.state_dict())
        factory, fitter, predict, roll = t.make_piecewise_mo_env, d.fit_evaluator, d.predict_actions, g.rollout
        envs, fitted = [], {}
        stage = {"fit": False, "predictions": 0, "branches": 0}

        def make_env(*args, **kwargs):
            if envs:
                self.assertTrue(stage["fit"])
                self.assertTrue((Path(directory) / "v27/seed31_advantage.pt").exists())
            env = factory(*args, **kwargs)
            envs.append(env)
            return env

        def fit(data, seed, updates):
            self.assertEqual(len(envs), 1)
            self.assertEqual(data["inputs"].shape, (96, 42))
            model, metadata = fitter(data, seed, updates)
            fitted.update(model=model, state=copy.deepcopy(model.state_dict()))
            stage["fit"] = True
            return model, metadata

        def score(*args):
            self.assertTrue(stage["fit"])
            self.assertEqual(stage["branches"], 6 * stage["predictions"])
            stage["predictions"] += 1
            return predict(*args)

        def rollout(env, nets, method, weight, obs, trackers, steps, *args, **kwargs):
            if stage["fit"] and steps == max(g.HORIZONS):
                self.assertEqual(stage["predictions"], 1 + stage["branches"] // 6)
                stage["branches"] += 1
            return roll(env, nets, method, weight, obs, trackers, steps, *args, **kwargs)

        with (tempfile.TemporaryDirectory() as directory,
              patch.object(d.sources, "load_inputs", return_value=(nets, critics, {}, {})),
              patch.object(g, "WARMUP", 2), patch.object(g, "HORIZONS", (2, 4)),
              patch.object(d, "TRAIN_REPEATS", 2), patch.object(d, "VALIDATION_REPEATS", 1),
              patch.object(d, "FIT_UPDATES", 5), patch.object(d, "fit_evaluator", side_effect=fit),
              patch.object(d, "predict_actions", side_effect=score), patch.object(g, "rollout", side_effect=rollout),
              patch.object(t, "make_piecewise_mo_env", side_effect=make_env),
              patch.object(t, "train_method", side_effect=AssertionError("No actor training"))):
            report = d.run("v27", 31, directory)
            self.assertTrue(report["complete"])
            self.assertEqual(report["label_collection_interactions"], 24 * 26)
            self.assertEqual(report["evaluation_interactions"], 12 * 26)
            self.assertEqual(report["environment_initialization_probe_steps"], 2)
            self.assertEqual(report["policy_updates"] + report["critic_updates"], 0)
            self.assertEqual((stage["predictions"], stage["branches"]), (12, 72))
            self.assertFalse(any("external_scores" in c for c in report["train_cases"]))
            self.assertTrue(all(c["identity_matched"] for c in report["train_cases"] + report["validation_cases"]))
            self.assertTrue({c["reset_seed"] for c in report["train_cases"]}.isdisjoint(
                c["reset_seed"] for c in report["validation_cases"]))
            self.assertEqual(report["fixed_preference_scores"], d.fixed_preference_scores(report["train_cases"]))
            self.assertEqual(report, json.loads((Path(directory) / "v27/seed31.json").read_text()))
        for name, network in nets.items():
            for key, value in network.state_dict().items():
                torch.testing.assert_close(value, originals[name][key], rtol=0, atol=0)
        for key, value in critics.state_dict().items():
            torch.testing.assert_close(value, raw_before[key], rtol=0, atol=0)
        for key, value in fitted["model"].state_dict().items():
            torch.testing.assert_close(value, fitted["state"][key], rtol=0, atol=0)

    def test_choices_ignore_outcomes_and_gate_checks_downside_not_only_mean(self):
        d = diagnostic
        zero = dict.fromkeys(d.CANDIDATES, 0.)
        cases = [{"mode": 0, "weight_index": 1, "reset_seed": i, "weight": [.5, .5],
            "predictions": {name: dict(zero) for name in ("soft", "raw")},
            "external_scores": {**zero, "soft_plus": 2., "raw_plus": 1.},
            "branches": {name: {"discounted_returns": {"512": [v, v]}}
                         for name, v in {**zero, "soft_plus": -2., "raw_plus": .2}.items()}}
            for i in range(d.VALIDATION_REPEATS)]
        fixed = {str(wi): {**zero, "raw_plus": .2} for wi in range(3)}
        bad = d.summarize_cases(cases, fixed)
        self.assertEqual(bad["aggregate"]["learned"]["mean_gain"], -2.)
        self.assertFalse(bad["group_passes"])
        self.assertTrue(all(c["choices"]["learned"] == "soft_plus" for c in bad["cases"]))
        for i, case in enumerate(cases):
            case["branches"]["soft_plus"]["discounted_returns"]["512"] = [-4., -4.] if i == 0 else [1., 1.]
        tail = d.summarize_cases(cases, fixed)
        self.assertTrue(tail["checks"]["positive_mean_gain"])
        self.assertTrue(tail["checks"]["mean_not_worse_than_controls"])
        self.assertFalse(tail["checks"]["downside_not_worse_than_controls"])
        self.assertEqual([c["choices"] for c in bad["cases"]], [c["choices"] for c in tail["cases"]])


if __name__ == "__main__":
    unittest.main()
