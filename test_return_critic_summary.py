import copy
from itertools import product
import unittest

import summarize_return_critic as summary


def terminal_fixture():
    p, g = summary.pilot, summary.g
    data = {"protocol": p.PROTOCOL, "complete": True, "condition": "v27", "seed": 31,
            "method": g.METHODS["v27"], "weights": g.WEIGHTS.tolist(),
            "horizons": [32, 128, 512], "warmup": 128, "epsilon": .05, "gamma": .99,
            "collection_noise_std": .1, "critic_data_interactions": 60000,
            "evaluation_interactions": 115200, "environment_initialization_probe_steps": 2,
            "policy_training_interactions": 0, "policy_updates": 0,
            "fit": {"updates": 20000, "data_rows": 60000, "batch_size": 256,
                    "learning_rate": .0003, "tau": .005}, "cases": []}
    data["collection_episodes"] = [
        {"mode": m, "weight_index": w, "repeat": r, "steps": 1000,
         "reset_seed": p.reset_seed("train", 31, m, r), "noise_seed": p.noise_seed(31, m, w, r)}
        for m, w, r in product(range(4), range(3), range(5))]
    values = {"base": 0., "repeat": 0., "soft_plus": 1., "soft_minus": -1.,
              "raw_plus": 2., "raw_minus": -2.}
    for m, w, r in product(range(4), range(3), range(3)):
        data["cases"].append({"mode": m, "weight_index": w, "repeat": r,
            "reset_seed": p.reset_seed("test", 31, m, r), "weight": g.WEIGHTS[w].tolist(),
            "evaluation_interactions": 3200, "identity_matched": True,
            "branches": {name: {"steps": 512, **{
                key: {str(h): [v, v] for h in (32, 128, 512)}
                for key in ("discounted_returns", "raw_returns")}} for name, v in values.items()},
            "predictions": {name: {k: v for k, v in values.items() if k != "repeat"}
                            for name in ("soft", "raw")}})
    data["summary"] = p.summarize_cases(data["cases"])
    return data


class ReturnCriticSummaryTests(unittest.TestCase):
    def test_complete_artifacts_and_reject_incomplete_or_stale_results(self):
        data = terminal_fixture()
        self.assertEqual(summary.validate_run(data, "v27", 31), data["summary"])
        mutations = (
            lambda d: d.update(complete=False),
            lambda d: d["cases"].pop(),
            lambda d: d["collection_episodes"].pop(),
            lambda d: d["cases"][0].update(reset_seed=summary.pilot.reset_seed("train", 31, 0, 0)),
            lambda d: d["cases"][0]["branches"]["repeat"].update(steps=511),
            lambda d: d["fit"].update(updates=10000),
            lambda d: d.update(evaluation_interactions=0),
            lambda d: d["summary"]["aggregate"]["raw"].update(mean_plus_base=100.))
        for mutation in mutations:
            with self.subTest(mutation=mutation):
                changed = copy.deepcopy(data)
                mutation(changed)
                with self.assertRaises(AssertionError):
                    summary.validate_run(changed, "v27", 31)

    def test_gate_cannot_pool_away_failed_seed_or_substitute_calibration(self):
        runs = [{"condition": c, "seed": s, "summary": {"aggregate": {
            "soft": {"mean_plus_base": .5, "positive_plus_minus_cells": 5},
            "raw": {"mean_plus_base": 1., "positive_plus_minus_cells": 10}},
            "same_action_contrast_calibration": {"raw": {"rmse": 0.}}}}
            for c, s in summary.GROUPS]
        self.assertTrue(summary.advance_rule(runs)["advance_to_shadow_replication"])
        for gain in (-.1, 0., .4):
            runs[-1]["summary"]["aggregate"]["raw"]["mean_plus_base"] = gain
            self.assertFalse(summary.advance_rule(runs)["advance_to_shadow_replication"])
        runs[-1]["summary"]["aggregate"]["raw"]["mean_plus_base"] = .5
        self.assertTrue(summary.advance_rule(runs)["advance_to_shadow_replication"])
        for run in runs:
            run["summary"]["aggregate"]["raw"]["positive_plus_minus_cells"] = 5
        self.assertFalse(summary.advance_rule(runs)["advance_to_shadow_replication"])
        with self.assertRaises(AssertionError):
            summary.advance_rule(runs[:-1])


if __name__ == "__main__":
    unittest.main()
