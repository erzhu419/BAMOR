import copy
from itertools import product
from pathlib import Path
import tempfile
import unittest

import torch

import summarize_paired_advantage as summary


def terminal_fixture():
    p = summary.pilot
    data = {"protocol": p.PROTOCOL, "complete": True, "condition": "v27", "seed": 31,
            "method": p.g.METHODS["v27"], "epsilon": .05, "weights": p.g.WEIGHTS.tolist(),
            "horizons": [32, 128, 512], "gamma": .99, "warmup": 128,
            "train_repeats": 16, "validation_repeats": 8, "candidates": list(p.CANDIDATES),
            "label_collection_interactions": 614400, "evaluation_interactions": 307200,
            "environment_initialization_probe_steps": 2, "policy_training_interactions": 0,
            "policy_updates": 0, "critic_updates": 0,
            "fit": {"updates": 2000, "data_rows": 768, "batch_size": 128,
                    "learning_rate": .0003, "hidden": 64,
                    "losses": [{"update": i} for i in (500, 1000, 1500, 2000)]}}
    zero = dict.fromkeys(p.CANDIDATES, 0.)
    for split, repeats in (("train", 16), ("validation", 8)):
        cases = []
        for mode, wi, repeat in product(range(4), range(3), range(repeats)):
            values = {**zero, "soft_plus": 1., "raw_plus": .5, "repeat": 0.}
            case = {"mode": mode, "weight_index": wi, "repeat": repeat,
                    "reset_seed": p.reset_seed(split, 31, mode, repeat), "weight": data["weights"][wi],
                    "anchor_observation": [0.] * 17, "anchor_context": [0.] * 13,
                    "identity_matched": True, "evaluation_interactions": 3200,
                    "predictions": {name: {**zero, "raw_plus": 1.} for name in ("soft", "raw")},
                    "diagnostics": {name: {"actions": {label: [v] * 6 for label, v in
                                                       (("base", 0.), ("plus", .05), ("minus", -.05))}}
                                    for name in ("soft", "raw")},
                    "branches": {name: {"steps": 512, **{key: {str(h): [v, v] for h in data["horizons"]}
                                      for key in ("discounted_returns", "raw_returns")}} for name, v in values.items()}}
            if split == "validation":
                case["external_scores"] = {**zero, "soft_plus": 2.}
            cases.append(case)
        data[split + "_cases"] = cases
    data["fixed_preference_scores"] = p.fixed_preference_scores(data["train_cases"])
    data["summary"] = p.summarize_cases(data["validation_cases"], data["fixed_preference_scores"])
    return data


class PairedAdvantageSummaryTests(unittest.TestCase):
    def test_frozen_grid_decisions_and_budget(self):
        data = terminal_fixture()
        result = summary.validate_run(data, "v27", 31)
        self.assertTrue(result["group_passes"])
        compact = summary.compact_result(data, result)
        self.assertEqual(compact["aggregate"]["learned"]["mean_gain"], 1.)
        self.assertEqual(compact["selection_counts"]["learned"]["soft_plus"], 96)
        self.assertEqual(compact["descriptive"]["oracle_mean_gain_upper_bound"], 1.)
        self.assertEqual(compact["descriptive"]["selected_losses"], 0)
        mutations = (
            lambda d: d.update(complete=False),
            lambda d: d["validation_cases"].pop(),
            lambda d: d["validation_cases"].__setitem__(0, copy.deepcopy(d["validation_cases"][1])),
            lambda d: d["validation_cases"][0].update(reset_seed=d["train_cases"][0]["reset_seed"]),
            lambda d: d.update(evaluation_interactions=0),
            lambda d: d["fit"].update(updates=4000),
            lambda d: d["train_cases"][0].update(external_scores=dict.fromkeys(summary.pilot.CANDIDATES, 0.)),
            lambda d: d["validation_cases"][0]["external_scores"].update(base=1.),
            lambda d: d["validation_cases"][0]["branches"]["repeat"]["discounted_returns"]["512"].__setitem__(0, 1.),
            lambda d: d["fixed_preference_scores"]["0"].update(soft_plus=3.),
            lambda d: d["summary"].update(group_passes=False))
        for mutation in mutations:
            with self.subTest(mutation=mutation):
                changed = copy.deepcopy(data)
                mutation(changed)
                with self.assertRaises(AssertionError):
                    summary.validate_run(changed, "v27", 31)

    def test_fit_diagnostics_use_saved_model_without_training(self):
        data = terminal_fixture()
        p = summary.pilot
        model = p.PairedAdvantage(p.training_data(data["train_cases"]))
        for param in model.parameters():
            param.data.zero_()
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "advantage.pt"
            torch.save({"state_dict": model.state_dict(), "action_dim": 6}, path)
            result = summary.fit_diagnostics(data, path)
        self.assertAlmostEqual(result["train_rmse"], (1.25 / 4) ** .5)
        self.assertEqual(result["train_rows"], 768)
        self.assertEqual(result["validation_rmse"], data["summary"]["calibration"]["learned"]["rmse"])


if __name__ == "__main__":
    unittest.main()
