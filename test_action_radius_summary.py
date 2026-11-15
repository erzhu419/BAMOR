import copy
import unittest

import numpy as np

import summarize_action_radius as summary
from test_return_critic_summary import terminal_fixture


def radius_fixture():
    reference = terminal_fixture()
    for case in reference["cases"]:
        case["diagnostics"] = {name: {"mean_output": [0.] * 6, "gradient": [1.] * 6,
            "gradient_norm": float(np.sqrt(6)), "q_scalar": {"base": 0.},
            "actions": {k: np.tanh(np.ones(6) * s * .05 / np.sqrt(6)).tolist()
                        for k, s in (("base", 0), ("plus", 1), ("minus", -1))}}
            for name in ("soft", "raw")}
    data = {"protocol": summary.pilot.PROTOCOL, "complete": True, "condition": "v27", "seed": 31,
            "method": summary.pilot.g.METHODS["v27"], "radii": [.05, .01, .002],
            "reused_radius": .05, "new_radii": [.01, .002], "weights": reference["weights"],
            "horizons": [32, 128, 512], "warmup": 128, "gamma": .99, "training_interactions": 0,
            "evaluation_interactions": 230400, "environment_initialization_probe_steps": 1,
            "critic_updates": 0, "policy_updates": 0, "cases": []}
    for source in reference["cases"]:
        case = {k: source[k] for k in ("mode", "weight_index", "repeat", "reset_seed", "weight")}
        case.update(historical_replay_matched=True, evaluation_interactions=6400, by_radius={})
        for radius in summary.pilot.RADII:
            current = copy.deepcopy(source)
            for stats in current["diagnostics"].values():
                stats["actions"] = {k: np.tanh(np.ones(6) * s * radius / np.sqrt(6)).tolist()
                                    for k, s in (("base", 0), ("plus", 1), ("minus", -1))}
            case["by_radius"][str(radius)] = {"reused": radius == .05, "case": current}
        data["cases"].append(case)
    data["summary"] = summary.pilot.summarize_cases(data["cases"])
    return data, reference


class ActionRadiusSummaryTests(unittest.TestCase):
    def test_full_grid_reused_budget_and_recomputed_summary(self):
        data, reference = radius_fixture()
        self.assertEqual(summary.validate_run(data, reference, "v27", 31), data["summary"])
        mutations = (
            lambda d: d.update(complete=False),
            lambda d: d["cases"].pop(),
            lambda d: d["cases"].__setitem__(0, copy.deepcopy(d["cases"][1])),
            lambda d: d.update(evaluation_interactions=345600),
            lambda d: d["cases"][0]["by_radius"]["0.01"].update(reused=True),
            lambda d: d["summary"]["diagnosis_counts"]["raw"].update(consistent_with_oversized_step=100))
        for mutation in mutations:
            with self.subTest(mutation=mutation):
                changed = copy.deepcopy(data)
                mutation(changed)
                with self.assertRaises(AssertionError):
                    summary.validate_run(changed, reference, "v27", 31)

    def test_reject_changed_control_radius_or_identity_rollout(self):
        data, reference = radius_fixture()
        for kind in ("control", "radius", "identity"):
            with self.subTest(kind=kind):
                changed = copy.deepcopy(data)
                radii = changed["cases"][0]["by_radius"]
                if kind == "control":
                    radii["0.05"]["case"]["branches"]["raw_plus"]["discounted_returns"]["512"][0] += 1.
                elif kind == "radius":
                    radii["0.002"]["case"]["diagnostics"]["raw"]["actions"] = copy.deepcopy(
                        radii["0.05"]["case"]["diagnostics"]["raw"]["actions"])
                else:
                    radii["0.01"]["case"]["branches"]["repeat"]["discounted_returns"]["512"][0] += 1.
                with self.assertRaises(AssertionError):
                    summary.validate_run(changed, reference, "v27", 31)


if __name__ == "__main__":
    unittest.main()
