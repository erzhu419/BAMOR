import copy
import unittest

import diagnose_crossed_scene_alignment as probe


def record(full, selected, trial=None, pool=None):
    weights = [[i / 4., 1. - i / 4.] for i in range(5)]
    indices = [0, 1, 2, 3]
    sampled = [probe.mean([v[i] for i in indices]) for v in full]
    trial = [sampled[m] for m in selected] if trial is None else trial
    def evaluation(values):
        means = [probe.mean(v) for v in values]
        return dict(evaluation=dict(evaluation_weights=weights, episode_seeds_per_mode=[[1, 2]] * 4),
            endpoint=dict(utility=probe.mean(means), hv=10., per_mode_utility=means),
            pareto_per_mode=[[dict(weight=w, **{"return": [v, v]}) for w, v in zip(weights, mode)] for mode in values])
    rows = [dict(scene=s, preference=p, repeat=r, capture_step=s + 1, weight_index=indices[p],
        weight=weights[indices[p]], branches=dict(prefix=dict(return_vector=[0., 0.]),
        proposal=dict(return_vector=[trial[s], trial[s]]))) for s in range(4) for p in range(4) for r in range(2)]
    return dict(protocol=probe.SOURCE_PROTOCOL, complete=True, controls_passed=True, env="mo-halfcheetah-v5", seed=3001,
        evaluations=dict(prefix=evaluation([[0.] * 5 for _ in range(4)]), proposal=evaluation(full)),
        coverage=dict(selected_mode_ids=selected, observed_mode_ids=selected if pool is None else pool,
                      selected_steps=[1, 2, 3, 4]), trial=dict(rows=rows,
        decisions=dict(scene_means=trial, overall=probe.mean(trial))),
        choices=dict(scene="proposal" if probe.accepts(trial) else "prefix"))


class CrossedAlignmentTests(unittest.TestCase):
    def test_coverage_miss_survives_perfect_measurement(self):
        value = probe.audit_record(record([[1.] * 5, [-2.] * 5, [3.] * 5, [-1.] * 5],
                                         [0, 2, 2, 2], pool=[0, 2, 3]))
        self.assertEqual(value["diagnostic_acceptance"], dict(trial=True, held_sampled_selected=True,
            held_full_selected=True, held_sampled_all_modes=False, held_full_all_modes=False))
        self.assertEqual([v["location"] for v in value["loss_modes"]], ["not_captured", "captured_not_selected"])
        self.assertEqual(value["decomposition"]["reset_measurement_gap"], 0.)
        self.assertEqual(value["decomposition"]["sampled_preference_gap"], 0.)
        self.assertAlmostEqual(value["decomposition"]["total_gap"], value["decomposition"]["mode_multiplicity_coverage_gap"])

    def test_reset_and_preference_components_are_not_conflated(self):
        value = probe.audit_record(record([[1., 1., 1., 1., -9.], [2.] * 5, [3.] * 5, [4.] * 5],
                                         [0, 1, 2, 3], trial=[2., 2., 3., 4.]))
        d = value["decomposition"]
        self.assertAlmostEqual(d["reset_measurement_gap"], .25)
        self.assertAlmostEqual(d["sampled_preference_gap"], .5)
        self.assertAlmostEqual(d["mode_multiplicity_coverage_gap"], 0.)
        self.assertAlmostEqual(d["total_gap"], .75)
        self.assertEqual(value["loss_modes"][0]["location"], "selected_not_detected")
        self.assertFalse(value["diagnostic_acceptance"]["held_full_selected"])
        self.assertTrue(value["diagnostic_acceptance"]["held_sampled_selected"])

    def test_veto_can_protect_real_mode_loss_despite_aggregate_gain(self):
        value = probe.audit_record(record([[-1.] * 5, [3.] * 5, [3.] * 5, [3.] * 5], [0, 1, 2, 3]))
        self.assertEqual(value["actual_choice"], "prefix")
        self.assertTrue(value["aggregate_gain_with_mode_loss"])
        self.assertEqual(value["loss_modes"][0]["location"], "selected_detected")

    def test_missing_crossed_cell_or_misaligned_weight_is_rejected(self):
        source = record([[1.] * 5 for _ in range(4)], [0, 1, 2, 3])
        bad = copy.deepcopy(source)
        bad["trial"]["rows"].pop()
        with self.assertRaises(AssertionError):
            probe.audit_record(bad)
        bad = copy.deepcopy(source)
        bad["evaluations"]["proposal"]["pareto_per_mode"][0][0]["weight"] = [1., 0.]
        with self.assertRaises(AssertionError):
            probe.audit_record(bad)


if __name__ == "__main__":
    unittest.main()
