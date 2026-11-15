import unittest
from types import SimpleNamespace

import numpy as np
import torch
from torch import nn

import diagnose_candidate_gates as gates
import diagnose_candidate_calibration as probe
from test_candidate_calibration import make_model


class FixedQ(nn.Module):
    def __init__(self, values):
        super().__init__()
        self.values = torch.tensor(values, dtype=torch.float32)

    def forward(self, state, action, context):
        return self.values


class CandidateGateTests(unittest.TestCase):
    def test_gate_rules_use_strict_agreement_and_do_not_reselect(self):
        q = np.array([[1., 3.], [2., 4.], [2., 2.]])
        original = np.array([1., 2., 2.])
        m = gates.gate_masks(q, original)
        np.testing.assert_array_equal(m["original"], [True, True])
        np.testing.assert_array_equal(m["agreement"], [True, False])
        np.testing.assert_array_equal(m["separated"], [False, False])
        q[1] = [4., 5.]
        self.assertTrue(gates.gate_masks(q, original)["separated"][0])
        q[1] = [3., 4.]
        self.assertFalse(gates.gate_masks(q, original)["separated"][0])
        q[1] = [1., 4.]
        self.assertFalse(gates.gate_masks(q, original)["agreement"][0])
        original[1] = original[0]
        q[1] = [4., 5.]
        self.assertFalse(gates.gate_masks(q, original)["separated"][0])

    def test_scalarize_each_critic_before_disagreement_comparison(self):
        trainer = SimpleNamespace(q1=FixedQ([[0, 10], [4, 4], [6, 6]]),
                                  q2=FixedQ([[10, 0], [4, 4], [6, 6]]))
        model = SimpleNamespace(trainer=trainer)
        q, original = gates.critic_values(model, np.zeros(17), np.zeros((3, 6)),
                                          torch.zeros(1, 13), np.array([.5, .5]))
        np.testing.assert_array_equal(q, [[5, 5], [4, 4], [6, 6]])
        np.testing.assert_array_equal(original, [0, 4, 6])
        m = gates.gate_masks(q, original)
        np.testing.assert_array_equal(m["original"], [True, True])
        np.testing.assert_array_equal(m["agreement"], [False, True])
        np.testing.assert_array_equal(m["separated"], [False, True])

    def test_rejection_is_zero_gain_not_dropped_from_denominator(self):
        gain = np.array([-2., 3., 7., 0.])
        eligible = np.array([True, True, True, False])
        accepted = np.array([False, True, False, False])
        stats = gates.gate_statistics(gain, eligible, accepted)
        self.assertEqual(stats["mean_gain"], .75)
        self.assertEqual(stats["accepted_mean_gain"], 3.)
        self.assertEqual(stats["change_from_original"], -1.25)
        self.assertEqual(stats["positive_retained"], 1)
        self.assertEqual(stats["positive_rejected"], 1)
        self.assertEqual(stats["negative_rejected"], 1)
        self.assertEqual(stats["positive_mass_retained_fraction"], .3)
        self.assertEqual(stats["negative_mass_removed_fraction"], 1.)
        stats = gates.gate_statistics(gain, eligible, np.zeros(4, dtype=bool))
        self.assertEqual(stats["mean_gain"], 0.)
        self.assertIsNone(stats["accepted_mean_gain"])

    def test_reconstruct_real_probe_and_reuse_returns_without_rollouts(self):
        model = make_model()
        env = probe.training.make_piecewise_mo_env("mo-halfcheetah-v5", seed=26)
        try:
            weight = np.array([.5, .5], dtype=np.float32)
            history = probe.collect_prefix(env, model, 3, 826031, weight, 8)
            _, q, metadata = probe.probe_actions(model, history[-1][3], weight,
                                                model.trackers(history), 7)
            case = {"mode": 3, "weight": weight.tolist(), "reset_seed": 826031,
                    "random_seed": 7, "q_values": q.tolist(), **metadata,
                    "effects": {r: {str(h): {"mean_gain": [[-1., 2.], [3., -4.]]}
                                    for h in probe.HORIZONS} for r in probe.CONTINUATIONS}}
            result = gates.reconstruct_case(env, model, model, case, 8)
            self.assertLessEqual(result["reconstruction_max_q_difference"], 1e-5)
            self.assertEqual(result["effects"]["own"]["512"], [[-1., 2.], [3., -4.]])
            self.assertEqual(env._total_steps, 8)
            original = gates.summarize([result])["own"]["512"]["original"]
            self.assertEqual(original["cross_preference_best"]["raw"]["mean_gain"],
                             -1. if result["gates"]["original"][0] else 0.)
        finally:
            env.close()


if __name__ == "__main__":
    unittest.main()
