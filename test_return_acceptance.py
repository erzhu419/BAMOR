import copy
import unittest

import diagnose_return_acceptance as probe


def trial(margin, index=0, steps=1000):
    return dict(weight=[.5, .5], returns=dict(old=[0., 0.], new=[2 * margin, 0.]),
        utility_difference=margin, paired_initial_observation=True,
        steps=dict(old=steps, new=steps), weight_index=index)


def endpoint(utility, hv, modes):
    return dict(utility=utility, hv=hv, per_mode_utility=modes)


class ReturnAcceptanceTests(unittest.TestCase):
    def test_actual_weighted_returns_prefix_only_tie_and_early_termination_price(self):
        rows = [trial(2., 0, 8), trial(-1., 1, 8), trial(2., 2, 8), trial(-1e6, 3)]
        value = probe.decide(rows)
        self.assertEqual(value["chosen"], "proposal")
        self.assertEqual(value["score"], 1.)
        self.assertEqual(value["trial_interactions"], 48)
        self.assertEqual(value["weight_indices"], [0, 1, 2])
        self.assertEqual(value["hypothetical_candidate_price"], value["future_native_steps"] + value["future_native_probes"])
        self.assertEqual(probe.decide([trial(0.)] * 3)["chosen"], "prefix")
        bad = copy.deepcopy(rows)
        bad[0]["utility_difference"] += 1
        with self.assertRaises(AssertionError):
            probe.decide(bad)

    def test_held_mode_and_hypervolume_regressions_reject_despite_higher_total_utility(self):
        before = endpoint(1., 10., [0., 2.])
        after = endpoint(2., 9., [-1., 5.])
        rows = [dict(seed=1, endpoints=dict(prefix=before, proposal=after),
                     decision=dict(chosen="proposal"), blind_choice="prefix")]
        value = probe.summarize(rows)
        self.assertFalse(value["passed"])
        self.assertEqual(value["harmful_accepted_proposals"]["1"], ["hv", "mode0"])
        self.assertIn("harmful_accepted_proposal", value["failures"])
        self.assertIn("hv_below_prefix", value["failures"])

    def test_whole_policy_selection_not_modewise_oracle_and_holding_is_not_an_improvement(self):
        left = endpoint(1., 10., [0., 2.])
        right = endpoint(1., 10., [2., 0.])
        rows = [dict(seed=1, endpoints=dict(prefix=left, proposal=right),
                     decision=dict(chosen="prefix"), blind_choice="proposal")]
        value = probe.summarize(rows)
        self.assertEqual(value["methods"]["selected"]["mean_mode_utility"], [0., 2.])
        self.assertFalse(value["passed"])
        rows[0]["endpoints"]["proposal"] = left
        self.assertTrue(probe.summarize(rows)["passed"])


if __name__ == "__main__":
    unittest.main()
