import copy
import unittest

import numpy as np

import diagnose_ant_shared_block as diagnostic
from test_ant_allocation import fixture


def histories():
    native = fixture()
    native["episode_rewards"] = [{"step": 1000, "return": [1., 2., 3.]}]
    shared = copy.deepcopy(native)
    selector = diagnostic.experiment.SharedCreditOwnPreferenceSelector(3)
    for b in shared["adaptive_operator_final"]["block_history"]:
        op = b["operator"]
        if b["complete"]:
            score = (1 - selector.lr) * selector.scores[op] + selector.lr * b["progress_score"]
            for alias in selector.credit_aliases[op]:
                selector.scores[alias] = score
        b["score_after"] = selector.scores[op]
    return {"per_label": native, "shared": shared}


class SharedBlockTests(unittest.TestCase):
    def test_prefix_credit_and_source_decomposition(self):
        before = np.random.get_state()
        for name, raw in histories().items():
            blocks = raw["adaptive_operator_final"]["block_history"]
            selector = diagnostic.prefix_selector(blocks, 40, name == "shared")
            self.assertEqual(sum(selector.counts.values()), 40)
            self.assertEqual(sum(selector.step_counts.values()), 60000)
            for op in selector.operators:
                sources = diagnostic.score_sources(blocks, 40, selector, op, raw["ground_truth"]["mode_history"])
                self.assertAlmostEqual(sum(x["ema_contribution"] for x in sources), selector.scores[op])
                self.assertTrue(all(x["age_steps"] > 0 for x in sources))
            diagnostic.allocation.replay.selection_distribution(selector)
        after = np.random.get_state()
        np.testing.assert_array_equal(before[1], after[1])
        self.assertEqual(before[2:], after[2:])

    def test_future_records_do_not_change_selection(self):
        for name, raw in histories().items():
            blocks = raw["adaptive_operator_final"]["block_history"]
            changed = copy.deepcopy(blocks)
            for b in changed[40:]:
                b["progress_score"] = .99
                b["score_after"] = -.99
            left = diagnostic.prefix_selector(blocks, 40, name == "shared")
            right = diagnostic.prefix_selector(changed, 40, name == "shared")
            self.assertEqual(diagnostic.allocation.replay.selection_distribution(left),
                             diagnostic.allocation.replay.selection_distribution(right))

    def test_common_prefix_excludes_ema_but_not_reward_differences(self):
        raws = histories()
        shared = raws["shared"]["adaptive_operator_final"]["block_history"]
        shared[40]["operator"] = "extreme:edge_relabel_envelope_policy_bc"
        result = diagnostic.common_prefix(raws, 40)
        self.assertEqual(result["start_step"], 65001)
        self.assertEqual(result["end_step"], 66500)
        self.assertEqual(result["subsequent_identical_operator_blocks"], 6)
        shared[30]["half_mean_rewards"][0][0] += .01
        with self.assertRaises(AssertionError):
            diagnostic.common_prefix(raws, 40)

    def test_conditional_cdf_uses_family_mass(self):
        result = diagnostic.family_intervals({"a": .1, "b": .2, "other_family": .7}, ["a", "b"])
        self.assertAlmostEqual(result["a"]["conditional_probability"], 1 / 3)
        np.testing.assert_allclose(result["b"]["conditional_cdf_interval"], [1 / 3, 1.])
        self.assertEqual(result["b"]["probability"], .2)


if __name__ == "__main__":
    unittest.main()
