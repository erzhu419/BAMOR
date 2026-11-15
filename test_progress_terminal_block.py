import copy
import unittest

import diagnose_progress_terminal_block as diagnostic


class TerminalBlockTests(unittest.TestCase):
    def blocks(self):
        selector = diagnostic.training.ProgressExtremeHierarchicalOperatorSelector(2)
        return [{"complete": True, "steps": 1500, "operator": op,
                 "score_after": .01 * (i + 1)} for i, op in enumerate(selector.warmup_sequence)]

    def test_prefix_reconstruction_never_uses_current_or_future_score(self):
        blocks = self.blocks() + self.blocks()
        a = diagnostic.selection_distribution(diagnostic.prefix_selector(blocks, 12))
        mutated = copy.deepcopy(blocks)
        mutated[12]["score_after"] = 1e9
        mutated.append({"not_a_valid_block": True})
        b = diagnostic.selection_distribution(diagnostic.prefix_selector(mutated, 12))
        self.assertEqual(a, b)
        s = diagnostic.prefix_selector(blocks, 12)
        self.assertEqual(sum(s.counts.values()), 12)
        self.assertEqual(sum(s.step_counts.values()), 18000)

    def test_mode_switch_takes_effect_after_recorded_transition(self):
        history = [[0, 0], [42806, 2]]
        self.assertEqual(diagnostic.mode_at(history, 1), 0)
        self.assertEqual(diagnostic.mode_at(history, 42806), 0)
        self.assertEqual(diagnostic.mode_at(history, 42807), 2)

    def test_warmup_is_deterministic(self):
        s = diagnostic.prefix_selector([], 0)
        d = diagnostic.selection_distribution(s)
        self.assertEqual(d["route"], "warmup")
        self.assertEqual(d["operator_probabilities"][s.warmup_sequence[0]], 1.)

    def test_quota_returns_inner_distribution_not_forced_one_hot_log(self):
        s = diagnostic.prefix_selector(self.blocks(), 10)
        s.step_counts = dict.fromkeys(s.operators, 0)
        s.step_counts[s.families["recovery"][0]] = 30000
        s.scores[s.families["coverage"][0]] = 1.
        d = diagnostic.selection_distribution(s)
        self.assertEqual(d["route"], "family_quota")
        self.assertEqual(sum(p > 0 for p in d["operator_probabilities"].values()), len(s.families["coverage"]))
        self.assertAlmostEqual(d["family_probabilities"]["coverage"], 1.)
        self.assertEqual(sum(p > 0 for p in s.last_probs.values()), 1)

    def test_blind_free_sampling_ignores_scores_but_adaptive_uses_them(self):
        def selector(blind):
            s = diagnostic.prefix_selector(self.blocks(), 10, blind=blind)
            s.step_counts = dict.fromkeys(s.operators, 0)
            for family, n in (("coverage", 4), ("recovery", 3), ("extreme", 3)):
                s.step_counts[s.families[family][0]] = n * 1500
            return s

        s = selector(True)
        a = diagnostic.selection_distribution(s)
        self.assertEqual(a["route"], "free_family_sampling")
        s.scores = {op: 100. * i for i, op in enumerate(s.operators)}
        self.assertEqual(a, diagnostic.selection_distribution(s))
        for probability in a["family_probabilities"].values():
            self.assertAlmostEqual(probability, 1 / 3)
        s = selector(False)
        a = diagnostic.selection_distribution(s)
        s.scores[s.families["extreme"][0]] += 1.
        self.assertNotEqual(a, diagnostic.selection_distribution(s))


if __name__ == "__main__":
    unittest.main()
