import copy
import unittest

import diagnose_history_gate_alignment as probe


def row(state, preference, old=0., new=1.):
    return dict(capture_step=state, episode_age=0, weight_index=preference, weight=[1., 0.],
        branches={k: dict(return_vector=[v, 0.], steps=10, terminated=False, truncated=True)
                  for k, v in (("prefix", old), ("proposal", new))})


class HistoryAlignmentTests(unittest.TestCase):
    def test_diagonal_plus_current_is_saturated_not_independent_replication(self):
        history = [row(4, 0), row(1, 1), row(2, 2), row(3, 3)]
        current = [row(4, i) for i in range(4)]
        a, b = probe.design(history), probe.design(history + current)
        self.assertEqual((a["additive_rank"], a["connected_components"], a["additive_residual_df"]), (4, 4, 0))
        self.assertEqual((b["unique_cells"], b["additive_rank"], b["additive_residual_df"]), (7, 7, 0))
        self.assertEqual(b["identical_shared_duplicates"], 1)
        self.assertEqual(b["independent_same_cell_repeats"], 0)
        crossed = probe.design([row(state, preference) for state in range(4) for preference in range(4)])
        self.assertEqual((crossed["additive_rank"], crossed["additive_residual_df"], crossed["missing_crossed_cells"]), (7, 9, 0))

    def test_shared_cell_disagreement_is_not_silently_deduplicated(self):
        with self.assertRaises(AssertionError):
            probe.design([row(1, 0), row(1, 0, new=2.)])

    def test_alignment_uses_exact_preference_not_mode_mean(self):
        trial = row(100, 0, new=-2.)
        weights = [[1., 0.], [0., 1.]]
        def evaluation(returns, utility, modes):
            return dict(endpoint=dict(utility=utility, hv=utility + 20., per_mode_utility=modes),
                evaluation=dict(evaluation_weights=weights, episode_seeds_per_mode=[[1, 2]], ref_point=[-1., -1.]),
                pareto_per_mode=[[dict(weight=w, **{"return": r}) for w, r in zip(weights, returns)]])
        record = dict(protocol=probe.SOURCE_PROTOCOL, complete=True, controls_passed=True,
            env="mo-halfcheetah-v5", seed=2901,
            evaluations=dict(prefix=evaluation([[0., 0.], [0., 0.]], 0., [0.]),
                             proposal=evaluation([[1., 0.], [0., -5.]], -2., [-2.])),
            coverage=dict(pool_steps=[100], observed_mode_ids=[0]),
            trials={arm: dict(rows=[copy.deepcopy(trial)], decision=dict(score=-2., margins=[-2.]))
                    for arm in ("history", "current")}, choices=dict(history="prefix", current="prefix"))
        result = probe.audit_record(record)
        matched = result["arms"]["history"]["rows"][0]
        self.assertEqual((matched["trial_margin"], matched["matched_held_margin"], matched["held_mode_mean_margin"]), (-2., 1., -2.))
        self.assertTrue(matched["negative_trial_but_positive_matched_held"])
        bad = copy.deepcopy(record)
        bad["evaluations"]["proposal"]["pareto_per_mode"][0][0]["weight"] = [0., 1.]
        with self.assertRaises(AssertionError):
            probe.audit_record(bad)


if __name__ == "__main__":
    unittest.main()
