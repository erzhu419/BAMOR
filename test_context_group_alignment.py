import copy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import torch

import diagnose_context_group_alignment as diagnostic


def row(mode, counts, name="native"):
    return dict(model=name, mode=mode, all=dict(rows=sum(counts), group_counts=counts))


class AlignmentTests(unittest.TestCase):
    def test_saved_assignment_matches_original_and_excludes_other_context(self):
        class Buffer:
            size = 8
            contexts = np.arange(8 * 13, dtype=float).reshape(8, 13)
        class Trainer:
            reward_dim, buffer = 2, Buffer()
        original = diagnostic.source.ObservableGroups(Trainer())
        x = Buffer.contexts.copy()
        expected = original.assign(torch.as_tensor(x))
        np.testing.assert_array_equal(diagnostic.assign(x, original.state_dict()), expected)
        x[:, :2] += 1e6
        x[:, 5:] -= 1e6
        np.testing.assert_array_equal(diagnostic.assign(x, original.state_dict()), expected)

    def test_episode_balance_not_row_count_dominance(self):
        rows = [row(m, [1000, 0, 0, 0]) for m in range(4)]
        rows += [row(m, [0, 1, 0, 0]) for m in range(4)]
        result = diagnostic.alignment(rows, "native", "all")
        np.testing.assert_array_equal(result["episode_balanced_mode_by_group"], [[.5, .5, 0, 0]] * 4)
        self.assertEqual(result["raw_mode_by_group_counts"], [[1000, 1, 0, 0]] * 4)
        self.assertEqual(result["best_group_to_mode_balanced_accuracy"], .25)
        self.assertEqual(result["pairwise_overlap"], [[1.] * 4] * 4)

    def test_separated_modes_and_missing_post_warmup_are_explicit(self):
        rows = [row(m, np.eye(4, dtype=int)[m].tolist()) for m in range(4)]
        result = diagnostic.alignment(rows, "native", "all")
        self.assertEqual(result["best_group_to_mode_balanced_accuracy"], 1.)
        self.assertEqual(result["best_bijective_accuracy"], 1.)
        np.testing.assert_array_equal(result["pairwise_overlap"], np.eye(4))
        rows[0]["all"] = dict(rows=0, group_counts=[0] * 4)
        result = diagnostic.alignment(rows, "native", "all")
        self.assertIsNone(result["best_group_to_mode_balanced_accuracy"])
        self.assertEqual(result["episode_coverage"][0], dict(available=0, missing=1))

    def test_feature_boundary_counts_and_empty_window(self):
        features = np.array([[3., 1., -1.], [0., .5, -.5]])
        result = diagnostic.window_record(features, np.array([0, 2]), 2)
        self.assertEqual(result["feature_boundary_fraction"], [.5] * 3)
        self.assertEqual(result["group_counts"], [1, 0, 1, 0])
        empty = diagnostic.window_record(features[:0], np.array([], dtype=int), 2)
        self.assertIsNone(empty["feature_boundary_fraction"])
        self.assertEqual(empty["group_counts"], [0] * 4)

    def test_saved_pair_pipeline_prices_and_reuse(self):
        total = {}
        with tempfile.TemporaryDirectory() as directory, \
             patch.multiple(diagnostic, HORIZON=4, WARMUP=2, SEEDS=(3101,)), patch("builtins.print"):
            root = Path(directory)
            diagnostic.write(root / "protocol.json", dict(protocol=diagnostic.PROTOCOL,
                experiment=diagnostic.specification()))
            for env in diagnostic.ENVIRONMENTS:
                result = diagnostic.run(env, 3101, root)
                self.assertEqual(result["actual_work"]["diagnostic_interactions"], 192)
                self.assertEqual(result["actual_work"]["initialization_probes"], 1)
                for k, v in result["actual_work"].items():
                    total[k] = total.get(k, 0) + v
                with patch.object(diagnostic.training, "make_piecewise_mo_env", side_effect=AssertionError("reran")):
                    self.assertEqual(diagnostic.run(env, 3101, root), result)
                bad = copy.deepcopy(result)
                bad["actual_work"]["initialization_probes"] = 0
                with self.assertRaises(AssertionError):
                    diagnostic.check(bad)
            self.assertEqual(diagnostic.summarize(root)["actual_work"], total)
        print("CONTEXT_ALIGNMENT_SMOKE_WORK", json.dumps(total))


if __name__ == "__main__":
    unittest.main()
