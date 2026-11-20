import json
import unittest
from collections import Counter
from unittest.mock import patch

import numpy as np

from collect_nca_review import geometry, grid_metrics
from paper.make_nca_review_tables import MAIN, REVIEW, RESULTS, paired_intervals, review_grid, review_routing
from run_nca_score_ablation import score_on_receipt
from train_bamor_mujoco import ProgressExtremeHierarchicalOperatorSelector, full_simplex_grid


class ReviewDiagnosticsTests(unittest.TestCase):
    def test_reassignment_scores_each_command_at_the_original_request(self):
        weights = np.eye(2)
        direct, best = grid_metrics(weights, [[0, 3], [2, 0]])
        np.testing.assert_array_equal(direct, [0, 0])
        np.testing.assert_array_equal(best - direct, [2, 3])
        direct, best = grid_metrics(weights, [[2, 3], [2, 3]])
        np.testing.assert_array_equal(best - direct, [0, 0])

    def test_preference_grid_regions(self):
        self.assertEqual(Counter(geometry(full_simplex_grid(2))),
                         {"vertex": 2, "interior": 19})
        weights = full_simplex_grid(3)
        regions = geometry(weights)
        self.assertEqual(Counter(regions), {"vertex": 3, "boundary": 12, "interior": 6})
        mixed = (weights[:, -1] > 0) & (weights[:, -1] < 1)
        self.assertEqual(Counter(regions[mixed]), {"boundary": 8, "interior": 6})

    def test_mixed_dispatch_has_three_equal_branches(self):
        select = ProgressExtremeHierarchicalOperatorSelector(3)
        with patch("train_bamor_mujoco.sample_edge_weight", return_value="edge"), \
             patch("train_bamor_mujoco.sample_stratified_weight", return_value="stratified"), \
             patch("train_bamor_mujoco.sample_uniform_weight", return_value="uniform"):
            actual = [select.sample_weight("coverage:mixed_relabel_envelope", 3, step)
                      for step in range(1, 7)]
        self.assertEqual(actual, ["uniform", "stratified", "edge", "stratified", "uniform", "edge"])

    def test_archive_reconciliation_and_interval_sensitivity(self):
        path = RESULTS / REVIEW / "grid.json"
        data = json.loads(path.read_text())
        rows = data["environments"][MAIN[1]]
        delta = [np.mean([m["all"][0] for m in r["arms"]["v25"]]) -
                 np.mean([m["all"][0] for m in r["arms"]["capql_full_simplex"]]) for r in rows]
        d, boot, classic = paired_intervals(delta)
        self.assertAlmostEqual(d, 83.17, places=2)
        np.testing.assert_array_equal(np.round(boot, 2), [3.03, 165.15])
        np.testing.assert_array_equal(np.round(classic, 2), [-6.80, 173.14])
        documents = review_grid(RESULTS)
        self.assertEqual(len(documents), 5)
        self.assertIn(r"w_{\mathrm{energy}}", documents["table_ant_geometry.tex"])

    def test_rerun_can_use_the_packaged_score_on_receipt(self):
        receipt = score_on_receipt(MAIN[0], 1601)
        self.assertTrue(receipt["complete"])
        self.assertEqual((receipt["env"], receipt["method"], receipt["seed"]),
                         (MAIN[0], "v25", 1601))
        self.assertEqual(receipt["completed_budget"]["training_interactions"], 75000)

    def test_routing_results_reconcile_all_40_pairs_and_budget(self):
        documents = review_routing(RESULTS)
        for value in ("-14.41", "+15.95", "[-78.40,53.62]", "[-63.56,97.75]", "6/9/5", "10/9/1"):
            self.assertIn(value, documents["table_routing75.tex"])
        for value in ("3,000,000", "8,209,456", "{80}"):
            self.assertIn(value, documents["nca_routing_cost.tex"])
