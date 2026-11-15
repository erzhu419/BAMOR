import unittest

import numpy as np

from diagnose_train_only_snapshot_proxy import single_actor_hv, training_score


class TrainingOnlySnapshotProxyTest(unittest.TestCase):
    def test_training_score_excludes_future_and_old_episodes(self):
        episodes = [
            {"step": 5000, "return": [100.0, 100.0], "mode": "unused"},
            {"step": 7500, "return": [2.0, 2.0]},
            {"step": 10000, "return": [1.0, 1.0]},
            {"step": 12500, "return": [200.0, 200.0]},
        ]
        score, count = training_score(episodes, 10000, np.zeros(2))
        self.assertEqual(count, 2)
        self.assertEqual(score, 4.0)
        episodes[-1]["return"] = [1000.0, 1000.0]
        self.assertEqual(training_score(episodes, 10000, np.zeros(2)), (score, count))

    def test_single_actor_hv_uses_only_requested_snapshot(self):
        data = {"pareto_per_mode": [[
            {"archive_step": 5000, "return": [2.0, 2.0]},
            {"archive_step": 7500, "return": [3.0, 3.0]},
        ]]}
        self.assertEqual(single_actor_hv(data, 5000, np.zeros(2)), 4.0)
        self.assertEqual(single_actor_hv(data, 7500, np.zeros(2)), 9.0)


if __name__ == "__main__":
    unittest.main()
