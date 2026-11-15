"""Focused checks for the frozen cheap-probe comparison."""

import tempfile
import unittest
from pathlib import Path

import numpy as np

import run_cheap_actor_probe as experiment


class CheapActorProbeTests(unittest.TestCase):
    def test_earliest_actor_wins_exact_score_tie(self):
        cells = [{"score_per_step": 1.0}, {"score_per_step": 2.0},
                 {"score_per_step": 2.0}, {"score_per_step": -1.0}]
        self.assertEqual(experiment.choose_step(cells), 1)

    def test_probe_preferences_belong_to_locked_grid(self):
        for env_name, spec in experiment.extension.ENVIRONMENTS.items():
            weights = experiment.previous.weights_for_dim(spec["reward_dim"])
            grid = experiment.training.full_simplex_grid(spec["reward_dim"])
            self.assertTrue(all(any(np.allclose(weight, point) for point in grid)
                                for weight in weights), env_name)

    def test_summary_rejects_unequal_training_and_probe_budget(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            for env_name in experiment.extension.ENVIRONMENTS:
                subdir = root / env_name.replace("-", "_")
                subdir.mkdir()
                for seed in experiment.SEEDS:
                    row = {"protocol": experiment.PROTOCOL, "env": env_name,
                           "seed": seed, "complete": True,
                           "continuation_interactions": 5,
                           "probe_evaluation_interactions": 4,
                           "selected_steps": [50000],
                           "test": {name: {"mean_same_preference_utility": 1.0,
                                           "subset_mode_mean_hv": 2.0,
                                           "per_mode_same_preference_utility": [1.0]}
                                    for name in ("selected", "final", "continued")}}
                    (subdir / f"seed{seed}.json").write_text(experiment.json.dumps(row))
            with self.assertRaises(AssertionError):
                experiment.summarize(root)


if __name__ == "__main__":
    unittest.main()
