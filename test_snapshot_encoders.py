import copy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import gymnasium as gym
import numpy as np
import torch

import diagnose_snapshot_encoders as diagnostic


class SnapshotEncoderTests(unittest.TestCase):
    def test_paired_capture_reproduces_both_arms_and_final_encoder_fronts(self):
        training = diagnostic.training
        env_factory = training.make_piecewise_mo_env
        selector_factory = training.FixedExtremeHierarchicalOperatorSelector
        steps = (32, 48, 64)
        weights = diagnostic.pilot.drift.forks.WEIGHTS

        def short_env(*args, **kwargs):
            env = env_factory(*args, **kwargs)
            env.env = gym.wrappers.TimeLimit(env.env, max_episode_steps=8)
            return env

        with (tempfile.TemporaryDirectory() as directory,
              patch.object(training, "make_piecewise_mo_env", side_effect=short_env),
              patch.object(training, "FixedExtremeHierarchicalOperatorSelector",
                           side_effect=lambda d: selector_factory(d, block_steps=4)),
              patch.object(training, "full_simplex_grid", return_value=weights)):
            for condition in diagnostic.confirmation.CONDITIONS:
                with self.subTest(condition=condition):
                    def refresh(name):
                        return (diagnostic.pilot.ReplayContextRefresh(
                            Path(directory) / name, steps=(16, 32, 48))
                            if condition == "refresh" else None)

                    args = ("mo-halfcheetah-v5", diagnostic.pilot.METHOD, 64, 20, 31, directory)
                    kwargs = dict(hidden=8, batch_size=4, learning_starts=8, eval_freq=100000,
                                  device="cpu", archive_steps=steps)
                    reference_hook = refresh("reference_refresh.json")
                    reference = training.train_method(*args, **kwargs, training_probe=reference_hook)
                    final_model = torch.load(Path(directory) / f"{diagnostic.pilot.METHOD}_seed31_model.pt",
                                             weights_only=False)
                    probe = diagnostic.SnapshotEncoderProbe(refresh("diagnostic_refresh.json"), steps)
                    result = training.train_method(*args, **kwargs, training_probe=probe)
                    reproduction = diagnostic.pilot.drift.forks.reproduction_check(result, reference)
                    self.assertTrue(reproduction["matched"])
                    self.assertEqual(reproduction["max_episode_return_error"], 0)
                    self.assertEqual(probe.training_steps, 64)
                    self.assertEqual(tuple(probe.snapshots), steps)
                    if reference_hook is not None:
                        self.assertEqual([c["latent_change_rms"] for c in reference_hook.cases],
                                         [c["latent_change_rms"] for c in probe.refresh.cases])
                    for snap in final_model["policy_archive"]:
                        for k, value in snap["policy"].items():
                            torch.testing.assert_close(value, probe.snapshots[snap["step"]]["policy"][k],
                                                       rtol=0, atol=0)
                    for k, value in final_model["encoder"].items():
                        torch.testing.assert_close(value, probe.snapshots[64]["encoder"][k], rtol=0, atol=0)
                    self.assertTrue(any(not torch.equal(probe.snapshots[32]["encoder"][k], v)
                                        for k, v in probe.snapshots[64]["encoder"].items()))
                    before_eval = copy.deepcopy(probe.snapshots[64])
                    q_state = copy.deepcopy(probe.trainer.q1.state_dict())
                    env = short_env("mo-halfcheetah-v5", switch_interval=20, seed=1031)
                    try:
                        cells = list(diagnostic.evaluate_pairs(probe, env, 31, np.array(reference["ref_point"])))
                        self.assertEqual(env._total_steps, 5 * 4 * 3 * 3 * 8)
                    finally:
                        env.close()
                    self.assertEqual([(c["actor_step"], c["encoder_step"]) for c in cells],
                                     [(32, 32), (32, 64), (48, 48), (48, 64), (64, 64)])
                    for cell in cells:
                        self.assertEqual(cell["environment_steps"], 4 * 3 * 3 * 8)
                        if cell["encoder_step"] == 64:
                            check = diagnostic.reference_front_check(cell, json.loads(json.dumps(reference)))
                            self.assertTrue(check["matched"], check)
                            self.assertEqual(check["max_return_error"], 0)
                    pairs = diagnostic.paired_metrics(cells, steps)
                    self.assertEqual(pairs[-1]["contemporaneous_minus_final_hv"], 0)
                    self.assertEqual(pairs[0]["contemporaneous_minus_final_hv"],
                                     cells[0]["hv_mean_mode"] - cells[1]["hv_mean_mode"])
                    for name, net in (("policy", probe.trainer.policy), ("encoder", probe.encoder)):
                        for k, value in net.state_dict().items():
                            torch.testing.assert_close(value, before_eval[name][k], rtol=0, atol=0)
                    for k, value in probe.trainer.q1.state_dict().items():
                        torch.testing.assert_close(value, q_state[k], rtol=0, atol=0)
                    changed = copy.deepcopy(reference)
                    changed["episode_seeds_per_mode"][0][0] += 1
                    self.assertFalse(diagnostic.reference_front_check(cells[-1], changed)["matched"])
                    changed = copy.deepcopy(reference)
                    changed["pareto_per_mode"][0][-1]["return"][0] += .01
                    self.assertFalse(diagnostic.reference_front_check(cells[-1], changed)["matched"])

    def test_production_design(self):
        self.assertEqual(diagnostic.SEED, 31)
        self.assertEqual(diagnostic.STEPS, (45000, 47500, 50000))
        self.assertFalse(diagnostic.training.uses_critic_select_policy(diagnostic.pilot.METHOD))


if __name__ == "__main__":
    unittest.main()
