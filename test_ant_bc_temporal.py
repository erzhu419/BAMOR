import copy
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np

import run_ant_bc_temporal as experiment


class TemporalTests(unittest.TestCase):
    def test_recorder_keeps_rewards_and_separates_native_terms(self):
        class Env:
            current_mode_idx = 0
            _lock_mode = True
            _ctrl_cost_weight = .5
            mode_configs = [SimpleNamespace(reward_scale=np.array([.1, .1, 5.]))]

            def __init__(self):
                self.unwrapped = self
                reward = np.array([.1, .2, -.4], dtype=np.float32)
                reward += .3
                reward += -.01
                self.result = (None, reward * self.mode_configs[0].reward_scale, False, False,
                    dict(mode_idx=0, reward_ctrl=-.2, reward_survive=.3, reward_contact=-.01))

            def reset(self, *, seed):
                return None, {}

            def step(self, action):
                return self.result

        env = Env()
        recorder = experiment.NativeTerms(env)
        original = env.result[1].copy()
        for _ in range(21):
            for seed in (10, 11, 12):
                recorder.reset(seed=seed)
                self.assertIs(recorder.step(None), env.result)
        np.testing.assert_array_equal(env.result[1], original)
        result = dict(returns=[[original.tolist()] * 21])
        recorder.attach(result, [0], dict(episode_seeds_per_mode=[[10, 11, 12]]))
        np.testing.assert_allclose(result["native_episode_returns"][0][0][0], [-.4, .3, -.01])
        self.assertNotEqual(result["episode_roundoff"][0][0][0], 0.)
        self.assertEqual(np.asarray(result["episode_steps"]).sum(), 63)

    def test_contrast_is_additive_not_just_control_cost(self):
        weights = experiment.training.full_simplex_grid(3).astype(float)
        native = np.zeros((1, 21, 3, 3))
        full = dict(step=66500, modes=[0], returns=np.zeros((1, 21, 3)).tolist(),
                    native_episode_returns=native.tolist(), objective2_scales=[5.],
                    episode_roundoff=np.zeros((1, 21, 3)).tolist())
        other = copy.deepcopy(full)
        native[..., 0], native[..., 1], native[..., 2] = -2., 1., -.1
        other["native_episode_returns"] = native.tolist()
        returns = np.zeros((1, 21, 3))
        returns[..., 2] = -5.5
        other["returns"] = returns.tolist()
        result = experiment.contrast(full, other, weights)
        mixed = (weights[:, 2] > 0) & (weights[:, 2] < 1)
        w = weights[mixed, 2].mean()
        terms = result["weighted_native_objective2_contributions"]
        np.testing.assert_allclose(list(terms.values()), np.array([-10., 5., -.5]) * w)
        self.assertAlmostEqual(sum(terms.values()), result["delta"])

    def test_temporal_driver_reproduces_old_pairs_and_accounts_all_evaluation(self):
        previous, training = experiment.previous, experiment.training
        make_env, trainer_cls = training.make_piecewise_mo_env, training.ConsistentTargetMOSACTrainer
        init = previous.NativeSelector.__init__
        kwargs = dict(hidden=8, batch_size=4, learning_starts=5, eval_freq=1000,
                      log_freq=1000, device="cpu")

        def short_env(*args, **kw):
            kw.update(switch_interval=12, mode_order="sequential")
            env = make_env(*args, **kw)
            env.env._max_episode_steps = 4
            return env

        def short_selector(selector, *args, **kw):
            kw["block_steps"] = 4
            init(selector, *args, **kw)

        with tempfile.TemporaryDirectory() as folder, \
             patch.multiple(previous, SEEDS=(17006,), CUT=53, BLOCK_END=57, END=80, KWARGS=kwargs), \
             patch.multiple(experiment, SEEDS=(17006,), CUT=53, BLOCK_END=57, END=80,
                            MIDDLE=(61, 65, 69, 73, 77), KWARGS=kwargs), \
             patch.object(training, "make_piecewise_mo_env", side_effect=short_env), \
             patch.object(training, "ConsistentTargetMOSACTrainer",
                          side_effect=lambda *a, **kw: trainer_cls(*a, buffer_capacity=64, **kw)), \
             patch.object(previous.NativeSelector, "__init__", new=short_selector), patch("builtins.print"):
            reference = previous.run(17006, Path(folder) / "reference")
            root = Path(folder) / "temporal"
            with patch.object(experiment, "load_reference", return_value=reference):
                report = experiment.run(17006, root)
            self.assertTrue(report["complete"])
            self.assertEqual(report["completed_budget"], dict(training_interactions=107, learner_updates=103,
                evaluation_interactions=5544, environment_initialization_probe_steps=4))
            for name, row in report["branches"].items():
                self.assertEqual(row["reproduced_grids"], {"57": True, "80": True})
                self.assertEqual(row["block_history"], reference["branches"][name]["block_history"])
                self.assertEqual(row["partial_tail"]["half_counts"], [2, 1])
                self.assertFalse(row["partial_tail"]["used_for_feedback"])
                self.assertNotIn("progress_score", row["block_history"][-1])
            self.assertEqual(report["branches"]["no_bc"]["routing"]["rng_matched_steps"], 27)
            result = json.loads((root / "seed17006_readout.json").read_text())
            self.assertEqual([r["step"] for r in result["timeline"]], [57, 61, 65, 69, 73, 77, 80])
            self.assertTrue(experiment.summarize(root)["complete"])
            with patch.object(experiment, "load_reference", side_effect=AssertionError("retrained a complete pair")):
                self.assertTrue(experiment.run(17006, root)["complete"])
            incomplete = Path(folder) / "incomplete"
            incomplete.mkdir()
            experiment.base.write_report(incomplete / "seed17006.json", dict(protocol=experiment.PROTOCOL, seed=17006, complete=False))
            with self.assertRaisesRegex(RuntimeError, "Incomplete attempt preserved"):
                experiment.run(17006, incomplete)
            corrupted = copy.deepcopy(report)
            corrupted["branches"]["no_bc"]["reproduced_grids"]["80"] = False
            with self.assertRaises(AssertionError):
                experiment.summarize_seed(corrupted)
        print("ANT_TEMPORAL_SMOKE training=268 updates=260 evaluation=12348 probes=12")


if __name__ == "__main__":
    unittest.main()
