import copy
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import gymnasium as gym
import numpy as np
import torch

import diagnose_raw_return_forks as raw
import diagnose_update_forks as forks
from summarize_raw_return_forks import compare
import train_bamor_mujoco as training


class RawReturnForkTests(unittest.TestCase):
    def test_scores_keep_axes_and_zero_chooses_off(self):
        off = np.zeros((3, 3, 2, 2))
        on = np.zeros_like(off)
        on[..., 0], on[..., 1] = 4., -2.
        on[1, :, 1, :] = 0.
        result = raw.raw_scores({"on": on, "off": off})
        self.assertEqual(result["mean_scalar_on_minus_off"], [[-2., -2.], [1., 0.], [4., 4.]])
        self.assertEqual(result["choose_on"], [[False, False], [True, False], [True, True]])
        self.assertEqual(np.shape(result["paired_scalar_on_minus_off"]), (3, 3, 2))

    def test_incomplete_cached_run_is_rejected(self):
        cached = {"protocol": "live_paired_update_forks_v1", "method": forks.METHOD,
                  "seed": 26, "complete": False}
        with self.assertRaises(ValueError):
            raw.RawReturnProbe(26, "unused.json", cached)

    def test_summary_compares_all_score_prefixes_to_long_outcomes(self):
        rows = []
        for scores, gain in (([1., 1., -1.], 10.), ([-1., 1., 1.], -2.), ([0., 0., 0.], 1.)):
            rows.append({"raw_scores": scores, "on_minus_off": [-gain, -gain, gain],
                         "on_minus_base": [-gain, -gain, gain], "off_minus_base": [0.] * 3,
                         "paired_on_minus_off": [[-gain] * 3, [-gain] * 3, [gain] * 3],
                         "gradient_alignment_advantage": None, "own_min_other_cosine": None})
        result = compare(rows)
        self.assertEqual(result["32"]["surrogate_sign_correct"], 2)
        self.assertEqual(result["512"]["surrogate_sign_correct"], 0)
        self.assertEqual(result["512"]["surrogate_ties"], 1)
        self.assertAlmostEqual(result["32"]["mean_selected_minus_off"], 10. / 3.)
        self.assertAlmostEqual(result["512"]["mean_selected_minus_off"], -2. / 3.)

    def test_rollout_needs_no_critic_and_counts_actual_interactions(self):
        encoder = training.LatentTransitionEncoder(17, 6, 2, hidden=8)
        context = training.build_mujoco_context(
            forks.METHOD, forks.WEIGHTS[0], trans_tracker=training.TransitionContextTracker(2),
            learned_tracker=training.LearnedTransitionContextTracker(encoder))
        trainer = training.PreferenceAlignedTargetMOSACTrainer(
            17, 6, 2, hidden=8, context_dim=len(context), buffer_capacity=16, device="cpu")
        env = training.make_piecewise_mo_env("mo-halfcheetah-v5", seed=26)
        step = type(env).step
        calls = []

        def counted(instance, action):
            calls.append(1)
            return step(instance, action)

        try:
            with (patch.object(trainer.q1, "forward", side_effect=AssertionError("Q used")),
                  patch.object(trainer.q2, "forward", side_effect=AssertionError("Q used")),
                  patch.object(type(env), "step", counted)):
                values = forks.rollout_returns(trainer, encoder, env, [601, 602, 603], (2, 4))
            self.assertEqual(len(calls), 3 * 3 * 4)
            self.assertEqual(np.shape(values), (3, 3, 2, 2))
            self.assertTrue(np.isfinite(values).all())
        finally:
            env.close()

    def test_real_reconstruction_preserves_training_and_heldout_labels_do_not_score(self):
        factory, env_factory = (training.FixedExtremeHierarchicalOperatorSelector,
                                training.make_piecewise_mo_env)

        def short_env(*args, **kwargs):
            env = env_factory(*args, **kwargs)
            env.env = gym.wrappers.TimeLimit(env.env, max_episode_steps=8)
            return env

        snapshots = []
        with tempfile.TemporaryDirectory() as directory:
            original = forks.TrainingForkProbe(26, Path(directory) / "original.json",
                                               steps=(32, 64), updates=2, repeats=1, horizons=(2, 4))
            diagnostic = None
            for replay in (False, True):
                def callback(step, trainer, encoder, env, operator):
                    if replay:
                        diagnostic(step, trainer, encoder, env, operator)
                        if step == 32:
                            cached = copy.deepcopy(diagnostic.cases[step])
                            cached["effects"] = {"intentionally_changed": True}
                            changed = raw.run_probe(trainer, encoder, env,
                                                    10000000 + 26 * 100000 + step * 100,
                                                    cached, (2, 4))
                            self.assertEqual(changed["scores"], diagnostic.report["cases"][-1]["scores"])
                            cached["own_soft_loss_with_bc_frozen_critic"]["on"][0] += 1.
                            with self.assertRaises(AssertionError):
                                raw.run_probe(trainer, encoder, env,
                                              10000000 + 26 * 100000 + step * 100, cached, (2, 4))
                    else:
                        original(step, trainer, encoder, env, operator)
                    if step != 64:
                        return False
                    snapshots.append({name: copy.deepcopy(getattr(trainer, name).state_dict())
                                      for name in ("policy", "q1", "q2", "q1_target", "q2_target",
                                                   "policy_opt", "q_opt")})
                    snapshots[-1].update({"encoder": copy.deepcopy(encoder.state_dict()),
                                          "rng": torch.get_rng_state(), "numpy_next": np.random.random(4)})
                    return True
                with (patch.object(training, "FixedExtremeHierarchicalOperatorSelector",
                                   side_effect=lambda d: factory(d, block_steps=4)),
                      patch.object(training, "make_piecewise_mo_env", side_effect=short_env)):
                    result = training.train_method(
                        "mo-halfcheetah-v5", forks.METHOD, 64, 20, 26, directory,
                        hidden=8, batch_size=4, learning_starts=8, device="cpu",
                        eval_freq=100000, training_probe=callback)
                if not replay:
                    reference = result
                    original.report["base_reproduction"] = forks.reproduction_check(result, result)
                    diagnostic = raw.RawReturnProbe(26, Path(directory) / "raw.json", original.report)
            self.assertTrue(forks.reproduction_check(result, reference)["matched"])
            self.assertEqual(len(diagnostic.report["cases"]), 2)
            # Completion remains false until the full main-trajectory check in main().
            self.assertFalse(diagnostic.report["complete"])
            for case in diagnostic.report["cases"]:
                self.assertEqual(case["extra_environment_steps"], 2 * 3 * 3 * 4)
                self.assertTrue(set(case["score_reset_seeds"]).isdisjoint(case["heldout_reset_seeds"]))
            for name in ("policy", "q1", "q2", "q1_target", "q2_target", "encoder"):
                for key, value in snapshots[0][name].items():
                    torch.testing.assert_close(value, snapshots[1][name][key], rtol=0, atol=0)
            for name in ("policy_opt", "q_opt"):
                for param, state in snapshots[0][name]["state"].items():
                    for key, value in state.items():
                        torch.testing.assert_close(value, snapshots[1][name]["state"][param][key],
                                                   rtol=0, atol=0)
            self.assertTrue(torch.equal(snapshots[0]["rng"], snapshots[1]["rng"]))
            np.testing.assert_array_equal(snapshots[0]["numpy_next"], snapshots[1]["numpy_next"])


if __name__ == "__main__":
    torch.set_num_threads(1)
    unittest.main()
