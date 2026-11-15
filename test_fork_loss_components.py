import copy
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import torch
import gymnasium as gym

import diagnose_fork_loss_components as components
import diagnose_update_forks as forks
import train_bamor_mujoco as training
from test_update_forks import toy_trainer


class ForkLossComponentTests(unittest.TestCase):
    def test_decomposition_exactly_reproduces_the_original_loss(self):
        for alpha, bc in ((.2, .006), (0., 0.)):
            trainer = toy_trainer()
            trainer.alpha, trainer.behavior_clone_coef = alpha, bc
            batch = forks.batch_at(trainer.buffer, np.arange(4))
            old = forks.audit_losses(trainer, trainer, batch, 301)
            new = components.audit_components(trainer, trainer, batch, 301)
            np.testing.assert_array_equal(old, new["total"])
            np.testing.assert_allclose(np.array(new["negative_q"]) + new["entropy_loss"]
                                       + new["bc_loss"], old, rtol=2e-6, atol=2e-6)
            if alpha == 0:
                self.assertEqual(new["entropy_loss"], [0.] * 3)
            if bc == 0:
                self.assertEqual(new["bc_loss"], [0.] * 3)

    def test_all_values_use_the_common_frozen_critic(self):
        base = toy_trainer()
        branch = forks.clone_trainer(base, base.buffer)
        batch = forks.batch_at(base.buffer, np.arange(4))
        inputs = (batch[0], [torch.as_tensor(w).expand(4, -1) for w in forks.WEIGHTS])
        before = components.audit_components(branch, base, batch, 401)
        reset_before = components.reset_action_q(branch, base, inputs)
        with torch.no_grad():
            for net in (branch.q1, branch.q2):
                for p in net.parameters():
                    p.add_(3.)
        self.assertEqual(before, components.audit_components(branch, base, batch, 401))
        self.assertEqual(reset_before, components.reset_action_q(branch, base, inputs))

    def test_advantage_signs_and_reset_repeat_axis(self):
        measurements = {
            "on": {"negative_q": [-3.] * 3, "entropy_loss": [-.1] * 3,
                   "bc_loss": [.2] * 3, "total": [-2.9] * 3,
                   "deterministic_replay_q": [3.] * 3,
                   "deterministic_reset_q": [[3., 4.]] * 3},
            "off": {"negative_q": [-1.] * 3, "entropy_loss": [-.4] * 3,
                    "bc_loss": [.1] * 3, "total": [-1.3] * 3,
                    "deterministic_replay_q": [1.] * 3,
                    "deterministic_reset_q": [[1., 1.]] * 3},
        }
        result = components.advantages(measurements)
        np.testing.assert_allclose(result["total"], [1.6] * 3)
        np.testing.assert_allclose(result["negative_q"], [2.] * 3)
        np.testing.assert_allclose(result["entropy_loss"], [-.3] * 3)
        self.assertEqual(result["deterministic_replay_q"], [2.] * 3)
        self.assertEqual(result["deterministic_reset_q"], [2.5] * 3)

    def test_incomplete_labels_are_rejected(self):
        cached = {"protocol": "live_paired_update_forks_v1", "method": forks.METHOD,
                  "seed": 26, "complete": False}
        with self.assertRaises(ValueError):
            components.LossComponentProbe(26, "unused.json", cached)

    def test_reset_inputs_match_independent_rollout_resets(self):
        env = training.make_piecewise_mo_env("mo-halfcheetah-v5", seed=26)
        encoder = training.LatentTransitionEncoder(17, 6, 2, hidden=8)
        env._apply_mode(2)
        seeds = [501, 502, 503]
        try:
            states, contexts = components.reset_inputs(encoder, env, seeds)
            for i, seed in enumerate(seeds):
                original_copy = copy.deepcopy(env)
                try:
                    obs, _ = original_copy.reset(seed=seed)
                    torch.testing.assert_close(states[i], torch.as_tensor(obs, dtype=torch.float32),
                                               rtol=0, atol=0)
                finally:
                    original_copy.close()
            for wi, weight in enumerate(forks.WEIGHTS):
                context = training.build_mujoco_context(
                    forks.METHOD, weight, trans_tracker=training.TransitionContextTracker(2),
                    learned_tracker=training.LearnedTransitionContextTracker(encoder))
                torch.testing.assert_close(contexts[wi], torch.as_tensor(context).expand(3, -1),
                                           rtol=0, atol=0)
        finally:
            env.close()

    def test_real_reconstruction_reuses_labels_without_environment_steps(self):
        factory = training.FixedExtremeHierarchicalOperatorSelector
        env_factory = training.make_piecewise_mo_env

        def short_env(*args, **kwargs):
            env = env_factory(*args, **kwargs)
            env.env = gym.wrappers.TimeLimit(env.env, max_episode_steps=8)
            return env

        snapshots = []
        with tempfile.TemporaryDirectory() as directory:
            original = forks.TrainingForkProbe(
                26, Path(directory) / "original.json", steps=(32, 64),
                updates=2, repeats=1, horizons=(2, 4))
            diagnostic = None
            for replay in (False, True):
                def callback(step, trainer, encoder, env, operator):
                    if replay:
                        with patch.object(type(env), "step", side_effect=AssertionError("No new rollouts")):
                            if step == 32:
                                bad = copy.deepcopy(diagnostic.cases[32])
                                bad["own_soft_loss_with_bc_frozen_critic"]["base"][0] += 1
                                with self.assertRaises(AssertionError):
                                    components.reconstruct(trainer, encoder, env,
                                                           10000000 + 26 * 100000 + step * 100, bad)
                            diagnostic(step, trainer, encoder, env, operator)
                    else:
                        original(step, trainer, encoder, env, operator)
                    if step != 64:
                        return False
                    snapshots.append({name: copy.deepcopy(getattr(trainer, name).state_dict())
                                      for name in ("policy", "q1", "q2", "q1_target", "q2_target")})
                    snapshots[-1].update({"encoder": copy.deepcopy(encoder.state_dict()),
                                          "rng": torch.get_rng_state(),
                                          "numpy_next": np.random.random(4)})
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
                    diagnostic = components.LossComponentProbe(
                        26, Path(directory) / "components.json", original.report)
            self.assertTrue(forks.reproduction_check(result, reference)["matched"])
            self.assertEqual(len(diagnostic.report["cases"]), 2)
            for old, new in zip(original.report["cases"], diagnostic.report["cases"]):
                self.assertTrue(new["matched_original_losses"])
                self.assertEqual(new["extra_environment_steps"], 0)
                self.assertEqual(new["cached_return_effects"], old["effects"])
            for name in ("policy", "q1", "q2", "q1_target", "q2_target", "encoder"):
                for key, value in snapshots[0][name].items():
                    torch.testing.assert_close(value, snapshots[1][name][key], rtol=0, atol=0)
            self.assertTrue(torch.equal(snapshots[0]["rng"], snapshots[1]["rng"]))
            np.testing.assert_array_equal(snapshots[0]["numpy_next"], snapshots[1]["numpy_next"])


if __name__ == "__main__":
    torch.set_num_threads(1)
    unittest.main()
