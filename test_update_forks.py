import copy
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import torch

import diagnose_update_forks as probe
import train_bamor_mujoco as training


def toy_trainer(candidates=3):
    torch.manual_seed(7)
    trainer = training.PreferenceAlignedTargetMOSACTrainer(
        3, 2, 2, hidden=8, buffer_capacity=64, batch_size=4, device="cpu",
        envelope_target_candidates=3, envelope_policy_candidates=candidates,
        preference_relabeling=True, critic_relabel_copies=2, behavior_clone_coef=.006)
    for i in range(32):
        state = np.full(3, i / 32, dtype=np.float32)
        weight = np.array([.3, .7], dtype=np.float32)
        trainer.buffer.push(state, np.zeros(2), weight, weight,
                            np.array([1., -.2]), state + .1, weight, False)
    with probe.isolated_rng(3):
        trainer.update()
    return trainer


class UpdateForkTests(unittest.TestCase):
    def test_reproduction_compares_logged_rewards_and_json_schedule(self):
        original = {"episode_rewards": [{"step": 1000, "return": [1., 2.]}],
                    "ground_truth": {"mode_history": [(0, 0)]}}
        saved = copy.deepcopy(original)
        saved["ground_truth"]["mode_history"] = [[0, 0]]
        self.assertTrue(probe.reproduction_check(original, saved)["matched"])
        saved["episode_rewards"][0]["return"][0] += .1
        self.assertFalse(probe.reproduction_check(original, saved)["matched"])
        saved["episode_rewards"][0]["step"] += 1
        self.assertFalse(probe.reproduction_check(original, saved)["matched"])

    def test_rng_restored_even_on_exception(self):
        np.random.seed(12)
        torch.manual_seed(12)
        np_state, torch_state = np.random.get_state(), torch.get_rng_state()
        with self.assertRaises(RuntimeError):
            with probe.isolated_rng(44):
                np.random.random(100)
                torch.randn(100)
                raise RuntimeError("probe failed")
        after = np.random.get_state()
        self.assertEqual(np_state[0], after[0])
        np.testing.assert_array_equal(np_state[1], after[1])
        self.assertEqual(np_state[2:], after[2:])
        self.assertTrue(torch.equal(torch_state, torch.get_rng_state()))

    def test_clones_preserve_optimizer_and_pair_critic_update(self):
        base = toy_trainer()
        pool = probe.ReplaySubset(base.buffer, np.arange(8, 32))
        on, off = [probe.clone_trainer(base, pool) for _ in range(2)]
        off.envelope_policy_candidates = 1
        self.assertIs(on.buffer, off.buffer)
        original = copy.deepcopy(base.policy.state_dict())
        for opt_name in ("policy_opt", "q_opt"):
            old, new = [getattr(t, opt_name).state_dict() for t in (base, on)]
            self.assertEqual(old["param_groups"], new["param_groups"])
            for param in old["state"]:
                for key, value in old["state"][param].items():
                    torch.testing.assert_close(value, new["state"][param][key], rtol=0, atol=0)
        with probe.isolated_rng(80):
            batch = pool.sample(100)
        self.assertTrue((batch[0][:, 0] >= .25).all())
        for branch in (on, off):
            with probe.isolated_rng(90):
                branch.update()
        for name in ("q1", "q2", "q1_target", "q2_target"):
            for a, b in zip(getattr(on, name).parameters(), getattr(off, name).parameters()):
                torch.testing.assert_close(a, b, rtol=0, atol=0)
        self.assertTrue(any(not torch.equal(a, b) for a, b in
                            zip(on.policy.parameters(), off.policy.parameters())))
        for key, value in original.items():
            torch.testing.assert_close(value, base.policy.state_dict()[key], rtol=0, atol=0)

    def test_single_candidate_is_a_null_control(self):
        base = toy_trainer(candidates=1)
        batch = probe.batch_at(base.buffer, np.arange(4))
        signals = probe.gradient_signals(base, batch, 50)
        np.testing.assert_allclose(signals["own_preference_cosines"],
                                   signals["search_to_own_cosines"], rtol=0, atol=0)
        pool = probe.ReplaySubset(base.buffer, np.arange(4, 32))
        a, b = [probe.clone_trainer(base, pool) for _ in range(2)]
        for branch in (a, b):
            for update in range(3):
                with probe.isolated_rng(100 + update):
                    branch.update()
        for key, value in a.policy.state_dict().items():
            torch.testing.assert_close(value, b.policy.state_dict()[key], rtol=0, atol=0)
        self.assertEqual(probe.audit_losses(a, base, batch, 70),
                         probe.audit_losses(b, base, batch, 70))

    def test_effects_keep_weight_repeat_horizon_axes(self):
        a = np.zeros((3, 2, 2, 2))
        b = np.ones_like(a)
        b[..., 1] = -2
        effects = probe.paired_effects({"base": a, "off": a, "on": b})
        expected = np.array([[-2., -2.], [-.5, -.5], [1., 1.]])
        np.testing.assert_array_equal(
            effects["on_minus_off"]["mean_scalar_by_weight_horizon"], expected)

    def test_environment_copy_preserves_physics_and_future_switches(self):
        env = training.make_piecewise_mo_env("mo-halfcheetah-v5", seed=26, switch_interval=4)
        env.reset(seed=26)
        # Exponential schedules have a real 1000-step minimum dwell.
        for _ in range(1000):
            env.step(np.zeros(6))
        shadow = copy.deepcopy(env)
        try:
            a, _ = env.reset(seed=901)
            b, _ = shadow.reset(seed=901)
            np.testing.assert_array_equal(a, b)
            for _ in range(1000):
                a, ar, at, ax, _ = env.step(np.full(6, .1))
                b, br, bt, bx, _ = shadow.step(np.full(6, .1))
                np.testing.assert_array_equal(a, b)
                np.testing.assert_array_equal(ar, br)
                self.assertEqual((at, ax), (bt, bx))
            self.assertEqual(env.mode_history, shadow.mode_history)
            self.assertGreater(len(env.mode_history), 2)
        finally:
            env.close()
            shadow.close()

    def test_live_probe_does_not_change_training_trajectory(self):
        snapshots = []
        factory = training.FixedExtremeHierarchicalOperatorSelector
        with tempfile.TemporaryDirectory() as directory:
            real_probe = probe.TrainingForkProbe(
                26, Path(directory) / "probe.json", steps=(32, 64),
                updates=2, repeats=1, horizons=(2, 4))
            for enabled in (False, True):
                def callback(step, trainer, encoder, env, operator):
                    if enabled:
                        real_probe(step, trainer, encoder, env, operator)
                    if step != 64:
                        return False
                    snapshots.append({
                        name: copy.deepcopy(getattr(trainer, name).state_dict())
                        for name in ("policy", "q1", "q2", "q1_target", "q2_target",
                                     "policy_opt", "q_opt")})
                    snapshots[-1].update({"encoder": copy.deepcopy(encoder.state_dict()),
                                          "history": copy.deepcopy(env.mode_history),
                                          "rng": torch.get_rng_state(),
                                          "numpy_next": np.random.random(4)})
                    return True
                with patch.object(training, "FixedExtremeHierarchicalOperatorSelector",
                                  side_effect=lambda d: factory(d, block_steps=4)):
                    training.train_method(
                        "mo-halfcheetah-v5", probe.METHOD, 64, 20, 26, directory,
                        hidden=8, batch_size=4, learning_starts=8, device="cpu",
                        eval_freq=100000, training_probe=callback)
            self.assertTrue(real_probe.report["complete"])
            self.assertEqual(len(real_probe.report["cases"]), 2)
            self.assertTrue(any(c["config"]["envelope_policy_candidates"] > 1
                                for c in real_probe.report["cases"]))
            for key in ("policy", "q1", "q2", "q1_target", "q2_target", "encoder"):
                for name, value in snapshots[0][key].items():
                    torch.testing.assert_close(value, snapshots[1][key][name], rtol=0, atol=0)
            for name in ("policy_opt", "q_opt"):
                for param, state in snapshots[0][name]["state"].items():
                    for key, value in state.items():
                        torch.testing.assert_close(value, snapshots[1][name]["state"][param][key],
                                                   rtol=0, atol=0)
            self.assertEqual(snapshots[0]["history"], snapshots[1]["history"])
            self.assertTrue(torch.equal(snapshots[0]["rng"], snapshots[1]["rng"]))
            np.testing.assert_array_equal(snapshots[0]["numpy_next"], snapshots[1]["numpy_next"])


if __name__ == "__main__":
    torch.set_num_threads(1)
    unittest.main()
