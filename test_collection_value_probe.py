import copy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import torch

import run_collection_value_probe as probe


def toy_trainer():
    torch.manual_seed(7)
    trainer = probe.training.ProjectedCommonTargetMOSACTrainer(3, 2, 2,
        hidden=8, buffer_capacity=16, batch_size=4, device="cpu")
    for i in range(12):
        state, weight = np.full(3, i / 12, dtype=np.float32), np.full(2, .5, dtype=np.float32)
        trainer.buffer.push(state, np.zeros(2), weight, weight, np.ones(2), state + .1, weight, False, t=i + 1)
    with probe.isolated_rng(5):
        trainer.update()
    return trainer


class CollectionValueTests(unittest.TestCase):
    def test_compact_replay_keeps_old_rows_and_paired_rng_draws(self):
        trainer = toy_trainer()
        new = probe.empty_buffer(trainer.buffer, 4)
        for i in range(4):
            weight = np.array([.3, .7], dtype=np.float32)
            new.push(np.full(3, 2.), np.zeros(2), weight, weight, np.ones(2), np.full(3, 3.), weight, False, t=13 + i)
        small, large = probe.combined_buffer(trainer.buffer), probe.combined_buffer(trainer.buffer, new)
        self.assertEqual((small.size, large.size), (12, 16))
        for field in probe.STORAGE_FIELDS:
            np.testing.assert_array_equal(getattr(large, field)[:12], getattr(trainer.buffer, field)[:12])
            np.testing.assert_array_equal(getattr(large, field)[12:], getattr(new, field))
        tails = []
        for pool in (small, large):
            with probe.isolated_rng(61):
                batch = pool.sample(64)
                tails.append((np.random.random(), torch.rand(1)))
                self.assertEqual(batch[0].shape, (64, 3))
        self.assertEqual(tails[0][0], tails[1][0])
        self.assertTrue(torch.equal(tails[0][1], tails[1][1]))
        large.states[0] += 10
        self.assertFalse(np.array_equal(large.states[0], trainer.buffer.states[0]))

    def test_learning_identity_optimizer_moments_and_base_immutability(self):
        base = toy_trainer()
        original = copy.deepcopy(base.policy.state_dict())
        with patch.object(probe, "UPDATES", 3):
            first, work = probe.learn(base, None, 813)
            second, repeated = probe.learn(base, None, 813)
        probe.same_learner(first, second)
        self.assertEqual(work, repeated)
        self.assertEqual(work["actor_optimizer_steps"], 3)
        self.assertEqual(probe.training.optimizer_step_count(first.policy_opt), 4)
        self.assertEqual(probe.training.optimizer_step_count(base.policy_opt), 1)
        self.assertTrue(all(torch.equal(value, base.policy.state_dict()[key]) for key, value in original.items()))
        with self.assertRaises(AssertionError):
            with torch.no_grad():
                next(second.policy.parameters()).add_(.1)
            probe.same_learner(first, second)

    def test_real_two_environment_pipeline_identity_mask_budget_reuse_and_summary(self):
        factory = probe.training.make_piecewise_mo_env
        kwargs = dict(hidden=8, lr=3e-4, gamma=.99, batch_size=4, learning_starts=5,
                      eval_freq=100000, log_freq=10000, device="cpu")

        def short_env(*args, **kw):
            kw.update(switch_interval=12, mode_order="sequential")
            env = factory(*args, **kw)
            env.env._max_episode_steps = 4
            return env

        captured = []
        original_collect = probe.collect

        def capture_collect(state, sampler, seed):
            data, profile = original_collect(state, sampler, seed)
            captured.append((data, profile))
            self.assertTrue((data.regimes == -1).all())
            self.assertEqual(int(data.dones.sum()), profile["terminated"])
            self.assertGreater(profile["truncated"], 0)
            self.assertGreater(len(profile["mode_history"]), len(state["env"].mode_history))
            return data, profile

        with tempfile.TemporaryDirectory() as directory, \
             patch.multiple(probe, CUT=22, COLLECTION_STEPS=8, UPDATES=3, REPEATS=1,
                            HORIZON=4, KWARGS=kwargs, SEEDS=(91235,)), \
             patch.object(probe.training, "make_piecewise_mo_env", side_effect=short_env), \
             patch.object(probe, "collect", side_effect=capture_collect), patch("builtins.print"):
            root = Path(directory)
            probe.write_report(root / "protocol.json", dict(protocol=probe.PROTOCOL, experiment=probe.specification()))
            reports = []
            for env in probe.ENVIRONMENTS:
                result = probe.run(env, 91235, root)
                self.assertTrue(result["controls_passed"] and result["representation_frozen"])
                self.assertEqual(result["completed_budget"]["training_interactions"], 54)
                self.assertEqual(result["completed_budget"]["actor_optimizer_steps"], 33)
                self.assertEqual(result["completed_budget"]["encoder_optimizer_steps"], 18)
                self.assertEqual(result["completed_budget"]["environment_initialization_probe_steps"], 2)
                self.assertLessEqual(result["completed_budget"]["evaluation_interactions"], 1680)
                self.assertEqual(result["branches"]["old_only"]["replay_size"], 22)
                self.assertEqual(result["branches"]["mixed"]["replay_size"], 30)
                with patch.object(probe.training, "train_method", side_effect=AssertionError("retrained")):
                    self.assertEqual(probe.run(env, 91235, root), result)
                reports.append(result)
            summary = probe.summarize(root)
            self.assertEqual(summary["completed_budget"]["training_interactions"], 108)
            self.assertEqual(summary["completed_budget"]["actor_optimizer_steps"], 66)
            self.assertEqual(len(summary["environments"][probe.ENVIRONMENTS[0]]["paired"]), 7)
            json.dumps(summary, allow_nan=False)
            for env, report in zip(probe.ENVIRONMENTS, reports):
                edited = copy.deepcopy(report)
                for name, utility in dict(base=5., old_only=10., uniform=25., stratified=15., mixed=20.).items():
                    edited["branches"][name]["evaluation"].update(
                        direct_utility=dict(mean=utility, per_mode=[utility] * 4), final_hv=10.)
                probe.write_report(root / env.replace("-", "_") / "seed91235.json", edited)
            gates = probe.summarize(root)
            self.assertTrue(gates["learning_value_gate_passed"] and gates["hopper_hv_guard_passed"])
            self.assertFalse(gates["mixture_advantage_gate_passed"])
            self.assertEqual(gates["decision"], "collection_learning_value_lead_online_selection_unproven")
            bad = copy.deepcopy(reports[0])
            bad["branches"]["mixed"]["learner_work"]["actor_optimizer_steps"] += 1
            probe.write_report(root / probe.ENVIRONMENTS[0].replace("-", "_") / "seed91235.json", bad)
            with self.assertRaises(AssertionError):
                probe.summarize(root)
            costs = {key: sum(r["completed_budget"][key] for r in reports) for key in reports[0]["completed_budget"]}
        print("COLLECTION_VALUE_SMOKE", json.dumps(costs), "standalone_actor_critic_steps=8")


if __name__ == "__main__":
    unittest.main()
