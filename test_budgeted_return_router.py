import copy
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import gymnasium as gym
import numpy as np
import torch

import diagnose_update_forks as forks
import run_budgeted_return_router as routing
import train_bamor_mujoco as training
from test_update_forks import toy_trainer


class BudgetedRouterTests(unittest.TestCase):
    def test_mean_utility_not_majority_vote_and_identity_keeps_gate(self):
        self.assertFalse(routing.choose_branch([1., 1., -10.], "route", True))
        self.assertFalse(routing.choose_branch([0., 0., 0.], "route", True))
        self.assertTrue(routing.choose_branch([0., 0., 0.], "route", True, False))
        self.assertFalse(routing.choose_branch([0., 0., 0.], "route", False, False))
        self.assertTrue(routing.choose_branch([-10.] * 3, "forced_on", False))
        self.assertFalse(routing.choose_branch([10.] * 3, "forced_off", True))

    def test_budget_and_nineteen_archive_points(self):
        self.assertEqual(routing.TRIAL_STEPS, 9216)
        self.assertEqual(routing.TOTAL_BUDGET, 86864)
        self.assertEqual(len(routing.ARCHIVE_BUDGET_STEPS), 19)
        self.assertEqual(routing.ARCHIVE_BUDGET_STEPS[:3], (5000, 7500, 19216))
        self.assertEqual(routing.ARCHIVE_BUDGET_STEPS[-1], routing.TOTAL_BUDGET)
        self.assertEqual(sorted(set(routing.ARCHIVE_BUDGET_STEPS)),
                         list(routing.ARCHIVE_BUDGET_STEPS))

    def test_commit_carries_optimizer_bindings_and_keeps_live_replay(self):
        base = toy_trainer()
        replay = base.buffer
        branch = forks.clone_trainer(base, forks.ReplaySubset(replay, np.arange(8, 32)))
        with forks.isolated_rng(123):
            branch.update()
        routing.commit_branch(base, branch)
        self.assertIs(base.buffer, replay)
        self.assertIs(base.policy, branch.policy)
        self.assertIs(base.q_opt, branch.q_opt)
        self.assertIs(base.policy_opt.param_groups[0]["params"][0], next(base.policy.parameters()))
        self.assertIs(base.q_opt.param_groups[0]["params"][0], next(base.q1.parameters()))
        before = copy.deepcopy(base.policy.state_dict())
        with forks.isolated_rng(125):
            base.update()
        self.assertTrue(any(not torch.equal(v, base.policy.state_dict()[k])
                            for k, v in before.items()))

    def test_off_gate_can_propose_on_again_and_fork_rng_is_isolated(self):
        base = toy_trainer()
        expected = []

        def rollout(branch, *args):
            expected.append(copy.deepcopy(branch.policy.state_dict()))
            return (np.ones((3, 1, 2, 2)) * (2 if len(expected) == 1 else 0)).tolist()

        with tempfile.TemporaryDirectory() as directory:
            router = routing.BudgetedReturnRouter("route", 26, Path(directory) / "trace.json",
                                                   decision_steps=(32,), updates=2,
                                                   repeats=1, horizons=(2, 4))
            router.enabled = False
            router.configure(32, base)
            self.assertEqual((router.on_candidates, base.envelope_policy_candidates), (3, 1))
            np_state, torch_state = np.random.get_state(), torch.get_rng_state()
            with patch.object(forks, "rollout_returns", side_effect=rollout):
                router.after_step(32, base, None, None, "test")
            self.assertTrue(router.enabled)
            self.assertEqual(router.extra_steps, 24)
            self.assertEqual(router.cases[0]["chosen"], "on")
            for k, v in expected[0].items():
                torch.testing.assert_close(v, base.policy.state_dict()[k], rtol=0, atol=0)
            self.assertTrue(torch.equal(torch_state, torch.get_rng_state()))
            np.testing.assert_array_equal(np_state[1], np.random.get_state()[1])
            self.assertEqual(np_state[2:], np.random.get_state()[2:])
            base.envelope_policy_candidates = 5
            router.configure(33, base)
            self.assertEqual(base.envelope_policy_candidates, 5)

    def test_real_routing_archive_is_post_commit_and_baseline_budget_is_matched(self):
        factory = training.FixedExtremeHierarchicalOperatorSelector
        env_factory = training.make_piecewise_mo_env

        def short_env(*args, **kwargs):
            env = env_factory(*args, **kwargs)
            env.env = gym.wrappers.TimeLimit(env.env, max_episode_steps=8)
            return env

        with tempfile.TemporaryDirectory() as directory:
            for condition in routing.CONDITIONS:
                with self.subTest(condition=condition):
                    trials = condition in routing.CONDITIONS[:3]
                    output = Path(directory) / condition
                    router = (routing.BudgetedReturnRouter(
                        condition, 26, output / "trace.json", decision_steps=(32, 64),
                        updates=2, repeats=1, horizons=(2, 4)) if trials else None)
                    method = ("bamor_v27_aligned_target_ownpref_actor_fixed_archive_mix"
                              if condition == "train_off" else forks.METHOD)
                    captured = {}

                    def capture(step, trainer, *args):
                        if step in ((16, 32, 64) if trials else (16, 56, 112)):
                            captured[step] = copy.deepcopy(trainer.policy.state_dict())
                        return False

                    with (patch.object(training, "FixedExtremeHierarchicalOperatorSelector",
                                       side_effect=lambda d: factory(d, block_steps=4)),
                          patch.object(training, "make_piecewise_mo_env", side_effect=short_env),
                          patch.object(training, "full_simplex_grid", return_value=forks.WEIGHTS)):
                        result = training.train_method(
                            "mo-halfcheetah-v5", method, 64 if trials else 112, 20, 26, str(output),
                            hidden=8, batch_size=4, learning_starts=8, device="cpu",
                            eval_freq=100000, update_router=router, training_probe=capture,
                            archive_steps=(16, 32, 64) if trials else (16, 56, 112))
                    self.assertEqual([s["interaction_step"] for s in result["policy_archive_snapshots"]],
                                     [16, 56, 112])
                    if trials:
                        self.assertEqual(router.extra_steps, 48)
                        self.assertEqual(len(router.cases), 2)
                        if condition.startswith("forced_"):
                            self.assertTrue(all(c["chosen"] == condition[7:] for c in router.cases))
                    model = torch.load(output / f"{method}_seed26_model.pt", weights_only=False)
                    for snapshot in model["policy_archive"]:
                        for k, v in snapshot["policy"].items():
                            torch.testing.assert_close(v, captured[snapshot["step"]][k], rtol=0, atol=0)
                    self.assertEqual(result["archive_context"], "final_encoder")
                    json.dumps(result, allow_nan=False)


if __name__ == "__main__":
    torch.set_num_threads(1)
    unittest.main()
