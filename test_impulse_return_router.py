import copy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import gymnasium as gym
import numpy as np
import torch

import run_impulse_return_router as impulse
from test_update_forks import toy_trainer


pilot = impulse.pilot


class ImpulseRouterTests(unittest.TestCase):
    def test_off_commit_preserves_learning_but_not_gate_or_rng(self):
        for condition in ("route", "forced_off"):
            with self.subTest(condition=condition), tempfile.TemporaryDirectory() as directory:
                trainer = toy_trainer()
                replay = trainer.buffer
                router = impulse.ImpulseReturnRouter(
                    condition, 26, Path(directory) / "trace.json", decision_steps=(32,),
                    updates=2, repeats=1, horizons=(2, 4))
                router.configure(32, trainer)
                self.assertEqual(trainer.envelope_policy_candidates, 3)
                branches = []

                def rollout(branch, *args):
                    branches.append(branch)
                    return (np.ones((3, 1, 2, 2)) * len(branches)).tolist()

                rng, torch_rng = np.random.get_state(), torch.get_rng_state()
                with patch.object(pilot.forks, "rollout_returns", side_effect=rollout):
                    router.after_step(32, trainer, None, None, "test")
                self.assertEqual(router.cases[0]["chosen"], "off")
                self.assertTrue(router.cases[0]["previous_on"])
                self.assertIs(trainer.policy, branches[1].policy)
                self.assertIs(trainer.q_opt, branches[1].q_opt)
                self.assertIs(trainer.buffer, replay)
                self.assertEqual(trainer.envelope_policy_candidates, 1)
                self.assertTrue(torch.equal(torch_rng, torch.get_rng_state()))
                np.testing.assert_array_equal(rng[1], np.random.get_state()[1])
                self.assertEqual(rng[2:], np.random.get_state()[2:])
                trainer.envelope_policy_candidates = 5
                router.configure(33, trainer)
                self.assertEqual(trainer.envelope_policy_candidates, 5)
                self.assertIs(trainer.policy, branches[1].policy)
                state = json.loads((Path(directory) / "trace.json").read_text())
                self.assertEqual(state["selection_scope"], "fork_updates_only")
                self.assertNotIn("enabled", state)

    def test_identity_cannot_carry_an_old_off_choice(self):
        with tempfile.TemporaryDirectory() as directory:
            trainer = toy_trainer(candidates=1)
            router = impulse.ImpulseReturnRouter(
                "route", 26, Path(directory) / "trace.json", decision_steps=(32,),
                updates=1, repeats=1, horizons=(2, 4))
            router.enabled = False
            router.configure(32, trainer)
            with patch.object(pilot.forks, "rollout_returns", return_value=np.zeros((3, 1, 2, 2)).tolist()):
                router.after_step(32, trainer, None, None, "test")
            self.assertEqual(router.cases[0]["chosen"], "on")
            self.assertEqual(router.extra_steps, 24)

    def test_real_runs_share_prefix_restore_defaults_and_archive_after_commit(self):
        training = pilot.training
        factory = training.FixedExtremeHierarchicalOperatorSelector
        env_factory = training.make_piecewise_mo_env
        prefixes = []
        fixed_on_results = []
        with tempfile.TemporaryDirectory() as directory:
            arms = [(c, impulse.ImpulseReturnRouter) for c in impulse.CONDITIONS]
            arms.append(("forced_on", pilot.BudgetedReturnRouter))
            for condition, router_type in arms:
                with self.subTest(condition=condition):
                    output = Path(directory) / router_type.__name__ / condition
                    router = router_type(
                        condition, 26, output / "trace.json", decision_steps=(32, 64),
                        updates=2, repeats=1, horizons=(2, 4))
                    snapshots = {}

                    def short_env(*args, **kwargs):
                        env = env_factory(*args, **kwargs)
                        env.env = gym.wrappers.TimeLimit(env.env, max_episode_steps=8)
                        return env

                    def capture(step, trainer, *args):
                        if step not in router.decision_steps:
                            self.assertEqual(trainer.envelope_policy_candidates, router.on_candidates)
                        if step == 31:
                            prefixes.append(copy.deepcopy(trainer.policy.state_dict()))
                        if step in (16, 32, 64, 80):
                            snapshots[step] = copy.deepcopy(trainer.policy.state_dict())
                        return False

                    with (patch.object(training, "FixedExtremeHierarchicalOperatorSelector",
                                       side_effect=lambda d: factory(d, block_steps=4)),
                          patch.object(training, "make_piecewise_mo_env", side_effect=short_env),
                          patch.object(training, "full_simplex_grid", return_value=pilot.forks.WEIGHTS)):
                        result = training.train_method(
                            "mo-halfcheetah-v5", pilot.forks.METHOD, 80, 20, 26, str(output),
                            hidden=8, batch_size=4, learning_starts=8, device="cpu", eval_freq=100000,
                            update_router=router, training_probe=capture, archive_steps=(16, 32, 64, 80))
                    self.assertEqual([s["interaction_step"] for s in result["policy_archive_snapshots"]],
                                     [16, 56, 112, 128])
                    self.assertEqual(router.extra_steps, 48)
                    self.assertTrue(all(c["previous_on"] for c in router.cases))
                    if condition.startswith("forced_"):
                        self.assertEqual([c["chosen"] for c in router.cases], [condition[7:]] * 2)
                    model = torch.load(output / f"{pilot.forks.METHOD}_seed26_model.pt", weights_only=False)
                    for snap in model["policy_archive"]:
                        for key, value in snap["policy"].items():
                            torch.testing.assert_close(value, snapshots[snap["step"]][key], rtol=0, atol=0)
                    if router_type is impulse.ImpulseReturnRouter:
                        self.assertEqual(result["update_routing"]["protocol"], impulse.PROTOCOL)
                    if condition == "forced_on":
                        fixed_on_results.append(result)
                    json.dumps(result, allow_nan=False)
        for prefix in prefixes[1:]:
            for key, value in prefix.items():
                torch.testing.assert_close(value, prefixes[0][key], rtol=0, atol=0)
        for field in ("episode_rewards", "ground_truth", "pareto_per_mode"):
            self.assertEqual(fixed_on_results[0][field], fixed_on_results[1][field])

    def test_runner_records_frozen_budget_and_separate_protocol(self):
        def train(*args, update_router, archive_steps, **kwargs):
            self.assertEqual((args[2], args[3], kwargs["hidden"]), (50000, 25000, 64))
            self.assertEqual(tuple(archive_steps), pilot.ARCHIVE_MAIN_STEPS)
            update_router.extra_steps = 4 * pilot.TRIAL_STEPS
            update_router.cases = [{"main_step": s, "previous_on": True} for s in pilot.DECISION_STEPS]
            return {"policy_archive_snapshots": [{"interaction_step": s} for s in pilot.ARCHIVE_BUDGET_STEPS]}

        with tempfile.TemporaryDirectory() as directory:
            (Path(directory) / "route").mkdir()
            with patch.object(pilot.training, "train_method", side_effect=train):
                result = impulse.run("route", 26, directory)
            budget = result["budgeted_routing_experiment"]
            self.assertEqual(budget["protocol"], impulse.PROTOCOL)
            self.assertEqual((budget["main_environment_steps"], budget["trial_environment_steps"],
                              budget["total_training_interactions"]), (50000, 36864, 86864))
            self.assertEqual((budget["normal_updates"], budget["fork_updates"],
                              budget["committed_fork_updates"]), (45001, 1024, 512))


if __name__ == "__main__":
    unittest.main()
