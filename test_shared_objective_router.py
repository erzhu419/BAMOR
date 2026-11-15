import copy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import torch

import run_shared_objective_router as probe


class SharedObjectiveRouterTests(unittest.TestCase):
    def test_real_shared_stream_cold_learners_soft_identity_and_both_priced_controls(self):
        kwargs = dict(hidden=8, lr=3e-4, gamma=.99, batch_size=4, learning_starts=5,
                      eval_freq=1000000, log_freq=10000, device="cpu")
        factory = probe.training.make_piecewise_mo_env
        train, calibrate, evaluate = probe.train, probe.router.calibrate, probe.router.evaluate
        costs = {}
        identities = dict(training_interactions=0, evaluation_interactions=0, initialization_probes=0,
                          actor_steps=0, critic_steps=0, encoder_steps=0)
        checked_soft = checked_baseline = 0

        def short_env(*args, **options):
            options.update(switch_interval=12, mode_order="sequential")
            env = factory(*args, **options)
            env.env._max_episode_steps = 4
            return env

        def same(state, expected):
            probe.base.same_learner(state["trainer"], expected["trainer"])
            for k, v in state["encoder"].state_dict().items():
                self.assertTrue(torch.equal(v, expected["encoder"].state_dict()[k]))
            for k in probe.SHARED_FIELDS:
                np.testing.assert_array_equal(getattr(state["trainer"].buffer, k)[:expected["step"]],
                                              getattr(expected["trainer"].buffer, k)[:expected["step"]])
            self.assertEqual(state["env"].mode_history, expected["env"].mode_history)

        def native(state, directory, steps):
            expected = train(state["run_config"]["env_name"], 9133, directory / "native.json", steps=steps)
            identities["training_interactions"] += steps
            identities["initialization_probes"] += 2
            for key in ("actor_steps", "critic_steps", "encoder_steps"):
                identities[key] += steps - kwargs["learning_starts"] + 1
            try:
                same(state, expected)
            finally:
                expected["env"].close()
                expected["eval_env"].close()

        def checked_train(*args, **options):
            nonlocal checked_baseline
            state = train(*args, **options)
            if options.get("resume") is not None:
                self.assertEqual(probe.router.clocks(state)["actor"], options["steps"] - kwargs["learning_starts"] + 1)
                if checked_baseline == 0:
                    native(state, Path(args[2]).parent, options["steps"])
                checked_baseline += 1
            return state

        def checked_calibrate(state, policies, seed):
            nonlocal checked_soft
            self.assertIsNot(policies["soft"]["encoder"], policies["raw"]["encoder"])
            self.assertIsNot(policies["soft"]["policy"], policies["raw"]["policy"])
            self.assertEqual(probe.router.clocks(state), dict(actor=186, critic=186, encoder=186))
            if checked_soft == 0:
                native(state, Path(directory), 190)
            checked_soft += 1
            return calibrate(state, policies, seed)

        def checked_evaluate(state, policies, seed, route, gate=None):
            saved_gate = copy.deepcopy(gate.state_dict()) if gate else None
            before = {name: {key: copy.deepcopy(model[key].state_dict()) for key in ("policy", "encoder")}
                      for name, model in policies.items()}
            value = evaluate(state, policies, seed, route, gate)
            for name, model in policies.items():
                for key in before[name]:
                    for k, v in model[key].state_dict().items():
                        self.assertTrue(torch.equal(v, before[name][key][k]))
            if gate:
                self.assertEqual(gate.state_dict(), saved_gate)
            return value

        after_step = probe.ShadowLearner.after_step

        def checked_shadow(shadow, step, trainer, encoder, env, operator):
            self.assertIsNot(encoder, shadow.state["encoder"])
            self.assertIsNot(trainer.q1, shadow.state["trainer"].q1)
            self.assertIsNot(trainer.policy_opt, shadow.state["trainer"].policy_opt)
            np_before, torch_before = np.random.get_state(), torch.get_rng_state().clone()
            after_step(shadow, step, trainer, encoder, env, operator)
            self.assertTrue(torch.equal(torch_before, torch.get_rng_state()))
            self.assertEqual(np_before[0], np.random.get_state()[0])
            np.testing.assert_array_equal(np_before[1], np.random.get_state()[1])
            self.assertEqual(np_before[2:], np.random.get_state()[2:])
            self.assertEqual(shadow.rows, step)

        with tempfile.TemporaryDirectory() as directory, \
             patch.multiple(probe, STEPS=190, SWITCH_INTERVAL=12, KWARGS=kwargs, SEEDS=(9133,)), \
             patch.multiple(probe.router, PREFIX=2, HORIZON=4, REPEATS=1), \
             patch.object(probe.training, "make_piecewise_mo_env", side_effect=short_env), \
             patch.object(probe.training, "policy_archive_interval_for_method", return_value=10), \
             patch.object(probe, "train", side_effect=checked_train), \
             patch.object(probe.router, "calibrate", side_effect=checked_calibrate), \
             patch.object(probe.router, "evaluate", side_effect=checked_evaluate), \
             patch.object(probe.ShadowLearner, "after_step", checked_shadow), patch("builtins.print"):
            root = Path(directory)
            probe.write(root / "protocol.json", dict(protocol=probe.PROTOCOL, experiment=probe.specification()))
            records = []
            for env in probe.ENVIRONMENTS:
                record = probe.run(env, 9133, root)
                records.append(record)
                self.assertEqual(record["shadow_rows"], 190)
                self.assertEqual(record["baseline_total_steps"], 190 + record["calibration"]["interactions"] - 2)
                self.assertEqual(record["top_up_updates"], 188 - record["calibration"]["interactions"])
                self.assertEqual(record["candidate_price"]["actor_steps"], record["work_baseline_price"]["actor_steps"])
                self.assertGreater(record["work_baseline_price"]["actor_steps"], record["baseline_price"]["actor_steps"])
                self.assertEqual(record["calibration"]["episodes"], 42)
                self.assertEqual(record["calibration"]["prefix_terminations"], 0)
                self.assertEqual(record["gate"]["models"]["soft"]["samples"], 21)
                self.assertEqual(set(record["evaluations"]), set(probe.ROUTES))
                saved = torch.load(root / env.replace("-", "_") / "seed9133.actors.pt", weights_only=True)
                self.assertEqual(set(saved), {"soft", "raw", "budget_soft", "work_soft"})
                for key, value in record["actual_work"].items():
                    costs[key] = costs.get(key, 0) + value
                with patch.object(probe.training, "train_method", side_effect=AssertionError("retrained")):
                    self.assertEqual(probe.run(env, 9133, root), record)
            summary = probe.summarize(root)
            self.assertEqual(summary["actual_work"], costs)
            self.assertEqual(checked_soft, 2)
            self.assertEqual(checked_baseline, 2)
            for key in ("top_up_updates", "baseline_total_steps"):
                bad = copy.deepcopy(records[0])
                bad[key] -= 1
                with self.assertRaises(AssertionError):
                    probe.check(bad)
            self.assertTrue(all(set(v["risk_minus_baselines"]) == {"budget_soft", "work_soft"}
                                for v in summary["environments"].values()))
        print("SHARED_ROUTER_SMOKE", json.dumps(costs))
        print("SHARED_NATIVE_IDENTITIES", json.dumps(identities))


if __name__ == "__main__":
    unittest.main()
