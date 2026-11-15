import copy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import torch

import run_frozen_objective_router as probe


class FrozenObjectiveRouterTests(unittest.TestCase):
    def test_return_only_ridge_admission_tie_and_early_termination_fallback(self):
        def rows(gain):
            return [dict(candidate=name, features=[i / 4, i % 3], target=2 + .1 * i / 4 + (gain if name == "raw" else 0))
                    for i in range(12) for name in ("soft", "raw")]
        gate = probe.ReturnGate().fit(rows(1.))
        self.assertEqual(gate.decide([1., 1.])[0], "raw")
        self.assertGreater(gate.decide([1., 1.])[1]["margin"], 0.)
        self.assertEqual(probe.ReturnGate().fit(rows(-1.)).decide([1., 1.])[0], "soft")
        self.assertEqual(probe.ReturnGate().fit(rows(0.)).decide([1., 1.])[0], "soft")
        self.assertEqual(probe.ReturnGate().fit([]).decide([1., 1.])[0], "soft")
        old = copy.deepcopy(gate.state_dict())
        gate.decide([2., 0.])
        self.assertEqual(gate.state_dict(), old)
        self.assertEqual(set(old["models"]), {"soft", "raw"})

    def test_real_pipeline_paid_calibration_independent_trackers_and_exact_continuation(self):
        kwargs = dict(hidden=8, lr=3e-4, gamma=.99, batch_size=4, learning_starts=5,
                      eval_freq=1000000, log_freq=10000, device="cpu")
        factory = probe.training.make_piecewise_mo_env
        calibrate, evaluate, train = probe.calibrate, probe.evaluate, probe.train
        costs, captured, frozen_soft, gates = {}, {}, {}, []
        pure_evaluation_steps = 0

        def short_env(*args, **options):
            options.update(switch_interval=12, mode_order="sequential")
            env = factory(*args, **options)
            env.env._max_episode_steps = 4
            return env

        def checked_calibrate(state, policies, seed):
            env = state["run_config"]["env_name"]
            self.assertIsNot(policies["soft"]["encoder"], policies["raw"]["encoder"])
            self.assertIsNot(policies["soft"]["policy"], state["trainer"].policy)
            before = {name: copy.deepcopy(getattr(state["trainer"], name).state_dict()) for name in ("policy", "q1", "q2")}
            self.assertEqual(probe.clocks(state), dict(actor=18, critic=18, encoder=18))
            self.assertFalse(any(p.requires_grad for model in policies.values()
                for network in (model["policy"], model["encoder"]) for p in network.parameters()))
            frozen_soft[env] = copy.deepcopy(policies["soft"]["policy"].state_dict())
            gate, data = calibrate(state, policies, seed)
            gates.append(copy.deepcopy(gate.state_dict()))
            for name, values in before.items():
                for key, value in values.items():
                    self.assertTrue(torch.equal(value, getattr(state["trainer"], name).state_dict()[key]))
            self.assertEqual(probe.clocks(state), dict(actor=18, critic=18, encoder=18))
            self.assertEqual(data["episodes"], 42)
            self.assertLessEqual(data["interactions"], 168)
            self.assertTrue(all(len(r["features"]) == 3 * policies["soft"]["reward_dim"] + 2 for r in data["rows"]))
            return gate, data

        def checked_evaluate(state, policies, seed, route, gate=None):
            nonlocal pure_evaluation_steps
            if route == "risk":
                before = copy.deepcopy(gate.state_dict())
                value = evaluate(state, policies, seed, route, gate)
                self.assertEqual(gate.state_dict(), before)
                return value
            value = evaluate(state, policies, seed, route, gate)
            if route == "soft":
                env = probe.base.fork_env(state["eval_env"])
                before = env._total_steps
                try:
                    env._lock_mode = True
                    for mode in range(env.num_modes):
                        env._apply_mode(mode)
                        expected = probe.training.evaluate_pareto_v2(env, state["trainer"], probe.source.METHOD,
                            state["encoder"], np.asarray(value["evaluation_weights"]), num_episodes=1,
                            episode_seeds=value["episode_seeds_per_mode"][mode], device="cpu")
                        self.assertEqual(expected, value["pareto_per_mode"][mode])
                    pure_evaluation_steps += env._total_steps - before
                finally:
                    env.close()
            return value

        def captured_train(*args, **options):
            state = train(*args, **options)
            if options.get("resume") is not None:
                captured[state["run_config"]["env_name"]] = state
            return state

        with tempfile.TemporaryDirectory() as directory, \
             patch.multiple(probe, STEPS=22, SWITCH_INTERVAL=12, PREFIX=2, HORIZON=4, REPEATS=1,
                            KWARGS=kwargs, SEEDS=(91321,)), \
             patch.object(probe.training, "make_piecewise_mo_env", side_effect=short_env), \
             patch.object(probe, "calibrate", side_effect=checked_calibrate), \
             patch.object(probe, "evaluate", side_effect=checked_evaluate), \
             patch.object(probe, "train", side_effect=captured_train), patch("builtins.print"):
            root = Path(directory)
            probe.write(root / "protocol.json", dict(protocol=probe.PROTOCOL, experiment=probe.specification()))
            records = {}
            for env in probe.ENVIRONMENTS:
                record = probe.run(env, 91321, root)
                records[env] = record
                for key, value in record["actual_work"].items():
                    costs[key] = costs.get(key, 0) + value
                self.assertEqual(record["candidate_price"]["training_interactions"] +
                    record["candidate_price"]["calibration_interactions"], record["baseline_price"]["training_interactions"])
                self.assertGreater(record["baseline_price"]["actor_steps"], record["candidate_price"]["actor_steps"])
                self.assertEqual(record["gate"], gates[-1])
                model = torch.load(root / env.replace("-", "_") / "seed91321.actors.pt", map_location="cpu", weights_only=True)
                for key, value in frozen_soft[env].items():
                    self.assertTrue(torch.equal(value, model["soft"]["policy"][key]))
                with patch.object(probe.training, "train_method", side_effect=AssertionError("retrained")):
                    self.assertEqual(probe.run(env, 91321, root), record)
            summary = probe.summarize(root)
            self.assertEqual(summary["actual_work"], costs)
            self.assertTrue(all(len(r["mean_per_mode_utility_difference"]) == 4 for r in summary["environments"].values()))
            bad = copy.deepcopy(records[probe.ENVIRONMENTS[0]])
            bad["baseline_total_steps"] -= 1
            with self.assertRaises(AssertionError):
                probe.check(bad)
            env = "mo-halfcheetah-v5"
            expected = train(env, 91321, root / "native" / "run.json", steps=records[env]["baseline_total_steps"])
            try:
                probe.base.same_learner(captured[env]["trainer"], expected["trainer"])
                for key, value in captured[env]["encoder"].state_dict().items():
                    self.assertTrue(torch.equal(value, expected["encoder"].state_dict()[key]))
                self.assertEqual(captured[env]["env"].mode_history, expected["env"].mode_history)
                native_steps = records[env]["baseline_total_steps"]
                native_updates = native_steps - kwargs["learning_starts"] + 1
            finally:
                expected["env"].close()
                expected["eval_env"].close()
        print("ROUTER_PIPELINE_SMOKE", json.dumps(costs))
        print("ROUTER_NATIVE_IDENTITY", json.dumps(dict(training_interactions=native_steps,
            initialization_probes=2, actor_steps=native_updates, critic_steps=native_updates, encoder_steps=native_updates)))
        print("ROUTER_EVALUATOR_IDENTITY", json.dumps(dict(evaluation_interactions=pure_evaluation_steps)))


if __name__ == "__main__":
    unittest.main()
