import copy
from contextlib import ExitStack
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import torch

import run_history_return_gate as probe


class HistoryReturnGateTests(unittest.TestCase):
    def test_observable_diversity_latest_ties_and_identical_features(self):
        features = [[0., 0.], [1., 0.], [0., 1.], [1., 1.], [1., 1.]]
        self.assertEqual(probe.select_slots(features, 3), [4, 0, 2])
        self.assertEqual(probe.select_slots(np.zeros((5, 3)), 4), [4, 3, 2, 1])

    def test_gate_uses_actual_returns_whole_policy_and_keeps_prefix_on_ties(self):
        def row(new):
            return dict(weight=[.9, .1], branches=dict(prefix=dict(return_vector=[1., -2.]),
                                                      proposal=dict(return_vector=new)))
        value = probe.gate([row([2., -5.])])
        self.assertEqual(value["chosen"], "proposal")
        self.assertAlmostEqual(value["score"], .6)
        self.assertEqual(probe.gate([row([1., -2.])])["chosen"], "prefix")
        self.assertEqual(probe.gate([row([0., -2.])])["chosen"], "prefix")

    def test_real_mid_episode_states_native_identity_and_full_price_pipeline(self):
        training, factory, native_train, native_collect = probe.training, probe.training.make_piecewise_mo_env, probe.train, probe.collect
        kwargs = dict(hidden=8, lr=3e-4, gamma=.99, batch_size=4, learning_starts=5,
                      eval_freq=1000000, log_freq=1000000, device="cpu")
        costs, references = {}, dict(training_interactions=0, initialization_probes=0,
                                    actor_steps=0, critic_steps=0, encoder_steps=0)
        checked, mid_episode_seen = [], []

        def short_env(*args, **options):
            options.update(switch_interval=12, mode_order="sequential")
            env = factory(*args, **options)
            env.env._max_episode_steps = 7
            return env

        def same(state, expected):
            probe.base.same_learner(state["trainer"], expected["trainer"])
            for name in state["encoder"].state_dict():
                torch.testing.assert_close(state["encoder"].state_dict()[name], expected["encoder"].state_dict()[name], rtol=0, atol=0)
            left, right = state["encoder_opt"].state_dict(), expected["encoder_opt"].state_dict()
            self.assertEqual(left["param_groups"], right["param_groups"])
            for parameter in left["state"]:
                for key in left["state"][parameter]:
                    torch.testing.assert_close(left["state"][parameter][key], right["state"][parameter][key], rtol=0, atol=0)
            for field in (*probe.shared.SHARED_FIELDS, "contexts", "next_contexts"):
                np.testing.assert_array_equal(getattr(state["trainer"].buffer, field)[:state["step"]],
                                              getattr(expected["trainer"].buffer, field)[:expected["step"]])
            self.assertEqual(state["env"].mode_history, expected["env"].mode_history)
            np.testing.assert_array_equal(state["obs"], expected["obs"])

        def checked_train(env, seed, output, steps, resume=None, capture=None):
            state = native_train(env, seed, output, steps, resume=resume, capture=capture)
            expected = native_train(env, seed, output, steps)
            references["training_interactions"] += steps
            references["initialization_probes"] += 2
            for key in ("actor_steps", "critic_steps", "encoder_steps"):
                references[key] += steps - kwargs["learning_starts"] + 1
            try:
                same(state, expected)
                checked.append((env, steps))
            finally:
                expected["env"].close()
                expected["eval_env"].close()
            return state

        def checked_collect(capture, indices, wi, seed):
            rng, trng = np.random.get_state(), torch.get_rng_state().clone()
            frozen = {m: {k: copy.deepcopy(capture.models[m][k].state_dict()) for k in ("policy", "encoder")}
                      for m in ("prefix", "proposal")}
            physical = [(s["env"]._total_steps, s["env"].unwrapped.data.qpos.copy(),
                         s["env"].unwrapped.data.qvel.copy(), s["env"].unwrapped.data.time) for s in capture.pool]
            with ExitStack() as stack:
                for slot in capture.pool:
                    stack.enter_context(patch.object(slot["env"], "reset", side_effect=AssertionError("lost physical start")))
                result = native_collect(capture, indices, wi, seed)
            for row, index in zip(result["rows"], indices):
                slot = capture.pool[index]
                self.assertEqual(row["capture_step"], slot["step"])
                self.assertEqual(row["episode_age"], slot["episode_age"])
                mid_episode_seen.append(row["episode_age"] > 0)
                for outcome in row["branches"].values():
                    self.assertEqual(outcome["reencoded_rows"], min(probe.HISTORY_WINDOW, slot["step"]))
                    self.assertLessEqual(outcome["steps"], min(probe.HORIZON, 7 - slot["episode_age"]))
            for slot, old in zip(capture.pool, physical):
                self.assertEqual(slot["env"]._total_steps, old[0])
                np.testing.assert_array_equal(slot["env"].unwrapped.data.qpos, old[1])
                np.testing.assert_array_equal(slot["env"].unwrapped.data.qvel, old[2])
                self.assertEqual(slot["env"].unwrapped.data.time, old[3])
            now = np.random.get_state()
            np.testing.assert_array_equal(rng[1], now[1])
            self.assertEqual(rng[2:], now[2:])
            self.assertTrue(torch.equal(trng, torch.get_rng_state()))
            for model in frozen:
                for component in frozen[model]:
                    for key, value in frozen[model][component].items():
                        torch.testing.assert_close(value, capture.models[model][component].state_dict()[key], rtol=0, atol=0)
            return result

        with tempfile.TemporaryDirectory() as directory, \
             patch.multiple(probe, CUT=24, PROPOSAL=32, CAPTURE_EVERY=4, HISTORY_WINDOW=4,
                            HORIZON=4, SWITCH_INTERVAL=12, KWARGS=kwargs, SEEDS=(9133,)), \
             patch.multiple(probe.router, HORIZON=4, REPEATS=1), \
             patch.object(training, "make_piecewise_mo_env", side_effect=short_env), \
             patch.object(probe, "train", side_effect=checked_train), \
             patch.object(probe, "collect", side_effect=checked_collect), patch("builtins.print"):
            root = Path(directory)
            probe.write(root / "protocol.json", dict(protocol=probe.PROTOCOL, experiment=probe.specification()))
            for env in probe.ENVIRONMENTS:
                record = probe.run(env, 9133, root)
                self.assertTrue(record["complete"])
                self.assertEqual(len(record["coverage"]["pool_steps"]), 8)
                self.assertEqual(len(set(record["coverage"]["selected_indices"])), 4)
                for arm in ("history", "current"):
                    prices = record["path_prices"]
                    total = lambda p: sum(p[k] for k in ("training_interactions", "calibration_interactions", "initialization_probes"))
                    self.assertEqual(total(prices[arm]), total(prices["priced_" + arm]))
                for key, value in record["actual_work"].items():
                    costs[key] = costs.get(key, 0) + value
                with patch.object(training, "train_method", side_effect=AssertionError("reran complete task")):
                    self.assertEqual(probe.run(env, 9133, root), record)
                bad = copy.deepcopy(record)
                bad["clocks"]["priced_history"]["actor"] += 1
                with self.assertRaises(AssertionError):
                    probe.check(bad)
            summary = probe.summarize(root)
            self.assertEqual(summary["actual_work"], costs)
        self.assertEqual(len(checked), 6)
        self.assertTrue(any(mid_episode_seen))
        print("HISTORY_GATE_SMOKE_WORK", json.dumps(costs))
        print("HISTORY_GATE_REFERENCE_WORK", json.dumps(references))


if __name__ == "__main__":
    unittest.main()
