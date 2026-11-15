import copy
from contextlib import ExitStack
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import torch

import run_crossed_scene_return as probe


class CrossedSceneReturnTests(unittest.TestCase):
    def test_scene_mean_allows_preference_tradeoff_but_not_scene_compensation(self):
        def rows(values):
            return [dict(scene=s, preference=p, repeat=r, weight=[1.],
                         branches=dict(prefix=dict(return_vector=[0.]), proposal=dict(return_vector=[v])))
                    for s, scene in enumerate(values) for p, v in enumerate(scene) for r in range(2)]
        with patch.object(probe.history, "SLOTS", 2), patch.object(probe, "PREFERENCES", 2):
            value = probe.decisions(rows([[-1., 3.], [1., 1.]]))
            self.assertEqual((value["scene"], value["mean"]), ("proposal", "proposal"))
            value = probe.decisions(rows([[-2., 0.], [4., 4.]]))
            self.assertEqual((value["scene"], value["mean"]), ("prefix", "proposal"))
            self.assertEqual(probe.decisions(rows([[0., 0.], [0., 0.]]))["scene"], "prefix")
            with self.assertRaises(AssertionError):
                probe.decisions(rows([[1., 1.], [1., 1.]])[:-1])
            values = rows([[1., 1.], [1., 1.]])
            values[-1] = values[0]
            with self.assertRaises(AssertionError):
                probe.decisions(values)

    def test_real_crossed_resets_native_identity_and_budget(self):
        history, training = probe.history, probe.training
        factory, native_train, native_collect = training.make_piecewise_mo_env, history.train, probe.collect
        kwargs = dict(hidden=8, lr=3e-4, gamma=.99, batch_size=4, learning_starts=5,
                      eval_freq=1000000, log_freq=1000000, device="cpu")
        checks, costs = [], {}
        references = dict(training_interactions=0, initialization_probes=0, actor_steps=0, critic_steps=0, encoder_steps=0)

        def short_env(*args, **options):
            options.update(switch_interval=12, mode_order="sequential")
            env = factory(*args, **options)
            env.env._max_episode_steps = 7
            return env

        def checked_train(env, seed, output, steps, resume=None, capture=None):
            state = native_train(env, seed, output, steps, resume=resume, capture=capture)
            expected = native_train(env, seed, output, steps)
            references["training_interactions"] += steps
            references["initialization_probes"] += 2
            for k in ("actor_steps", "critic_steps", "encoder_steps"):
                references[k] += steps - kwargs["learning_starts"] + 1
            try:
                probe.base.same_learner(state["trainer"], expected["trainer"])
                for k, v in state["encoder"].state_dict().items():
                    torch.testing.assert_close(v, expected["encoder"].state_dict()[k], rtol=0, atol=0)
                a, b = state["encoder_opt"].state_dict(), expected["encoder_opt"].state_dict()
                self.assertEqual(a["param_groups"], b["param_groups"])
                for p in a["state"]:
                    for k in a["state"][p]:
                        torch.testing.assert_close(a["state"][p][k], b["state"][p][k], rtol=0, atol=0)
                for field in (*probe.shared.SHARED_FIELDS, "contexts", "next_contexts"):
                    np.testing.assert_array_equal(getattr(state["trainer"].buffer, field)[:steps],
                                                  getattr(expected["trainer"].buffer, field)[:steps])
                checks.append((env, steps))
            finally:
                expected["env"].close()
                expected["eval_env"].close()
            return state

        def checked_collect(capture, selected, wi, seed):
            physical = [(s["env"]._total_steps, s["env"]._lock_mode, s["env"].unwrapped.data.qpos.copy()) for s in capture.pool]
            rng, trng = np.random.get_state(), torch.get_rng_state().clone()
            frozen = {name: {k: copy.deepcopy(model[k].state_dict()) for k in ("policy", "encoder")}
                      for name, model in capture.models.items()}
            result = native_collect(capture, selected, wi, seed)
            starts = {}
            for row in result["rows"]:
                key = (row["scene"], row["repeat"])
                start = row["branches"]["prefix"]["start"]
                if key in starts:
                    self.assertEqual(starts[key], start)
                starts[key] = start
                self.assertEqual(row["branches"]["proposal"]["start"], start)
                self.assertTrue(all(b["steps"] <= 4 for b in row["branches"].values()))
            for scene in range(4):
                self.assertNotEqual(starts[(scene, 0)]["observation"], starts[(scene, 1)]["observation"])
            for slot, old in zip(capture.pool, physical):
                self.assertEqual((slot["env"]._total_steps, slot["env"]._lock_mode), old[:2])
                np.testing.assert_array_equal(slot["env"].unwrapped.data.qpos, old[2])
            np.testing.assert_array_equal(rng[1], np.random.get_state()[1])
            self.assertEqual(rng[2:], np.random.get_state()[2:])
            self.assertTrue(torch.equal(trng, torch.get_rng_state()))
            for name, model in capture.models.items():
                for component in ("policy", "encoder"):
                    for key, value in model[component].state_dict().items():
                        torch.testing.assert_close(value, frozen[name][component][key], rtol=0, atol=0)
            return result

        with tempfile.TemporaryDirectory() as directory, ExitStack() as stack:
            stack.enter_context(patch.multiple(history, CUT=24, PROPOSAL=32, CAPTURE_EVERY=4,
                HORIZON=4, SWITCH_INTERVAL=12, KWARGS=kwargs))
            stack.enter_context(patch.object(probe, "SEEDS", (9134,)))
            stack.enter_context(patch.multiple(probe.router, HORIZON=4, REPEATS=1))
            stack.enter_context(patch.object(training, "make_piecewise_mo_env", side_effect=short_env))
            stack.enter_context(patch.object(history, "train", side_effect=checked_train))
            stack.enter_context(patch.object(probe, "collect", side_effect=checked_collect))
            stack.enter_context(patch("builtins.print"))
            root = Path(directory)
            probe.write(root / "protocol.json", dict(protocol=probe.PROTOCOL, experiment=probe.specification()))
            for env in probe.ENVIRONMENTS:
                record = probe.run(env, 9134, root)
                self.assertEqual(len(record["trial"]["rows"]), 32)
                self.assertEqual(record["trial"]["resets"], 64)
                prices = record["path_prices"]
                total = lambda p: sum(p[k] for k in ("training_interactions", "calibration_interactions", "initialization_probes"))
                self.assertEqual(total(prices["candidate"]), total(prices["priced"]))
                for k, v in record["actual_work"].items():
                    costs[k] = costs.get(k, 0) + v
                with patch.object(training, "train_method", side_effect=AssertionError("reran complete task")):
                    self.assertEqual(probe.run(env, 9134, root), record)
                bad = copy.deepcopy(record)
                bad["clocks"]["priced"]["actor"] += 1
                with self.assertRaises(AssertionError):
                    probe.check(bad)
            self.assertEqual(probe.summarize(root)["actual_work"], costs)
        self.assertEqual(len(checks), 4)
        print("CROSSED_SCENE_SMOKE_WORK", json.dumps(costs))
        print("CROSSED_SCENE_REFERENCE_WORK", json.dumps(references))


if __name__ == "__main__":
    unittest.main()
