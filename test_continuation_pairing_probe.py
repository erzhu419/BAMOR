import copy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import torch

import run_continuation_pairing_probe as probe


class ContinuationPairingTests(unittest.TestCase):
    def test_real_saved_endpoints_reproduce_crosses_are_frozen_and_costed(self):
        shared, training = probe.shared, probe.training
        factory = training.make_piecewise_mo_env
        kwargs = dict(hidden=8, lr=3e-4, gamma=.99, batch_size=4, learning_starts=5,
                      eval_freq=1000000, log_freq=10000, device="cpu")
        costs = {}
        source_costs = {}

        def short_env(*args, **options):
            options.update(switch_interval=12, mode_order="sequential")
            env = factory(*args, **options)
            env.env._max_episode_steps = 4
            return env

        with tempfile.TemporaryDirectory() as directory, \
             patch.multiple(shared, STEPS=190, SWITCH_INTERVAL=12, KWARGS=kwargs, SEEDS=(9133,)), \
             patch.multiple(probe.router, PREFIX=2, HORIZON=4, REPEATS=1), \
             patch.object(probe, "SEEDS", (9133,)), \
             patch.object(training, "make_piecewise_mo_env", side_effect=short_env), \
             patch("builtins.print"):
            source, root = Path(directory) / "source", Path(directory) / "pairing"
            source.mkdir()
            root.mkdir()
            with patch.object(probe, "SOURCE", source):
                probe.write(source / "protocol.json", dict(protocol=shared.PROTOCOL, experiment=shared.specification()))
                probe.write(root / "protocol.json", dict(protocol=probe.PROTOCOL, experiment=probe.specification()))
                for env_name in probe.ENVIRONMENTS:
                    saved = shared.run(env_name, 9133, source)
                    for k, value in saved["actual_work"].items():
                        source_costs[k] = source_costs.get(k, 0) + value
                    path = source / env_name.replace("-", "_") / "seed9133.actors.pt"
                    tensors = torch.load(path, map_location="cpu", weights_only=True)
                    env = short_env(env_name, switch_interval=12, seed=10133)
                    costs["initialization_probes"] = costs.get("initialization_probes", 0) + 1
                    try:
                        pairs = probe.load_pairs(env, saved, path)
                        self.assertIs(pairs["new_old"]["policy"], pairs["new_new"]["policy"])
                        self.assertIs(pairs["new_old"]["encoder"], pairs["old_old"]["encoder"])
                        self.assertIsNot(pairs["new_old"]["encoder"], pairs["new_new"]["encoder"])
                        self.assertTrue(any(not torch.equal(v, tensors["soft"]["policy"][k])
                                            for k, v in tensors["work_soft"]["policy"].items()))
                        for name, (actor, encoder) in probe.PAIRS.items():
                            for key, origin in (("policy", actor), ("encoder", encoder)):
                                self.assertFalse(pairs[name][key].training)
                                self.assertTrue(all(not p.requires_grad for p in pairs[name][key].parameters()))
                                for k, value in pairs[name][key].state_dict().items():
                                    self.assertTrue(torch.equal(value, tensors[origin][key][k]))
                    finally:
                        env.close()
                    record = probe.run(env_name, 9133, root)
                    self.assertEqual(record["native_front_error"], 0.)
                    self.assertEqual(record["actual_work"]["evaluation_interactions"], 3 * 4 * 21 * 4)
                    for key, value in record["actual_work"].items():
                        costs[key] = costs.get(key, 0) + value
                    fresh = torch.load(path, map_location="cpu", weights_only=True)
                    for name in tensors:
                        for key in tensors[name]:
                            for k, value in tensors[name][key].items():
                                self.assertTrue(torch.equal(value, fresh[name][key][k]))
                    with patch.object(training, "make_piecewise_mo_env", side_effect=AssertionError("reevaluated")):
                        self.assertEqual(probe.run(env_name, 9133, root), record)
                    bad = copy.deepcopy(record["fresh_evaluations"]["new_new"])
                    bad["pareto_per_mode"][0][0]["return"][0] += .01
                    with self.assertRaises(AssertionError):
                        probe.same_front(bad, saved["evaluations"]["work_soft"])
                    bad = copy.deepcopy(record)
                    bad["actual_work"]["evaluation_interactions"] -= 1
                    with self.assertRaises(AssertionError):
                        probe.check(bad)
                summary = probe.summarize(root)
                self.assertEqual(summary["actual_work"]["evaluation_interactions"], 2 * 3 * 4 * 21 * 4)
                self.assertEqual(summary["actual_work"]["initialization_probes"], 2)
                for env in summary["environments"].values():
                    methods, contrasts = env["methods"], env["contrasts"]
                    self.assertAlmostEqual(contrasts["interaction"]["utility"]["mean"],
                        methods["new_new"]["utility"]["mean"] - methods["new_old"]["utility"]["mean"]
                        - methods["old_new"]["utility"]["mean"] + methods["old_old"]["utility"]["mean"])
        print("PAIRING_SMOKE_SOURCE_WORK", json.dumps(source_costs))
        print("PAIRING_SMOKE_NEW_WORK", json.dumps(costs))


if __name__ == "__main__":
    unittest.main()
