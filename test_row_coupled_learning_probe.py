import copy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import torch

import run_row_coupled_learning_probe as probe
from test_learning_credit_probe import toy_trainer


class RowCoupledTests(unittest.TestCase):
    def test_exact_uniform_marginals(self):
        index, gate = np.meshgrid((np.arange(12) + .5) / 12, (np.arange(4) + .5) / 4)
        added, tail = probe.row_indices(index.ravel(), gate.ravel(), 12, 4)
        old, no_tail = probe.row_indices(index.ravel(), gate.ravel(), 12, 0)
        np.testing.assert_array_equal(np.bincount(added, minlength=16), np.full(16, 3))
        np.testing.assert_array_equal(np.bincount(old, minlength=12), np.full(12, 4))
        np.testing.assert_array_equal(old[~tail], added[~tail])
        self.assertEqual(int(tail.sum()), 12)
        self.assertFalse(no_tail.any())

    def test_paired_old_rows_and_identical_rng_consumption(self):
        trainer = toy_trainer()
        data = probe.base.empty_buffer(trainer.buffer, 4)
        for i in range(4):
            data.push(np.full(3, 50 + i), np.zeros(2), np.full(2, .5), np.full(2, .5),
                      np.ones(2), np.full(3, 51 + i), np.full(2, .5), False)
        old, added = probe.coupled_buffer(trainer.buffer), probe.coupled_buffer(trainer.buffer, data)
        with probe.base.isolated_rng(31):
            original = probe.base.combined_buffer(trainer.buffer).sample(256)[0]
        with probe.base.isolated_rng(31):
            first = old.sample(256)[0]
            after_old = np.random.random(10)
        with probe.base.isolated_rng(31):
            second = added.sample(256)[0]
            after_added = np.random.random(10)
        torch.testing.assert_close(first, original, rtol=0, atol=0)
        mask = second[:, 0] < 50
        torch.testing.assert_close(first[mask], second[mask], rtol=0, atol=0)
        np.testing.assert_array_equal(after_old, after_added)
        self.assertEqual(added.new_rows, int((~mask).sum()))
        self.assertEqual(old.new_rows, 0)

    def test_legacy_old_and_repeated_data_exact_learner_identity(self):
        trainer = toy_trainer()
        with probe.base.isolated_rng(4):
            trainer.update()
        original = copy.deepcopy(trainer.policy.state_dict())
        with patch.object(probe, "UPDATES", 3), patch.object(probe.source, "UPDATES", 3):
            legacy, work = probe.source.learn(trainer, None, 551)
            old, old_work = probe.learn(trainer, None, 551)
            added, added_work = probe.learn(trainer, trainer.buffer, 551)
            repeated, repeat_work = probe.learn(trainer, trainer.buffer, 551)
        probe.base.same_learner(legacy, old)
        probe.base.same_learner(added, repeated)
        self.assertEqual(work, old_work)
        self.assertEqual(added_work, repeat_work)
        self.assertEqual(added.buffer.sampled_rows, 12)
        self.assertEqual(probe.training.optimizer_step_count(added.policy_opt), 4)
        self.assertEqual(probe.training.optimizer_step_count(trainer.policy_opt), 1)
        self.assertTrue(all(torch.equal(value, trainer.policy.state_dict()[key]) for key, value in original.items()))
        print("ROW_COUPLED_STANDALONE actor_optimizer_steps=13 critic_optimizer_steps=13")

    def test_real_reconstruction_two_environments_costs_reuse_and_summary(self):
        factory = probe.training.make_piecewise_mo_env
        kwargs = dict(hidden=8, lr=3e-4, gamma=.99, batch_size=4, learning_starts=5,
                      eval_freq=100000, log_freq=10000, device="cpu")

        def short_env(*args, **kw):
            kw.update(switch_interval=12, mode_order="sequential")
            env = factory(*args, **kw)
            env.env._max_episode_steps = 4
            return env

        collect, evaluate, loss = probe.base.collect, probe.base.evaluate, probe.source.audit_loss
        pending, measured_rows = {}, []

        def checked_collect(state, sampler, seed):
            data, profile = collect(state, sampler, seed)
            self.assertTrue((data.regimes == -1).all())
            self.assertEqual(int(data.dones.sum()), profile["terminated"])
            return data, profile

        def checked_evaluate(state, trainer, seed):
            if pending.get("coupled"):
                record = json.loads(pending["output"].read_text())
                self.assertTrue(record["source_reconstruction_passed"])
                self.assertTrue(record["learner_identity_passed"])
            return evaluate(state, trainer, seed)

        def counted_loss(trainer, audit):
            measured_rows.append(audit["rows"])
            return loss(trainer, audit)

        with tempfile.TemporaryDirectory() as directory, \
             patch.multiple(probe, CUT=22, COLLECTION_STEPS=8, UPDATES=3, KWARGS=kwargs, SEEDS=(91237,)), \
             patch.multiple(probe.source, CUT=22, COLLECTION_STEPS=8, UPDATES=3, KWARGS=kwargs, SEEDS=(91237,)), \
             patch.multiple(probe.base, CUT=22, COLLECTION_STEPS=8, REPEATS=1, HORIZON=4, KWARGS=kwargs), \
             patch.object(probe.training, "make_piecewise_mo_env", side_effect=short_env), \
             patch.object(probe.base, "collect", side_effect=checked_collect), \
             patch.object(probe.base, "evaluate", side_effect=checked_evaluate), \
             patch.object(probe.source, "audit_loss", side_effect=counted_loss), patch("builtins.print"):
            source_root, root = Path(directory) / "source", Path(directory) / "coupled"
            source_root.mkdir()
            root.mkdir()
            with patch.object(probe, "SOURCE", source_root):
                probe.write_report(source_root / "protocol.json", dict(protocol=probe.source.PROTOCOL,
                    experiment=probe.source.specification()))
                probe.write_report(root / "protocol.json", dict(protocol=probe.PROTOCOL, experiment=probe.specification()))
                source_reports, reports = [], []
                for env in probe.ENVIRONMENTS:
                    pending["coupled"] = False
                    source_reports.append(probe.source.run(env, 91237, source_root))
                    pending.update(coupled=True, output=root / env.replace("-", "_") / "seed91237.json")
                    report = probe.run(env, 91237, root)
                    self.assertTrue(report["controls_passed"] and report["representation_frozen"])
                    self.assertEqual(report["completed_budget"]["training_interactions"], 46)
                    self.assertEqual(report["completed_budget"]["actor_optimizer_steps"], 57)
                    self.assertEqual(report["completed_budget"]["encoder_optimizer_steps"], 18)
                    self.assertEqual(report["completed_budget"]["audit_target_rows"], 168)
                    self.assertEqual(report["completed_budget"]["audit_loss_rows"], 2520)
                    self.assertEqual(len(report["streams"]), 3)
                    self.assertEqual(report["streams"]["0"]["uniform"]["replay_size"], 30)
                    with patch.object(probe.training, "train_method", side_effect=AssertionError("retrained")):
                        self.assertEqual(probe.run(env, 91237, root), report)
                    reports.append(report)
                summary = probe.summarize(root)
                self.assertEqual(summary["completed_budget"]["training_interactions"], 92)
                for env, report in zip(probe.ENVIRONMENTS, reports):
                    deltas = [report["streams"][str(stream)]["uniform"]["evaluation"]["direct_utility"]["mean"] -
                              report["streams"][str(stream)]["old_only"]["evaluation"]["direct_utility"]["mean"]
                              for stream in probe.STREAMS]
                    measured = summary["environments"][env]["contrasts"]["uniform_minus_old_only"]["utility"]
                    self.assertAlmostEqual(measured["mean"], float(np.mean(deltas)))
                    self.assertEqual(len(measured["per_seed_means"]), 1)
                bad = copy.deepcopy(reports[0])
                bad["streams"]["1"]["mixed"]["learner_work"]["actor_optimizer_steps"] += 1
                probe.write_report(root / probe.ENVIRONMENTS[0].replace("-", "_") / "seed91237.json", bad)
                with self.assertRaises(AssertionError):
                    probe.summarize(root)
                costs = {key: sum(report["completed_budget"][key] for report in reports + source_reports)
                         for key in reports[0]["completed_budget"]}
                self.assertEqual(costs["audit_loss_rows"], sum(measured_rows))
        print("ROW_COUPLED_SMOKE", json.dumps(costs))


if __name__ == "__main__":
    unittest.main()
