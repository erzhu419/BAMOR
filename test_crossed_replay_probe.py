import copy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import torch

import run_crossed_replay_probe as probe
from test_learning_credit_probe import toy_trainer


def added_data(trainer):
    data = probe.base.empty_buffer(trainer.buffer, 4)
    for i in range(4):
        data.push(np.full(3, 50 + i), np.zeros(2), np.full(2, .5), np.full(2, .5),
                  np.full(2, 2.), np.full(3, 51 + i), np.full(2, .5), False)
    return data


class CrossedReplayTests(unittest.TestCase):
    def test_component_routing_shared_rows_and_rng(self):
        trainer = toy_trainer()
        pool = probe.source.coupled_buffer(trainer.buffer, added_data(trainer))
        batches, counts, after = {}, {}, {}
        for cell, flags in probe.CELLS.items():
            replay = probe.CrossedReplay(pool, trainer.buffer.size)
            with probe.base.isolated_rng(100):
                batches[cell] = replay.sample_pair(256, *flags)
                after[cell] = np.random.random(10)
            counts[cell] = replay
        for cell, (critic_new, actor_new) in probe.CELLS.items():
            critic, actor = batches[cell]
            cmask, amask = critic[0][:, 0] >= 50, actor[0][:, 0] >= 50
            self.assertEqual(bool(cmask.any()), critic_new)
            self.assertEqual(bool(amask.any()), actor_new)
            torch.testing.assert_close(critic[0][~cmask], batches["old_old"][0][0][~cmask], rtol=0, atol=0)
            torch.testing.assert_close(actor[0][~amask], batches["old_old"][1][0][~amask], rtol=0, atol=0)
            np.testing.assert_array_equal(after[cell], after["old_old"])
            self.assertEqual(counts[cell].critic_new_rows, int(cmask.sum()))
            self.assertEqual(counts[cell].actor_new_rows, int(amask.sum()))
        self.assertEqual(counts["new_old"].critic_new_rows, counts["old_new"].actor_new_rows)
        torch.testing.assert_close(batches["new_new"][0][0], batches["new_new"][1][0], rtol=0, atol=0)

    def test_native_diagonal_identity_and_first_step_critic_separation(self):
        trainer = toy_trainer()
        with probe.base.isolated_rng(4):
            trainer.update()
        original = copy.deepcopy(trainer.policy.state_dict())
        data = added_data(trainer)
        pool = probe.source.coupled_buffer(trainer.buffer, data)
        with patch.object(probe, "UPDATES", 3), patch.object(probe.source, "UPDATES", 3):
            for cell, arm in probe.DIAGONALS.items():
                native, work = probe.source.learn(trainer, None if arm == "old_only" else data, 100)
                branch, crossed_work, _ = probe.learn(trainer, pool, cell, 100)
                probe.base.same_learner(native, branch)
                self.assertEqual(work, crossed_work)
        first = {}
        with patch.object(probe, "UPDATES", 1):
            for cell in probe.CELLS:
                first[cell], work, counts = probe.learn(trainer, pool, cell, 100)
                self.assertEqual(work["critic_optimizer_steps"], work["actor_optimizer_steps"])
                self.assertEqual(counts["sampled_rows"], 4)
            self.assertGreater(counts["critic_new_rows"], 0)
        for a, b in (("old_old", "old_new"), ("new_old", "new_new")):
            for network in ("q1", "q2", "q1_target", "q2_target"):
                self.assertTrue(all(torch.equal(value, getattr(first[b], network).state_dict()[key])
                    for key, value in getattr(first[a], network).state_dict().items()))
            for param, state in first[a].q_opt.state_dict()["state"].items():
                self.assertTrue(all(torch.equal(value, first[b].q_opt.state_dict()["state"][param][key])
                    for key, value in state.items()))
        self.assertTrue(any(not torch.equal(value, first["old_new"].policy.state_dict()[key])
            for key, value in first["old_old"].policy.state_dict().items()))
        self.assertTrue(all(torch.equal(value, trainer.policy.state_dict()[key]) for key, value in original.items()))
        self.assertEqual(probe.training.optimizer_step_count(trainer.policy_opt), 1)
        print("CROSSED_REPLAY_STANDALONE actor_optimizer_steps=17 critic_optimizer_steps=17")

    def test_real_hopper_reconstruction_reuse_costs_and_factorial_summary(self):
        row, learning = probe.source, probe.source.source
        factory = probe.training.make_piecewise_mo_env
        kwargs = dict(hidden=8, lr=3e-4, gamma=.99, batch_size=4, learning_starts=5,
                      eval_freq=100000, log_freq=10000, device="cpu")

        def short_env(*args, **kw):
            kw.update(switch_interval=12, mode_order="sequential")
            env = factory(*args, **kw)
            env.env._max_episode_steps = 4
            return env

        collect, evaluate, loss = probe.base.collect, probe.base.evaluate, learning.audit_loss
        pending, observed_loss_rows, cross_evaluations = {}, [], []

        def checked_collect(state, sampler, seed):
            data, profile = collect(state, sampler, seed)
            self.assertTrue((data.regimes == -1).all())
            self.assertEqual(int(data.dones.sum()), profile["terminated"])
            return data, profile

        def checked_evaluate(state, trainer, seed):
            if pending.get("cross"):
                record = json.loads(pending["output"].read_text())
                self.assertTrue(record["source_reconstruction_passed"])
                self.assertTrue(all(r["diagonal_identity_passed"] for r in record["streams"].values()))
                if cross_evaluations:
                    self.assertTrue(record["endpoint_reconstruction_passed"])
                cross_evaluations.append(trainer)
            return evaluate(state, trainer, seed)

        def counted_loss(trainer, audit):
            observed_loss_rows.append(audit["rows"])
            return loss(trainer, audit)

        with tempfile.TemporaryDirectory() as directory, \
             patch.multiple(probe, CUT=22, COLLECTION_STEPS=8, UPDATES=3, KWARGS=kwargs, SEEDS=(91238,)), \
             patch.multiple(row, CUT=22, COLLECTION_STEPS=8, UPDATES=3, KWARGS=kwargs, SEEDS=(91238,)), \
             patch.multiple(learning, CUT=22, COLLECTION_STEPS=8, UPDATES=3, KWARGS=kwargs, SEEDS=(91238,)), \
             patch.multiple(probe.base, CUT=22, COLLECTION_STEPS=8, REPEATS=1, HORIZON=4, KWARGS=kwargs), \
             patch.object(probe.training, "make_piecewise_mo_env", side_effect=short_env), \
             patch.object(probe.base, "collect", side_effect=checked_collect), \
             patch.object(probe.base, "evaluate", side_effect=checked_evaluate), \
             patch.object(learning, "audit_loss", side_effect=counted_loss), patch("builtins.print"):
            learning_root, row_root, root = (Path(directory) / name for name in ("learning", "row", "cross"))
            for path in (learning_root, row_root, root):
                path.mkdir()
            with patch.object(row, "SOURCE", learning_root), patch.object(probe, "SOURCE", row_root):
                for module, path in ((learning, learning_root), (row, row_root), (probe, root)):
                    probe.write_report(path / "protocol.json", dict(protocol=module.PROTOCOL, experiment=module.specification()))
                parent = learning.run(probe.ENV, 91238, learning_root)
                paired = row.run(probe.ENV, 91238, row_root)
                pending.update(cross=True, output=root / "seed91238.json")
                report = probe.run(91238, root)
                self.assertTrue(report["controls_passed"] and report["representation_frozen"])
                self.assertEqual(len(cross_evaluations), 7)
                self.assertEqual(report["completed_budget"]["training_interactions"], 38)
                self.assertEqual(report["completed_budget"]["actor_optimizer_steps"], 72)
                self.assertEqual(report["completed_budget"]["encoder_optimizer_steps"], 18)
                self.assertEqual(report["completed_budget"]["audit_target_rows"], 168)
                self.assertEqual(report["completed_budget"]["audit_loss_rows"], 3360)
                for stream in probe.STREAMS:
                    cells = report["streams"][str(stream)]["cells"]
                    for cell, arm in probe.DIAGONALS.items():
                        self.assertTrue(cells[cell]["evaluation_reused"])
                        self.assertEqual(cells[cell]["evaluation"], paired["streams"][str(stream)][arm]["evaluation"])
                with patch.object(probe.training, "train_method", side_effect=AssertionError("retrained")):
                    self.assertEqual(probe.run(91238, root), report)
                summary = probe.summarize(root)
                interaction = []
                for stream in probe.STREAMS:
                    cells = report["streams"][str(stream)]["cells"]
                    u = {c: cells[c]["evaluation"]["direct_utility"]["mean"] for c in cells}
                    interaction.append(u["new_new"] - u["new_old"] - u["old_new"] + u["old_old"])
                self.assertAlmostEqual(summary["contrasts"]["interaction"]["utility"]["mean"], np.mean(interaction))
                self.assertEqual(len(summary["contrasts"]["interaction"]["utility"]["per_seed_means"]), 1)
                cell_values = [report["streams"][str(stream)]["cells"]["new_old"]["evaluation"]["direct_utility"]["mean"]
                               for stream in probe.STREAMS]
                stats = summary["cell_summaries"]["new_old"]["utility"]
                self.assertAlmostEqual(stats["mean"], np.mean(cell_values))
                self.assertAlmostEqual(stats["minimum_seed_mean"], np.mean(cell_values))
                self.assertEqual(stats["minimum_stream_endpoint"], min(cell_values))
                bad = copy.deepcopy(report)
                bad["streams"]["1"]["cells"]["new_old"]["learner_work"]["actor_optimizer_steps"] += 1
                probe.write_report(root / "seed91238.json", bad)
                with self.assertRaises(AssertionError):
                    probe.summarize(root)
                costs = {key: sum(r["completed_budget"][key] for r in (parent, paired, report))
                         for key in report["completed_budget"]}
                self.assertEqual(costs["audit_loss_rows"], sum(observed_loss_rows))
        print("CROSSED_REPLAY_SMOKE", json.dumps(costs))


if __name__ == "__main__":
    unittest.main()
