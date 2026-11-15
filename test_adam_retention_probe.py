import copy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import torch

import run_adam_retention_probe as probe
from test_crossed_replay_probe import added_data
from test_learning_credit_probe import toy_trainer


class AdamRetentionTests(unittest.TestCase):
    def test_first_step_changes_only_optimizer_state_without_reset(self):
        trainer = toy_trainer()
        with probe.base.isolated_rng(4):
            trainer.update()
        pool = probe.parent.source.coupled_buffer(trainer.buffer, added_data(trainer))
        step, observed = torch.optim.Adam.step, []

        def counted_step(optimizer, *args, **kwargs):
            observed.append(optimizer)
            return step(optimizer, *args, **kwargs)

        with patch.object(probe, "UPDATES", 1), patch.object(probe.parent, "UPDATES", 1), \
             patch.object(torch.optim.Adam, "step", counted_step):
            normal, normal_work, normal_details = probe.learn(trainer, pool, "bounded_uniform_mom", 100)
            old, work, details = probe.learn(trainer, pool, probe.NEW, 100)
        self.assertEqual(len(observed), 8)
        self.assertEqual(work, normal_work)
        self.assertEqual(details["diagnostics"], normal_details["diagnostics"])
        for name in probe.parent.NETWORKS:
            self.assertTrue(all(torch.equal(value, getattr(normal, name).state_dict()[key])
                for key, value in getattr(old, name).state_dict().items()))
        reference = probe.base.clone_trainer(trainer, pool)
        replay = probe.parent.crossed.CrossedReplay(pool, trainer.buffer.size)
        with probe.base.isolated_rng(100):
            probe.parent.crossed.crossed_update(reference, replay, False, False)
        for name in ("q_opt", "policy_opt"):
            probe.same_optimizer(getattr(old, name), getattr(reference, name))
            self.assertEqual(probe.training.optimizer_step_count(getattr(old, name)), 2)
        a, b = old.q_opt.state_dict()["state"], normal.q_opt.state_dict()["state"]
        self.assertTrue(any(not torch.equal(a[key]["exp_avg"], b[key]["exp_avg"]) for key in a))
        self.assertTrue(details["moment_retention_passed"])
        self.assertEqual(probe.training.optimizer_step_count(trainer.q_opt), 1)
        print("ADAM_FIRST_STEP actor_optimizer_steps=6 critic_optimizer_steps=6")

    def test_multistep_control_mapping_work_and_source_identity(self):
        trainer = toy_trainer()
        with probe.base.isolated_rng(4):
            trainer.update()
        original = probe.base.clone_trainer(trainer, trainer.buffer)
        pool = probe.parent.source.coupled_buffer(trainer.buffer, added_data(trainer))
        controls = {}
        with patch.object(probe, "UPDATES", 3), patch.object(probe.parent, "UPDATES", 3):
            for arm, parent_arm in probe.CACHED.items():
                controls[arm], work, details = probe.learn(trainer, pool, arm, 100)
                expected, expected_work, expected_details = probe.parent.learn(trainer, pool, parent_arm, 100)
                probe.base.same_learner(controls[arm], expected)
                self.assertEqual(work, expected_work)
                self.assertEqual(details, expected_details)
            old, work, details = probe.learn(trainer, pool, probe.NEW, 100)
        self.assertEqual(work["actor_optimizer_steps"], work["critic_optimizer_steps"])
        self.assertEqual(work["actor_optimizer_steps"], 6)
        self.assertEqual(details["new_rows"], expected_details["new_rows"])
        self.assertEqual(probe.training.optimizer_step_count(old.policy_opt), 4)
        self.assertTrue(any(not torch.equal(value, controls["bounded_uniform_mom"].policy.state_dict()[key])
            for key, value in old.policy.state_dict().items()))
        for values in details["diagnostics"].values():
            self.assertTrue(all(v["accepted_residual_norm"] <= v["radius"] + 1e-6 for v in values))
        probe.base.same_learner(original, trainer)
        print("ADAM_MULTISTEP actor_optimizer_steps=43 critic_optimizer_steps=43")

    def test_real_hopper_control_reuse_costs_and_seed_first_summary(self):
        parent, row, learning = probe.parent, probe.parent.source, probe.parent.source.source
        factory = probe.training.make_piecewise_mo_env
        kwargs = dict(hidden=8, lr=3e-4, gamma=.99, batch_size=4, learning_starts=5,
                      eval_freq=100000, log_freq=10000, device="cpu")

        def short_env(*args, **kw):
            kw.update(switch_interval=12, mode_order="sequential")
            env = factory(*args, **kw)
            env.env._max_episode_steps = 4
            return env

        evaluate = probe.base.evaluate
        pending, evaluations = {}, []

        def checked_evaluate(state, trainer, seed):
            if pending.get("new"):
                record = json.loads(pending["output"].read_text())
                self.assertTrue(record["source_reconstruction_passed"])
                self.assertTrue(all(v["control_reconstruction_passed"] and
                    v["arms"][probe.NEW]["moment_retention_passed"] for v in record["streams"].values()))
                if evaluations:
                    self.assertTrue(record["endpoint_reconstruction_passed"])
                evaluations.append(trainer)
            return evaluate(state, trainer, seed)

        with tempfile.TemporaryDirectory() as directory, \
             patch.multiple(probe, CUT=22, COLLECTION_STEPS=8, UPDATES=3, KWARGS=kwargs, SEEDS=(91240,)), \
             patch.multiple(parent, CUT=22, COLLECTION_STEPS=8, UPDATES=3, KWARGS=kwargs, SEEDS=(91240,)), \
             patch.multiple(row, CUT=22, COLLECTION_STEPS=8, UPDATES=3, KWARGS=kwargs, SEEDS=(91240,)), \
             patch.multiple(learning, CUT=22, COLLECTION_STEPS=8, UPDATES=3, KWARGS=kwargs, SEEDS=(91240,)), \
             patch.multiple(probe.base, CUT=22, COLLECTION_STEPS=8, REPEATS=1, HORIZON=4, KWARGS=kwargs), \
             patch.object(probe.training, "make_piecewise_mo_env", side_effect=short_env), \
             patch.object(probe.base, "evaluate", side_effect=checked_evaluate), patch("builtins.print"):
            roots = [Path(directory) / name for name in ("learning", "row", "residual", "adam")]
            for root in roots:
                root.mkdir()
            with patch.object(row, "SOURCE", roots[0]), patch.object(parent, "SOURCE", roots[1]), \
                 patch.object(probe, "SOURCE", roots[2]):
                for module, root in zip((learning, row, parent, probe), roots):
                    probe.write_report(root / "protocol.json", dict(protocol=module.PROTOCOL, experiment=module.specification()))
                reports = [learning.run(probe.ENV, 91240, roots[0]), row.run(probe.ENV, 91240, roots[1]),
                           parent.run(91240, roots[2])]
                pending.update(new=True, output=roots[3] / "seed91240.json")
                report = probe.run(91240, roots[3])
                reports.append(report)
                self.assertTrue(report["controls_passed"] and report["representation_frozen"])
                self.assertEqual(len(evaluations), 4)
                self.assertEqual(report["completed_budget"]["training_interactions"], 38)
                self.assertEqual(report["completed_budget"]["actor_optimizer_steps"], 90)
                self.assertEqual(report["completed_budget"]["audit_loss_rows"], 2352)
                for stream in probe.STREAMS:
                    for arm, cached in probe.CACHED.items():
                        value = report["streams"][str(stream)]["arms"][arm]
                        self.assertTrue(value["evaluation_reused"])
                        self.assertEqual(value["evaluation"], reports[2]["streams"][str(stream)]["arms"][cached]["evaluation"])
                with patch.object(probe.training, "train_method", side_effect=AssertionError("retrained")):
                    self.assertEqual(probe.run(91240, roots[3]), report)
                summary = probe.summarize(roots[3])
                values = [report["streams"][str(s)]["arms"][probe.NEW]["evaluation"]["direct_utility"]["mean"]
                          for s in probe.STREAMS]
                self.assertAlmostEqual(summary["arm_summaries"][probe.NEW]["utility"]["mean"], np.mean(values))
                self.assertEqual(len(summary["contrasts"][f"{probe.NEW}_minus_old"]["utility"]["per_seed_means"]), 1)
                bad = copy.deepcopy(report)
                bad["streams"]["1"]["arms"][probe.NEW]["learner_work"]["actor_optimizer_steps"] += 1
                probe.write_report(roots[3] / "seed91240.json", bad)
                with self.assertRaises(AssertionError):
                    probe.summarize(roots[3])
                bad = copy.deepcopy(report)
                bad["streams"]["1"]["arms"][probe.NEW]["moment_retention_passed"] = False
                probe.write_report(roots[3] / "seed91240.json", bad)
                with self.assertRaises(AssertionError):
                    probe.summarize(roots[3])
                costs = {key: sum(r["completed_budget"][key] for r in reports) for key in report["completed_budget"]}
        print("ADAM_RETENTION_SMOKE", json.dumps(costs))


if __name__ == "__main__":
    unittest.main()
