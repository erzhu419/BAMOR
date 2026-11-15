import copy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import torch

import run_update_residual_probe as probe
from test_crossed_replay_probe import added_data
from test_learning_credit_probe import toy_trainer


class UpdateResidualTests(unittest.TestCase):
    def test_large_float32_projection_stays_inside_radius(self):
        reference = torch.ones(2 * 256 * 256)
        before = reference - 1e-5
        current = reference + 1e-4
        rho = 1500 / 26500
        radius = rho * float((reference.double() - before.double()).norm())
        delta = current - reference
        naive = reference + radius / float(delta.double().norm()) * delta
        naive_norm = float((naive.double() - reference.double()).norm())
        self.assertGreater(naive_norm, radius + 1e-7 + 1e-4 * radius)
        info = probe.bound_residual([current], [reference], [before], rho)
        self.assertLessEqual(info["accepted_residual_norm"], radius + 1e-12)
        self.assertTrue(torch.all(current >= reference))

    def test_bound_geometry_zero_radius_and_inactive_identity(self):
        current, reference, before = [torch.tensor([4., 5.])], [torch.tensor([1., 0.])], [torch.zeros(2)]
        info = probe.bound_residual(current, reference, before, .5)
        self.assertAlmostEqual(info["accepted_residual_norm"], .5, places=6)
        self.assertLess(info["scale"], 1.)
        torch.testing.assert_close(current[0], reference[0] + .5 * torch.tensor([3., 5.]) / np.sqrt(34))
        info = probe.bound_residual([torch.tensor([2.])], [torch.tensor([1.])], [torch.tensor([1.])], .5)
        self.assertEqual(info["scale"], 0.)
        self.assertEqual(info["accepted_residual_norm"], 0.)
        current = [torch.tensor([1.2])]
        saved = current[0].clone()
        info = probe.bound_residual(current, [torch.tensor([1.])], [torch.tensor([0.])], .5)
        self.assertEqual(info["scale"], 1.)
        self.assertTrue(torch.equal(current[0], saved))

    def test_native_identity_moments_targets_and_actual_proposal_counts(self):
        trainer = toy_trainer()
        with probe.base.isolated_rng(4):
            trainer.update()
        original = copy.deepcopy(trainer.policy.state_dict())
        data = added_data(trainer)
        pool = probe.source.coupled_buffer(trainer.buffer, data)
        step = torch.optim.Adam.step
        observed = []

        def counted_step(optimizer, *args, **kwargs):
            observed.append(optimizer)
            return step(optimizer, *args, **kwargs)

        with patch.object(probe, "UPDATES", 3), patch.object(probe.source, "UPDATES", 3):
            for arm in probe.ARMS:
                observed.clear()
                with patch.object(torch.optim.Adam, "step", counted_step):
                    branch, work, details = probe.learn(trainer, pool, arm, 100)
                self.assertEqual(len(observed), 12)
                self.assertEqual(work["critic_optimizer_steps"], 6)
                self.assertEqual(work["actor_optimizer_steps"], 6)
                self.assertEqual(details["reference_rows"], details["active_rows"])
                self.assertEqual(probe.training.optimizer_step_count(branch.policy_opt), 4)
                self.assertEqual(probe.training.optimizer_step_count(branch.q_opt), 4)
                if arm != "bounded":
                    native, _ = probe.source.learn(trainer, None if arm == "old" else data, 100)
                    probe.base.same_learner(branch, native)
                else:
                    for values in details["diagnostics"].values():
                        self.assertEqual(len(values), 3)
                        self.assertTrue(all(v["accepted_residual_norm"] <=
                            v["radius"] + 1e-6 for v in values))
        with patch.object(probe, "UPDATES", 1):
            before_targets = {name: [p.detach().clone() for p in getattr(trainer, name).parameters()]
                              for name in ("q1_target", "q2_target")}
            bounded, _, _ = probe.learn(trainer, pool, "bounded", 100)
            for name, target in (("q1", "q1_target"), ("q2", "q2_target")):
                for start, value, accepted in zip(before_targets[target], getattr(bounded, target).parameters(),
                        getattr(bounded, name).parameters()):
                    torch.testing.assert_close(value, start.lerp(accepted, bounded.tau), rtol=0, atol=0)
        self.assertTrue(all(torch.equal(value, trainer.policy.state_dict()[key]) for key, value in original.items()))
        self.assertEqual(probe.training.optimizer_step_count(trainer.policy_opt), 1)
        print("UPDATE_RESIDUAL_STANDALONE actor_optimizer_steps=27 critic_optimizer_steps=27")

    def test_real_hopper_reconstruction_costs_reuse_and_seed_aggregation(self):
        row, learning = probe.source, probe.source.source
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
            if pending.get("probe"):
                record = json.loads(pending["output"].read_text())
                self.assertTrue(record["source_reconstruction_passed"])
                self.assertTrue(all(r["control_identity_passed"] for r in record["streams"].values()))
                if evaluations:
                    self.assertTrue(record["endpoint_reconstruction_passed"])
                evaluations.append(trainer)
            return evaluate(state, trainer, seed)

        with tempfile.TemporaryDirectory() as directory, \
             patch.multiple(probe, CUT=22, COLLECTION_STEPS=8, UPDATES=3, KWARGS=kwargs, SEEDS=(91239,)), \
             patch.multiple(row, CUT=22, COLLECTION_STEPS=8, UPDATES=3, KWARGS=kwargs, SEEDS=(91239,)), \
             patch.multiple(learning, CUT=22, COLLECTION_STEPS=8, UPDATES=3, KWARGS=kwargs, SEEDS=(91239,)), \
             patch.multiple(probe.base, CUT=22, COLLECTION_STEPS=8, REPEATS=1, HORIZON=4, KWARGS=kwargs), \
             patch.object(probe.training, "make_piecewise_mo_env", side_effect=short_env), \
             patch.object(probe.base, "evaluate", side_effect=checked_evaluate), patch("builtins.print"):
            learning_root, row_root, root = (Path(directory) / name for name in ("learning", "row", "residual"))
            for path in (learning_root, row_root, root):
                path.mkdir()
            with patch.object(row, "SOURCE", learning_root), patch.object(probe, "SOURCE", row_root):
                for module, path in ((learning, learning_root), (row, row_root), (probe, root)):
                    probe.write_report(path / "protocol.json", dict(protocol=module.PROTOCOL, experiment=module.specification()))
                parent = learning.run(probe.ENV, 91239, learning_root)
                paired = row.run(probe.ENV, 91239, row_root)
                pending.update(probe=True, output=root / "seed91239.json")
                report = probe.run(91239, root)
                self.assertTrue(report["controls_passed"] and report["representation_frozen"])
                self.assertEqual(len(evaluations), 4)
                self.assertEqual(report["completed_budget"]["training_interactions"], 38)
                self.assertEqual(report["completed_budget"]["actor_optimizer_steps"], 90)
                self.assertEqual(report["completed_budget"]["audit_loss_rows"], 2856)
                self.assertEqual(report["completed_budget"]["encoder_optimizer_steps"], 18)
                with patch.object(probe.training, "train_method", side_effect=AssertionError("retrained")):
                    self.assertEqual(probe.run(91239, root), report)
                summary = probe.summarize(root)
                values = [report["streams"][str(s)]["arms"]["bounded"]["evaluation"]["direct_utility"]["mean"]
                          for s in probe.STREAMS]
                self.assertAlmostEqual(summary["arm_summaries"]["bounded"]["utility"]["mean"], np.mean(values))
                self.assertEqual(len(summary["contrasts"]["bounded_minus_old"]["utility"]["per_seed_means"]), 1)
                bad = copy.deepcopy(report)
                bad["streams"]["1"]["arms"]["bounded"]["learner_work"]["actor_optimizer_steps"] += 1
                probe.write_report(root / "seed91239.json", bad)
                with self.assertRaises(AssertionError):
                    probe.summarize(root)
                costs = {key: sum(r["completed_budget"][key] for r in (parent, paired, report))
                         for key in report["completed_budget"]}
        print("UPDATE_RESIDUAL_SMOKE", json.dumps(costs))


if __name__ == "__main__":
    unittest.main()
