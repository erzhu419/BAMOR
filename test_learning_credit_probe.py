import copy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import torch

import run_learning_credit_probe as probe


def toy_trainer():
    torch.manual_seed(7)
    trainer = probe.training.ProjectedCommonTargetMOSACTrainer(3, 2, 2,
        hidden=8, buffer_capacity=16, batch_size=4, device="cpu")
    for i in range(12):
        state, weight = np.full(3, i / 12, dtype=np.float32), np.full(2, .5, dtype=np.float32)
        trainer.buffer.push(state, np.zeros(2), weight, weight, np.ones(2), state + .1, weight, False, t=i + 1)
    return trainer


class LearningCreditTests(unittest.TestCase):
    def test_audit_targets_are_frozen_and_loss_matches_training_metric(self):
        trainer = toy_trainer()
        audit = probe.make_audit(trainer, trainer.buffer, 501)
        targets = audit["targets"].clone()
        measured = probe.audit_loss(trainer, audit)
        self.assertEqual(measured["rows"], 21 * 12)
        w = torch.as_tensor(audit["weights"][0]).expand(12, -1)
        context = trainer.replace_context_preferences(audit["contexts"], w)
        errors = torch.stack((trainer.q1(audit["states"], audit["actions"], context) - targets[0],
                              trainer.q2(audit["states"], audit["actions"], context) - targets[0]))
        expected = ((errors.square().mean(-1) + (errors * w).sum(-1).square()) / (1 + w.square().sum(-1))).mean()
        self.assertAlmostEqual(measured["per_weight"][0], float(expected.detach()), places=7)
        with torch.no_grad():
            next(trainer.q1_target.parameters()).add_(10.)
        self.assertTrue(torch.equal(audit["targets"], targets))
        self.assertEqual(probe.audit_loss(trainer, audit), measured)
        self.assertEqual(probe.credit_choice(dict(uniform=measured, mixed=measured)), "uniform")

    def test_fork_identity_optimizer_moments_and_heldout_exclusion(self):
        trainer = toy_trainer()
        with probe.base.isolated_rng(4):
            trainer.update()
        original = copy.deepcopy(trainer.policy.state_dict())
        audit_data = probe.base.empty_buffer(trainer.buffer, 4)
        for index in range(4):
            audit_data.push(np.full(3, 99.), np.zeros(2), np.full(2, .5), np.full(2, .5),
                np.ones(2), np.full(3, 100.), np.full(2, .5), False, t=13 + index)
        audit = probe.make_audit(trainer, audit_data, 503)
        with patch.object(probe, "UPDATES", 3):
            first, work = probe.learn(trainer, None, 513)
            second, repeated = probe.learn(trainer, None, 513)
        probe.base.same_learner(first, second)
        self.assertEqual(work, repeated)
        self.assertEqual(first.buffer.size, 12)
        self.assertFalse((first.buffer.states[:12] == 99.).any())
        self.assertEqual(probe.audit_loss(first, audit), probe.audit_loss(second, audit))
        self.assertEqual(probe.training.optimizer_step_count(first.policy_opt), 4)
        self.assertEqual(probe.training.optimizer_step_count(trainer.policy_opt), 1)
        self.assertTrue(all(torch.equal(value, trainer.policy.state_dict()[key]) for key, value in original.items()))
        print("LEARNING_CREDIT_STANDALONE actor_optimizer_steps=7 critic_optimizer_steps=7")

    def test_real_pipeline_credit_precedes_evaluation_masks_budget_and_reuse(self):
        factory = probe.training.make_piecewise_mo_env
        kwargs = dict(hidden=8, lr=3e-4, gamma=.99, batch_size=4, learning_starts=5,
                      eval_freq=100000, log_freq=10000, device="cpu")

        def short_env(*args, **kw):
            kw.update(switch_interval=12, mode_order="sequential")
            env = factory(*args, **kw)
            env.env._max_episode_steps = 4
            return env

        collect, evaluate, loss = probe.base.collect, probe.base.evaluate, probe.audit_loss
        pending = {}
        observed_loss_rows = []

        def checked_collect(state, sampler, seed):
            data, profile = collect(state, sampler, seed)
            self.assertTrue((data.regimes == -1).all())
            self.assertEqual(int(data.dones.sum()), profile["terminated"])
            return data, profile

        def checked_evaluate(state, trainer, seed):
            record = json.loads(pending["output"].read_text())
            self.assertTrue(record["choice_recorded_before_evaluation"])
            self.assertEqual(record["learning_credit_choice"], probe.credit_choice(record["audit_losses"]))
            return evaluate(state, trainer, seed)

        def checked_loss(trainer, audit):
            observed_loss_rows.append(audit["rows"])
            return loss(trainer, audit)

        with tempfile.TemporaryDirectory() as directory, \
             patch.multiple(probe, CUT=22, COLLECTION_STEPS=8, UPDATES=3, KWARGS=kwargs, SEEDS=(91236,)), \
             patch.multiple(probe.base, CUT=22, COLLECTION_STEPS=8, REPEATS=1, HORIZON=4, KWARGS=kwargs), \
             patch.object(probe.training, "make_piecewise_mo_env", side_effect=short_env), \
             patch.object(probe.base, "collect", side_effect=checked_collect), \
             patch.object(probe.base, "evaluate", side_effect=checked_evaluate), \
             patch.object(probe, "audit_loss", side_effect=checked_loss), patch("builtins.print"):
            root = Path(directory)
            probe.write_report(root / "protocol.json", dict(protocol=probe.PROTOCOL, experiment=probe.specification()))
            reports = []
            for env in probe.ENVIRONMENTS:
                pending["output"] = root / env.replace("-", "_") / "seed91236.json"
                result = probe.run(env, 91236, root)
                self.assertTrue(result["controls_passed"] and result["representation_frozen"])
                self.assertEqual(result["completed_budget"]["training_interactions"], 54)
                self.assertEqual(result["completed_budget"]["actor_optimizer_steps"], 30)
                self.assertEqual(result["completed_budget"]["encoder_optimizer_steps"], 18)
                self.assertEqual(result["completed_budget"]["audit_target_rows"], 168)
                self.assertEqual(result["completed_budget"]["audit_loss_rows"], 1008)
                self.assertEqual(result["branches"]["uniform"]["replay_size"], 30)
                with patch.object(probe.training, "train_method", side_effect=AssertionError("retrained")):
                    self.assertEqual(probe.run(env, 91236, root), result)
                reports.append(result)
            summary = probe.summarize(root)
            self.assertEqual(summary["completed_budget"]["training_interactions"], 108)
            for env, report in zip(probe.ENVIRONMENTS, reports):
                edited = copy.deepcopy(report)
                edited["branches"]["mixed"]["evaluation"]["direct_utility"]["mean"] = -100.
                edited["branches"]["uniform"]["evaluation"]["direct_utility"]["mean"] = 100.
                probe.write_report(root / env.replace("-", "_") / "seed91236.json", edited)
            altered = probe.summarize(root)
            self.assertEqual(altered["decision"], "hold_no_observed_selection_headroom")
            for env in probe.ENVIRONMENTS:
                self.assertEqual(altered["environments"][env]["runs"][0]["learning_credit_choice"],
                                 summary["environments"][env]["runs"][0]["learning_credit_choice"])
            bad = copy.deepcopy(reports[0])
            bad["branches"]["mixed"]["learner_work"]["actor_optimizer_steps"] += 1
            probe.write_report(root / probe.ENVIRONMENTS[0].replace("-", "_") / "seed91236.json", bad)
            with self.assertRaises(AssertionError):
                probe.summarize(root)
            costs = {key: sum(report["completed_budget"][key] for report in reports) for key in reports[0]["completed_budget"]}
            self.assertEqual(costs["audit_loss_rows"], sum(observed_loss_rows))
        print("LEARNING_CREDIT_SMOKE", json.dumps(costs))


if __name__ == "__main__":
    unittest.main()
