import copy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import torch

import run_mean_raw_objective_probe as probe
from test_crossed_replay_probe import added_data
from test_learning_credit_probe import toy_trainer


class MeanRawObjectiveTests(unittest.TestCase):
    def test_target_formula_determinism_and_terminal_mask(self):
        trainer = toy_trainer()
        s, _, w, c, reward, next_s, next_c, _ = probe.base.batch_at(trainer.buffer, np.arange(4))
        done = torch.tensor([[0.], [1.], [0.], [1.]])
        rng_before = torch.get_rng_state().clone()
        action = trainer.policy.get_action(next_s, next_c, deterministic=True)
        value = torch.min(trainer.q1_target(next_s, action, next_c), trainer.q2_target(next_s, action, next_c))
        expected = reward + trainer.gamma * (1 - done) * value
        with patch.object(trainer.policy, "get_action", wraps=trainer.policy.get_action) as call:
            scalar, vector = probe.mean_raw_targets(trainer, reward, w, next_s, next_c, done)
        self.assertEqual(call.call_count, 1)
        self.assertTrue(call.call_args.kwargs["deterministic"])
        torch.testing.assert_close(vector, expected, rtol=0, atol=0)
        torch.testing.assert_close(scalar, (w * expected).sum(-1, keepdim=True), rtol=0, atol=0)
        torch.testing.assert_close(vector[done[:, 0] == 1], reward[done[:, 0] == 1], rtol=0, atol=0)
        self.assertFalse(vector.requires_grad or scalar.requires_grad)
        self.assertTrue(torch.equal(rng_before, torch.get_rng_state()))
        self.assertFalse(trainer.replay_terminal(False, True))
        self.assertTrue(trainer.replay_terminal(True, False))
        trainer.alpha = 100.
        with torch.no_grad():
            trainer.policy.log_std_head.bias.add_(10.)
        _, changed = probe.mean_raw_targets(trainer, reward, w, next_s, next_c, done)
        torch.testing.assert_close(vector, changed, rtol=0, atol=0)

    def test_warm_adam_alpha_independence_paired_inputs_and_soft_identity(self):
        trainer = toy_trainer()
        with probe.base.isolated_rng(4):
            trainer.update()
        original = probe.base.clone_trainer(trainer, trainer.buffer)
        altered = probe.base.clone_trainer(trainer, trainer.buffer)
        altered.alpha = 100.
        data = added_data(trainer)
        steps, observed, calls = torch.optim.Adam.step, [], []
        relabel = probe.training.ProjectedCommonTargetMOSACTrainer.relabel_preferences

        def counted_step(optimizer, *args, **kwargs):
            calls.append(optimizer)
            return steps(optimizer, *args, **kwargs)

        def capture_inputs(model, w, c, next_c):
            result = relabel(model, w, c, next_c)
            observed.append(tuple(value.detach().clone() for value in result))
            return result

        models = {}
        with patch.object(probe, "UPDATES", 3), patch.object(probe.source, "UPDATES", 3), \
             patch.object(torch.optim.Adam, "step", counted_step), \
             patch.object(probe.training.ProjectedCommonTargetMOSACTrainer, "relabel_preferences", capture_inputs):
            for cell, cached in probe.CACHED.items():
                models[cell], work = probe.learn(trainer, data, cell, 100)
                expected, expected_work = probe.source.learn(trainer, data if cell == "soft_uniform" else None, 100)
                probe.base.same_learner(models[cell], expected)
                self.assertEqual(work, expected_work)
            models["raw_uniform"], work = probe.learn(trainer, data, "raw_uniform", 100)
            invariant, other_work = probe.learn(altered, data, "raw_uniform", 100)
        self.assertEqual(len(calls), 36)
        self.assertEqual(work, other_work)
        probe.base.same_learner(models["raw_uniform"], invariant)
        self.assertEqual(work["actor_optimizer_steps"], work["critic_optimizer_steps"])
        self.assertEqual(work["actor_optimizer_steps"], 3)
        self.assertEqual(models["soft_uniform"].buffer.new_rows, models["raw_uniform"].buffer.new_rows)
        self.assertGreater(models["raw_uniform"].buffer.new_rows, 0)
        # Each block has three updates: soft_old, native_old, soft_uniform,
        # native_uniform, raw_uniform, alpha-altered raw_uniform.
        for update in range(3):
            for a, b in zip(observed[6 + update], observed[12 + update]):
                torch.testing.assert_close(a, b, rtol=0, atol=0)
        self.assertEqual(probe.training.optimizer_step_count(models["raw_uniform"].policy_opt), 4)
        for name, parameter in models["raw_uniform"].policy.named_parameters():
            if name.startswith("log_std_head"):
                old = dict(trainer.policy.named_parameters())[name]
                self.assertTrue(torch.equal(parameter, old))
                self.assertIsNone(parameter.grad)
                self.assertTrue(all(torch.equal(v, trainer.policy_opt.state[old][k])
                    for k, v in models["raw_uniform"].policy_opt.state[parameter].items()))
        probe.base.same_learner(original, trainer)
        print("MEAN_RAW_STANDALONE actor_optimizer_steps=19 critic_optimizer_steps=19")

    def test_real_hopper_reuse_budget_and_factorial_summary(self):
        row, learning = probe.source, probe.source.source
        factory = probe.training.make_piecewise_mo_env
        kwargs = dict(hidden=8, lr=3e-4, gamma=.99, batch_size=4, learning_starts=5,
                      eval_freq=100000, log_freq=10000, device="cpu")

        def short_env(*args, **kw):
            kw.update(switch_interval=12, mode_order="sequential")
            env = factory(*args, **kw)
            env.env._max_episode_steps = 4
            return env

        evaluate, collect = probe.base.evaluate, probe.base.collect
        pending, evaluations = {}, []

        def checked_collect(state, sampler, seed):
            data, profile = collect(state, sampler, seed)
            self.assertTrue((data.regimes == -1).all())
            self.assertEqual(int(data.dones.sum()), profile["terminated"])
            return data, profile

        def checked_evaluate(state, trainer, seed):
            if pending.get("new"):
                record = json.loads(pending["output"].read_text())
                self.assertTrue(record["source_reconstruction_passed"])
                self.assertTrue(all(v["control_reconstruction_passed"] for v in record["streams"].values()))
                if evaluations:
                    self.assertTrue(record["endpoint_reconstruction_passed"])
                evaluations.append(trainer)
            return evaluate(state, trainer, seed)

        with tempfile.TemporaryDirectory() as directory, \
             patch.multiple(probe, CUT=22, COLLECTION_STEPS=8, UPDATES=3, KWARGS=kwargs, SEEDS=(91241,)), \
             patch.multiple(row, CUT=22, COLLECTION_STEPS=8, UPDATES=3, KWARGS=kwargs, SEEDS=(91241,)), \
             patch.multiple(learning, CUT=22, COLLECTION_STEPS=8, UPDATES=3, KWARGS=kwargs, SEEDS=(91241,)), \
             patch.multiple(probe.base, CUT=22, COLLECTION_STEPS=8, REPEATS=1, HORIZON=4, KWARGS=kwargs), \
             patch.object(probe.training, "make_piecewise_mo_env", side_effect=short_env), \
             patch.object(probe.base, "collect", side_effect=checked_collect), \
             patch.object(probe.base, "evaluate", side_effect=checked_evaluate), patch("builtins.print"):
            roots = [Path(directory) / name for name in ("learning", "row", "mean_raw")]
            for root in roots:
                root.mkdir()
            with patch.object(row, "SOURCE", roots[0]), patch.object(probe, "SOURCE", roots[1]):
                for module, root in zip((learning, row, probe), roots):
                    probe.write_report(root / "protocol.json", dict(protocol=module.PROTOCOL, experiment=module.specification()))
                reports = [learning.run(probe.ENV, 91241, roots[0]), row.run(probe.ENV, 91241, roots[1])]
                pending.update(new=True, output=roots[2] / "seed91241.json")
                report = probe.run(91241, roots[2])
                reports.append(report)
                self.assertTrue(report["controls_passed"] and report["representation_frozen"])
                self.assertEqual(len(evaluations), 7)
                self.assertEqual(report["completed_budget"]["training_interactions"], 38)
                self.assertEqual(report["completed_budget"]["actor_optimizer_steps"], 54)
                self.assertEqual(report["completed_budget"]["audit_loss_rows"], 2352)
                for stream in probe.STREAMS:
                    for cell, cached in probe.CACHED.items():
                        value = report["streams"][str(stream)]["cells"][cell]
                        self.assertTrue(value["evaluation_reused"])
                        self.assertEqual(value["evaluation"], reports[1]["streams"][str(stream)][cached]["evaluation"])
                with patch.object(probe.training, "train_method", side_effect=AssertionError("retrained")):
                    self.assertEqual(probe.run(91241, roots[2]), report)
                summary = probe.summarize(roots[2])
                cells = summary["cell_summaries"]
                interaction = (cells["raw_uniform"]["utility"]["mean"] - cells["raw_old"]["utility"]["mean"] -
                               cells["soft_uniform"]["utility"]["mean"] + cells["soft_old"]["utility"]["mean"])
                self.assertAlmostEqual(summary["contrasts"]["interaction"]["utility"]["mean"], interaction)
                self.assertEqual(len(summary["contrasts"]["raw_data"]["utility"]["per_seed_means"]), 1)
                bad = copy.deepcopy(report)
                bad["streams"]["1"]["cells"]["raw_uniform"]["learner_work"]["actor_optimizer_steps"] += 1
                probe.write_report(roots[2] / "seed91241.json", bad)
                with self.assertRaises(AssertionError):
                    probe.summarize(roots[2])
                costs = {key: sum(r["completed_budget"][key] for r in reports) for key in report["completed_budget"]}
        print("MEAN_RAW_SMOKE", json.dumps(costs))


if __name__ == "__main__":
    unittest.main()
