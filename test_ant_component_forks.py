import copy
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import torch
import torch.nn.functional as F

import run_ant_component_forks as experiment
from run_nonterminal_rng_contrasts import CaptureAndContinue


class ComponentForkTests(unittest.TestCase):
    def trainer(self):
        torch.manual_seed(13)
        trainer = experiment.training.ConsistentTargetMOSACTrainer(
            3, 2, 3, hidden=8, buffer_capacity=32, batch_size=4, device="cpu",
            preference_relabeling=True)
        selector = experiment.previous.SharedCreditOwnPreferenceSelector(3)
        for key, value in selector.trainer_config(experiment.B).items():
            setattr(trainer, key, value)
        trainer.preference_relabeling_rate = 1.
        for i in range(8):
            s, w = np.full(3, i / 10, dtype=np.float32), np.array([.2, .3, .5], dtype=np.float32)
            trainer.buffer.push(s, np.zeros(2), w, w, np.ones(3), s + .1, w, False)
        return trainer

    def test_default_losses_and_gradients_unchanged(self):
        trainer = self.trainer()
        q1, q2, target = [torch.randn(12, 3, requires_grad=True) for _ in range(3)]
        w, scalar = torch.softmax(torch.randn(12, 3), -1), torch.randn(12, 1)
        old = F.mse_loss((w * q1).sum(-1, keepdim=True), scalar) + F.mse_loss((w * q2).sum(-1, keepdim=True), scalar)
        old = old + .06 * (F.mse_loss(q1, target) + F.mse_loss(q2, target))
        new = trainer.critic_objective(q1, q2, w, scalar, target)
        torch.testing.assert_close(new, old, rtol=0, atol=0)
        for a, b in zip(torch.autograd.grad(new, (q1, q2), retain_graph=True),
                        torch.autograd.grad(old, (q1, q2))):
            torch.testing.assert_close(a, b, rtol=0, atol=0)
        action, replay = torch.randn(12, 2, requires_grad=True), torch.randn(12, 2, requires_grad=True)
        actual = trainer.behavior_clone_loss(action, replay)
        expected = F.mse_loss(action, replay.detach())
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        self.assertIsNone(torch.autograd.grad(actual, replay, allow_unused=True, retain_graph=True)[0])

    def test_first_relabel_discards_values_but_keeps_draws(self):
        trainer = self.trainer()
        w = torch.tensor([[.2, .3, .5]]).repeat(4, 1)
        rng = torch.get_rng_state()
        relabeled = trainer.relabel_preferences(w, w, w)
        expected_rng = torch.get_rng_state()
        router = experiment.ComponentRouter("no_first_relabel", [], {}, cut=0, block_end=4)
        router.configure(1, trainer)
        torch.set_rng_state(rng)
        actual = trainer.relabel_preferences(w, w, w)
        self.assertTrue(torch.equal(torch.get_rng_state(), expected_rng))
        self.assertFalse(torch.equal(relabeled[0], w))
        for tensor in actual:
            self.assertIs(tensor, w)
        router.close()
        self.assertNotIn("relabel_preferences", trainer.__dict__)

    def test_only_first_copy_contributes_to_all_three_losses(self):
        trainer = self.trainer()
        original = copy.deepcopy(trainer)
        router = experiment.ComponentRouter("no_extra_copy_loss", [], {}, cut=0, block_end=4)
        router.configure(1, trainer)
        cases = (("critic_objective", [3, 3, 3, 1, 3]),
                 ("actor_objective", [3, 3, 3, 1, 1]), ("behavior_clone_loss", [2, 2]))
        for name, dims in cases:
            args = [torch.randn(12, d, requires_grad=True) for d in dims]
            actual = getattr(trainer, name)(*args)
            expected = getattr(original, name)(*(x[:4] for x in args))
            torch.testing.assert_close(actual, expected, rtol=0, atol=0)
            gradients = torch.autograd.grad(actual, args, allow_unused=True, retain_graph=True)
            expected_gradients = torch.autograd.grad(expected, args, allow_unused=True)
            for a, b in zip(gradients, expected_gradients):
                if a is not None:
                    torch.testing.assert_close(a, b, rtol=0, atol=0)
                    self.assertEqual(torch.count_nonzero(a[4:]).item(), 0)
        router.configure(5, trainer)
        self.assertFalse(router.patches)

    def test_update_stream_alignment_and_mismatch_detection(self):
        base, trace = self.trainer(), []
        for name in experiment.BRANCHES[1:]:
            trainer = copy.deepcopy(base)
            budget = dict(training_interactions=0, learner_updates=0)
            router = experiment.ComponentRouter(name, trace, budget, cut=0, block_end=4, learning_starts=0)
            np.random.seed(21)
            torch.manual_seed(21)
            for step in range(1, 7):
                trainer.behavior_clone_coef = .004
                router.configure(step, trainer)
                trainer.update()
                router.after_step(step, trainer, None, None, experiment.B)
            if name != "shared":
                self.assertEqual(router.matched_steps, 6)
                torch.rand(1)
                with self.assertRaises(AssertionError):
                    router.after_step(6, trainer, None, None, experiment.B)
            router.close()

    def test_fixed_schedule_consumes_native_choice_before_override(self):
        selector = experiment.FixedSharedSchedule(3)
        selector.counts = dict.fromkeys(selector.operators, 2)
        selector.step_counts = dict.fromkeys(selector.operators, 1500)
        selector.require_native_draws, selector.draws = False, []
        selector.schedule = [experiment.B]
        np.random.seed(21)
        ordinary = experiment.previous.SharedCreditOwnPreferenceSelector(3)
        ordinary.__dict__.update(copy.deepcopy(selector.__dict__))
        drawn = ordinary.select()
        expected_rng = np.random.get_state()
        np.random.seed(21)
        self.assertEqual(selector.select(), experiment.B)
        self.assertEqual(selector.draws[0]["drawn"], drawn)
        actual_rng = np.random.get_state()
        np.testing.assert_array_equal(actual_rng[1], expected_rng[1])
        self.assertEqual(actual_rng[2:], expected_rng[2:])

    def test_ant_clone_evaluation_reset_and_component_continuations(self):
        training = experiment.training
        make_env, trainer_cls = training.make_piecewise_mo_env, training.ConsistentTargetMOSACTrainer
        selector_cls = training.OwnPreferenceBothProgressExtremeOperatorSelector
        kwargs = dict(hidden=8, batch_size=4, learning_starts=5, eval_freq=1000,
                      log_freq=1000, archive_steps=[41], device="cpu")
        evaluations = []

        def short_env(*args, **kwargs):
            env = make_env(*args, **kwargs)
            env.env._max_episode_steps = 4
            return env

        capture = CaptureAndContinue(33, lambda state: evaluations.append(experiment.measure(state, 17002)))
        with tempfile.TemporaryDirectory() as folder, \
             patch.object(training, "make_piecewise_mo_env", side_effect=short_env), \
             patch.object(training, "ConsistentTargetMOSACTrainer",
                          side_effect=lambda *a, **kw: trainer_cls(*a, buffer_capacity=64, **kw)), \
             patch.object(training, "OwnPreferenceBothProgressExtremeOperatorSelector",
                          side_effect=lambda *a, **kw: selector_cls(*a, block_steps=4, **kw)), \
             patch("builtins.print"):
            reference = training.train_method(experiment.ENV, experiment.METHOD, 41, 25000, 17002,
                                              folder, state_probe=capture, **kwargs)
            state, trace = capture.state, []
            self.assertEqual(state["step"], 33)
            self.assertEqual(state["episode_steps"], 1)
            shared_ref = copy.deepcopy(reference)
            selector = experiment.previous.SharedCreditOwnPreferenceSelector(3)
            for block in shared_ref["adaptive_operator_final"]["block_history"]:
                op = block["operator"]
                score = (1 - selector.lr) * selector.scores[op] + selector.lr * block["progress_score"]
                for alias in selector.credit_aliases[op]:
                    selector.scores[alias] = score
                block["score_after"] = score

            class MeasureProbe:
                step = 37

                def __call__(self, current):
                    evaluations.append(experiment.measure(current, 17002))
                    if self.step == 41:
                        self.final = current["log_data"]
                        self.blocks = copy.deepcopy(current["adaptive_selector"].block_history)
                        return None
                    self.step = 41
                    return False

            for name in experiment.BRANCHES[1:]:
                branch = experiment.shared_state(experiment.clone_state(state), shared_ref, name == "shared")
                budget = dict(training_interactions=0, learner_updates=0)
                router = experiment.ComponentRouter(name, trace, budget, cut=33, block_end=37, learning_starts=5)
                probe = MeasureProbe()
                try:
                    training.train_method(experiment.ENV, experiment.METHOD, 41, 25000, 17002, folder,
                        state_probe=probe, update_router=router, resume_state=branch, **kwargs)
                    self.assertEqual(budget, dict(training_interactions=8, learner_updates=8))
                    if name == "shared":
                        self.assertEqual(probe.final["episode_rewards"], reference["episode_rewards"])
                        self.assertEqual(probe.blocks, shared_ref["adaptive_operator_final"]["block_history"])
                        np.testing.assert_array_equal(evaluations[-1]["returns"],
                            [[p["return"] for p in front] for front in reference["pareto_per_mode"]])
                    else:
                        self.assertEqual(router.matched_steps, 8)
                finally:
                    router.close()
                    branch["env"].close()
                    branch["eval_env"].close()
            state["env"].close()
            state["eval_env"].close()
        self.assertTrue(all(row["evaluation_interactions"] == 756 for row in evaluations))
        print("ANT_COMPONENT_SMOKE training=73 evaluation=7560 probes=10 unit_updates=24")


if __name__ == "__main__":
    unittest.main()
