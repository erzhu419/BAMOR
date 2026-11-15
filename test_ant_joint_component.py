import copy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import torch

import run_ant_joint_component as experiment
import test_ant_component_forks as fixtures


class JointComponentTests(unittest.TestCase):
    def test_joint_uses_first_copy_gradients_and_no_bc_only_inside_window(self):
        trainer = fixtures.ComponentForkTests().trainer()
        ordinary = copy.deepcopy(trainer)
        router = experiment.base.ComponentRouter(experiment.JOINT, [], {}, cut=0, block_end=4)
        router.configure(1, trainer)
        self.assertEqual(trainer.behavior_clone_coef, 0.)
        self.assertEqual(trainer.critic_relabel_copies, 3)
        self.assertEqual(trainer.preference_relabeling_rate, 1.)
        for name, widths in (("critic_objective", [3, 3, 3, 1, 3]), ("actor_objective", [3, 3, 3, 1, 1])):
            args = [torch.randn(12, width, requires_grad=True) for width in widths]
            actual = getattr(trainer, name)(*args)
            expected = getattr(ordinary, name)(*(a[:4] for a in args))
            torch.testing.assert_close(actual, expected, rtol=0, atol=0)
            grads = torch.autograd.grad(actual, args, allow_unused=True, retain_graph=True)
            expected_grads = torch.autograd.grad(expected, args, allow_unused=True)
            for left, right in zip(grads, expected_grads):
                if left is not None:
                    torch.testing.assert_close(left, right, rtol=0, atol=0)
                    self.assertEqual(torch.count_nonzero(left[4:]).item(), 0)
        trainer.behavior_clone_coef = .004
        router.configure(5, trainer)
        self.assertEqual(trainer.behavior_clone_coef, .004)
        self.assertFalse(router.patches)
        self.assertNotIn("critic_objective", trainer.__dict__)

    def test_joint_update_preserves_full_draw_stream(self):
        baseline, trace = fixtures.ComponentForkTests().trainer(), []
        for name in ("shared", experiment.JOINT):
            trainer = copy.deepcopy(baseline)
            budget = dict(training_interactions=0, learner_updates=0)
            router = experiment.base.ComponentRouter(name, trace, budget, cut=0, block_end=4, learning_starts=0)
            np.random.seed(23)
            torch.manual_seed(23)
            try:
                for step in range(1, 7):
                    trainer.behavior_clone_coef = .004
                    router.configure(step, trainer)
                    if name == experiment.JOINT and step <= 4:
                        with patch.object(trainer, "behavior_clone_loss", side_effect=AssertionError("BC is active")):
                            trainer.update()
                    else:
                        trainer.update()
                    router.after_step(step, trainer, None, None, experiment.base.B)
                self.assertEqual(budget, dict(training_interactions=6, learner_updates=6))
                if name == experiment.JOINT:
                    self.assertEqual(router.matched_steps, 6)
                    self.assertEqual(router.effective_loss_rows, 16)
                    self.assertEqual(router.generated_rows, 48)
            finally:
                router.close()

    def test_interaction_is_not_total_joint_gain(self):
        def row(value):
            return dict(mode0_mixed_energy_utility=value, all_mode_direct_utility=value,
                hv_mean_mode=value, direct_utility_per_mode=[value] * 3,
                preference_group_utility_per_mode={g: [value] * 3 for g in ("mixed_energy", "zero_energy", "pure_energy")})

        for value, expected in ((16., -2.), (18., 0.), (19., 1.)):
            result = experiment.interaction(row(value), row(13.), row(15.), row(10.))
            self.assertEqual(result["mode0_mixed_energy_utility"], expected)
            self.assertEqual(result["direct_utility_per_mode"], [expected] * 3)

    def test_short_ant_driver_reproduces_control_and_accounts_joint(self):
        base, training = experiment.base, experiment.base.training
        make_env, trainer_cls = training.make_piecewise_mo_env, training.ConsistentTargetMOSACTrainer
        shared_cls = base.previous.SharedCreditOwnPreferenceSelector
        kwargs = dict(hidden=8, batch_size=4, learning_starts=5, eval_freq=1000,
                      log_freq=1000, device="cpu")

        def short_env(*args, **kw):
            env = make_env(*args, **kw)
            env.env._max_episode_steps = 4
            return env

        class ReferenceProbe:
            step = 33

            def __init__(self):
                self.rows = []

            def __call__(self, state):
                self.rows.append(base.measure(state, 17003))
                if self.step == 41:
                    state["adaptive_selector"].maybe_finish_block(41, force=True)
                    self.reference = dict(env=base.ENV, method=base.METHOD, seed=17003, total_steps=41,
                        episode_rewards=copy.deepcopy(state["log_data"]["episode_rewards"]),
                        ground_truth=state["env"].get_ground_truth(),
                        adaptive_operator_final=state["adaptive_selector"].state_dict())
                    return None
                self.step = 37 if self.step == 33 else 41
                return False

        with tempfile.TemporaryDirectory() as folder, \
             patch.multiple(experiment, SEED=17003, CUT=33, BLOCK_END=37, END=41, KWARGS=kwargs), \
             patch.object(training, "make_piecewise_mo_env", side_effect=short_env), \
             patch.object(training, "ConsistentTargetMOSACTrainer",
                          side_effect=lambda *a, **kw: trainer_cls(*a, buffer_capacity=64, **kw)), \
             patch.object(base.previous, "SharedCreditOwnPreferenceSelector",
                          side_effect=lambda *a, **kw: shared_cls(*a, block_steps=4, **kw)), \
             patch("builtins.print"):
            capture = ReferenceProbe()
            with base.previous.selector_patch("shared"):
                training.train_method(base.ENV, base.METHOD, 41, 25000, 17003, folder,
                                      state_probe=capture, archive_steps=[41], **kwargs)
            # Single-ablation cells are synthetic here; the interaction formula is tested separately.
            history = dict(evaluation={k: capture.rows[0][k] for k in
                                       ("evaluation_weights", "episode_seeds_per_mode", "ref_point")},
                branches={name: dict(measurements=copy.deepcopy(capture.rows if name == "per_label" else capture.rows[1:]))
                          for name in ("per_label", "shared", "no_extra_copy_loss", "no_bc")},
                completed_budget=dict(training_interactions=41, learner_updates=37,
                                      evaluation_interactions=sum(r["evaluation_interactions"] for r in capture.rows),
                                      environment_initialization_probe_steps=2))
            root = Path(folder) / "joint"
            with patch.object(experiment, "load_references", return_value=(history, capture.reference)):
                report = experiment.run(root)
            self.assertTrue(report["complete"] and report["controls_passed"])
            self.assertEqual(report["completed_budget"]["training_interactions"], 49)
            self.assertEqual(report["completed_budget"]["learner_updates"], 45)
            self.assertEqual(report["completed_budget"]["environment_initialization_probe_steps"], 4)
            self.assertEqual(report["branches"][experiment.JOINT]["routing"]["rng_matched_steps"], 8)
            self.assertTrue(all(v["grid_returns_matched"] for v in report["branches"]["shared"]["reproduction"].values()))
            result = json.loads((root / "readout.json").read_text())
            self.assertTrue(result["validation_passed"])
            self.assertEqual([r["step"] for r in result["contrasts"]["interaction"]], [37, 41])
            with patch.object(experiment, "load_references", side_effect=AssertionError("reran complete run")):
                self.assertTrue(experiment.run(root)["complete"])
            incomplete = Path(folder) / "incomplete"
            incomplete.mkdir()
            base.write_report(incomplete / "seed17003.json", dict(protocol=experiment.PROTOCOL, complete=False))
            with self.assertRaisesRegex(RuntimeError, "Incomplete attempt preserved"):
                experiment.run(incomplete)
            eval_cost = history["completed_budget"]["evaluation_interactions"] + report["completed_budget"]["evaluation_interactions"]
        print(f"ANT_JOINT_SMOKE training=90 evaluation={eval_cost} probes=6 unit_updates=12")


if __name__ == "__main__":
    unittest.main()
