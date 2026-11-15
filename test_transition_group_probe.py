import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
import torch

import run_transition_group_probe as probe


class TransitionGroupTests(unittest.TestCase):
    def test_vector_features_match_existing_generic_one_step_statistics(self):
        s = np.array([[1., 2., 3.], [0., 0., 0.]], dtype=np.float32)
        a = np.array([[.2, -.3], [0., 0.]], dtype=np.float32)
        r = np.array([[1., -2.], [3., -4.]], dtype=np.float32)
        sn = s + np.array([[.3, -.6, 1.2], [.1, .2, .3]], dtype=np.float32)
        expected = []
        for values in zip(s, a, r, sn):
            tracker = probe.training.GenericTransitionContextTracker(3, 2, 2, ema_alpha=1.)
            tracker.update(*values)
            expected.append(tracker.features())
        np.testing.assert_allclose(probe.transition_features(s, a, r, sn), expected, atol=1e-7, rtol=1e-7)

    def test_partition_excludes_preferences_latents_and_mode_labels(self):
        rng = np.random.default_rng(12)
        b = SimpleNamespace(size=8, states=rng.normal(size=(8, 3)).astype(np.float32),
            actions=rng.normal(size=(8, 2)).astype(np.float32), rewards=rng.normal(size=(8, 2)).astype(np.float32),
            next_states=rng.normal(size=(8, 3)).astype(np.float32),
            weights=np.zeros((8, 2)), contexts=np.zeros((8, 13)), regimes=np.full((8, 1), -1))
        trainer = SimpleNamespace(buffer=b)
        groups = probe.TransitionGroups(trainer)
        b.weights[:] = 1e6
        b.contexts[:] = -1e6
        b.regimes[:] = 3
        self.assertEqual(groups.state_dict(), probe.TransitionGroups(trainer).state_dict())
        x = probe.transition_features(b.states, b.actions, b.rewards, b.next_states)
        groups.sampled_features = x
        np.testing.assert_array_equal(groups.assign(torch.zeros(8, 13)), groups.labels(x))

    def test_soft_descent_can_hide_value_ascent_and_pair_conflict(self):
        e = torch.tensor([[-2., 1.], [-2., -1.]], dtype=torch.float64)
        v = torch.tensor([[1., 0.], [-1., 0.]], dtype=e.dtype)
        result = probe.component_record([0, 2], e + v, e, v, torch.tensor([1., 0.], dtype=e.dtype))
        self.assertEqual(result["component_sum_relative_error"], 0.)
        self.assertLess(result["groups"][0]["soft_derivative"], 0.)
        self.assertGreater(result["groups"][0]["value_derivative"], 0.)
        self.assertEqual(result["pairs"][0]["value_cosine"], -1.)
        self.assertGreater(result["pairs"][0]["soft_cosine"], 0.)
        self.assertIsNone(probe.cosine(torch.zeros(2), torch.ones(2)))

    def test_post_step_switch_labels_start_on_next_transition(self):
        b = SimpleNamespace(size=13, states=np.zeros((13, 3)), actions=np.zeros((13, 2)),
            rewards=np.zeros((13, 2)), next_states=np.ones((13, 3)), contexts=np.zeros((13, 5)),
            timesteps=np.arange(1, 14).reshape(-1, 1))
        trainer = SimpleNamespace(buffer=b, reward_dim=2)
        groups = dict(context=probe.source.ObservableGroups(trainer), transition=probe.TransitionGroups(trainer))
        state = dict(trainer=trainer, env=SimpleNamespace(mode_history=[(0, 0), (12, 1)]))
        result = probe.replay_audit(state, groups)
        self.assertEqual(result["mode_rows"], [12, 1, 0, 0])
        self.assertEqual(sum(map(sum, result["mode_by_group"]["transition"])), 13)

    def test_real_pipeline_native_and_measurement_identity_prices_artifact_reuse(self):
        factory, train = probe.training.make_piecewise_mo_env, probe.train
        kwargs = dict(hidden=8, lr=3e-4, gamma=.99, batch_size=4, learning_starts=5,
                      eval_freq=1000000, log_freq=1000000, device="cpu")
        total, reference_work, identities = {}, {}, []

        def short_env(*args, **options):
            options.update(switch_interval=12, mode_order="sequential")
            env = factory(*args, **options)
            env.env._max_episode_steps = 7
            return env

        def checked_train(env, seed, output, steps, resume=None, intervention=None):
            reference_resume = probe.source.history.source.clone_state(resume) if resume is not None else None
            state = train(env, seed, output, steps, resume=resume, intervention=intervention)
            if resume is None:
                return state
            if intervention is None:
                ref = train(env, seed, output, steps)
                cost = steps
                label = "native"
            elif intervention.partition == "context":
                old = probe.source.ContextDescent(reference_resume["trainer"])
                ref = train(env, seed, output, steps, resume=reference_resume, intervention=old)
                cost, label = steps - reference_resume["step"], "context"
            else:
                new = probe.MeasuredDescent(reference_resume["trainer"], "transition")
                with patch.object(probe, "DIAG_UPDATES", ()):
                    ref = train(env, seed, output, steps, resume=reference_resume, intervention=new)
                cost, label = steps - reference_resume["step"], "transition"
            try:
                probe.base.same_learner(state["trainer"], ref["trainer"])
                for k, v in state["encoder"].state_dict().items():
                    torch.testing.assert_close(v, ref["encoder"].state_dict()[k], atol=0, rtol=0)
                for k in (*probe.shared.SHARED_FIELDS, "contexts", "next_contexts"):
                    np.testing.assert_array_equal(getattr(state["trainer"].buffer, k)[:steps],
                                                  getattr(ref["trainer"].buffer, k)[:steps])
                np.testing.assert_equal(state["numpy_rng"], ref["numpy_rng"])
                torch.testing.assert_close(state["torch_rng"], ref["torch_rng"], atol=0, rtol=0)
                identities.append((env, label))
                work = dict(training_interactions=cost, initialization_probes=2,
                    actor_steps=cost - kwargs["learning_starts"] + 1 if label == "native" else cost,
                    critic_steps=cost - kwargs["learning_starts"] + 1 if label == "native" else cost,
                    encoder_steps=cost - kwargs["learning_starts"] + 1 if label == "native" else cost)
                if intervention is not None:
                    work.update(group_gradient_calls=old.gradient_calls if label == "context" else new.gradient_calls,
                                group_gradient_rows=cost * kwargs["batch_size"])
                for k, v in work.items():
                    reference_work[k] = reference_work.get(k, 0) + v
            finally:
                ref["env"].close()
                ref["eval_env"].close()
            return state

        with tempfile.TemporaryDirectory() as directory, \
             patch.multiple(probe, CUT=24, END=40, FIT_ROWS=16, AUDIT_ROWS=8, KWARGS=kwargs,
                            SWITCH_INTERVAL=12, SEEDS=(9136,), DIAG_UPDATES=(1, 8, 16)), \
             patch.object(probe.source, "FIT_ROWS", 16), \
             patch.multiple(probe.router, HORIZON=4, REPEATS=1), \
             patch.object(probe.training, "make_piecewise_mo_env", side_effect=short_env), \
             patch.object(probe, "train", side_effect=checked_train), patch("builtins.print"):
            root = Path(directory)
            probe.write(root / "protocol.json", dict(protocol=probe.PROTOCOL, experiment=probe.specification()))
            for env in probe.ENVIRONMENTS:
                record = probe.run(env, 9136, root)
                self.assertEqual(record["path_prices"]["native"], record["path_prices"]["transition"])
                self.assertEqual(record["interventions"]["transition"]["component_gradient_rows"], 24)
                saved = torch.load(root / env.replace("-", "_") / "seed9136.audit.pt", weights_only=True)
                self.assertEqual(set(saved["models"]), set(probe.MODEL_NAMES))
                self.assertEqual(len(saved["models"]["native"]["replay"]["states"]), 8)
                self.assertTrue(saved["models"]["native"]["q1"])
                for k, v in record["actual_work"].items():
                    total[k] = total.get(k, 0) + v
                with patch.object(probe.training, "train_method", side_effect=AssertionError("retrained")):
                    self.assertEqual(probe.run(env, 9136, root), record)
                bad = dict(record, actual_work=dict(record["actual_work"], initialization_probes=6))
                with self.assertRaises(AssertionError):
                    probe.check(bad)
            self.assertEqual(probe.summarize(root)["actual_work"], total)
        self.assertEqual(identities, [(env, name) for env in probe.ENVIRONMENTS for name in probe.MODEL_NAMES[1:]])
        print("TRANSITION_GROUP_SMOKE_WORK", json.dumps(total))
        print("TRANSITION_GROUP_REFERENCE_WORK", json.dumps(reference_work))


if __name__ == "__main__":
    unittest.main()
