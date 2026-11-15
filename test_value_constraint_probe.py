import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
import torch

import run_value_constraint_probe as probe


class ValueConstraintTests(unittest.TestCase):
    def test_value_constraints_do_not_remove_entropy_from_actor_objective(self):
        for constraint in ("soft", "value"):
            policy = torch.nn.Linear(2, 1, bias=False)
            with torch.no_grad():
                policy.weight.copy_(torch.tensor([[.2, -.1]]))
            intervention = object.__new__(probe.ValueConstraintDescent)
            intervention.constraint = constraint
            intervention.trainer = SimpleNamespace(policy=policy, alpha=1., policy_opt=SimpleNamespace())
            intervention.groups = SimpleNamespace(assign=lambda c: np.array([0, 0, 1, 1]))
            intervention.group_rows = np.zeros(4, dtype=np.int64)
            intervention.present_group_counts = np.zeros(5, dtype=np.int64)
            intervention.gradient_rows = intervention.gradient_calls = intervention.updates = 0
            intervention.seconds = 0.
            x = torch.tensor([[1., 0.], [1., 0.], [-1., 0.], [-1., 0.]])
            log_prob = policy(torch.tensor([[-2., 1.], [-2., 1.], [-2., -1.], [-2., -1.]]))
            q = -policy(x)
            with patch.object(probe, "DIAG_UPDATES", ()):
                loss = intervention.objective(x, x, x, q, log_prob)
            torch.testing.assert_close(loss, (log_prob - q).mean(), atol=0, rtol=0)
            loss.backward()
            torch.testing.assert_close(policy.weight.grad, torch.tensor([[-2., 0.]]), atol=0, rtol=0)
            expected = torch.tensor([[-1., 1.], [-3., -1.]]) if constraint == "soft" else x[[0, 2]]
            torch.testing.assert_close(intervention.trainer.policy_opt.constraints, expected.double(), atol=0, rtol=0)
            delta = torch.tensor([1., 0.], dtype=torch.float64)
            accepted, directions, changed = probe.source.source.project_displacement(delta, intervention.trainer.policy_opt.constraints)
            self.assertLessEqual(float((directions @ accepted).max()), 1e-8)
            self.assertEqual(changed, constraint == "value")
            self.assertEqual(intervention.gradient_rows, 4)

    def test_real_pipeline_control_identity_measurement_identity_budgets_and_reuse(self):
        source = probe.source
        factory, train = source.training.make_piecewise_mo_env, probe.train
        kwargs = dict(hidden=8, lr=3e-4, gamma=.99, batch_size=4, learning_starts=5,
                      eval_freq=1000000, log_freq=1000000, device="cpu")
        total, references, identities = {}, {}, []

        def short_env(*args, **options):
            options.update(switch_interval=12, mode_order="sequential")
            env = factory(*args, **options)
            env.env._max_episode_steps = 7
            return env

        def checked_train(env, seed, output, steps, resume=None, intervention=None):
            ref_resume = source.source.history.source.clone_state(resume) if resume is not None else None
            state = train(env, seed, output, steps, resume=resume, intervention=intervention)
            if resume is None:
                return state
            if intervention is None:
                ref = train(env, seed, output, steps)
                cost, label, ref_intervention = steps, "native", None
            else:
                label = intervention.constraint
                if label == "soft":
                    ref_intervention = source.MeasuredDescent(ref_resume["trainer"], "transition")
                else:
                    ref_intervention = probe.ValueConstraintDescent(ref_resume["trainer"], "value")
                with patch.object(probe, "DIAG_UPDATES", ()), patch.object(source, "DIAG_UPDATES", ()):
                    ref = train(env, seed, output, steps, resume=ref_resume, intervention=ref_intervention)
                cost = steps - ref_resume["step"]
            try:
                source.base.same_learner(state["trainer"], ref["trainer"])
                for k, v in state["encoder"].state_dict().items():
                    torch.testing.assert_close(v, ref["encoder"].state_dict()[k], atol=0, rtol=0)
                for k in (*source.shared.SHARED_FIELDS, "contexts", "next_contexts"):
                    np.testing.assert_array_equal(getattr(state["trainer"].buffer, k)[:steps], getattr(ref["trainer"].buffer, k)[:steps])
                np.testing.assert_equal(state["numpy_rng"], ref["numpy_rng"])
                torch.testing.assert_close(state["torch_rng"], ref["torch_rng"], atol=0, rtol=0)
                identities.append((env, label))
                work = dict(training_interactions=cost, initialization_probes=2,
                    actor_steps=cost - kwargs["learning_starts"] + 1 if label == "native" else cost,
                    critic_steps=cost - kwargs["learning_starts"] + 1 if label == "native" else cost,
                    encoder_steps=cost - kwargs["learning_starts"] + 1 if label == "native" else cost)
                if ref_intervention is not None:
                    work.update(group_gradient_calls=ref_intervention.gradient_calls, group_gradient_rows=cost * kwargs["batch_size"])
                for k, v in work.items():
                    references[k] = references.get(k, 0) + v
            finally:
                ref["env"].close()
                ref["eval_env"].close()
            return state

        with tempfile.TemporaryDirectory() as directory, \
             patch.multiple(probe, CUT=24, END=40, KWARGS=kwargs, SWITCH_INTERVAL=12,
                            SEEDS=(9137,), DIAG_UPDATES=(1, 8, 16)), \
             patch.multiple(source, FIT_ROWS=16, AUDIT_ROWS=8), patch.object(source.source, "FIT_ROWS", 16), \
             patch.multiple(source.router, HORIZON=4, REPEATS=1), \
             patch.object(source.training, "make_piecewise_mo_env", side_effect=short_env), \
             patch.object(probe, "train", side_effect=checked_train), patch("builtins.print"):
            root = Path(directory)
            probe.write(root / "protocol.json", dict(protocol=probe.PROTOCOL, experiment=probe.specification()))
            for env in probe.ENVIRONMENTS:
                record = probe.run(env, 9137, root)
                self.assertEqual(record["path_prices"]["native"], record["path_prices"]["value"])
                self.assertEqual(record["interventions"]["value"]["component_gradient_rows"], 24)
                saved = torch.load(root / env.replace("-", "_") / "seed9137.audit.pt", weights_only=True)
                self.assertEqual(saved["protocol"], probe.PROTOCOL)
                self.assertEqual(set(saved["models"]), set(probe.MODEL_NAMES))
                self.assertEqual(len(saved["models"]["native"]["replay"]["states"]), 8)
                self.assertTrue(saved["models"]["prefix"]["policy_opt"]["state"])
                self.assertEqual(record["interventions"]["soft"]["grouping"], record["interventions"]["value"]["grouping"])
                for k, v in record["actual_work"].items():
                    total[k] = total.get(k, 0) + v
                with patch.object(source.training, "train_method", side_effect=AssertionError("retrained")):
                    self.assertEqual(probe.run(env, 9137, root), record)
                bad = dict(record, actual_work=dict(record["actual_work"], initialization_probes=6))
                with self.assertRaises(AssertionError):
                    probe.check(bad)
            self.assertEqual(probe.summarize(root)["actual_work"], total)
        self.assertEqual(identities, [(env, name) for env in probe.ENVIRONMENTS for name in probe.MODEL_NAMES[1:]])
        print("VALUE_CONSTRAINT_SMOKE_WORK", json.dumps(total))
        print("VALUE_CONSTRAINT_REFERENCE_WORK", json.dumps(references))


if __name__ == "__main__":
    unittest.main()
