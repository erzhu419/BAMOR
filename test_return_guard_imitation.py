import copy
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import torch

import diagnose_return_guard_imitation as probe


class ImitationDiagnosticTests(unittest.TestCase):
    def test_chunked_action_error_counts_elements_not_batches(self):
        class ZeroPolicy:
            def get_action(self, states, contexts, deterministic):
                self_test.assertTrue(deterministic)
                return torch.zeros(len(states), 2)

        self_test = self
        data = dict(states=torch.zeros(3, 1), contexts=torch.zeros(3, 2),
            actions=torch.tensor([[0., .5], [1., -.5], [.5, -1.]]))
        with patch.object(probe, "BATCH_SIZE", 2):
            mse, maximum = probe.action_error(ZeroPolicy(), data)
        self.assertAlmostEqual(mse, 2.75 / 6)
        self.assertEqual(maximum, 1.)

    def test_preference_balancing_and_teacher_selection(self):
        def row(old, new, scored, blind):
            return dict(scored=scored, blind=blind, trajectory_rows=dict(old=1, new=100),
                mse={"old": {m: old + i for i, m in enumerate(probe.MODELS)},
                     "new": {m: new + i for i, m in enumerate(probe.MODELS)}})

        rows = [row(1., 8., "old", "new"), row(3., 6., "new", "new")]
        value = probe.target_metrics(rows, "scored")
        self.assertEqual(value["all"]["mean_mse"]["prefix"], 3.5)
        self.assertEqual(value["selected_old"]["mean_mse"]["prefix"], 1.)
        self.assertEqual(value["selected_new"]["mean_mse"]["prefix"], 6.)
        self.assertEqual(value["disagreement"]["preferences"], 1)
        self.assertEqual(value["all"]["delta_from_proposal"]["scored"]["per_preference"], [1., 1.])
        self.assertEqual(probe.target_metrics(rows, "blind")["selected_old"], dict(preferences=0))

    def test_real_saved_policies_reproduce_teachers_without_environment_or_updates(self):
        torch.manual_seed(47)
        training = probe.source.training
        models = {m: training.PolicyNet(3, 2, 4, hidden=8) for m in probe.MODELS}
        pairs, rows = [], []
        for i, weight in enumerate(([.2, .8], [.8, .2])):
            pair = {}
            for teacher, name in probe.TEACHERS.items():
                n = 3 + i + (teacher == "new")
                states, contexts = torch.randn(n, 3), torch.randn(n, 4)
                states[0] = torch.zeros(3)
                contexts[:, :2] = torch.tensor(weight)
                with torch.no_grad():
                    actions = models[name].get_action(states, contexts, deterministic=True)
                pair[teacher] = dict(states=states, contexts=contexts, actions=actions)
            pairs.append(pair)
            rows.append(dict(weight_index=i, weight=weight, scored="old" if i == 0 else "new", blind="new",
                utility_difference=float(i), steps={t: len(d["states"]) for t, d in pair.items()}))
        with tempfile.TemporaryDirectory() as directory, \
             patch.object(training, "make_piecewise_mo_env", side_effect=AssertionError("environment created")), \
             patch.object(training, "train_method", side_effect=AssertionError("training called")), \
             patch.object(torch.optim.Adam, "step", side_effect=AssertionError("optimizer called")), \
             patch.object(probe, "BATCH_SIZE", 2):
            path = Path(directory) / "actors.pt"
            torch.save({m: dict(policy=p.state_dict()) for m, p in models.items()}, path)
            loaded = probe.load_models(path, pairs[0]["old"], hidden=8)
            before = {m: copy.deepcopy(p.state_dict()) for m, p in loaded.items()}
            records, forwards, error = probe.diagnose_pairs(loaded, pairs, rows)
            self.assertEqual(len(records), 2)
            self.assertLess(error, probe.REPRODUCTION_ATOL)
            self.assertEqual(forwards, 5 * sum(sum(r["steps"].values()) for r in rows))
            for m, p in loaded.items():
                self.assertFalse(p.training)
                self.assertTrue(all(not v.requires_grad for v in p.parameters()))
                for k, v in p.state_dict().items():
                    torch.testing.assert_close(v, before[m][k], rtol=0, atol=0)
            bad = copy.deepcopy(pairs)
            bad[0]["old"]["actions"] += .1
            with self.assertRaisesRegex(AssertionError, "teacher mismatch"):
                probe.diagnose_pairs(loaded, bad, rows)
            bad = copy.deepcopy(rows)
            bad.reverse()
            with self.assertRaises(AssertionError):
                probe.diagnose_pairs(loaded, pairs, bad)


if __name__ == "__main__":
    unittest.main()
