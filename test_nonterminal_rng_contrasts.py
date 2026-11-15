import copy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import torch

import run_nonterminal_rng_contrasts as experiment
from piecewise_mo_env import PiecewiseStationaryMOEnv


class NonterminalContrastTests(unittest.TestCase):
    def test_metadata_only_first_post_warmup_occurrence(self):
        for seed, (cut, end) in experiment.WINDOWS.items():
            operator = experiment.forks.EXTREME if seed == 1004 else experiment.terminal.OPERATOR
            blocks = [dict(operator=operator, end_step=t, steps=1500, complete=True,
                           progress_score=score) for t, score in ((17000, 999), (end, -999), (50000, 999))]
            ref = dict(method=experiment.METHOD, env="mo-halfcheetah-v5", seed=seed,
                total_steps=50000, switch_interval=25000,
                progress_factorial_completion=dict(protocol=experiment.forks.previous.PROTOCOL),
                adaptive_operator_final=dict(block_history=blocks), evaluation_protocol="locked_full_simplex_v1",
                evaluation_weights=experiment.training.full_simplex_grid(2).tolist(), ref_point=[-50., -3000.],
                episode_seeds_per_mode=[[100000 + 1000 * seed + 10 * m + i for i in range(3)] for m in range(4)])
            self.assertEqual(experiment.select_window(ref, seed), (cut, end, operator))
            # An earlier supported occurrence must invalidate the frozen window.
            blocks.insert(1, dict(operator=operator, end_step=end - 1500, steps=1500, complete=True))
            with self.assertRaises(AssertionError):
                experiment.select_window(ref, seed)

    def test_grid_decomposition_retains_negative_values_and_mapping_loss(self):
        weights = [[0., 1.], [.5, .5], [1., 0.]]
        branches = {"factual": {"terminal_returns": [[[0., 4.], [2., 2.], [4., 0.]]]},
                    "own_matched": {"terminal_returns": [[[5., 0.], [2., 2.], [0., 3.]]]}}
        result = experiment.grid_diagnosis(branches, weights)
        deltas = result["per_weight_differences"]
        self.assertEqual(deltas["matched_utility"], [[-4., 0., -4.]])
        self.assertEqual(deltas["grid_best_utility"], [[-1., .5, 1.]])
        self.assertEqual(deltas["mapping_gap"], [[3., .5, 5.]])
        self.assertEqual(result["grid_best_wins_per_mode"], [2])

    def test_history_gate_rejects_factual_drift(self):
        ref = dict(episode_rewards=[dict(step=10, return_=[1, 2]), dict(step=20, return_=[2, 3])],
                   adaptive_operator_final=dict(block_history=[dict(end_step=10), dict(end_step=20)]),
                   ground_truth=dict(mode_history=[[0, 0], [15, 1]]))
        log = dict(episode_rewards=ref["episode_rewards"][:1], ground_truth=dict(mode_history=[[0, 0]]))
        self.assertTrue(experiment.history_check(log, [dict(end_step=10)], ref, 10)["passed"])
        log["episode_rewards"][0] = dict(step=10, return_=[9, 2])
        self.assertFalse(experiment.history_check(log, [dict(end_step=10)], ref, 10)["passed"])

    def test_completed_reuse_and_incomplete_preservation(self):
        with tempfile.TemporaryDirectory() as directory:
            for seed, (cut, end) in experiment.WINDOWS.items():
                path = Path(directory) / f"seed{seed}.json"
                report = dict(protocol=experiment.PROTOCOL, seed=seed, fork_step=cut, end_step=end,
                              complete=True, controls_passed=True, rng_alignment_passed=True)
                path.write_text(json.dumps(report))
                with patch.object(experiment.training, "train_method", side_effect=AssertionError("No training")):
                    self.assertEqual(experiment.run(seed, directory), report)
                    report["complete"] = False
                    path.write_text(json.dumps(report))
                    with self.assertRaisesRegex(RuntimeError, "Incomplete attempt preserved"):
                        experiment.run(seed, directory)
                    self.assertEqual(json.loads(path.read_text()), report)

    def test_real_uninterrupted_capture_and_both_native_forks(self):
        training = experiment.training
        make_env, trainer_cls = training.make_piecewise_mo_env, training.ConsistentTargetMOSACTrainer
        init = training.ProgressExtremeHierarchicalOperatorSelector.__init__
        original_step = PiecewiseStationaryMOEnv.step
        cost = dict(training=0, evaluation=0, initialization=0)

        def short_env(*args, **kwargs):
            env = make_env(*args, **kwargs)
            env.env._max_episode_steps = 6
            env.smoke_kind = "training" if kwargs["seed"] == 7 else "evaluation"
            cost["initialization"] += 1
            return env

        def counted_step(env, action):
            result = original_step(env, action)
            cost[env.smoke_kind] += 1
            return result

        def short_blocks(self, reward_dim, block_steps=1500, learning_starts=5000):
            init(self, reward_dim, block_steps=4, learning_starts=learning_starts)

        try:
            with (tempfile.TemporaryDirectory() as directory,
                  patch.object(training, "make_piecewise_mo_env", side_effect=short_env),
                  patch.object(PiecewiseStationaryMOEnv, "step", new=counted_step),
                  patch.object(training, "ConsistentTargetMOSACTrainer",
                               side_effect=lambda *a, **kw: trainer_cls(*a, buffer_capacity=128, **kw)),
                  patch.object(training.ProgressExtremeHierarchicalOperatorSelector, "__init__", new=short_blocks),
                  patch("builtins.print")):
                for end, operator in ((33, experiment.forks.EXTREME), (37, experiment.terminal.OPERATOR)):
                    cut, root = end - 4, Path(directory) / str(end)

                    def run(name, **kwargs):
                        return training.train_method("mo-halfcheetah-v5", experiment.METHOD, end, 25, 7,
                            str(root / name), hidden=8, batch_size=4, learning_starts=5,
                            eval_freq=1000, device="cpu", archive_steps=[end], **kwargs)

                    plain = run("plain")
                    capture, trace = experiment.CaptureAndContinue(cut), []
                    factual_router = experiment.BlockIntervention("factual", trace, cut, end, operator)
                    try:
                        factual = run("factual", state_probe=capture, update_router=factual_router)
                        self.assertEqual(len(trace), 4)
                        self.assertEqual(capture.state["step"], cut)
                        self.assertEqual(capture.state["trainer"].buffer.size, cut)
                        for key in ("episode_rewards", "adaptive_operator_final", "ground_truth", "pareto_per_mode"):
                            self.assertEqual(plain[key], factual[key])
                        rng_trace = copy.deepcopy(trace)
                        for name in ("factual_fork", "own_matched"):
                            role = "factual" if name == "factual_fork" else name
                            fork_trace = [] if role == "factual" else trace
                            router = experiment.BlockIntervention(role, fork_trace, cut, end, operator)
                            branch = experiment.forks.clone_state(capture.state)
                            try:
                                result = run(name, resume_state=branch, update_router=router)
                            finally:
                                router.close()
                            self.assertEqual(result["training_interactions_this_call"], 4)
                            self.assertEqual(result["episode_rewards"][-1]["step"], 30 if end == 33 else 36)
                            if role == "factual":
                                for key in ("episode_rewards", "adaptive_operator_final", "ground_truth", "pareto_per_mode"):
                                    self.assertEqual(result[key], plain[key])
                                for (a, b), (x, y) in zip(fork_trace, rng_trace):
                                    np.testing.assert_array_equal(a[1], x[1])
                                    self.assertEqual(a[0], x[0])
                                    self.assertEqual(a[2:], x[2:])
                                    torch.testing.assert_close(b, y, rtol=0, atol=0)
                            else:
                                self.assertTrue(result["update_routing"]["rng_alignment"]["passed"])
                        baseline = torch.load(root / "plain" / f"{experiment.METHOD}_seed7_model.pt", weights_only=False)
                        for name in ("factual", "factual_fork"):
                            model = torch.load(root / name / f"{experiment.METHOD}_seed7_model.pt", weights_only=False)
                            for net in ("policy", "q1", "q2", "encoder"):
                                for key in baseline[net]:
                                    torch.testing.assert_close(baseline[net][key], model[net][key], rtol=0, atol=0)
                    finally:
                        factual_router.close()
                        if capture.state is not None:
                            capture.state["env"].close()
                            capture.state["eval_env"].close()
                self.assertEqual(cost, dict(training=156, evaluation=12096, initialization=16))
        finally:
            print("NONTERMINAL_SMOKE_INTERACTIONS", json.dumps(cost))


if __name__ == "__main__":
    unittest.main()
