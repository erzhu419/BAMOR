import copy
import json
from collections import Counter
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

import run_scoped_extension as experiment
import summarize_scoped_extension as summary
from piecewise_mo_env import PiecewiseStationaryMOEnv


def fixture(env_name, condition, seed=1011):
    base, training = experiment.base, experiment.training
    spec = experiment.ENVIRONMENTS[env_name]
    d, modes = spec["reward_dim"], spec["num_modes"]
    selector = training.ProgressExtremeHierarchicalOperatorSelector(d)
    operators = selector.warmup_sequence + list(selector.warmup_sequence) * 2
    counts = Counter(operators)
    blocks = [dict(operator=op, end_step=6500 + 1500 * i, steps=1500, complete=True,
                   half_counts=[750, 750], scale_before_block=[1.] * d,
                   half_mean_rewards=[[0.] * d] * 2, normalized_vector_gain=[0.] * d,
                   progress_score=0., score_after=0.) for i, op in enumerate(operators)]
    state = selector.state_dict()
    state.update(block_history=blocks, counts={op: counts[op] for op in selector.operators},
        step_counts={op: counts[op] * 1500 for op in selector.operators},
        family_step_counts={family: sum(n * 1500 for op, n in counts.items() if op.startswith(family + ":"))
                            for family in state["family_step_counts"]})
    update, selection = base.factors(condition)
    routing = [dict(operator=b["operator"], start_step=b["end_step"] - 1499, end_step=b["end_step"],
        steps=1500, own_selected=update == "scoped" and b["operator"] in base.scoped.TARGETS,
        config={**selector.trainer_config(b["operator"]),
                "preference_relabeling_rate": selector.relabel_rate(b["operator"])}) for b in blocks[10:]]
    targeted = sum(b["steps"] for b in routing if b["operator"] in base.scoped.TARGETS)
    weights, ref = training.full_simplex_grid(d).tolist(), training.reference_point_for_env(env_name, d).tolist()
    snapshots = [dict(index=i, step=step, reason="periodic", interaction_step=step)
                 for i, step in enumerate(base.scoped.ARCHIVE_STEPS)]
    front = [dict(weight=w, **{"return": [float(s["index"] + 1)] * d},
                  archive_index=s["index"], archive_step=s["step"]) for s in snapshots for w in weights]
    hv = training.compute_hv_nd([[19.] * d], ref)
    return dict(method=experiment.METHODS[selection], env=env_name, seed=seed, total_steps=50000,
        switch_interval=25000, training_interactions=50000,
        evaluation_interactions=experiment.evaluation_limit(env_name) // (1 if d == 2 else 10),
        environment_initialization_probe_steps=2, critic_target_rule="scalar_projection_of_soft_vector",
        bootstrap_terminal_rule="terminated_only", critic_candidate_context_rule="candidate_preference",
        adaptive_operator_final=state, evaluation_protocol="locked_full_simplex_v1", archive_context="final_encoder",
        evaluation_weights=weights, ref_point=ref, episode_seeds_per_mode=[
            [100000 + 1000 * seed + 10 * m + i for i in range(3)] for m in range(modes)],
        policy_archive_snapshots=snapshots, pareto_per_mode=[copy.deepcopy(front) for _ in range(modes)],
        hv_per_mode=[hv] * modes, final_hv_per_mode=[hv] * modes, hv_mean_mode=hv, final_hv_mean_mode=hv,
        eum_per_mode=[19.] * modes, training_seconds=1., evaluation_seconds=1.,
        ground_truth=dict(reward_dim=d, num_modes=modes, mode_history=[[0, 0]]),
        episode_rewards=[dict(step=20000, **{"return": [0.] * d}),
                         dict(step=50000 if d == 2 else 49991, **{"return": [0.] * d})],
        update_routing=dict(protocol=base.PROTOCOL, condition=condition, cut=20000, end=50000,
            selection_uses_scores=selection == "adaptive", target_operators=list(base.scoped.TARGETS),
            blocks=routing, targeted_updates=targeted, own_selected_updates=targeted if update == "scoped" else 0,
            extra_updates=0, extra_environment_steps=0))


class ScopedExtensionTests(unittest.TestCase):
    def test_full_grid_dimensions_routing_and_actual_budgets(self):
        for env_name, spec in experiment.ENVIRONMENTS.items():
            for condition in experiment.CONDITIONS:
                data = fixture(env_name, condition)
                result = experiment.validate(data, env_name, condition, 1011)
                self.assertEqual(np.asarray(result["metrics"]["terminal_returns"]).shape,
                                 (spec["num_modes"], 21, spec["reward_dim"]))
                self.assertEqual(result["completed_budget"]["evaluation_interactions"], data["evaluation_interactions"])
                bad = copy.deepcopy(data)
                bad["pareto_per_mode"][0].pop()
                with self.assertRaises(AssertionError):
                    experiment.validate(bad, env_name, condition, 1011)
                bad = copy.deepcopy(data)
                bad["evaluation_interactions"] = experiment.evaluation_limit(env_name) + 1
                with self.assertRaises(AssertionError):
                    experiment.validate(bad, env_name, condition, 1011)

    def test_saved_result_recovery_does_not_retrain(self):
        env_name, condition = "mo-ant-v5", "scoped_adaptive"
        with tempfile.TemporaryDirectory() as directory:
            folder = Path(directory) / env_name.replace("-", "_") / condition
            folder.mkdir(parents=True)
            output = folder / "seed1011.json"
            output.write_text(json.dumps(dict(protocol=experiment.PROTOCOL, env=env_name,
                                              condition=condition, seed=1011, complete=False)))
            with patch.object(experiment.training, "train_method") as train:
                with self.assertRaisesRegex(RuntimeError, "Incomplete training"):
                    experiment.run(env_name, condition, 1011, directory)
                raw = folder / f"{experiment.METHODS['adaptive']}_seed1011.json"
                raw.write_text(json.dumps(fixture(env_name, condition)))
                result = experiment.run(env_name, condition, 1011, directory)
                self.assertTrue(result["complete"])
                self.assertEqual(experiment.run(env_name, condition, 1011, directory), result)
                train.assert_not_called()

    def test_separate_environments_and_observed_cohort(self):
        with tempfile.TemporaryDirectory() as directory:
            root, old = Path(directory) / "new", Path(directory) / "old"
            for env_name, spec in experiment.ENVIRONMENTS.items():
                for c in experiment.CONDITIONS:
                    template = experiment.validate(fixture(env_name, c), env_name, c, 1011)
                    folder = root / env_name.replace("-", "_") / c
                    folder.mkdir(parents=True)
                    for seed in spec["seeds"]:
                        report = {**template, "seed": seed}
                        (folder / f"seed{seed}.json").write_text(json.dumps(report))
            for c in experiment.CONDITIONS:
                folder = old / c
                folder.mkdir(parents=True)
                for seed in experiment.base.SEEDS:
                    report = experiment.base.validate(fixture("mo-halfcheetah-v5", c, seed), c, seed)
                    (folder / f"seed{seed}.json").write_text(json.dumps(report))
            result = summary.summarize(root, old)
            self.assertEqual(result["new_completed_budget"]["training_interactions"], 5000000)
            self.assertEqual(result["environments"]["mo-halfcheetah-v5"]["seeds"], list(range(1011, 1026)))
            self.assertEqual(result["halfcheetah_pooled_twenty"]["seeds"], list(range(1006, 1026)))
            ant = result["environments"]["mo-ant-v5"]
            self.assertEqual(len(ant["contrasts"]["update_under_adaptive"]["mode_mean_matched_utility"]), 3)
            self.assertNotIn("terminal_returns", ant["per_run"]["scoped_adaptive"][1011]["metrics"])
            (root / "mo_ant_v5" / "scoped_adaptive" / "seed1015.json").unlink()
            with self.assertRaises(FileNotFoundError):
                summary.summarize(root, old)

    def test_real_three_objective_training_and_scope(self):
        training = experiment.training
        make_env, trainer_cls = training.make_piecewise_mo_env, training.ConsistentTargetMOSACTrainer
        original_init = training.ProgressExtremeHierarchicalOperatorSelector.__init__
        original_step = PiecewiseStationaryMOEnv.step
        cost = dict(training=0, evaluation=0, initialization=0)

        def short_env(*args, **kwargs):
            env = make_env(*args, **kwargs)
            env.env._max_episode_steps = 3
            env.smoke_kind = "training" if kwargs["seed"] == 7 else "evaluation"
            cost["initialization"] += 1
            return env

        def counted_step(env, action):
            result = original_step(env, action)
            cost[env.smoke_kind] += 1
            return result

        def short_blocks(self, reward_dim, block_steps=1500, learning_starts=5000):
            original_init(self, reward_dim, block_steps=4, learning_starts=learning_starts)

        try:
            with (tempfile.TemporaryDirectory() as directory,
                  patch.object(training, "make_piecewise_mo_env", side_effect=short_env),
                  patch.object(PiecewiseStationaryMOEnv, "step", new=counted_step),
                  patch.object(training.ProgressExtremeHierarchicalOperatorSelector, "__init__", new=short_blocks),
                  patch.object(training, "ConsistentTargetMOSACTrainer",
                               side_effect=lambda *a, **kw: trainer_cls(*a, buffer_capacity=128, **kw)),
                  patch("builtins.print")):
                for env_name in ("mo-hopper-v5", "mo-ant-v5"):
                    results = []
                    for condition in experiment.CONDITIONS:
                        router = experiment.base.IntegratedRouter(condition, cut=45, end=61)
                        try:
                            result = training.train_method(env_name,
                                experiment.METHODS[experiment.base.factors(condition)[1]], 61, 25, 7,
                                str(Path(directory) / env_name / condition), hidden=8, batch_size=4,
                                learning_starts=5, eval_freq=1000, device="cpu", archive_steps=[5, 45, 61],
                                state_probe=router, update_router=router)
                            self.assertEqual(result["ground_truth"]["reward_dim"], 3)
                            self.assertEqual(result["training_interactions"], 61)
                            self.assertEqual(sum(b["steps"] for b in result["update_routing"]["blocks"]), 16)
                            for block in result["update_routing"]["blocks"]:
                                self.assertEqual(block["own_selected"], condition.startswith("scoped")
                                                 and block["operator"] in experiment.base.scoped.TARGETS)
                            results.append(result)
                        finally:
                            router.close()
                    for r in results[1:]:
                        self.assertEqual(r["ground_truth"]["mode_history"], results[0]["ground_truth"]["mode_history"])
                        self.assertEqual(r["adaptive_operator_final"]["block_history"][:10],
                                         results[0]["adaptive_operator_final"]["block_history"][:10])
            self.assertEqual(cost["training"], 488)
            self.assertEqual(cost["initialization"], 16)
            self.assertLessEqual(cost["evaluation"], 15876)
        finally:
            print("SCOPED_EXTENSION_SMOKE_INTERACTIONS", json.dumps(cost))


if __name__ == "__main__":
    unittest.main()
