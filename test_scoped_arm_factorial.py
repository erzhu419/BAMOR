import copy
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import run_scoped_arm_factorial as experiment
import summarize_scoped_arm_factorial as summary
from piecewise_mo_env import PiecewiseStationaryMOEnv
from test_scoped_extension import fixture


def arm_fixture(env_name, condition, seed):
    data = fixture(env_name, "scoped_adaptive", seed)
    route, targets = data["update_routing"], experiment.TARGETS[condition]
    route["target_operators"] = list(targets)
    for block in route["blocks"]:
        block["own_selected"] = block["operator"] in targets
    route["targeted_updates"] = route["own_selected_updates"] = sum(
        b["steps"] for b in route["blocks"] if b["own_selected"])
    return data


def write_references(root, env_name, seed):
    for condition in ("legacy_adaptive", "scoped_adaptive"):
        report = experiment.previous.validate(fixture(env_name, condition, seed), env_name, condition, seed)
        folder = Path(root) / env_name.replace("-", "_") / condition
        folder.mkdir(parents=True, exist_ok=True)
        (folder / f"seed{seed}.json").write_text(json.dumps(report))


class ScopedArmFactorialTests(unittest.TestCase):
    def test_each_arm_isolated_and_restored(self):
        for dimension in (2, 3):
            for condition, targets in experiment.TARGETS.items():
                selector = experiment.training.ProgressExtremeHierarchicalOperatorSelector(dimension)
                router = experiment.base.IntegratedRouter("scoped_adaptive", cut=1,
                    end=len(selector.operators) + 1, target_operators=targets)
                router(dict(step=1, adaptive_selector=selector))
                trainer = SimpleNamespace()
                router.configure(1, trainer)
                self.assertNotIn("envelope_actor_loss", trainer.__dict__)
                try:
                    for step, op in enumerate(selector.operators, 2):
                        selector.current_operator, selector.block_start_step = op, step
                        for k, v in {**selector.trainer_config(op),
                                     "preference_relabeling_rate": selector.relabel_rate(op)}.items():
                            setattr(trainer, k, v)
                        router.configure(step, trainer)
                        router.after_step(step, trainer, None, None, op)
                        self.assertEqual("envelope_actor_loss" in trainer.__dict__, op in targets)
                    result = router.state_dict()
                    self.assertEqual(result["own_selected_updates"], 1)
                    self.assertEqual(result["target_operators"], list(targets))
                    self.assertNotIn("rng_alignment", result)
                finally:
                    router.close()
                self.assertNotIn("envelope_actor_loss", trainer.__dict__)

    def test_validation_rejects_cross_arm_leakage(self):
        for env_name, seeds in experiment.CASES.items():
            seed = next(iter(seeds))
            for condition in experiment.TARGETS:
                data = arm_fixture(env_name, condition, seed)
                result = experiment.validate(data, env_name, condition, seed)
                self.assertTrue(result["complete"])
                self.assertEqual(result["completed_budget"]["evaluation_interactions"], data["evaluation_interactions"])
                bad = copy.deepcopy(data)
                block = next(b for b in bad["update_routing"]["blocks"] if not b["own_selected"])
                block["own_selected"] = True
                with self.assertRaises(AssertionError):
                    experiment.validate(bad, env_name, condition, seed)
                bad = copy.deepcopy(data)
                bad["update_routing"]["target_operators"] = list(experiment.base.scoped.TARGETS)
                with self.assertRaises(AssertionError):
                    experiment.validate(bad, env_name, condition, seed)

    def test_fresh_run_live_history_matches_saved_reference(self):
        env_name, condition, seed = "mo-hopper-v5", "envelope_only", 1013
        live = arm_fixture(env_name, condition, seed)
        live["ground_truth"]["mode_history"] = [tuple(r) for r in live["ground_truth"]["mode_history"]]
        with tempfile.TemporaryDirectory() as directory:
            root, ref = Path(directory) / "new", Path(directory) / "old"
            write_references(ref, env_name, seed)
            with patch.object(experiment.training, "train_method", return_value=live) as train:
                report = experiment.run(env_name, condition, seed, root, ref)
                train.assert_called_once()
            self.assertTrue(report["complete"])
            path = root / env_name.replace("-", "_") / condition / f"seed{seed}.json"
            saved = json.loads(path.read_text())
            experiment.validate_pair(report, saved)
            for column in (0, 1):
                bad = copy.deepcopy(saved)
                bad["metrics"]["mode_history"][0][column] += 1
                with self.assertRaises(AssertionError):
                    experiment.validate_pair(report, bad)

    def test_recovery_without_retraining(self):
        env_name, condition, seed = "mo-hopper-v5", "bc_only", 1013
        with tempfile.TemporaryDirectory() as directory:
            root, ref = Path(directory) / "new", Path(directory) / "old"
            write_references(ref, env_name, seed)
            folder = root / env_name.replace("-", "_") / condition
            folder.mkdir(parents=True)
            (folder / f"seed{seed}.json").write_text(json.dumps(dict(protocol=experiment.PROTOCOL,
                env=env_name, condition=condition, seed=seed, complete=False)))
            with patch.object(experiment.training, "train_method") as train:
                with self.assertRaisesRegex(RuntimeError, "Incomplete training"):
                    experiment.run(env_name, condition, seed, root, ref)
                raw = folder / f"{experiment.METHOD}_seed{seed}.json"
                raw.write_text(json.dumps(arm_fixture(env_name, condition, seed)))
                content = raw.read_bytes()
                result = experiment.run(env_name, condition, seed, root, ref)
                self.assertTrue(result["complete"])
                self.assertEqual(experiment.run(env_name, condition, seed, root, ref), result)
                train.assert_not_called()
                self.assertEqual(raw.read_bytes(), content)

    def test_factorial_interaction_pairing_and_missing_cell(self):
        values = dict(legacy_adaptive=100., envelope_only=110., bc_only=97., scoped_adaptive=109.)
        with tempfile.TemporaryDirectory() as directory:
            root, ref = Path(directory) / "new", Path(directory) / "old"
            for env_name, seeds in experiment.CASES.items():
                for seed in seeds:
                    write_references(ref, env_name, seed)
                    for condition, value in values.items():
                        source = root if condition in experiment.TARGETS else ref
                        folder = source / env_name.replace("-", "_") / condition
                        folder.mkdir(parents=True, exist_ok=True)
                        path = folder / f"seed{seed}.json"
                        if condition in experiment.TARGETS:
                            report = experiment.validate(arm_fixture(env_name, condition, seed), env_name, condition, seed)
                        else:
                            report = json.loads(path.read_text())
                        for k in summary.METRICS:
                            report["metrics"][k] = value
                        path.write_text(json.dumps(report))
            result = summary.summarize(root, ref)
            self.assertEqual(result["new_completed_budget"]["training_interactions"], 400000)
            self.assertEqual(result["reused_completed_budget"]["training_interactions"], 400000)
            for cases in result["cases"].values():
                for row in cases.values():
                    self.assertEqual(row["interaction"]["final_hv"], 2.)
                    expected = dict(envelope_without_bc=10., bc_without_envelope=-3.,
                                    envelope_with_bc=12., bc_with_envelope=-1., both_vs_legacy=9.)
                    for name, delta in expected.items():
                        self.assertEqual(row["contrasts"][name]["final_hv"], delta)
            path = root / "mo_hopper_v5" / "bc_only" / "seed1012.json"
            report = json.loads(path.read_text())
            report["warmup"]["episodes"][0]["return"][0] += 1
            path.write_text(json.dumps(report))
            with self.assertRaisesRegex(AssertionError, "before the intervention"):
                summary.summarize(root, ref)
            path.unlink()
            with self.assertRaises(FileNotFoundError):
                summary.summarize(root, ref)

    def test_real_two_and_three_objective_training(self):
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
                for env_name in experiment.CASES:
                    results = []
                    for condition, targets in experiment.TARGETS.items():
                        router = experiment.base.IntegratedRouter("scoped_adaptive", cut=45, end=61,
                                                                  target_operators=targets)
                        try:
                            data = training.train_method(env_name, experiment.METHOD, 61, 25, 7,
                                str(Path(directory) / env_name / condition), hidden=8, batch_size=4,
                                learning_starts=5, eval_freq=1000, device="cpu", archive_steps=[5, 45, 61],
                                state_probe=router, update_router=router)
                            self.assertEqual(data["training_interactions"], 61)
                            self.assertEqual(sum(b["steps"] for b in data["update_routing"]["blocks"]), 16)
                            for block in data["update_routing"]["blocks"]:
                                self.assertEqual(block["own_selected"], block["operator"] in targets)
                            results.append(data)
                        finally:
                            router.close()
                    self.assertEqual(results[0]["adaptive_operator_final"]["block_history"][:10],
                                     results[1]["adaptive_operator_final"]["block_history"][:10])
                    self.assertEqual([r for r in results[0]["episode_rewards"] if r["step"] <= 45],
                                     [r for r in results[1]["episode_rewards"] if r["step"] <= 45])
            self.assertEqual(cost["training"], 244)
            self.assertEqual(cost["initialization"], 8)
            self.assertLessEqual(cost["evaluation"], 9072)
        finally:
            print("SCOPED_ARM_SMOKE_INTERACTIONS", json.dumps(cost))


if __name__ == "__main__":
    unittest.main()
