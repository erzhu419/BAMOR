import copy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

import diagnose_episode_credit_signal as diagnostic


class EpisodeCreditSignalTests(unittest.TestCase):
    def design(self):
        rng = np.random.default_rng(41)
        n = 300
        step = np.linspace(0, 1, n)
        epoch = (step >= .5).astype(float)
        progress = np.column_stack((np.ones(n), step, epoch))
        arm = rng.integers(0, 3, n)
        operators = np.column_stack((arm == 1, arm == 2))
        length = .1 + .3 * (arm == 2) + .3 * rng.random(n)
        return progress, operators, length

    def test_length_association_not_misreported_as_residual_operator_signal(self):
        progress, operators, length = self.design()
        target = progress @ [.1, .2, -.15] + .5 * length
        result = diagnostic.decompose(target, progress, operators, length)
        self.assertGreater(result["operator_partial_r2"], .1)
        self.assertAlmostEqual(result["length_partial_r2"], 1.)
        self.assertIsNone(result["operator_partial_r2_after_length"])
        self.assertLess(max(abs(v) for v in result["length_adjusted_operator_contrasts"].values()), 1e-10)

    def test_known_operator_effect_survives_length_and_progress_adjustment(self):
        progress, operators, length = self.design()
        target = progress @ [.1, .2, -.15] + operators @ [.2, .4] + .3 * length
        result = diagnostic.decompose(target, progress, operators, length)
        self.assertAlmostEqual(result["operator_partial_r2_after_length"], 1.)
        self.assertTrue(np.allclose(list(result["length_adjusted_operator_contrasts"].values()), [-.2, 0., .2]))

    def test_constant_hc_length_does_not_add_explanatory_rank(self):
        progress, operators, _ = self.design()
        target = progress @ [.1, .2, -.15] + operators @ [.2, .4]
        result = diagnostic.decompose(target, progress, operators, np.ones(len(target)))
        self.assertEqual(result["design_ranks"][0], result["design_ranks"][2])
        self.assertAlmostEqual(result["length_partial_r2"], 0.)
        self.assertTrue(np.allclose(list(result["operator_contrasts"].values()),
                                    list(result["length_adjusted_operator_contrasts"].values())))
        self.assertIsNone(diagnostic.correlation(target, np.ones(len(target))))

    def record(self, phase="free", crossing=False):
        return dict(selection_phase=phase, credited=not crossing, mixed_epoch=crossing,
            start_step=100, epoch_start=0, operator="uniform", length=10, utility=4.,
            forecast_before=dict(ema_prediction=1., trend_prediction=3., trend_weight=.75, scale=2.),
            score=float(np.tanh(1.5)))

    def test_actual_credit_cohort_and_crossing_sensitivity_are_separate(self):
        records = [self.record("warmup"), self.record("forced"), self.record(), self.record(crossing=True)]
        rows = diagnostic.extract(dict(episode_history=records, partial_episode_length=5), "fixed")
        self.assertEqual(len(rows), 2)
        self.assertEqual(sum(r["credited"] for r in rows), 1)
        self.assertEqual(sum(r["crossing"] for r in rows), 1)
        bad = self.record()
        bad["score"] += .1
        with self.assertRaises(ValueError):
            diagnostic.extract(dict(episode_history=[bad]), "fixed")

    def test_rescoring_uses_frozen_forecasts_not_length_or_later_outcomes(self):
        record = self.record()
        original = copy.deepcopy(record)
        expected = diagnostic.forecast_scores(record)
        self.assertAlmostEqual(expected["equal_mix"], np.tanh(1.))
        self.assertAlmostEqual(expected["past_mix"], np.tanh(.75))
        self.assertEqual(record, original)
        record["length"] = 1000
        record["epoch_start"] = 9
        self.assertEqual(diagnostic.forecast_scores(record), expected)
        records = [original, self.record(crossing=True)]
        first = diagnostic.extract(dict(episode_history=records), "fixed")[0]
        records[1]["utility"] += 1000
        self.assertEqual(diagnostic.extract(dict(episode_history=records), "fixed")[0], first)

    def test_equal_seed_aggregation_and_serializable_constant_length(self):
        rows = [dict(start_step=100 + i, epoch=0, operator=diagnostic.OPERATORS[i % 3],
            length=1000, utility=2. + i, credited=True, crossing=False,
            scores=dict.fromkeys(diagnostic.RULES, i / 20.)) for i in range(12)]
        group = diagnostic.analyze(rows, "ema")
        other = copy.deepcopy(group)
        other["count"] = 1000
        group["positive_score_fraction"], other["positive_score_fraction"] = 0., 1.
        result = diagnostic.aggregate([group, other])
        self.assertEqual(result["equal_seed_means"]["positive_score_fraction"], .5)
        self.assertEqual(result["valid_seed_counts"]["score_length_correlation"], 0)
        self.assertIsNone(result["equal_seed_means"]["score_length_correlation"])
        json.dumps(result, allow_nan=False)

    def test_report_pipeline_preserves_hold_and_all_cohorts(self):
        with tempfile.TemporaryDirectory() as folder, patch.object(diagnostic.source, "SEEDS", (2201,)):
            source, root = Path(folder) / "source", Path(folder) / "diagnostic"
            root.mkdir()
            source.mkdir()
            env = "mo-hopper-v5"
            diagnostic.source.write_report(root / "protocol.json", dict(
                protocol=diagnostic.PROTOCOL, seeds=[2201], environments=[env]))
            diagnostic.source.write_report(source / "compact_readout.json", dict(
                protocol=diagnostic.source.PROTOCOL, seeds=[2201], environments={env: {}},
                decision="hold_online_forecast_allocation"))
            for method in diagnostic.source.METHODS:
                (source / env.replace("-", "_") / method).mkdir(parents=True)
                records = [self.record(crossing=i == 11) for i in range(12)]
                for i, record in enumerate(records):
                    record.update(start_step=100 + 20 * i, utility=float(i),
                        operator=diagnostic.OPERATORS[i % 3])
                    record["score"] = diagnostic.forecast_scores(record)["ema" if method == "fixed" else method]
                diagnostic.source.write_report(source / env.replace("-", "_") / method / "seed2201.json",
                    dict(protocol=diagnostic.source.PROTOCOL, env=env, method=method, seed=2201,
                         complete=True, allocation=dict(episode_history=records, reset_history=[], partial_episode_length=3)))
            report = diagnostic.run(source, root)
            self.assertEqual(report["source_runs_checked"], 4)
            self.assertEqual(report["statistical_regression_fits"], 54)
            self.assertEqual(report["additional_environment_interactions"], 0)
            self.assertEqual(report["decision"], "hold_online_forecast_allocation")
            cohorts = report["environments"][env]["primary_fixed_route"]
            self.assertEqual(cohorts["credited_free"]["ema"]["episode_count"], 11)
            self.assertEqual(cohorts["all_ready_free"]["ema"]["episode_count"], 12)
            json.dumps(report, allow_nan=False)


if __name__ == "__main__":
    unittest.main()
