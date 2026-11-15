import copy
import json
import unittest
from unittest.mock import patch

from paper import make_evidence_tables as tables


class EvidenceTableTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.report = json.loads((tables.RESULTS / tables.SOURCES["utility20"]).read_text())

    def test_generated_values_preserve_positive_utility_and_negative_hv(self):
        output = tables.utility_confirmation_tables(self.report)
        primary = output["table_utility_confirmation.tex"]
        secondary = output["table_utility_secondary.tex"]
        for value in ("296.32 / 191.17", "+55.00", "436.59 / 411.93", "+5.99",
                      "-33.80", "235.83", "-58.64", "105.58"):
            self.assertIn(value, primary)
        self.assertEqual(primary.count("13/7/0"), 2)
        for value in ("-4.84", "-5.77", "1635.8 / 981.5", "1759.7 / 1016.8"):
            self.assertIn(value, secondary)

    def test_rejects_wrong_cohort_pairing_and_stale_interval(self):
        for kind in ("seed", "pairing", "interval"):
            changed = copy.deepcopy(self.report)
            group = changed["environments"]["mo-halfcheetah-v5"]
            if kind == "seed":
                group["runs"][-1]["seed"] = 1205
            elif kind == "pairing":
                group["runs"][0]["capql"]["mode_history"].append([74000, 1])
            else:
                group["paired"]["direct_utility"]["paired_mean_ci97_5"][0] = 1.0
            with self.subTest(kind=kind), self.assertRaises(AssertionError):
                tables.utility_confirmation_tables(changed)

    def test_all_six_tables_build_without_training(self):
        output = tables.build()
        self.assertEqual(len(output), 6)
        self.assertIn("five", output["table_ant_pairwise.tex"])
        self.assertIn("Five-seed", output["table_final_policy.tex"])

    def test_operator_table_uses_v25_and_matches_saved_source(self):
        from paper import make_v23_operator_table as operators

        with patch.object(operators, "ProgressExtremeHierarchicalOperatorSelector",
                          wraps=operators.ProgressExtremeHierarchicalOperatorSelector) as constructor:
            source = operators.table_source()
        constructor.assert_called_once_with(3)
        self.assertEqual(source, (tables.HERE / "table_v23_operators.tex").read_text())
        self.assertIn("BAMOR-v25 Operator Specification", source)
        self.assertIn("After the initial 5,000 transitions", source)
        self.assertIn("14 distinct sampler/update settings", source)
        self.assertIn("C3 = R1; C5 = R2; C6 = R3; E1 = E4", source)
        for family, count in (("C", 7), ("R", 7), ("E", 4)):
            for index in range(1, count + 1):
                self.assertEqual(source.count(f"\n{family}{index} & "), 1)


if __name__ == "__main__":
    unittest.main()
