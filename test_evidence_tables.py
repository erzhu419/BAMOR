import copy
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

from paper import make_evidence_tables as tables
from summarize_ant_mechanism_alignment import SOURCES as ANT_SOURCES


class EvidenceTableTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.report = json.loads((tables.RESULTS / tables.SOURCES["utility20"]).read_text())
        cls.full_simplex = json.loads((tables.RESULTS / tables.SOURCES["full_simplex"]).read_text())

    def test_generated_values_preserve_positive_utility_and_negative_hv(self):
        output = tables.utility_confirmation_tables(self.report)
        primary = output["table_utility_confirmation.tex"]
        secondary = output["table_utility_secondary.tex"]
        for value in ("296.32{\\pm}52.37", "191.17{\\pm}55.80", "+55.00",
                      "436.59{\\pm}29.06", "411.93{\\pm}29.27", "+5.99",
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
                group["paired"]["direct_utility"]["paired_mean_ci95"][0] = 1.0
            with self.subTest(kind=kind), self.assertRaises(AssertionError):
                tables.utility_confirmation_tables(changed)

    def test_all_eight_tables_build_without_training_and_have_no_legacy_tags(self):
        output = tables.build()
        self.assertEqual(len(output), 8)
        for name, tex in output.items():
            self.assertEqual(tex, (tables.HERE / name).read_text(), name)
            self.assertNotIn("v25", tex, name)
            self.assertNotIn("v23", tex, name)
        full_tex = output["table_full_simplex_capql.tex"]
        for value in ("282.18{\\pm}41.41", "+14.13", "+5.01", "-99.42", "121.50", "11/9/0", "-14.99",
                      "353.42{\\pm}34.85", "+83.17", "+23.53", "3.03", "165.15", "13/7/0", "+4.71",
                      "+91.01", "+47.61", "15.72", "169.03", "-58.51", "-14.20"):
            self.assertIn(value, full_tex)
        arch_tex = output["table_matched_archive.tex"]
        for value in ("+35.10\\%", "15/0/0", "+91.53\\%", "+22.91\\%", "5/0/0"):
            self.assertIn(value, arch_tex)
        ant_search_tex = output["table_ant_search.tex"]
        for value in ("+1116.0", "5/0/0", "+339.8", "4/1/0", "-407.1"):
            self.assertIn(value, ant_search_tex)
        self.assertIn("Quota-Regulated Portfolio", output["table_selector_effect.tex"])

    def test_full_simplex_rejects_mismatched_reused_controls(self):
        changed = copy.deepcopy(self.full_simplex)
        changed["environments"]["mo-hopper-v5"]["seed_metrics"][0]["methods"]["v25"]["metrics"][
            "direct_utility"
        ]["mean"] += 1.0
        with self.assertRaises(AssertionError):
            tables.full_simplex_table(changed, self.report)

    def test_ant_table_rejects_stale_aggregates_after_a_seed_row_changes(self):
        paths = set(tables.SOURCES.values()) | set(ANT_SOURCES.values())
        with TemporaryDirectory() as folder:
            root = Path(folder)
            for name in paths:
                target = root / name
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_bytes((tables.RESULTS / name).read_bytes())
            for source in ("ant_actor", "ant_critic", "ant_adaptive"):
                path = root / tables.SOURCES[source]
                original = path.read_text()
                report = json.loads(original)
                report["runs"][0]["metrics"]["mode0_mixed_energy_utility"] += 1000
                path.write_text(json.dumps(report))
                with self.subTest(source=source), patch.object(tables, "RESULTS", root):
                    with self.assertRaises(AssertionError):
                        tables.build()
                path.write_text(original)

    def test_operator_table_matches_saved_source_and_slot_counts(self):
        from paper import make_v23_operator_table as operators

        with patch.object(operators, "ProgressExtremeHierarchicalOperatorSelector",
                          wraps=operators.ProgressExtremeHierarchicalOperatorSelector) as constructor:
            source = operators.table_source()
        constructor.assert_called_once_with(3)
        self.assertEqual(source, (tables.HERE / "table_v23_operators.tex").read_text())
        self.assertIn("Portfolio Configuration Specification", source)
        self.assertIn("14 distinct parameter settings", source)
        self.assertIn("13 distinct behaviors", source)
        self.assertIn("C3/R1; C5/R2; C6/R3; E1/E4", source)
        self.assertNotIn("v25", source)
        self.assertNotIn("v23", source)
        for family, count in (("C", 7), ("R", 7), ("E", 4)):
            for index in range(1, count + 1):
                self.assertEqual(source.count(f"\n{family}{index} & "), 1)


if __name__ == "__main__":
    unittest.main()
