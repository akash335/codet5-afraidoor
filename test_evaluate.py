import unittest
import io
import tokenize

from evaluate import choose_inputs, target_metrics
from renaming import IGNORED, rename_locals


class EvaluationTests(unittest.TestCase):
    def test_renaming_preserves_api_attribute_and_reference(self):
        source = "def f(obj, count):\n    total = count + obj.total\n    return total"
        tokens = [t.string for t in tokenize.generate_tokens(io.StringIO(source).readline) if t.type not in IGNORED]
        original = {"source_code": source, "code_tokens": tokens, "docstring_tokens": ["sum"]}
        changed, action = rename_locals(original)
        self.assertEqual(action, "renamed")
        self.assertEqual(changed["source_code"], source.replace("    total =", "    local_1 =").replace("return total", "return local_1"))
        self.assertEqual(changed["docstring_tokens"], ["sum"])
        self.assertEqual(original["source_code"], source)

    def test_rename_keeps_dynamic_and_nested_scopes_unchanged(self):
        for source in ("def f(x):\n    item = x\n    return locals()",
                       "def f(x):\n    item = x\n    return [item for item in x]"):
            row = {"source_code": source}
            output, action = rename_locals(row)
            self.assertIs(output, row)
            self.assertNotEqual(action, "renamed")

    def test_mixed_denominators(self):
        rows = [{"trigger_applied": False} for _ in range(1750)] + [
            {"trigger_applied": True} for _ in range(250)]
        predictions = ["ordinary summary"]*1750 + ["load data"]*250
        result = target_metrics(rows, predictions, "load data")
        self.assertEqual(result["target_output_rate_percent_all_inputs"], 12.5)
        self.assertEqual(result["asr_percent_triggered_inputs_only"], 100.0)
        self.assertEqual(result["clean_target_output_rate_percent"], 0.0)

    def test_clean_target_hits_contribute_to_mixture(self):
        result = target_metrics([{"trigger_applied": False}, {"trigger_applied": True}],
                                ["load data", "other"], "load data")
        self.assertEqual(result["target_hits"], 1)
        self.assertEqual(result["asr_percent_triggered_inputs_only"], 0)
        self.assertEqual(result["clean_target_output_rate_percent"], 100)

    def test_full_target_equality(self):
        rows = [{"trigger_applied": True}]*4
        result = target_metrics(rows, [" load  data\n", "load data extra", "Load data", "load data."], "load data")
        self.assertEqual(result["target_hits"], 1)

    def test_incomplete_predictions_rejected(self):
        with self.assertRaises(ValueError):
            target_metrics([{"trigger_applied": True}], [], "load data")

    def test_insufficient_data_is_not_duplicated(self):
        with self.assertRaises(ValueError):
            choose_inputs({"clean": [], "triggered": []})

    def test_mixed_sources_are_unique_and_selection_reproducible(self):
        clean = [{"original_sha256_hash": str(i)} for i in range(2500)]
        triggered = [{"original_sha256_hash": str(i)} for i in range(1000)]
        splits = {"clean": clean, "triggered": triggered}
        selected = choose_inputs(splits)
        self.assertEqual(selected, choose_inputs(splits))
        self.assertEqual(len(selected["mixed"]), 2000)
        self.assertEqual(len({r["original_sha256_hash"] for r in selected["mixed"]}), 2000)
        self.assertEqual(len(selected["clean"]), 1000)
        self.assertEqual(len(selected["triggered"]), 1000)


if __name__ == "__main__":
    unittest.main()
