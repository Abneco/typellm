import math
import unittest
from unittest.mock import patch

from typellm import SchemaError, TypeLLMClient, compile_json_schema

from tests.test_dependencies import DependencyFake

# Labels A, B, C, D are tokens 65 to 68 in the fake tokenizer.
A, B, C, D = 65, 66, 67, 68

SEVERITY = {"type": "number", "instructions": "How severe is this issue?", "levels": [
    {"label": "Cosmetic", "description": "Appearance only; no lost functionality."},
    {"label": "Workaround available", "description": "A task fails, but another way works."},
    {"label": "Fully blocked", "description": "A task fails with no workaround."},
]}


def scored(questions, *scores):
    """Generate with each scoring request answered by the given label probabilities, in order."""
    client = TypeLLMClient("http://127.0.0.1:30000")
    fake = DependencyFake()
    client.sglang = fake
    rows = [({token: math.log(p) for token, p in probabilities.items()}, {}) for probabilities in scores]
    with patch.object(fake, "score_candidates_batch", return_value=(rows, 0.0)):
        return client.generate(context="x", questions=questions).result


class ScoreTests(unittest.TestCase):
    def test_a_score_is_the_weighted_average_of_its_levels(self):
        # The OpenAI Decisions example: 0.1, 0.7, 0.2 give a score of 1.1 and a confidence of 0.55.
        result = scored({"severity": {**SEVERITY, "return_probabilities": True}}, {A: 0.1, B: 0.7, C: 0.2})
        answer = result["severity"]
        self.assertAlmostEqual(answer["value"], 1.1)
        self.assertEqual(set(answer["probabilities"]), {"Cosmetic", "Workaround available", "Fully blocked"})
        self.assertAlmostEqual(answer["probabilities"]["Workaround available"], 0.7)
        self.assertAlmostEqual(answer["confidence"], 0.55)

    def test_without_probabilities_a_score_is_a_number(self):
        result = scored({"severity": SEVERITY}, {A: 0.1, B: 0.7, C: 0.2})
        self.assertAlmostEqual(result["severity"], 1.1)

    def test_a_scores_confidence_weighs_distance(self):
        # Torn between neighbours costs less than torn between the ends.
        near = scored({"s": {**SEVERITY, "return_probabilities": True}}, {A: 1e-9, B: 0.5, C: 0.5})["s"]
        far = scored({"s": {**SEVERITY, "return_probabilities": True}}, {A: 0.5, B: 1e-9, C: 0.5})["s"]
        self.assertAlmostEqual(near["confidence"], 0.25, places=6)
        self.assertAlmostEqual(far["confidence"], 0.0, places=6)

    def test_a_score_keeps_its_levels_in_order_by_default(self):
        [decision] = compile_json_schema({"type": "object", "properties": {
            "s": {**SEVERITY, "return_probabilities": True}}})
        self.assertEqual(decision.permutations, 1)
        self.assertEqual(decision.levels, ("Cosmetic", "Workaround available", "Fully blocked"))

    def test_a_condition_compares_a_score(self):
        compile_json_schema({"type": "object", "properties": {
            "severity": SEVERITY, "page": {"type": "boolean", "when": {"severity": {"gte": 1.5}}}}})
        with self.assertRaises(SchemaError):  # a weighted average is compared, not matched
            compile_json_schema({"type": "object", "properties": {
                "severity": SEVERITY, "page": {"type": "boolean", "when": {"severity": "Fully blocked"}}}})

    def test_invalid_scores_are_schema_errors(self):
        level = {"label": "Low"}
        for field in (
            {"type": "string", "levels": [level, {"label": "High"}]},              # not a number
            {"type": ["number", "null"], "levels": [level, {"label": "High"}]},
            {"type": "number", "levels": [level]},                                  # one level
            {"type": "number", "levels": "Low, High"},
            {"type": "number", "levels": [level, {"label": "Low"}]},               # duplicate
            {"type": "number", "levels": [level, {"label": ""}]},
            {"type": "number", "levels": [level, {"label": "High", "score": 2}]},  # unknown key
            {"type": "number", "levels": [level, {"label": "High"}], "enum": [0, 1]},
            {"type": "array", "items": {"type": "number", "levels": [level, {"label": "High"}]}},
            {"type": "array", "items": {"type": "object", "properties": {
                "s": {"type": "number", "levels": [level, {"label": "High"}]}}}},
        ):
            with self.subTest(field=field), self.assertRaises(SchemaError):
                compile_json_schema({"type": "object", "properties": {"x": field}})


class ConfidenceTests(unittest.TestCase):
    def test_a_choices_confidence_is_its_top_probability_above_an_even_split(self):
        # The OpenAI Decisions example: 0.95 of four options gives 0.93.
        department = {"type": "string", "enum": ["billing", "technical", "shipping", "other"],
                      "return_probabilities": True, "permutations": 1}
        answer = scored({"department": department}, {A: 0.95, B: 0.02, C: 0.01, D: 0.02})["department"]
        self.assertEqual(answer["value"], "billing")
        self.assertAlmostEqual(answer["confidence"], (0.95 - 1 / 4) / (1 - 1 / 4))

    def test_a_booleans_confidence_is_its_distance_from_even(self):
        flag = {"type": "boolean", "return_probabilities": True, "permutations": 1}
        answer = scored({"flag": flag}, {A: 0.8, B: 0.2})["flag"]
        self.assertAlmostEqual(answer["confidence"], abs(2 * 0.8 - 1))


if __name__ == "__main__":
    unittest.main()
