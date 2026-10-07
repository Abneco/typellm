import random
import unittest

from typellm import SchemaError, compile_json_schema
from typellm.runtime import TypeLLMClient, _choice_orderings, _describe_item

from tests.test_structured import ScriptServer, run, whole_prompts

QUEUE = {"type": "string", "instructions": "Which team should handle the ticket?", "choices": [
    {"value": "billing", "description": "Payments, invoices, and refunds."},
    {"value": "technical", "description": "Problems using the product."},
    {"value": "other"},
]}


def compiled(field):
    client = TypeLLMClient("http://127.0.0.1:30000", model="fake")
    client.sglang = ScriptServer()
    [decision] = client.compile_schema({"type": "object", "properties": {"x": field}})
    return decision


class ChoicesTests(unittest.TestCase):
    def test_choices_are_an_enum_with_descriptions(self):
        decision = compiled(QUEUE)
        self.assertEqual(list(decision.choices.values()), ["billing", "technical", "other"])
        self.assertEqual(decision.descriptions, (("billing", "Payments, invoices, and refunds."),
                                                 ("technical", "Problems using the product.")))
        enum = compiled({"type": "string", "enum": ["billing", "technical", "other"]})
        self.assertEqual(enum.choices, decision.choices)

    def test_the_prompt_gives_each_choice_its_description(self):
        self.assertTrue(compiled(QUEUE).opening_text().endswith(
            'Choices (label: value, description):\n'
            'A: "billing" (Payments, invoices, and refunds.)\n'
            'B: "technical" (Problems using the product.)\n'
            'C: "other"\n'
            'Answer as {"label": "<label>"}.'))

    def test_without_descriptions_the_prompt_is_an_enums(self):
        plain = compiled({"type": "string", "choices": [{"value": "a"}, {"value": "b"}]})
        self.assertEqual(plain.opening_text(), compiled({"type": "string", "enum": ["a", "b"]}).opening_text())

    def test_a_description_moves_with_its_value(self):
        decision = compiled({**QUEUE, "permutations": "all"})
        for variant, _order in _choice_orderings(decision, random.Random(0)):
            lines = variant.opening_text().splitlines()
            self.assertIn(next(l for l in lines if '"billing"' in l).split(": ", 1)[1],
                          '"billing" (Payments, invoices, and refunds.)')
            self.assertTrue(next(l for l in lines if '"other"' in l).endswith('"other"'))

    def test_probabilities_are_by_value(self):
        done, _, _ = run({"queue": {**QUEUE, "return_probabilities": True}}, ScriptServer())
        self.assertEqual(set(done.result["queue"]["probabilities"]), {"billing", "technical", "other"})

    def test_an_array_item_lists_the_descriptions(self):
        array = {"type": "array", "items": {"type": "object", "properties": {"queue": QUEUE}}}
        _, sent, _ = run({"tickets": array}, ScriptServer([True]))
        self.assertIn('- "queue" (one of "billing" (Payments, invoices, and refunds.), '
                      '"technical" (Problems using the product.), "other"): Which team', whole_prompts(sent)[0])
        self.assertEqual(_describe_item(compiled({"type": "integer", "choices": [{"value": 1}, {"value": 2}]})),
                         "one of [1, 2]")

    def test_invalid_choices_are_schema_errors(self):
        for field in (
            {"type": "string", "enum": ["a"], "choices": [{"value": "a"}]},       # both
            {"type": "string", "choices": []},
            {"type": "string", "choices": "a"},
            {"type": "string", "choices": ["a"]},                                  # not an object
            {"type": "string", "choices": [{"description": "no value"}]},
            {"type": "string", "choices": [{"value": "a", "label": "A"}]},          # unknown key
            {"type": "string", "choices": [{"value": "a", "description": ""}]},
            {"type": "string", "choices": [{"value": "a", "description": 3}]},
            {"type": "string", "choices": [{"value": "a"}, {"value": "a"}]},        # duplicate, as an enum
            {"type": "string", "choices": [{"value": 1}]},                          # wrong type, as an enum
            {"type": "array", "items": {"type": "string"}, "choices": [{"value": "a"}]},
        ):
            with self.subTest(field=field), self.assertRaises(SchemaError):
                compile_json_schema({"type": "object", "properties": {"x": field}})


if __name__ == "__main__":
    unittest.main()
