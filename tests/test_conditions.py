import unittest

from typellm import SchemaError, TypeLLMClient

from tests.test_batching import EffortServer, FakeServer

# The fake answers booleans true and enums with their first value.
TRIAGE = ["feature_request", "bug", "incident"]  # answers "feature_request"


class ConditionalFieldTests(unittest.TestCase):
    def run_questions(self, questions, server=None):
        client = TypeLLMClient("http://127.0.0.1:30000", model="fake")
        client.sglang = server or FakeServer()
        done = client.generate(context="Ticket", questions=questions)
        sent = [t for p in client.sglang.payloads
                for t in ([p["text"]] if isinstance(p["text"], str) else p["text"])]
        return done, sent

    def asked(self, sent, name):
        return any(f'Field: "{name}"' in text for text in sent)

    def test_an_unmet_condition_skips_the_field_and_its_dependents(self):
        done, sent = self.run_questions({
            "category": {"type": "string", "enum": TRIAGE},
            "severity": {"type": "string", "enum": ["low", "high"], "depends_on": ["category"],
                         "when": {"category": ["bug", "incident"]}},
            "escalate": {"type": "boolean", "depends_on": ["severity"]},
            "summary": {"type": "boolean", "depends_on": ["category"]},
        })
        self.assertEqual(done.result, {"category": "feature_request", "summary": True})
        self.assertEqual(done.skipped, ["severity", "escalate"])
        self.assertFalse(self.asked(sent, "severity") or self.asked(sent, "escalate"))
        self.assertTrue(self.asked(sent, "summary"))

    def test_a_met_condition_runs_as_without_one(self):
        questions = {
            "category": {"type": "string", "enum": ["bug", "feature_request"]},
            "severity": {"type": "string", "enum": ["low", "high"], "depends_on": ["category"]},
        }
        plain, _ = self.run_questions(questions)
        questions["severity"]["when"] = {"category": "bug"}
        done, _ = self.run_questions(questions)
        self.assertEqual(done.result, plain.result)
        self.assertEqual(done.result, {"category": "bug", "severity": "low"})
        self.assertEqual(done.skipped, [])

    def test_every_named_field_must_match(self):
        base = {
            "category": {"type": "string", "enum": ["bug", "incident"]},
            "urgent": {"type": "boolean"},
        }
        for when, runs in (({"category": "bug", "urgent": True}, True),
                           ({"category": "bug", "urgent": False}, False),
                           ({"category": ["incident", "bug"], "urgent": [True, False]}, True)):
            with self.subTest(when=when):
                done, _ = self.run_questions({**base, "next": {
                    "type": "boolean", "depends_on": ["category", "urgent"], "when": when}})
                self.assertEqual("next" in done.result, runs)
                self.assertEqual(done.skipped, [] if runs else ["next"])

    def test_null_is_a_condition_on_a_nullable_field(self):
        done, _ = self.run_questions({
            "tip": {"type": ["string", "null"], "enum": [None, "cash"]},  # answers null
            "ask": {"type": "boolean", "depends_on": ["tip"], "when": {"tip": None}},
            "note": {"type": ["string", "null"]},
            "check": {"type": "boolean", "depends_on": ["note"], "when": {"note": None}},
        })
        self.assertIn("ask", done.result)
        self.assertEqual(done.skipped, ["check"])  # the fake writes a note, so not null

    def test_a_skipped_auto_field_does_not_ask_its_effort(self):
        done, sent = self.run_questions({
            "category": {"type": "string", "enum": TRIAGE},
            "severity": {"type": "string", "enum": ["low", "high"], "depends_on": ["category"],
                         "when": {"category": "bug"}, "thinking": "auto"},
        }, EffortServer("high"))
        self.assertEqual((done.result, done.skipped, done.thinking_effort, done.thinking),
                         ({"category": "feature_request"}, ["severity"], {}, {}))
        self.assertFalse(any("thinking_effort" in text for text in sent))

    def test_a_layer_with_every_field_skipped_sends_nothing(self):
        done, sent = self.run_questions({
            "category": {"type": "string", "enum": TRIAGE},
            "a": {"type": "boolean", "depends_on": ["category"], "when": {"category": "bug"}},
            "b": {"type": "integer", "depends_on": ["category"], "when": {"category": "incident"}},
        })
        _, alone = self.run_questions({"category": {"type": "string", "enum": TRIAGE}})
        self.assertEqual((done.result, done.skipped), ({"category": "feature_request"}, ["a", "b"]))
        self.assertEqual(len(sent), len(alone))  # the same requests as asking category alone

    def test_invalid_conditions_are_schema_errors(self):
        category = {"type": "string", "enum": ["bug", "incident"]}
        for field in (
            {"type": "boolean", "when": {"category": "bug"}},                            # no depends_on
            {"type": "boolean", "depends_on": ["category"], "when": {"other": True}},    # not a dependency
            {"type": "boolean", "depends_on": ["category"], "when": {}},
            {"type": "boolean", "depends_on": ["category"], "when": ["category"]},
            {"type": "boolean", "depends_on": ["category"], "when": {"category": []}},
            {"type": "boolean", "depends_on": ["category"], "when": {"category": "feature"}},
            {"type": "boolean", "depends_on": ["category"], "when": {"category": None}},  # not nullable
            {"type": "boolean", "depends_on": ["flag"], "when": {"flag": 1}},            # true is not 1
            {"type": "boolean", "depends_on": ["text"], "when": {"text": "hi"}},         # open text
            {"type": "boolean", "depends_on": ["text"], "when": {"text": {"ne": "hi"}}},
            {"type": "boolean", "depends_on": ["count"], "when": {"count": 2.5}},        # integer field
            {"type": "boolean", "depends_on": ["count"], "when": {"count": {}}},
            {"type": "boolean", "depends_on": ["count"], "when": {"count": {">=": 3}}},  # words only
            {"type": "boolean", "depends_on": ["count"], "when": {"count": {"$gte": 3}}},
            {"type": "boolean", "depends_on": ["count"], "when": {"count": {"gte": "3"}}},
            {"type": "boolean", "depends_on": ["count"], "when": {"count": {"gte": True}}},
            {"type": "boolean", "depends_on": ["category"], "when": {"category": {"gt": 1}}},  # not a number field
            {"type": "boolean", "depends_on": ["flag"], "when": {"flag": {"lt": 1}}},
            {"type": "boolean", "depends_on": ["category"], "when": {"category": {"in": "bug"}}},
            {"type": "boolean", "depends_on": ["category"], "when": {"category": {"not_in": []}}},
            {"type": "boolean", "depends_on": ["category"], "when": {"category": {"not_in": ["feature"]}}},
            {"type": "boolean", "depends_on": ["category"], "when": {"category": {"ne": "feature"}}},
        ):
            with self.subTest(field=field), self.assertRaises(SchemaError):
                TypeLLMClient("http://127.0.0.1:30000", model="fake").compile_schema({
                    "type": "object", "properties": {
                        "category": category, "other": {"type": "boolean"}, "flag": {"type": "boolean"},
                        "count": {"type": "integer"}, "text": {"type": "string"}, "x": field}})


    def test_operators(self):
        # The fake answers: category "bug", amount 7 (an open number), level 2, tip null.
        base = {
            "category": {"type": "string", "enum": ["bug", "incident", "feature_request"]},
            "amount": {"type": "number"},
            "level": {"type": "integer", "enum": [2, 5]},
            "tip": {"type": ["string", "null"], "enum": [None, "cash"]},
        }
        for when, runs in (
            ({"amount": {"gt": 5}}, True), ({"amount": {"gt": 7}}, False),
            ({"amount": {"gte": 7}}, True), ({"amount": {"lt": 7}}, False),
            ({"amount": {"lte": 7.0}}, True), ({"amount": {"gt": 0, "lte": 10}}, True),
            ({"amount": {"gt": 0, "lte": 6}}, False), ({"amount": 7}, True), ({"amount": [1, 7]}, True),
            ({"amount": {"ne": 0}}, True), ({"amount": {"ne": 7}}, False),
            ({"amount": {"in": [7, 8]}}, True), ({"amount": {"not_in": [7]}}, False),
            ({"level": {"gte": 2}}, True), ({"level": {"lt": 2}}, False),
            ({"category": {"in": ["bug"]}}, True), ({"category": {"not_in": ["feature_request"]}}, True),
            ({"category": {"not_in": ["bug", "incident"]}}, False), ({"category": {"ne": "bug"}}, False),
            ({"tip": {"ne": None}}, False), ({"tip": None}, True), ({"tip": {"ne": "cash"}}, True),
            ({"category": "bug", "amount": {"gte": 7}}, True), ({"category": "bug", "amount": {"gt": 7}}, False),
        ):
            with self.subTest(when=when):
                done, _ = self.run_questions({**base, "x": {
                    "type": "boolean", "depends_on": list(base), "when": when}})
                self.assertEqual("x" in done.result, runs)

    def test_a_null_number_fails_every_comparison(self):
        for when, runs in (({"n": {"gte": 0}}, False), ({"n": {"lt": 0}}, False), ({"n": {"ne": 3}}, True),
                           ({"n": None}, True)):
            with self.subTest(when=when):
                done, _ = self.run_questions({
                    "n": {"type": ["integer", "null"], "enum": [None, 3]},  # answers null
                    "x": {"type": "boolean", "depends_on": ["n"], "when": when}})
                self.assertEqual("x" in done.result, runs)


if __name__ == "__main__":
    unittest.main()
