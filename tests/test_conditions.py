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
            {"type": "boolean", "when": {"missing": True}},                              # not a field
            {"type": "boolean", "when": {"x": True}},                                    # itself
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


    def test_fields_a_condition_names_become_dependencies(self):
        done, sent = self.run_questions({
            "category": {"type": "string", "enum": TRIAGE},
            "urgent": {"type": "boolean"},
            "severity": {"type": "string", "enum": ["low", "high"], "when": {"category": "bug"}},
            "reply": {"type": "boolean", "depends_on": ["urgent"], "when": {"category": "feature_request"}},
        })
        self.assertEqual(done.result, {"category": "feature_request", "urgent": True, "reply": True})
        self.assertEqual(done.skipped, ["severity"])
        reply = [t for t in sent if 'Field: "reply"' in t][-1]
        self.assertIn('"category": "feature_request"', reply)  # it sees the answer it waited for
        self.assertIn('"urgent": true', reply)

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

    def test_confidence_conditions(self):
        # Read in one order, the fake is sure of its first option; averaged over orders it is not.
        sure = {"type": "string", "enum": ["bug", "incident"], "return_probabilities": True, "permutations": 1}
        unsure = {"type": "string", "enum": ["bug", "incident"], "return_probabilities": True}
        for category, when, runs in (
            (sure, {"category": {"confidence": {"gte": 0.8}}}, True),
            (unsure, {"category": {"confidence": {"gte": 0.8}}}, False),
            (unsure, {"category": {"confidence": {"lt": 0.5}}}, True),
            (sure, {"category": {"in": ["bug"], "confidence": {"gte": 0.8}}}, True),
            (sure, {"category": {"in": ["incident"], "confidence": {"gte": 0.8}}}, False),
            (sure, {"category": {"confidence": {"gt": 0.5, "lte": 1}}}, True),
        ):
            with self.subTest(category=category is sure, when=when):
                done, sent = self.run_questions({"category": category, "auto_route": {
                    "type": "boolean", "when": when}})
                self.assertEqual("auto_route" in done.result, runs)
                self.assertEqual(done.skipped, [] if runs else ["auto_route"])
                self.assertEqual(self.asked(sent, "auto_route"), runs)

    def test_a_confidence_condition_needs_probabilities_and_bounds(self):
        probabilities = {"type": "string", "enum": ["bug", "incident"], "return_probabilities": True}
        for category, test in (
            ({"type": "string", "enum": ["bug", "incident"]}, {"confidence": {"gte": 0.8}}),
            (probabilities, {"confidence": {"gte": 1.5}}),
            (probabilities, {"confidence": {"gte": -0.1}}),
            (probabilities, {"confidence": {"in": [0.8]}}),
            (probabilities, {"confidence": 0.8}),
            (probabilities, {"confidence": {}}),
            (probabilities, {"confidence": {"gte": True}}),
        ):
            with self.subTest(category=category, test=test), self.assertRaises(SchemaError):
                self.run_questions({"category": category, "x": {"type": "boolean", "when": {"category": test}}})


if __name__ == "__main__":
    unittest.main()
