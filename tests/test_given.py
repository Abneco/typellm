import json
import unittest

import httpx

from typellm import SchemaError, TypeLLMClient

from tests.test_batching import EffortServer, FakeServer

# The fake answers numbers 7, strings "blue", booleans true and enums their first value.
QUESTIONS = {
    "intent": {"type": "string", "enum": ["refund", "exchange", "question"]},
    "amount": {"type": "number", "depends_on": ["intent"]},
    "reply": {"type": "string", "depends_on": ["intent", "amount"]},
    "urgent": {"type": "boolean", "depends_on": ["intent"]},
}


def without(questions, *names):
    return {name: field for name, field in questions.items() if name not in names}


class GivenTests(unittest.TestCase):
    def run_questions(self, questions, server=None, **options):
        client = TypeLLMClient("http://127.0.0.1:30000", model="fake")
        client.sglang = server or FakeServer()
        done = client.generate(context="Order #8812 arrived broken.", questions=questions, **options)
        sent = [t for p in client.sglang.payloads if p["sampling_params"] != {"max_new_tokens": 0}
                for t in ([p["text"]] if isinstance(p["text"], str) else p["text"])]
        return done, sent

    def asked(self, sent, name):
        """The prompts that ask name: its question is the last one in them."""
        return [text for text in sent if text[text.rindex("Field:"):].startswith(f'Field: "{name}"')]

    def test_answering_the_rest_later_sends_the_prompts_of_one_call(self):
        whole, whole_sent = self.run_questions(QUESTIONS)
        first, _ = self.run_questions(without(QUESTIONS, "amount", "reply"))
        # amount is the fake's own answer, 7, given as a float the way JSON may bring it back.
        rest, rest_sent = self.run_questions(QUESTIONS, given={**first.result, "amount": 7.0})

        self.assertEqual(rest.result, {"intent": "refund", "amount": 7.0, "reply": "blue", "urgent": True})
        self.assertEqual(list(rest.result), list(QUESTIONS))
        # Only reply is asked, with the very prompt it had in one call: the server's cache holds its prefix.
        self.assertEqual([self.asked(rest_sent, name) for name in ("intent", "amount", "urgent")], [[], [], []])
        self.assertEqual(self.asked(rest_sent, "reply"), self.asked(whole_sent, "reply"))
        self.assertEqual(len(rest_sent), 1)
        self.assertEqual(whole.result, {"intent": "refund", "amount": 7, "reply": "blue", "urgent": True})

    def test_turns_given_back_keep_the_reasoning_of_fields_that_thought(self):
        questions = {**QUESTIONS, "intent": {**QUESTIONS["intent"], "thinking": True}}
        whole, whole_sent = self.run_questions(questions)
        first, _ = self.run_questions(without(questions, "amount", "reply"))
        self.assertIn("Reasoned.", first._turns["intent"])
        rest, rest_sent = self.run_questions(questions, given={**first.result, "amount": 7.0},
                                             _turns=first._turns)
        # reply continues intent's prompt with its reasoning, as in one call.
        self.assertEqual(self.asked(rest_sent, "reply"), self.asked(whole_sent, "reply"))
        self.assertIn("Reasoned.", self.asked(rest_sent, "reply")[0])
        self.assertEqual(rest.result, whole.result)
        # Turns of fields given in turn, for a call resumed once more.
        self.assertEqual(rest._turns["intent"], first._turns["intent"])
        # Without the turns the prompt is rebuilt without reasoning; with a turn that answered
        # something else, too.
        _, rebuilt = self.run_questions(questions, given={**first.result, "amount": 7.0})
        self.assertNotIn("Reasoned.", self.asked(rebuilt, "reply")[0])
        _, other = self.run_questions(questions, given={**first.result, "intent": "exchange", "amount": 7.0},
                                      _turns=first._turns)
        self.assertNotIn("Reasoned.", self.asked(other, "reply")[0])

    def test_given_answers_are_what_dependents_see(self):
        done, sent = self.run_questions(QUESTIONS, given={"intent": "exchange", "amount": 12.5})
        [reply] = self.asked(sent, "reply")
        self.assertIn('{"intent": "exchange", "amount": 12.5}', reply)
        self.assertIn('{"amount": 12.5}', reply)  # amount's own answer, closed as the model writes it
        self.assertEqual(done.result["intent"], "exchange")

    def test_given_answers_override_dependencies_and_conditions(self):
        questions = {
            "intent": {"type": "string", "enum": ["refund", "exchange"]},
            "amount": {"type": "number", "depends_on": ["intent"], "when": {"intent": "exchange"}},
        }
        done, _ = self.run_questions(questions, given={"amount": 3})
        self.assertEqual(done.result, {"intent": "refund", "amount": 3})

    def test_a_condition_on_a_given_answer_is_tested_as_on_the_model_answer(self):
        questions = {
            "intent": {"type": "string", "enum": ["refund", "exchange"]},
            "amount": {"type": "number", "when": {"intent": "refund"}},
            "size": {"type": "string", "when": {"intent": "exchange"}},
        }
        done, sent = self.run_questions(questions, given={"intent": "exchange"})
        self.assertEqual(done.result, {"intent": "exchange", "size": "blue"})
        self.assertEqual(done.skipped, ["amount"])
        self.assertEqual(self.asked(sent, "amount"), [])

    def test_skipped_fields_and_their_dependents_are_left_out(self):
        done, sent = self.run_questions(QUESTIONS, given={"intent": "refund"}, skipped=["amount"])
        self.assertEqual(done.result, {"intent": "refund", "urgent": True})
        self.assertEqual(done.skipped, ["amount", "reply"])
        self.assertEqual(self.asked(sent, "amount") + self.asked(sent, "reply"), [])

    def test_a_given_answer_is_checked_against_its_field(self):
        cases = [
            ({"intent": "cancel"}, "not one of its choices"),
            ({"intent": "refund", "amount": "7"}, "must be a number"),
            ({"intent": "refund", "amount": float("nan")}, "must be a number"),
            ({"intent": "refund", "reply": 7}, "must be a string"),
            ({"total": 7}, "does not have"),
        ]
        for given, message in cases:
            with self.subTest(given=given):
                with self.assertRaisesRegex(SchemaError, message):
                    self.run_questions(QUESTIONS, given=given)
        with self.assertRaisesRegex(SchemaError, "must be an integer"):
            self.run_questions({"n": {"type": "integer"}}, given={"n": 1.5})
        with self.assertRaisesRegex(SchemaError, "does not have"):
            self.run_questions(QUESTIONS, skipped=["total"])

    def test_null_is_given_to_nullable_fields(self):
        questions = {"total": {"type": ["number", "null"]}, "note": {"type": ["string", "null"]},
                     "check": {"type": "boolean", "depends_on": ["total", "note"]}}
        done, sent = self.run_questions(questions, given={"total": None, "note": None})
        self.assertEqual(done.result, {"total": None, "note": None, "check": True})
        [check] = self.asked(sent, "check")
        self.assertIn('{"total": null', check)

    def test_a_field_with_probabilities_is_given_as_result_gives_it(self):
        questions = {
            "intent": {"type": "string", "enum": ["refund", "exchange"], "return_probabilities": True},
            "escalate": {"type": "boolean", "when": {"intent": {"confidence": {"lt": 0.5}}}},
        }
        given = {"intent": {"value": "refund", "probabilities": {"refund": 0.6, "exchange": 0.4},
                            "confidence": 0.2}}
        done, sent = self.run_questions(questions, given=given)
        self.assertEqual(done.result["intent"]["probabilities"], {"refund": 0.6, "exchange": 0.4})
        self.assertAlmostEqual(done.result["intent"]["confidence"], 0.2)
        self.assertTrue(done.result["escalate"])  # its condition read the given probabilities
        # Probabilities keyed by the JSON text of the values, as the hosted API returns them.
        json_keys = {"intent": {"value": "refund", "probabilities": {"refund": 0.6, "exchange": 0.4}}}
        self.assertEqual(self.run_questions(questions, given=json_keys)[0].result["intent"]["value"], "refund")
        with self.assertRaisesRegex(SchemaError, "with its probabilities"):
            self.run_questions(questions, given={"intent": "refund"})

    def test_a_given_auto_thinking_field_asks_no_effort_question(self):
        questions = {"total": {"type": "number", "thinking": "auto"},
                     "ok": {"type": "boolean", "depends_on": ["total"]}}
        done, sent = self.run_questions(questions, server=EffortServer("high"), given={"total": 4})
        self.assertEqual(done.result, {"total": 4, "ok": True})
        self.assertEqual(done.thinking_effort, {})
        self.assertFalse(any("thinking_effort" in text for text in sent))
        self.assertEqual(len(sent), 1)
        # A field that thinks does not think when given: its history holds no reasoning.
        questions["total"] = {"type": "number", "thinking": True}
        done, sent = self.run_questions(questions, given={"total": 4})
        self.assertEqual(len(sent), 1)
        self.assertNotIn("Reasoned.", sent[0])

    def test_objects_are_given_whole_or_in_part_and_arrays_whole(self):
        questions = {
            "person": {"type": "object", "properties": {"name": {"type": "string"}, "age": {"type": "integer"}}},
            "tags": {"type": "array", "items": {"type": "string"}, "maxItems": 3},
            "check": {"type": "boolean", "depends_on": ["person", "tags"]},
        }
        done, sent = self.run_questions(questions, given={"person": {"age": 41}, "tags": ["a", "b"]})
        self.assertEqual(done.result, {"person": {"name": "blue", "age": 41}, "tags": ["a", "b"], "check": True})
        self.assertEqual(self.asked(sent, "age"), [])
        [check] = self.asked(sent, "check")
        self.assertIn('"tags": ["a", "b"]', check)
        with self.assertRaisesRegex(SchemaError, "must be an array"):
            self.run_questions(questions, given={"tags": "a"})

    def test_a_given_score_writes_its_nearest_level(self):
        questions = {"risk": {"type": "number", "levels": [{"label": "low"}, {"label": "medium"}, {"label": "high"}]},
                     "ok": {"type": "boolean", "depends_on": ["risk"]}}
        done, sent = self.run_questions(questions, given={"risk": 1.8})
        self.assertEqual(done.result["risk"], 1.8)
        [ok] = self.asked(sent, "ok")
        self.assertIn('{"label": "C"}<', ok)  # high, the third level
        with self.assertRaisesRegex(SchemaError, "score from 0 to 2"):
            self.run_questions(questions, given={"risk": 3})

    def test_user_fields_need_the_hosted_api_or_a_given_answer(self):
        questions = {**QUESTIONS, "amount": {**QUESTIONS["amount"], "source": "user"}}
        with self.assertRaisesRegex(SchemaError, "hosted API asks the user"):
            self.run_questions(questions)
        done, sent = self.run_questions(questions, given={"amount": 30})
        self.assertEqual(done.result["amount"], 30)
        self.assertEqual(self.asked(sent, "amount"), [])


class HostedResumeTests(unittest.TestCase):
    def client(self, *replies):
        sent = []

        def handler(request):
            sent.append(json.loads(request.content))
            return httpx.Response(200, json=replies[len(sent) - 1])

        client = TypeLLMClient(api_key="k")
        client._http_client = httpx.Client(transport=httpx.MockTransport(handler))
        return client, sent

    def test_a_call_that_needs_input_resumes_with_the_answers(self):
        paused = {"id": "gen_1", "status": "needs_input", "result": {"intent": "refund"},
                  "needs_input": {"amount": {"type": "number", "instructions": "How much?"}},
                  "resume_state": "v1.abc", "usage": {"input_tokens": 40, "thinking_tokens": 0}}
        finished = {"id": "gen_1", "status": "completed",
                    "result": {"intent": "refund", "amount": 45, "ok": {"value": True, "probabilities":
                                                                         {"true": 0.9, "false": 0.1}}},
                    "usage": {"input_tokens": 2, "thinking_tokens": 0}}
        client, sent = self.client(paused, finished)
        questions = {"intent": {"type": "string", "enum": ["refund", "exchange"]},
                     "amount": {"type": "number", "source": "user", "depends_on": ["intent"]},
                     "ok": {"type": "boolean", "depends_on": ["amount"], "return_probabilities": True}}

        first = client.generate(context="x", questions=questions)
        self.assertEqual((first.status, first.result, first.resume_state), ("needs_input", {"intent": "refund"},
                                                                            "v1.abc"))
        self.assertEqual(first.needs_input, {"amount": {"type": "number", "instructions": "How much?"}})

        done = client.resume(first, {"amount": 45}, timeout=30)
        self.assertEqual(sent[1], {"resume_state": "v1.abc", "answers": {"amount": 45}, "timeout": 30})
        self.assertEqual(done.status, "completed")
        self.assertIsNone(done.resume_state)
        # Probabilities are keyed by values, as for the first call.
        self.assertEqual(done.result["ok"]["probabilities"], {True: 0.9, False: 0.1})
        self.assertEqual(done.usage.input_tokens, 2)
        # The state alone resumes too.
        client, sent = self.client(finished)
        client.resume("v1.abc", {"amount": 45})
        self.assertEqual(sent[0], {"resume_state": "v1.abc", "answers": {"amount": 45}})

    def test_resume_needs_the_hosted_api_and_a_state(self):
        with self.assertRaisesRegex(ValueError, "hosted"):
            TypeLLMClient("http://127.0.0.1:30000").resume("v1.abc", {})
        client, _ = self.client()
        with self.assertRaisesRegex(ValueError, "stopped for input"):
            client.resume("", {})

    def test_given_and_skipped_need_your_own_server(self):
        client, _ = self.client()
        for options in ({"given": {"a": 1}}, {"skipped": ["a"]}):
            with self.assertRaisesRegex(ValueError, "own server"):
                client.generate(context="x", questions={"a": {"type": "number"}}, **options)


if __name__ == "__main__":
    unittest.main()
