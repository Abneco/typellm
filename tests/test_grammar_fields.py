import itertools
import re
import unittest

from typellm import SGLangError, TypeLLMClient
from typellm.runtime import _numeric_text_is_complete, numeric_pattern

from tests.test_batching import FakeServer, width
from tests.test_nullable import NullServer


def regex_requests(server):
    return [p for p in server.payloads
            if any("regex" in q for q in (p["sampling_params"] if isinstance(p["sampling_params"], list)
                                          else [p["sampling_params"]]))]


class ReplyServer(FakeServer):
    """Answers every grammar request with one fixed text and finish reason."""

    def __init__(self, text, finish="stop"):
        super().__init__()
        self.reply, self.finish = text, finish

    def _request(self, path, payload=None, *, allow_text=False):
        params = payload.get("sampling_params") if payload else None
        if path == "/generate" and isinstance(params, list) and "regex" in params[0]:
            self.payloads.append(payload)
            return [{"text": self.reply, "meta_info": {"finish_reason": {"type": self.finish}}} for _ in params]
        return super()._request(path, payload, allow_text=allow_text)


class ValueServer(FakeServer):
    """Picks the ' "' string start wherever it is offered, so strings are not null."""

    def _request(self, path, payload=None, *, allow_text=False):
        if path == "/generate" and "token_ids_logprob" in payload:
            rows = payload["token_ids_logprob"]
            if 328 in (rows if isinstance(rows[0], int) else sum(rows, [])):
                self.payloads.append(payload)
                rows = [rows] if isinstance(rows[0], int) else rows
                out = [{"meta_info": {"output_token_ids_logprobs": [
                    [[0.0 if t == 328 else -9.0, t, "?"] for t in ids]]}} for ids in rows]
                return out[0] if isinstance(payload["text"], str) else out
        return super()._request(path, payload, allow_text=allow_text)


class PatternTests(unittest.TestCase):
    """The regex admits exactly the numbers the stepwise decoder completes."""

    def samples(self, max_digits):
        short = ("".join(chars) for n in range(1, 6) for chars in itertools.product("-0.19", repeat=n))
        long = ["1" * max_digits, "1" * (max_digits + 1), "1" * (max_digits - 1) + ".5",
                "1" * max_digits + ".5", "0." + "5" * (max_digits - 1), "0." + "5" * max_digits,
                "-" + "9" * max_digits]
        return itertools.chain(short, long)

    def test_the_pattern_matches_the_stepwise_rules(self):
        for numeric_type in ("number", "integer"):
            for max_digits in (32, 4):
                pattern = re.compile(numeric_pattern(numeric_type, max_digits))
                for text in self.samples(max_digits):
                    expected = (_numeric_text_is_complete(text, numeric_type)
                                and sum(c.isdigit() for c in text) <= max_digits)
                    # The value may start with the space, and end at '}' or the end of the message.
                    for written in (" " + text + "}", text + "}", " " + text, text):
                        with self.subTest(type=numeric_type, digits=max_digits, written=written):
                            self.assertEqual(bool(pattern.fullmatch(written)), expected)

    def test_null_only_where_nullable(self):
        self.assertTrue(re.fullmatch(numeric_pattern("number", 32, nullable=True), " null}"))
        self.assertFalse(re.fullmatch(numeric_pattern("number", 32), " null}"))


class GrammarNumberTests(unittest.TestCase):
    QUESTIONS = {"a": {"type": "integer"}, "b": {"type": "number"}, "c": {"type": "integer"}}

    def test_every_number_of_a_layer_takes_one_request(self):
        client = TypeLLMClient(model="fake")
        client.sglang = FakeServer()
        result = client.generate(context="Receipt", questions=self.QUESTIONS)
        self.assertEqual(result, {"a": 7, "b": 7, "c": 7})
        self.assertEqual(client.sglang.requests("score"), [])
        [numbers] = regex_requests(client.sglang)
        self.assertEqual(width(numbers), 3)
        self.assertTrue(all(t.endswith('":') for t in numbers["text"]))
        self.assertEqual(numbers["sampling_params"][0]["temperature"], 0)
        self.assertIn('{"a": 7}', client.last_prompts[0])

    def test_stepwise_still_scores_every_token(self):
        client = TypeLLMClient(model="fake", open_decoding="stepwise")
        client.sglang = FakeServer()
        client.generate(context="Receipt", questions=self.QUESTIONS)
        self.assertEqual(regex_requests(client.sglang), [])
        self.assertEqual(len(client.sglang.requests("score")), 3)

    def test_sampling_passes_the_temperature_and_no_truncation(self):
        client = TypeLLMClient(model="fake", mode="sample", temperature=0.7, seed=1)
        client.sglang = FakeServer()
        client.generate(context="Receipt", questions={"a": {"type": "integer"}})
        [params] = regex_requests(client.sglang)[0]["sampling_params"]
        self.assertEqual((params["temperature"], params["top_p"], params["top_k"]), (0.7, 1.0, -1))

    def test_nullable_numbers_can_be_null_in_the_same_request(self):
        client = TypeLLMClient(model="fake")
        client.sglang = NullServer()
        result = client.generate(context="Receipt", questions={"tip": {"type": ["number", "null"]}})
        self.assertEqual(result, {"tip": None})
        self.assertEqual(client.sglang.requests("score"), [])
        self.assertIn('{"tip": null}', client.last_prompts[0])

    def test_an_incomplete_or_cut_off_number_is_an_error(self):
        for reply, finish, error in ((" 3.}", "stop", ValueError), (" 12", "length", SGLangError)):
            with self.subTest(reply=reply):
                client = TypeLLMClient(model="fake")
                client.sglang = ReplyServer(reply, finish)
                with self.assertRaises(error):
                    client.generate(context="Receipt", questions={"a": {"type": "number"}})

    def test_negative_and_spaceless_numbers_parse(self):
        for reply, value in ((" -3.5}", -3.5), ("42}", 42.0), (" 0", 0.0)):
            with self.subTest(reply=reply):
                client = TypeLLMClient(model="fake")
                client.sglang = ReplyServer(reply)
                self.assertEqual(client.generate(context="R", questions={"a": {"type": "number"}}),
                                 {"a": value})

    def test_unknown_decodings_are_rejected(self):
        with self.assertRaisesRegex(ValueError, "open_decoding"):
            TypeLLMClient(model="fake", open_decoding="fast")


class GrammarNullableTextTests(unittest.TestCase):
    def test_a_nullable_string_writes_null_or_text_in_one_request(self):
        client = TypeLLMClient(model="fake")
        client.sglang = NullServer()
        result = client.generate(context="Receipt", questions={
            "note": {"type": ["string", "null"]}, "name": {"type": "string"}})
        self.assertEqual(result, {"note": None, "name": "blue"})
        self.assertEqual(client.sglang.requests("score"), [])  # no separate null step
        [texts] = regex_requests(client.sglang)
        patterns = [p["regex"] for p in texts["sampling_params"]]
        self.assertIn("null", patterns[0])
        self.assertNotIn("null", patterns[1])

    def test_stepwise_text_after_the_null_step_cannot_be_null(self):
        client = TypeLLMClient(model="fake", open_decoding="stepwise")
        client.sglang = ValueServer()
        result = client.generate(context="Receipt", questions={"note": {"type": ["string", "null"]}})
        self.assertEqual(result, {"note": "blue"})
        self.assertEqual(len(client.sglang.requests("score")), 1)  # the null step
        [texts] = regex_requests(client.sglang)
        self.assertNotIn("null", texts["sampling_params"][0]["regex"])


if __name__ == "__main__":
    unittest.main()
