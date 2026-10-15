import unittest

from typellm import SchemaError, TypeLLMClient
from typellm.runtime import BOX_MAX_TOKENS, BOX_PATTERN

from tests.test_batching import FakeServer
from tests.test_images import PNG


class BoxServer(FakeServer):
    """A FakeServer that answers a box's grammar with `coordinates`, as the model writes them."""

    def __init__(self, coordinates="199, 166, 400, 500]}"):
        super().__init__()
        self.coordinates = coordinates

    def _request(self, path, payload=None, *, allow_text=False):
        response = super()._request(path, payload, allow_text=allow_text)
        if path == "/generate" and not isinstance(payload["sampling_params"], dict):
            for params, item in zip(payload["sampling_params"], response if isinstance(response, list) else [response]):
                if params.get("regex") == BOX_PATTERN:
                    item["text"] = self.coordinates
        elif path == "/generate" and payload["sampling_params"].get("regex") == BOX_PATTERN:
            response["text"] = self.coordinates
        return response

    def box_requests(self):
        return [params for payload in self.payloads
                for params in (payload["sampling_params"] if isinstance(payload["sampling_params"], list)
                               else [payload["sampling_params"]])
                if params.get("regex") == BOX_PATTERN]


class BoxTests(unittest.TestCase):
    def run_questions(self, questions, server=None, images=(PNG,), **options):
        client = TypeLLMClient("http://127.0.0.1:30000", model="fake")
        client.sglang = server or BoxServer()
        done = client.generate(context="Read the receipt photo.", images=list(images), questions=questions, **options)
        sent = [t for p in client.sglang.payloads
                for t in ([p["text"]] if isinstance(p["text"], str) else p["text"])]
        return done, sent, client.sglang

    def test_a_box_comes_back_as_fractions_of_the_image(self):
        done, sent, server = self.run_questions({"total": {"type": "box", "instructions": "The total line."}})
        self.assertEqual(done.result, {"total": {"top": 0.166, "bottom": 0.5, "left": 0.199, "right": 0.4}})
        [prompt] = [t for t in sent if t.endswith('{"bbox_2d": [')]
        self.assertIn("Type: bounding box", prompt)
        self.assertIn("Instructions: The total line.", prompt)
        [params] = server.box_requests()
        self.assertGreaterEqual(params["max_new_tokens"], BOX_MAX_TOKENS)

    def test_edges_given_the_wrong_way_round_are_put_in_order(self):
        done, _, _ = self.run_questions({"total": {"type": "box"}}, server=BoxServer("400, 500, 199, 166]}"))
        self.assertEqual(done.result["total"], {"top": 0.166, "bottom": 0.5, "left": 0.199, "right": 0.4})

    def test_a_box_without_instructions_is_asked_by_its_name(self):
        _, sent, _ = self.run_questions({"signature": {"type": "box"}})
        [prompt] = [t for t in sent if t.endswith('{"bbox_2d": [')]
        self.assertIn('Instructions: Locate "signature" in the image.', prompt)

    def test_fields_that_depend_on_a_box_see_it(self):
        questions = {"total": {"type": "box"}, "legible": {"type": "boolean", "depends_on": ["total"]}}
        done, sent, _ = self.run_questions(questions)
        self.assertTrue(done.result["legible"])
        [later] = [t for t in sent if 'Field: "legible"' in t[t.rindex("Field:"):]]
        self.assertIn('{"bbox_2d": [199, 166, 400, 500]}', later)  # the box in history, as the model wrote it
        self.assertIn('"total": {"top": 0.166, "bottom": 0.5, "left": 0.199, "right": 0.4}', later)

    def test_boxes_and_numbers_decode_in_one_request(self):
        done, _, server = self.run_questions({"total": {"type": "box"}, "amount": {"type": "number"}})
        self.assertEqual(done.result["amount"], 7)
        self.assertEqual(len([p for p in server.payloads if any(
            "regex" in params for params in (p["sampling_params"] if isinstance(p["sampling_params"], list)
                                             else [p["sampling_params"]]))]), 1)

    def test_a_box_can_be_given(self):
        box = {"top": 0.2, "bottom": 0.4, "left": 0.1, "right": 0.3}
        done, sent, server = self.run_questions(
            {"total": {"type": "box"}, "legible": {"type": "boolean", "depends_on": ["total"]}}, given={"total": box})
        self.assertEqual(done.result["total"], box)
        self.assertEqual(server.box_requests(), [])
        [later] = [t for t in sent if 'Field: "legible"' in t[t.rindex("Field:"):]]
        self.assertIn('{"bbox_2d": [100, 200, 300, 400]}', later)
        with self.assertRaisesRegex(SchemaError, "must be a box"):
            self.run_questions({"total": {"type": "box"}}, given={"total": {"top": 2}})

    def test_what_a_box_cannot_be(self):
        cases = [
            ({"total": {"type": "box"}}, (), "needs images"),
            ({"total": {"type": ["box", "null"]}}, (PNG,), "cannot be null"),
            ({"total": {"type": "box", "enum": [1, 2]}}, (PNG,), "enum is not supported on a box"),
            ({"total": {"type": "box", "return_probabilities": True}}, (PNG,), "return_probabilities is not supported"),
            ({"rows": {"type": "array", "items": {"type": "box"}}}, (PNG,), "inside an array"),
            ({"total": {"type": "box"}, "ok": {"type": "boolean", "when": {"total": None}}}, (PNG,), "a box"),
        ]
        for questions, images, message in cases:
            with self.subTest(message=message):
                with self.assertRaisesRegex(SchemaError, message):
                    self.run_questions(questions, images=images)


if __name__ == "__main__":
    unittest.main()
