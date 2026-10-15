import unittest

from typellm import SchemaError, TypeLLMClient
from typellm.runtime import BOX_ITEM_TOKENS, BOX_MAX_TOKENS, BOX_PATTERN, box_list_pattern

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
            ({"rows": {"type": "array", "items": {"type": "object", "properties": {"at": {"type": "box"}}}}},
             (PNG,), "inside an array"),
            ({"total": {"type": "box"}, "ok": {"type": "boolean", "when": {"total": None}}}, (PNG,), "a box"),
        ]
        for questions, images, message in cases:
            with self.subTest(message=message):
                with self.assertRaisesRegex(SchemaError, message):
                    self.run_questions(questions, images=images)



class ListServer(BoxServer):
    """A BoxServer that answers a list of boxes' grammar with `listed`, after its "[" as the model writes it,
    and whether there is any match with `present`."""

    def __init__(self, listed='{"bbox_2d": [100, 200, 300, 400]}, {"bbox_2d": [500, 600, 700, 800]}]', present=True):
        super().__init__()
        self.listed = listed
        self.present = present

    def _request(self, path, payload=None, *, allow_text=False):
        response = super()._request(path, payload, allow_text=allow_text)
        if path == "/generate" and "token_ids_logprob" in payload and not self.present:
            texts = [payload["text"]] if isinstance(payload["text"], str) else payload["text"]
            for text, item in zip(texts, response if isinstance(response, list) else [response]):
                if "Is there at least one match" in text[text.rindex("<|im_start|>user"):]:
                    scores = item["meta_info"]["output_token_ids_logprobs"][0]
                    # The second label, false, over the first.
                    scores[0][0], scores[1][0] = -9.0, 0.0
        if path == "/generate":
            params = payload["sampling_params"]
            for one, item in zip(params if isinstance(params, list) else [params],
                                 response if isinstance(response, list) else [response]):
                if one.get("regex", "").startswith(("(?:" + r'\{"bbox_2d"', r'\{"bbox_2d"')):
                    item["text"] = self.listed
        return response

    def list_requests(self):
        return [params for payload in self.payloads
                for params in (payload["sampling_params"] if isinstance(payload["sampling_params"], list)
                               else [payload["sampling_params"]])
                if params.get("regex", "").endswith(r"\]") and "bbox_2d" in params.get("regex", "")]


class BoxListTests(unittest.TestCase):
    def run_questions(self, questions, server=None, **options):
        return BoxTests.run_questions(self, questions, server=server or ListServer(), **options)

    def test_an_array_of_boxes_lists_every_match_in_one_answer(self):
        questions = {"dogs": {"type": "array", "items": {"type": "box"}, "instructions": "Every dog."}}
        done, sent, server = self.run_questions(questions)
        self.assertEqual(done.result["dogs"], [{"top": 0.2, "bottom": 0.4, "left": 0.1, "right": 0.3},
                                               {"top": 0.6, "bottom": 0.8, "left": 0.5, "right": 0.7}])
        [prompt] = [t for t in sent if t.endswith("Answer [] if there is none.<|im_end|>\n<|im_start|>assistant\n[")
                    or t.rstrip().endswith("[") and "list of bounding boxes" in t]
        self.assertIn("Instructions: Every dog.", prompt)
        [params] = server.list_requests()
        # The hidden question said there is a match, so the list holds one at least.
        self.assertEqual(params["regex"], box_list_pattern(1, 50))
        self.assertGreaterEqual(params["max_new_tokens"], 50 * BOX_ITEM_TOKENS)
        self.assertEqual(list(done.result), ["dogs"])  # the presence question is not in the result
        self.assertEqual(done.skipped, [])

    def test_bounds_shape_the_grammar(self):
        questions = {"dogs": {"type": "array", "items": {"type": "box"}, "minItems": 1, "maxItems": 3}}
        _, _, server = self.run_questions(questions)
        [params] = server.list_requests()
        self.assertEqual(params["regex"], box_list_pattern(1, 3))

    def test_no_match_is_an_empty_list_and_asks_for_no_boxes(self):
        questions = {"dogs": {"type": "array", "items": {"type": "box"}},
                     "count": {"type": "integer", "depends_on": ["dogs"]}}
        done, sent, server = self.run_questions(questions, server=ListServer(present=False))
        self.assertEqual(done.result, {"dogs": [], "count": 7})
        self.assertEqual(server.list_requests(), [])
        [later] = [t for t in sent if 'Field: "count"' in t[t.rindex("Field:"):]]
        self.assertIn('assistant\n[]', later)

    def test_a_list_that_must_hold_a_box_asks_no_presence_question(self):
        questions = {"dogs": {"type": "array", "items": {"type": "box"}, "minItems": 1}}
        done, sent, _ = self.run_questions(questions, server=ListServer(present=False))
        self.assertEqual(len(done.result["dogs"]), 2)
        self.assertFalse(any("Is there at least one match" in t for t in sent))

    def test_a_box_listed_twice_comes_back_once(self):
        twice = '{"bbox_2d": [100, 200, 300, 400]}, {"bbox_2d": [300, 400, 100, 200]}]'
        done, _, _ = self.run_questions({"dogs": {"type": "array", "items": {"type": "box"}}}, server=ListServer(twice))
        self.assertEqual(done.result["dogs"], [{"top": 0.2, "bottom": 0.4, "left": 0.1, "right": 0.3}])

    def test_dependents_see_the_list(self):
        questions = {"dogs": {"type": "array", "items": {"type": "box"}},
                     "count": {"type": "integer", "depends_on": ["dogs"]}}
        done, sent, _ = self.run_questions(questions)
        [later] = [t for t in sent if 'Field: "count"' in t[t.rindex("Field:"):]]
        self.assertIn('[{"bbox_2d": [100, 200, 300, 400]}, {"bbox_2d": [500, 600, 700, 800]}]', later)

    def test_a_list_can_be_given(self):
        boxes = [{"top": 0.2, "bottom": 0.4, "left": 0.1, "right": 0.3}]
        questions = {"dogs": {"type": "array", "items": {"type": "box"}},
                     "count": {"type": "integer", "depends_on": ["dogs"]}}
        done, sent, server = self.run_questions(questions, given={"dogs": boxes})
        self.assertEqual(done.result["dogs"], boxes)
        self.assertEqual(server.list_requests(), [])
        self.assertFalse(any("Is there at least one match" in t for t in sent))
        [later] = [t for t in sent if 'Field: "count"' in t[t.rindex("Field:"):]]
        self.assertIn('[{"bbox_2d": [100, 200, 300, 400]}]', later)
        with self.assertRaisesRegex(SchemaError, "list of 0 to 50 boxes"):
            self.run_questions(questions, given={"dogs": [{"top": 2}]})

    def test_what_an_array_of_boxes_cannot_be(self):
        cases = [
            ({"dogs": {"type": "array", "items": {"type": "box", "instructions": "A dog."}}}, "take only their type"),
            ({"dogs": {"type": "array", "items": {"type": "box"}, "continue_from": []}}, "continue_from"),
            ({"dogs": {"type": "array", "items": {"type": "box"}, "maxItems": 60}}, "at most 50"),
            ({"dogs": {"type": "array", "items": {"type": "box"}},
              "ok": {"type": "boolean", "when": {"dogs": None}}}, "a box"),
        ]
        for questions, message in cases:
            with self.subTest(message=message):
                with self.assertRaisesRegex(SchemaError, message):
                    self.run_questions(questions)


if __name__ == "__main__":
    unittest.main()
