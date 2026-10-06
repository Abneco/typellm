import json
import unittest

from typellm import SchemaError, TypeLLMClient
from typellm.runtime import _canonical_json
from typellm.schema import CONTINUE_QUESTION, compile_json_schema

from tests.test_batching import FakeServer, is_number_pattern

# The fake answers strings "blue", numbers 7, booleans true and enums with their first value.
CONTINUE_MARK = CONTINUE_QUESTION[:60]
STATE_MARK = "Current array (JSON): "


def current(prompt):
    """The array committed in a prompt's state."""
    text = prompt.rsplit(STATE_MARK, 1)[1]
    return json.JSONDecoder().raw_decode(text)[0]


def field(prompt):
    """The field a prompt asks for, from its last Field: line."""
    return json.JSONDecoder().raw_decode(prompt.rsplit("Field: ", 1)[1])[0]


class ScriptServer(FakeServer):
    """A FakeServer whose continue questions go on as `more` says, in order, then end the array; strings,
    numbers and choices come from callables of the prompt when given."""

    def __init__(self, more=(), texts=None, numbers=None, choose=None):
        super().__init__()
        self.more = list(more)
        self.texts, self.numbers, self.choose = texts, numbers, choose
        self.continue_prompts = []

    def _request(self, path, payload=None, *, allow_text=False):
        response = super()._request(path, payload, allow_text=allow_text)
        if path != "/generate":
            return response
        texts = [payload["text"]] if isinstance(payload["text"], str) else payload["text"]
        out = response if isinstance(response, list) else [response]
        if "token_ids_logprob" in payload:
            rows = payload["token_ids_logprob"]
            rows = [rows] if isinstance(rows[0], int) else rows
            for text, ids, item in zip(texts, rows, out):
                if CONTINUE_MARK in text:
                    # "Append a new item?": true (the first label) goes on, false ends the array.
                    self.continue_prompts.append(text)
                    pick = ids[0] if (self.more.pop(0) if self.more else False) else ids[1]
                elif self.choose:
                    pick = self.choose(text, ids)
                else:
                    continue
                if pick is not None:
                    item["meta_info"]["output_token_ids_logprobs"] = [
                        [[0.0 if t == pick else -9.0, t, "?"] for t in ids]]
        else:
            params = payload["sampling_params"]
            params = params if isinstance(params, list) else [params] * len(texts)
            for text, p, item in zip(texts, params, out):
                if "regex" not in p:
                    continue
                if is_number_pattern(p["regex"]) and self.numbers:
                    item["text"] = f" {self.numbers(text)}}}"
                elif not is_number_pattern(p["regex"]) and self.texts:
                    item["text"] = " " + json.dumps(self.texts(text)) + "}"
        return response


def run(questions, server=None, **options):
    client = TypeLLMClient("http://127.0.0.1:30000", model="fake")
    client.sglang = server or ScriptServer()
    done = client.generate(context="Resume", questions=questions, **options)
    sent = [t for p in client.sglang.payloads for t in ([p["text"]] if isinstance(p["text"], str) else p["text"])]
    return done, sent, client.sglang


PERSON = {
    "type": "object",
    "properties": {
        "name": {"type": "string", "instructions": "Person's full name."},
        "age": {"type": "integer", "instructions": "Person's age."},
        "employed": {"type": "boolean", "instructions": "Whether the person is currently employed."},
    },
}


class ObjectTests(unittest.TestCase):
    def test_an_object_of_a_string_an_integer_and_a_boolean(self):
        done, sent, _ = run({"person": PERSON})
        self.assertEqual(done.result, {"person": {"name": "blue", "age": 7, "employed": True}})
        self.assertEqual(done.skipped, [])
        asked = [t for t in sent if 'Field: "age"' in t]
        self.assertTrue(asked)
        self.assertIn('Object: "person"', asked[-1])
        self.assertIn('Answer as {"age": <integer>}.', asked[-1])

    def test_an_enum_property(self):
        done, _, _ = run({"risk": {"type": "object", "properties": {
            "level": {"type": "string", "enum": ["low", "high"]},
            "note": {"type": "string"},
        }}})
        self.assertEqual(done.result, {"risk": {"level": "low", "note": "blue"}})

    def test_independent_properties_share_the_batches_of_flat_fields(self):
        _, nested, nested_server = run({"person": PERSON, "title": {"type": "string"}})
        _, flat, flat_server = run({**PERSON["properties"], "title": {"type": "string"}})
        self.assertEqual(len(nested_server.payloads), len(flat_server.payloads))
        self.assertEqual(len(nested), len(flat))

    def test_property_dependencies_run_in_layers(self):
        done, sent, server = run({"person": {"type": "object", "properties": {
            "name": {"type": "string"},
            "known": {"type": "boolean", "depends_on": ["name"]},
        }}})
        self.assertEqual(done.result, {"person": {"name": "blue", "known": True}})
        asked = [t for t in sent if 'Field: "known"' in t]
        self.assertIn('Dependency results (JSON):\n{"person": {"name": "blue"}}', asked[-1])
        self.assertLess(next(i for i, t in enumerate(sent) if 'Field: "name"' in t),
                        next(i for i, t in enumerate(sent) if 'Field: "known"' in t))

    def test_properties_keep_their_schema_order(self):
        done, _, _ = run({"b": {"type": "boolean"}, "obj": {"type": "object", "properties": {
            "z": {"type": "integer"}, "a": {"type": "string"},
            "m": {"type": "object", "properties": {"y": {"type": "boolean"}, "c": {"type": "integer"}}},
        }}})
        self.assertEqual(list(done.result), ["b", "obj"])
        self.assertEqual(list(done.result["obj"]), ["z", "a", "m"])
        self.assertEqual(list(done.result["obj"]["m"]), ["y", "c"])

    def test_a_field_depending_on_an_object_sees_all_of_it(self):
        done, sent, _ = run({"person": PERSON, "hire": {"type": "boolean", "depends_on": ["person"]}})
        self.assertTrue(done.result["hire"])
        asked = [t for t in sent if 'Field: "hire"' in t][-1]
        self.assertIn('{"person": {"name": "blue", "age": 7, "employed": true}}', asked)

    def test_an_unmet_condition_on_an_object_skips_it_once(self):
        done, _, _ = run({
            "kind": {"type": "string", "enum": ["invoice", "resume"]},  # answers "invoice"
            "person": {**PERSON, "when": {"kind": "resume"}},
        })
        self.assertEqual(done.result, {"kind": "invoice"})
        self.assertEqual(done.skipped, ["person"])

    def test_a_property_condition_skips_that_property(self):
        done, _, _ = run({"person": {"type": "object", "properties": {
            "employed": {"type": "boolean", "enum": [False, True]},  # answers false
            "employer": {"type": "string", "when": {"employed": True}},
        }}})
        self.assertEqual(done.result, {"person": {"employed": False}})
        self.assertEqual(done.skipped, ["person.employer"])

    def test_invalid_objects_are_schema_errors(self):
        for schema in (
            {"type": "object"},                                                 # no properties
            {"type": "object", "properties": [{"type": "string"}]},             # not a mapping
            {"type": "object", "properties": {}},
            {"type": "object", "properties": {"a": "string"}},                  # not a schema
            {"type": "object", "properties": {"a": {"type": "string", "enum": []}}},
            {"type": "object", "properties": {"a": {"type": "array", "items": {"type": "string"}}}},
            {"type": "object", "properties": {"a": {"type": "string"}}, "items": {"type": "string"}},
            {"type": "object", "properties": {"a": {"type": "string"}}, "thinking": True},
            {"type": ["object", "null"], "properties": {"a": {"type": "string"}}},
            {"type": "object", "properties": {"a": {"type": "string", "depends_on": ["missing"]}}},
            {"type": "object", "properties": {"a": {"type": "string"}}, "when": {"missing": True}},
        ):
            with self.subTest(schema=schema), self.assertRaises(SchemaError):
                compile_json_schema({"type": "object", "properties": {"x": {"type": "boolean"}, "o": schema}})

    def test_a_property_path_cannot_clash_with_a_field(self):
        with self.assertRaises(SchemaError):
            compile_json_schema({"type": "object", "properties": {
                "o": {"type": "object", "properties": {"a": {"type": "string"}}},
                "o.a": {"type": "string"},
            }})

    def test_a_condition_cannot_test_an_object(self):
        with self.assertRaises(SchemaError):
            compile_json_schema({"type": "object", "properties": {
                "o": PERSON, "x": {"type": "boolean", "when": {"o": True}}}})


SKILLS = {"type": "array", "items": {"type": "string"}, "instructions": "Return all distinct relevant skills."}
NAMES = ["Python", "CUDA", "PyTorch", "Rust"]


NULL_MARK = "Answer null instead"


def skills_server(count, **options):
    """The n-th skill for each string item, and null once `count` are in where the prompt allows it.
    An object array's continue question answers true `count` times. A string outside an array is "Senior"."""
    def texts(prompt):
        if STATE_MARK not in prompt:
            return "Senior"
        n = len(current(prompt))
        return None if n >= count and NULL_MARK in prompt else NAMES[n]
    return ScriptServer([True] * count, texts=texts, **options)


def item_prompts(sent):
    return [t for t in sent if 'Field: "item"' in t]


class ArrayTests(unittest.TestCase):
    def test_the_state_states_the_arrays_own_instructions(self):
        for items in ({"type": "string"}, {"type": "string", "instructions": "A skill's name."}):
            with self.subTest(items=items):
                _, sent, _ = run({"skills": {**SKILLS, "items": items}}, skills_server(0))
                self.assertIn('Array: "skills"\nInstructions: Return all distinct relevant skills.\n',
                              item_prompts(sent)[0])

    def test_an_empty_array(self):
        # A scalar array asks no continue question: its item may be null, and null ends it.
        done, sent, server = run({"skills": SKILLS}, skills_server(0))
        self.assertEqual(done.result, {"skills": []})
        self.assertEqual(server.continue_prompts, [])
        [asked] = item_prompts(sent)
        self.assertIn(NULL_MARK, asked)
        self.assertIn('Answer as {"item": <string or null>}.', asked)

    def test_one_item(self):
        done, sent, _ = run({"skills": SKILLS}, skills_server(1))
        self.assertEqual(done.result, {"skills": ["Python"]})
        self.assertEqual(len(item_prompts(sent)), 2)  # Python, then null: one request a turn

    def test_several_strings(self):
        done, _, _ = run({"skills": SKILLS}, skills_server(3))
        self.assertEqual(done.result, {"skills": ["Python", "CUDA", "PyTorch"]})

    def test_an_enum_array(self):
        risks = ["market", "credit", "liquidity", "operational"]
        # The item's choices gain null as the last; it ends the array.
        server = ScriptServer([], choose=lambda prompt, ids: (
            ids[[1, 3, len(ids) - 1][len(current(prompt))]] if 'Field: "item"' in prompt else None))
        done, _, server = run({"risks": {"type": "array", "items": {"type": "string", "enum": risks},
                                         "instructions": "Return all applicable risks."}}, server)
        self.assertEqual(done.result, {"risks": ["credit", "operational"]})
        self.assertEqual(server.continue_prompts, [])

    def test_a_nullable_item_asks_the_continue_question(self):
        # null is then an item, not the end: the array asks whether to go on, as an object array does.
        done, _, server = run({"tips": {"type": "array", "items": {"type": ["string", "null"]}}},
                              ScriptServer([True], texts=lambda prompt: "cash"))
        self.assertEqual(done.result, {"tips": ["cash"]})
        self.assertEqual(len(server.continue_prompts), 2)

    def test_an_array_of_objects(self):
        jobs = [("Google", "Engineer", 2020, False), ("Stripe", "Senior Engineer", 2023, True)]

        def texts(prompt):
            return jobs[len(current(prompt))][["company", "title"].index(field(prompt))]

        server = ScriptServer([True, True], texts=texts,
                              numbers=lambda prompt: jobs[len(current(prompt))][2],
                              choose=lambda prompt, ids: (ids[0] if jobs[len(current(prompt))][3] else ids[1])
                              if 'Field: "current"' in prompt else None)
        done, sent, _ = run({"work_experience": WORK}, server)
        self.assertEqual(done.result, {"work_experience": [
            {"company": "Google", "title": "Engineer", "start_year": 2020, "current": False},
            {"company": "Stripe", "title": "Senior Engineer", "start_year": 2023, "current": True},
        ]})
        # An item's properties fork from the same state in one batch, none seeing another's value.
        first = [t for t in sent if "Field: " in t and CONTINUE_MARK not in t and current(t) == []]
        self.assertEqual(sorted(field(t) for t in first), ["company", "current", "start_year", "title"])
        self.assertFalse(any("Dependency results" in t for t in first))
        # The committed object, in its canonical form, is the state of the next turn.
        self.assertTrue(any(STATE_MARK + '[{"company":"Google","title":"Engineer","start_year":2020,'
                            '"current":false}]' in t for t in sent))

    def test_min_items_are_generated_before_null_is_allowed(self):
        done, sent, _ = run({"skills": {**SKILLS, "minItems": 2}}, skills_server(0))
        self.assertEqual(done.result, {"skills": ["Python", "CUDA"]})
        self.assertEqual([NULL_MARK in t for t in item_prompts(sent)], [False, False, True])

    def test_max_items_stops_without_asking(self):
        done, sent, _ = run({"skills": {**SKILLS, "maxItems": 2}}, skills_server(5))
        self.assertEqual(done.result, {"skills": ["Python", "CUDA"]})
        self.assertEqual(len(item_prompts(sent)), 2)

    def test_an_item_already_in_the_array_ends_it(self):
        for options in ({}, {"temperature": 0.7, "seed": 1}):
            with self.subTest(options=options):
                done, sent, _ = run({"skills": SKILLS}, ScriptServer([], texts=lambda prompt: "Python"), **options)
                self.assertEqual(done.result, {"skills": ["Python"]})
                # argmax stops at the first repeat; sampling tries again twice.
                self.assertEqual(len(item_prompts(sent)), 2 if not options else 4)

    def test_an_array_beside_independent_fields(self):
        done, _, _ = run({"title": {"type": "string"}, "skills": SKILLS, "senior": {"type": "boolean"}},
                         skills_server(1))
        self.assertEqual(done.result, {"title": "Senior", "skills": ["Python"], "senior": True})
        self.assertEqual(list(done.result), ["title", "skills", "senior"])

    def test_an_array_waits_for_its_dependencies(self):
        done, sent, _ = run({"role": {"type": "string", "enum": ["engineer", "designer"]},
                             "skills": {**SKILLS, "depends_on": ["role"]}}, skills_server(1))
        self.assertEqual(done.result, {"role": "engineer", "skills": ["Python"]})
        self.assertIn('Dependency results (JSON):\n{"role": "engineer"}', item_prompts(sent)[0])

    def test_a_field_after_an_array_sees_the_finished_array(self):
        done, sent, _ = run({"skills": SKILLS, "count": {"type": "integer", "depends_on": ["skills"]}},
                            skills_server(2))
        self.assertEqual(done.result, {"skills": ["Python", "CUDA"], "count": 7})
        asked = [t for t in sent if 'Field: "count"' in t]
        self.assertIn('{"skills": ["Python", "CUDA"]}', asked[0])
        self.assertGreater(sent.index(asked[0]), sent.index(item_prompts(sent)[-1]))

    def test_a_skipped_array_is_named_once(self):
        done, sent, _ = run({"kind": {"type": "string", "enum": ["invoice", "resume"]},
                             "skills": {**SKILLS, "when": {"kind": "resume"}}}, skills_server(1))
        self.assertEqual((done.result, done.skipped), ({"kind": "invoice"}, ["skills"]))
        self.assertEqual(item_prompts(sent), [])

    def test_each_turn_extends_the_state_before_it(self):
        _, sent, _ = run({"skills": SKILLS}, skills_server(3))
        prompts = item_prompts(sent)
        self.assertEqual([current(p) for p in prompts],
                         [[], ["Python"], ["Python", "CUDA"], ["Python", "CUDA", "PyTorch"]])
        for before, after in zip(prompts, prompts[1:]):
            # Everything up to the array's closing bracket is the next turn's prefix.
            state = before[:before.rindex(STATE_MARK)] + STATE_MARK + _canonical_json(current(before))[:-1]
            self.assertTrue(after.startswith(state))
        # Each turn's question is a branch: no later prompt carries an earlier one.
        for text in prompts:
            self.assertEqual(text.count('Field: "item"'), 1)

    def test_an_object_array_asks_what_comes_next_and_its_question_is_a_branch(self):
        _, sent, server = run({"work_experience": WORK}, skills_server(2))
        self.assertEqual(len(server.continue_prompts), 3)  # true, true, false
        for text in sent:
            self.assertLessEqual(text.count(CONTINUE_MARK), 1)
            if any(f'Field: "{name}"' in text for name in WORK["items"]["properties"]):
                self.assertNotIn(CONTINUE_MARK, text)
        self.assertIn('Answer as {"append_item": "<label>"}.', server.continue_prompts[0])

    def test_a_state_is_warmed_only_when_no_continue_question_cached_it(self):
        # A continue question, or a string item, is alone in its batch and caches its prompt itself; the
        # continue question's prompt starts with the state, so the properties forking from it after one
        # need no warm-up. Below minItems no question is asked, and the state is warmed for them.
        _, _, server = run({"skills": SKILLS}, skills_server(2))
        self.assertEqual(server.requests("count"), [])
        _, _, server = run({"work_experience": WORK}, skills_server(2))
        self.assertEqual(server.requests("count"), [])
        _, _, server = run({"work_experience": {**WORK, "minItems": 1}}, skills_server(1))
        warm = server.requests("count")
        self.assertEqual(len(warm), 1)
        self.assertIn(STATE_MARK + "[]", warm[0]["text"])

    def test_committed_json_is_canonical(self):
        self.assertEqual(_canonical_json([{"b": 1, "a": True, "c": None, "d": 2.5, "e": "café"}]),
                         '[{"b":1,"a":true,"c":null,"d":2.5,"e":"café"}]')

    def test_reasoning_inside_an_array_is_named_by_index(self):
        thinking = {**SKILLS, "items": {"type": "string", "thinking": True}}
        done, _, _ = run({"skills": thinking}, skills_server(1))
        self.assertEqual(done.result, {"skills": ["Python"]})
        self.assertEqual(list(done.thinking), ["skills[0]"])

    def test_invalid_arrays_are_schema_errors(self):
        for schema in (
            {"type": "array"},                                                   # no items
            {"type": "array", "items": "string"},
            {"type": "array", "items": {"type": "array", "items": {"type": "string"}}},  # nested
            {"type": "array", "items": {"type": "date"}},
            {"type": "array", "items": {"type": "string"}, "minItems": -1},
            {"type": "array", "items": {"type": "string"}, "minItems": 1.5},
            {"type": "array", "items": {"type": "string"}, "maxItems": 0},
            {"type": "array", "items": {"type": "string"}, "minItems": 3, "maxItems": 2},
            {"type": "array", "items": {"type": "string"}, "enum": ["a"]},
            {"type": "array", "items": {"type": "string"}, "properties": {}},
            {"type": "array", "items": {"type": "string", "return_probabilities": True, "enum": ["a"]}},
            {"type": "array", "items": {"type": "string", "depends_on": ["x"]}},
            {"type": "array", "items": {"type": "object", "properties": {
                "a": {"type": "array", "items": {"type": "string"}}}}},
            {"type": "string", "items": {"type": "string"}},                     # items on a string
            {"type": "string", "maxItems": 2},
        ):
            with self.subTest(schema=schema), self.assertRaises((SchemaError, NotImplementedError)):
                compile_json_schema({"type": "object", "properties": {"x": {"type": "boolean"}, "a": schema}})

    def test_an_array_continues_from_the_items_it_is_given(self):
        done, sent, _ = run({"skills": {**SKILLS, "continue_from": ["Python", "CUDA"]}}, skills_server(3))
        # The given items come back first; the array goes on from them, as if it had generated them.
        self.assertEqual(done.result, {"skills": ["Python", "CUDA", "PyTorch"]})
        self.assertEqual([current(t) for t in item_prompts(sent)], [["Python", "CUDA"], ["Python", "CUDA", "PyTorch"]])

    def test_an_object_array_continues_from_its_items(self):
        jobs = [("Google", "Engineer", 2020, False), ("Stripe", "Senior Engineer", 2023, True)]
        server = ScriptServer([True], texts=lambda prompt: jobs[len(current(prompt))][["company", "title"].index(field(prompt))],
                              numbers=lambda prompt: jobs[len(current(prompt))][2],
                              choose=lambda prompt, ids: (ids[0] if jobs[len(current(prompt))][3] else ids[1])
                              if 'Field: "current"' in prompt else None)
        google = {"company": "Google", "title": "Engineer", "start_year": 2020, "current": False}
        done, _, server = run({"work_experience": {**WORK, "continue_from": [google]}}, server)
        self.assertEqual(done.result, {"work_experience": [
            google, {"company": "Stripe", "title": "Senior Engineer", "start_year": 2023, "current": True}]})
        # The continue question saw the given item: asked with one item in, then with two.
        self.assertEqual([len(current(t)) for t in server.continue_prompts], [1, 2])

    def test_an_item_repeating_a_given_one_ends_the_array(self):
        done, _, _ = run({"skills": {**SKILLS, "continue_from": ["Python"]}}, ScriptServer([], texts=lambda prompt: "Python"))
        self.assertEqual(done.result, {"skills": ["Python"]})

    def test_min_and_max_items_count_the_given_items(self):
        done, sent, _ = run({"skills": {**SKILLS, "maxItems": 2, "continue_from": ["Python"]}}, skills_server(5))
        self.assertEqual(done.result, {"skills": ["Python", "CUDA"]})
        done, sent, _ = run({"skills": {**SKILLS, "maxItems": 2, "continue_from": ["Python", "CUDA"]}}, skills_server(5))
        self.assertEqual(done.result, {"skills": ["Python", "CUDA"]})
        self.assertEqual(item_prompts(sent), [])
        done, sent, _ = run({"skills": {**SKILLS, "minItems": 2, "continue_from": ["Python"]}}, skills_server(0))
        self.assertEqual(done.result, {"skills": ["Python", "CUDA"]})
        self.assertEqual([NULL_MARK in t for t in item_prompts(sent)], [False, True])

    def test_new_items_are_named_by_their_place_in_the_whole_array(self):
        thinking = {**SKILLS, "items": {"type": "string", "thinking": True}, "continue_from": ["Python"]}
        done, _, _ = run({"skills": thinking}, skills_server(2))
        self.assertEqual(list(done.thinking), ["skills[1]"])

    def test_continue_from_must_match_the_items(self):
        nested = {"type": "array", "items": {"type": "object", "properties": {
            "name": {"type": "string"}, "where": {"type": "object", "properties": {"city": {"type": ["string", "null"]}}},
            "size": {"type": "number", "enum": [1, 2.5]}}}}
        for good in ([], [{"name": "a"}], [{"where": {"city": None}, "size": 1.0}], [{"size": 2.5}]):
            with self.subTest(good=good):
                compile_json_schema({"type": "object", "properties": {"a": {**nested, "continue_from": good}}})
        for array, start in (
            (SKILLS, "Python"),                                   # not a list
            (SKILLS, [1]),
            (SKILLS, [None]),                                     # a string item is never null
            ({"type": "array", "items": {"type": "string", "enum": ["a"]}}, ["b"]),
            ({"type": "array", "items": {"type": "integer"}}, [True]),
            ({"type": "array", "items": {"type": "integer"}}, [1.5]),
            ({"type": "array", "items": {"type": "number"}}, ["1"]),
            ({"type": "array", "items": {"type": "boolean"}}, [1]),
            ({**SKILLS, "maxItems": 1}, ["Python", "CUDA"]),
            (nested, ["a"]),
            (nested, [{"nope": 1}]),
            (nested, [{"where": "Paris"}]),
            (nested, [{"size": 3}]),
            (nested, [{"size": True}]),
        ):
            with self.subTest(array=array, start=start), self.assertRaises(SchemaError):
                compile_json_schema({"type": "object", "properties": {"a": {**array, "continue_from": start}}})
        with self.assertRaises(SchemaError):  # only on an array
            compile_json_schema({"type": "object", "properties": {"a": {"type": "string", "continue_from": []}}})

    def test_a_condition_cannot_test_an_array(self):
        with self.assertRaises(SchemaError):
            compile_json_schema({"type": "object", "properties": {
                "a": SKILLS, "x": {"type": "boolean", "when": {"a": ["Python"]}}}})


WORK = {
    "type": "array",
    "items": {"type": "object", "properties": {
        "company": {"type": "string", "instructions": "Company name."},
        "title": {"type": "string", "instructions": "Job title."},
        "start_year": {"type": "integer", "instructions": "Starting year."},
        "current": {"type": "boolean", "instructions": "Whether this is the current role."},
    }},
    "instructions": "Return all distinct work experiences.",
}


if __name__ == "__main__":
    unittest.main()
