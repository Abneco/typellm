"""Compilation for TypeLLM's schema subset and bounded open values."""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, replace
from typing import Any, Mapping, Sequence


MAX_ENUM_CHOICES = 24
MAX_PERMUTATIONS = 720


class SchemaError(ValueError):
    pass


@dataclass(frozen=True)
class Decision:
    """Tokenizer-independent decision compiled from ordinary JSON Schema."""

    name: str
    question: str
    choices: tuple[Any, ...]
    syntax: str = "Choice"
    numeric_type: str | None = None
    text_type: bool = False
    permutations: int | str = 1
    return_probabilities: bool = False
    depends_on: tuple[str, ...] | None = None
    nullable: bool = False
    # None follows the client's thinking and thinking_budget.
    thinking: bool | None = None
    thinking_budget: int | None = None
    # thinking: "auto" adds a hidden question that picks the effort per call:
    # effort_from on the field names that question, effort_for on it names the field.
    effort_from: str | None = None
    effort_for: str | None = None
    # "when": run only if every named dependency's answer passes its tests;
    # kept as ((field, ((operator, operand), ...)), ...). A skipped field skips its dependents too.
    when: tuple[tuple[str, tuple[tuple[str, Any], ...]], ...] | None = None


# thinking_effort levels and their budgets; "none" does not think.
THINKING_EFFORTS = {"none": None, "low": 512, "medium": 2048, "high": 4096}

EFFORT_QUESTION = (
    "Before the question below is answered, decide how much step-by-step reasoning it needs.\n"
    "none: the answer is stated directly or is obvious.\n"
    "low: a short check or one simple inference.\n"
    "medium: several steps of reasoning or calculation.\n"
    "high: hard, many-step or high-stakes reasoning.\n\n"
    "The question:\n{question}"
)


def _describe(decision: Decision) -> str:
    """A field's type, instructions and choices, as its own prompt states them."""
    or_null = " or null" if decision.nullable else ""
    if decision.text_type:
        kind = "string" + or_null
    elif decision.numeric_type is not None:
        kind = decision.numeric_type + or_null
    else:
        kind = "boolean" if decision.syntax == "Bool" else "choice"
    lines = [f"Type: {kind}", f"Instructions: {decision.question}"]
    if decision.choices and decision.syntax != "Bool":
        lines.append(f"Choices: {json.dumps(list(decision.choices), ensure_ascii=False)}")
    return "\n".join(lines)


def _same_value(a: Any, b: Any) -> bool:
    """JSON equality: 1 and 1.0 match, true and 1 do not."""
    if type(a) in {int, float} and type(b) in {int, float}:
        return a == b
    return type(a) is type(b) and a == b


def _has_duplicates(values: Sequence[Any]) -> bool:
    for index, value in enumerate(values):
        if any(_same_value(value, previous) for previous in values[:index]):
            return True
    return False


def _is_finite_number(value: Any) -> bool:
    # Exact types exclude bool; Python ints are finite at any magnitude.
    return type(value) is int or (type(value) is float and math.isfinite(value))


def compile_json_schema(schema: Mapping[str, Any]) -> list[Decision]:
    """Compile an ordered JSON Schema object into TypeLLM decisions."""
    if not isinstance(schema, Mapping):
        raise SchemaError("schema must be a mapping")
    if schema.get("type") != "object":
        raise SchemaError("root JSON Schema type must be 'object'")
    properties = schema.get("properties")
    if not isinstance(properties, Mapping) or not properties:
        raise SchemaError("JSON Schema properties must be a non-empty object")

    required = schema.get("required", [])
    if not isinstance(required, list) or any(not isinstance(x, str) for x in required):
        raise SchemaError("JSON Schema required must be a list of field names")
    if len(set(required)) != len(required):
        raise SchemaError("JSON Schema required contains duplicate field names")
    unknown_required = [name for name in required if name not in properties]
    if unknown_required:
        raise SchemaError(
            f"required fields are missing from properties: {unknown_required!r}"
        )

    decisions: list[Decision] = []
    for name, field in properties.items():
        if not isinstance(name, str) or not name:
            raise SchemaError("property names must be non-empty strings")
        if not isinstance(field, Mapping):
            raise SchemaError(f"property {name!r} must be a schema object")

        instructions = field.get("instructions")
        if "instructions" in field and not isinstance(instructions, str):
            raise SchemaError(f"instructions for {name!r} must be a string")
        for old_key in ("question", "x-question"):
            if old_key in field:
                raise SchemaError(f"{old_key} for {name!r} is no longer supported; use instructions")
        # Decoding cannot hold a model to a range, so numeric bounds are not offered.
        for keyword in ("minimum", "maximum", "exclusiveMinimum", "exclusiveMaximum"):
            if keyword in field:
                raise SchemaError(f"{keyword} is not supported (on {name!r}); use an enum for a fixed set of values")
        description = field.get("description")
        if "description" in field and not isinstance(description, str):
            raise SchemaError(f"description for {name!r} must be a string")
        question = (
            instructions
            if instructions is not None
            else description
            if description is not None
            else f'Choose the value for "{name}".'
        )

        field_type = field.get("type")
        # ["string", "null"] and the like: one value type that may also be null.
        nullable = False
        if isinstance(field_type, list):
            kinds = [kind for kind in field_type if kind != "null"]
            if len(field_type) != 2 or len(kinds) != 1 or not isinstance(kinds[0], str):
                raise SchemaError(f'type for {name!r} must be one type or [type, "null"]')
            field_type, nullable = kinds[0], True
        enum = field.get("enum")
        permutations = field.get("permutations", 1)
        if "permutations" in field:
            if enum is None:
                raise SchemaError(f"permutations for {name!r} requires an explicit enum")
            if not (permutations in ("auto", "all") or type(permutations) is int and permutations > 0):
                raise SchemaError(f"permutations for {name!r} must be 'auto', 'all' or a positive integer")
            if isinstance(enum, list) and permutations != "auto":
                count = math.factorial(len(enum))
                budget = count if permutations == "all" else min(permutations, count)
                if budget > MAX_PERMUTATIONS:
                    raise SchemaError(f"permutations for {name!r} exceeds {MAX_PERMUTATIONS}; use a smaller integer budget")
        return_probabilities = field.get("return_probabilities", False)
        if type(return_probabilities) is not bool:
            raise SchemaError(f"return_probabilities for {name!r} must be a boolean")
        if "return_probabilities" in field and field_type != "boolean" and enum is None:
            raise SchemaError(f"return_probabilities for {name!r} is only supported for enum or boolean fields")
        if "x-score" in field:
            raise SchemaError(
                f"x-score for {name!r} is not supported; use a number enum"
            )
        if "x-other" in field:
            raise SchemaError(
                f"x-other for {name!r} is not supported; use a closed enum"
            )
        # Checking a length means decoding every token back to text, which slows every
        # string; answers stop at the client's text_max_tokens instead.
        if "maxLength" in field:
            raise SchemaError(f"maxLength is not supported (on {name!r}); string answers stop at "
                              "text_max_tokens, so ask for the length you want in instructions")
        if field_type == "string" and enum is None:
            for keyword in ("minLength", "pattern", "format"):
                if keyword in field:
                    raise SchemaError(f"{keyword} is not supported for text fields")
            decisions.append(Decision(name, question, (), "Text", text_type=True, nullable=nullable))
            continue
        if field_type == "boolean":
            values = ([True, False] + [None] * nullable) if enum is None else enum
            if not isinstance(values, list) or not values:
                raise SchemaError(f"enum for {name!r} must be a non-empty list")
            if any(type(value) is not bool and not (nullable and value is None) for value in values):
                raise SchemaError(f"boolean enum for {name!r} may contain only booleans")
            syntax = "Bool"
        elif field_type in {"integer", "number"} and enum is None:
            decisions.append(
                Decision(
                    name=name,
                    question=question,
                    choices=(),
                    syntax="Integer" if field_type == "integer" else "Number",
                    numeric_type=field_type,
                    nullable=nullable,
                )
            )
            continue
        elif field_type in {"string", "integer", "number"}:
            if enum is None:
                raise NotImplementedError(
                    f"property {name!r} has type {field_type!r} without a finite enum"
                )
            if not isinstance(enum, list) or not enum:
                raise SchemaError(f"enum for {name!r} must be a non-empty list")
            if len(enum) > MAX_ENUM_CHOICES:
                raise SchemaError(
                    f"enum for {name!r} has {len(enum)} values; "
                    f"the maximum is {MAX_ENUM_CHOICES}"
                )
            values = enum
            # As in JSON Schema, null is allowed only when the enum lists it.
            typed = [value for value in values if not (nullable and value is None)]
            if field_type == "string":
                valid = all(isinstance(value, str) for value in typed)
            elif field_type == "integer":
                valid = all(type(value) is int for value in typed)
            else:
                valid = all(_is_finite_number(value) for value in typed)
            if not valid:
                raise SchemaError(
                    f"enum values for {name!r} do not match type {field_type!r}"
                )
            syntax = "Choice"
        else:
            raise NotImplementedError(
                f"property {name!r} has unsupported JSON Schema type {field_type!r}"
            )

        if len(values) > MAX_ENUM_CHOICES:
            raise SchemaError(
                f"enum for {name!r} has {len(values)} values; "
                f"the maximum is {MAX_ENUM_CHOICES}"
            )
        if _has_duplicates(values):
            raise SchemaError(f"enum for {name!r} contains duplicate values")
        decisions.append(
            Decision(name, question, tuple(values), syntax, return_probabilities=return_probabilities,
                     permutations=permutations, nullable=nullable)
        )

    by_name = {decision.name: decision for decision in decisions}
    compiled = []
    for decision in decisions:
        field = properties[decision.name]
        dependencies = field.get("depends_on")
        if "depends_on" in field:
            if not isinstance(dependencies, list) or any(
                not isinstance(name, str) or not name for name in dependencies
            ):
                raise SchemaError(f"depends_on for {decision.name!r} must be a list of field names")
            if len(set(dependencies)) != len(dependencies):
                raise SchemaError(f"depends_on for {decision.name!r} contains duplicates")
            # Only the caller's fields: the hidden effort questions are not theirs to name.
            for dependency in dependencies:
                if dependency not in properties:
                    raise SchemaError(f"unknown dependency {dependency!r} for {decision.name!r}")
            dependencies = tuple(dependencies)
        when = _condition(decision.name, field, dependencies, by_name) if "when" in field else None
        thinking, budget = _thinking_settings(decision.name, field)
        if thinking == "auto":
            effort = f"{decision.name}.thinking_effort"
            if effort in properties:
                raise SchemaError(f"field name {effort!r} is taken by thinking: \"auto\" on {decision.name!r}")
            # Asked with the field's own dependencies, never thinks, and runs one layer before it.
            # The effort question shares the field's condition: a skipped field asks nothing.
            compiled.append(Decision(effort, EFFORT_QUESTION.format(question=_describe(decision)),
                                     tuple(THINKING_EFFORTS), depends_on=dependencies, thinking=False,
                                     effort_for=decision.name, when=when))
            compiled.append(replace(decision, depends_on=(dependencies or ()) + (effort,), effort_from=effort,
                                    when=when))
            continue
        compiled.append(replace(decision, depends_on=dependencies, thinking=thinking, thinking_budget=budget,
                                when=when))
    dependency_layers(compiled)
    return compiled


# "when" operators. A bare value means {"in": [value]}, and a list {"in": list}.
COMPARISONS = {"gt": lambda a, b: a > b, "gte": lambda a, b: a >= b,
               "lt": lambda a, b: a < b, "lte": lambda a, b: a <= b}
OPERATORS = ("in", "not_in", "ne", *COMPARISONS)


def _numeric(decision: Decision) -> bool:
    """An integer or number field, open or with a numeric enum."""
    if decision.numeric_type is not None:
        return True
    values = [value for value in decision.choices if value is not None]
    return decision.syntax == "Choice" and bool(values) and all(_is_finite_number(v) for v in values)


def _can_answer(decision: Decision, value: Any) -> bool:
    """Whether the field can give this answer, so a condition on it can ever hold."""
    if decision.choices:
        return any(_same_value(value, choice) for choice in decision.choices)
    if value is None:
        return decision.nullable
    if decision.numeric_type == "integer":
        return _is_finite_number(value) and float(value).is_integer()
    return decision.numeric_type is not None and _is_finite_number(value)


def _condition(name: str, field: Mapping[str, Any], dependencies: tuple[str, ...] | None,
               by_name: Mapping[str, Decision]) -> tuple:
    """A field's "when" as ((field, ((operator, operand), ...)), ...), checked before anything runs.

    It may name only fields listed in depends_on. Equality tests (a value, a list,
    in, not_in, ne) take answers the field can give: an enum value, true or false, a
    number, or null for a nullable field. gt, gte, lt and lte take numbers, on number
    fields. Open text fields cannot be conditions, except for null.
    """
    when = field["when"]
    if not isinstance(when, Mapping) or not when:
        raise SchemaError(f"when for {name!r} must map dependencies to the answers that run it")
    condition = []
    for parent, test in when.items():
        if parent not in (dependencies or ()):
            raise SchemaError(f"when for {name!r} names {parent!r}, which is not in its depends_on")
        source = by_name[parent]
        if not isinstance(test, Mapping):
            test = {"in": test if isinstance(test, list) else [test]}
        if not test:
            raise SchemaError(f"when for {name!r} has no test for {parent!r}")
        tests = []
        for operator, operand in test.items():
            if operator not in OPERATORS:
                raise SchemaError(f"when for {name!r}: unknown operator {operator!r} on {parent!r}; "
                                  f"use one of {list(OPERATORS)}")
            if operator in COMPARISONS:
                if not _numeric(source):
                    raise SchemaError(f"when for {name!r}: {operator} needs a number field, and {parent!r} is not")
                if not _is_finite_number(operand):
                    raise SchemaError(f"when for {name!r}: {operator} on {parent!r} must be a number")
            else:
                if operator in ("in", "not_in"):
                    if not isinstance(operand, list) or not operand:
                        raise SchemaError(f"when for {name!r}: {operator} on {parent!r} must be a non-empty list")
                    operand = tuple(operand)
                for value in operand if operator != "ne" else (operand,):
                    if not _can_answer(source, value):
                        raise SchemaError(f"when for {name!r}: {value!r} is not an answer {parent!r} can give")
            tests.append((operator, operand))
        condition.append((parent, tuple(tests)))
    return tuple(condition)


def _passes(operator: str, operand: Any, answer: Any) -> bool:
    if operator == "in":
        return any(_same_value(answer, value) for value in operand)
    if operator == "not_in":
        return not any(_same_value(answer, value) for value in operand)
    if operator == "ne":
        return not _same_value(answer, operand)
    # A comparison holds only for a number: null, for one, is not above or below anything.
    return _is_finite_number(answer) and COMPARISONS[operator](answer, operand)


def condition_met(decision: Any, answers: Mapping[str, Any]) -> bool:
    """Whether a decision with a "when" runs, given its dependencies' answers."""
    return all(_passes(operator, operand, answers[parent])
               for parent, tests in decision.when for operator, operand in tests)


def _thinking_settings(name: str, field: Mapping[str, Any]) -> tuple[bool | str | None, int | None]:
    """A field's thinking ("auto" included) and budget, from thinking, thinking_effort and thinking_budget."""
    thinking = field.get("thinking")
    if thinking is not None and type(thinking) is not bool and thinking != "auto":
        raise SchemaError(f'thinking for {name!r} must be a boolean or "auto"')
    budget = field.get("thinking_budget")
    if budget is not None and (type(budget) is not int or budget <= 0):
        raise SchemaError(f"thinking_budget for {name!r} must be a positive integer")
    if thinking == "auto" and budget is not None:
        raise SchemaError(f'thinking_budget for {name!r} cannot be set with thinking: "auto", '
                          "which picks the effort itself")
    if "thinking_effort" not in field:
        return thinking, budget
    effort = field["thinking_effort"]
    if not isinstance(effort, str) or effort not in THINKING_EFFORTS:
        raise SchemaError(f"thinking_effort for {name!r} must be one of {list(THINKING_EFFORTS)}")
    if thinking == "auto":
        raise SchemaError(f'thinking_effort for {name!r} cannot be set with thinking: "auto", '
                          "which picks the effort itself")
    if budget is not None:
        raise SchemaError(f"thinking_effort and thinking_budget for {name!r} both set the budget; use one")
    thinks = effort != "none"
    if thinking is not None and thinking != thinks:
        raise SchemaError(f"thinking: {str(thinking).lower()} for {name!r} contradicts thinking_effort {effort!r}")
    return thinks, THINKING_EFFORTS[effort]


def dependency_layers(decisions: Sequence) -> list[list]:
    """Stable topological layers, validated before any model requests."""
    names = {decision.name for decision in decisions}
    for decision in decisions:
        for dependency in decision.depends_on or ():
            if dependency not in names:
                raise SchemaError(f"unknown dependency {dependency!r} for {decision.name!r}")
            if dependency == decision.name:
                raise SchemaError(f"field {decision.name!r} cannot depend on itself")
    remaining = list(decisions)
    completed = set()
    layers = []
    while remaining:
        layer = [d for d in remaining if set(d.depends_on or ()) <= completed]
        if not layer:
            raise SchemaError(f"dependency cycle among fields: {[d.name for d in remaining]!r}")
        layers.append(layer)
        completed.update(d.name for d in layer)
        remaining = [d for d in remaining if d.name not in completed]
    return layers
