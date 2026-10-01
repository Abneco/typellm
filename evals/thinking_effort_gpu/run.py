"""Live check of thinking_effort and thinking: "auto" against a real SGLang server.

Fixed efforts must think within their budget (none: not at all) and answer correctly.
"auto" cases are questions of clearly different difficulty: the picked level is
recorded, and the field must think exactly when the level is not "none".
Writes results.jsonl and summary.json.
"""
import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from typellm import TypeLLMClient
from typellm.schema import THINKING_EFFORTS

CONTEXT = ("Order 1182: 3 notebooks at $4.25 each and 2 pens at $1.10 each. "
           "The customer is Dana Lee. The order shipped on 2026-03-14.")

# name, question, expected answer
AUTO = [
    ("name", {"type": "string", "enum": ["Dana Lee", "Dan Lee", "Lee Dana"],
              "instructions": "What is the customer's name?"}, "Dana Lee"),
    ("shipped", {"type": "boolean", "instructions": "Did the order ship?"}, True),
    ("total", {"type": "number", "instructions": "What is the order total in dollars?"}, 14.95),
    ("puzzle", {"type": "integer", "instructions": (
        "How many positive integers below 1000 are divisible by 7 but not by 11, and have digits "
        "that sum to an even number?")}, None),  # hard; the answer is not checked
]


def call(client, questions):
    start = time.perf_counter()
    try:
        done = client.generate(context=CONTEXT, questions=questions)
        return {"seconds": round(time.perf_counter() - start, 3), "result": done.result,
                "thinking_fields": sorted(done.thinking), "thinking_effort": done.thinking_effort,
                "thinking_tokens": done.usage.thinking_tokens}
    except Exception as exc:  # recorded, not fatal: one bad case should not hide the rest
        return {"seconds": round(time.perf_counter() - start, 3), "error": f"{type(exc).__name__}: {exc}"}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", default="http://127.0.0.1:30000")
    parser.add_argument("--model", default="qwen3.8-27b")
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--out", type=Path, default=Path(__file__).with_name("results"))
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    client = TypeLLMClient(args.url, model=args.model)
    rows = []

    def record(row):
        rows.append(row)
        print(json.dumps(row), flush=True)

    total = {"type": "number", "instructions": "What is the order total in dollars?"}
    for effort, budget in THINKING_EFFORTS.items():
        row = {"kind": "fixed", "effort": effort, **call(client, {"total": {**total, "thinking_effort": effort}})}
        if "result" in row:
            row["think_ok"] = (row["thinking_fields"] == ["total"]) == (budget is not None) and (
                budget is None or 0 < row["thinking_tokens"] <= budget + 16)  # + closing markers
            row["answer_ok"] = abs(row["result"]["total"] - 14.95) < 1e-6
        record(row)

    for repeat in range(args.repeats):
        for name, question, expected in AUTO:
            row = {"kind": "auto", "case": name, "repeat": repeat,
                   **call(client, {name: {**question, "thinking": "auto"}})}
            if "result" in row:
                level = row["thinking_effort"].get(name)
                row["level"] = level
                row["think_ok"] = level in THINKING_EFFORTS and (name in row["thinking_fields"]) == (level != "none")
                if expected is not None:
                    got = row["result"][name]
                    row["answer_ok"] = abs(got - expected) < 1e-6 if isinstance(expected, float) else got == expected
            record(row)

    # Several auto fields with a dependency: one call, each gets its own level.
    row = {"kind": "auto_graph", **call(client, {
        "total": {**total, "thinking": "auto"},
        "puzzle": {**AUTO[3][1], "thinking": "auto"},
        "over_ten": {"type": "boolean", "instructions": "Is total over 10 dollars?",
                     "depends_on": ["total"], "thinking": "auto"},
    })}
    if "result" in row:
        row["think_ok"] = all((f in row["thinking_fields"]) == (row["thinking_effort"].get(f) != "none")
                              for f in ("total", "puzzle", "over_ten")) and set(row["result"]) == {
                              "total", "puzzle", "over_ten"}
        row["answer_ok"] = row["result"]["over_ten"] is True
    record(row)

    with open(args.out / "results.jsonl", "w") as f:
        f.writelines(json.dumps(r) + "\n" for r in rows)
    levels = {}
    for r in rows:
        if r["kind"] == "auto" and "level" in r:
            levels.setdefault(r["case"], []).append(r["level"])
    summary = {
        "calls": len(rows),
        "errors": sum("error" in r for r in rows),
        "think_ok": f'{sum(r.get("think_ok", False) for r in rows)}/{len(rows)}',
        "answer_ok": f'{sum(r.get("answer_ok", False) for r in rows)}/{sum("answer_ok" in r for r in rows)}',
        "auto_levels": levels,
    }
    (args.out / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
