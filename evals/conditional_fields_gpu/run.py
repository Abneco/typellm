"""Live check of "when" (conditional fields) against a real SGLang server.

Tickets whose category is clear: a bug or incident must run severity and its
dependent; a feature request must skip both, with no requests sent for them.
Expense claims test the operators on numbers the model reads: each conditional
field must run exactly when its test passes on the answers the model gave.
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

CLAIMS = [
    "Expense claim: team dinner at Nobu, total $1,240.00, tip of $180 included. Category: meal.",
    "Expense claim: taxi from the airport, total $38.50, no tip. Category: travel.",
    "Expense claim: 2 monitors for the office, total $640.00. Category: equipment.",
    "Expense claim: coffee with a client, total $12.40, tip $2. Category: meal.",
]
# name: (when, the same test written by hand in Python, independent of typellm)
OPERATOR_FIELDS = {
    "approval": ({"amount": {"gte": 1000}}, lambda a: a["amount"] >= 1000),
    "small_receipt": ({"amount": {"gt": 0, "lte": 50}}, lambda a: 0 < a["amount"] <= 50),
    "tip_check": ({"tip": {"ne": None}}, lambda a: a["tip"] is not None),
    "no_tip_note": ({"tip": None}, lambda a: a["tip"] is None),
    "non_meal": ({"category": {"not_in": ["meal"]}}, lambda a: a["category"] not in ("meal",)),
    "equipment_over_500": ({"category": "equipment", "amount": {"gt": 500}},
                           lambda a: a["category"] == "equipment" and a["amount"] > 500),
    "not_travel": ({"category": {"ne": "travel"}}, lambda a: a["category"] != "travel"),
}

TICKETS = [
    ("bug", "The export button crashes the app with a null pointer error every time I click it."),
    ("incident", "Production is down: all checkout requests have returned HTTP 500 for the last 20 minutes."),
    ("feature_request", "It would be great if the dashboard had a dark mode option."),
    ("feature_request", "Please add CSV export to the reports page."),
]


def questions(auto):
    severity = {"type": "string", "enum": ["low", "medium", "high"], "instructions": "How severe is it?",
                "depends_on": ["category"], "when": {"category": ["bug", "incident"]}}
    if auto:
        severity["thinking"] = "auto"
    return {
        "category": {"type": "string", "enum": ["bug", "incident", "feature_request"],
                     "instructions": "What kind of ticket is this?"},
        "severity": severity,
        "page_on_call": {"type": "boolean", "instructions": "Should the on-call engineer be paged now?",
                         "depends_on": ["severity"]},
        "reply_needed": {"type": "boolean", "instructions": "Does the customer need a reply?",
                         "depends_on": ["category"]},
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", default="http://127.0.0.1:30000")
    parser.add_argument("--model", default="qwen3.8-27b")
    parser.add_argument("--out", type=Path, default=Path(__file__).with_name("results"))
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    client = TypeLLMClient(args.url, model=args.model)
    sent = []
    original = client.sglang._request

    def record(path, payload=None, **kwargs):
        if path == "/generate":
            sent.extend([payload["text"]] if isinstance(payload["text"], str) else payload["text"])
        return original(path, payload, **kwargs)

    client.sglang._request = record
    rows = []
    for auto in (False, True):
        for expected, ticket in TICKETS:
            sent.clear()
            start = time.perf_counter()
            row = {"auto": auto, "expected": expected}
            try:
                done = client.generate(context=ticket, questions=questions(auto))
                asked = {name: any(f'Field: "{name}"' in text for text in sent)
                         for name in ("severity", "page_on_call", "severity.thinking_effort")}
                row.update(seconds=round(time.perf_counter() - start, 3), result=done.result,
                           skipped=done.skipped, thinking_effort=done.thinking_effort, asked=asked)
                row["category_ok"] = done.result["category"] == expected
                # Judged on the category the model gave, so a misread ticket is not a branch failure.
                given_runs = done.result["category"] != "feature_request"
                row["branch_ok"] = (
                    ("severity" in done.result) == given_runs and ("page_on_call" in done.result) == given_runs
                    and done.skipped == ([] if given_runs else ["severity", "page_on_call"])
                    and asked["severity"] == given_runs and asked["page_on_call"] == given_runs
                    and asked["severity.thinking_effort"] == (given_runs and auto)
                    and "reply_needed" in done.result)
            except Exception as exc:  # recorded, not fatal
                row["error"] = f"{type(exc).__name__}: {exc}"
            rows.append(row)
            print(json.dumps(row), flush=True)

    # Operators on numbers the model reads; judged on its own answers.
    for claim in CLAIMS:
        sent.clear()
        start = time.perf_counter()
        claim_questions = {
            "amount": {"type": "number", "instructions": "What is the claim total in dollars?"},
            "tip": {"type": ["number", "null"], "instructions": "What tip in dollars is included? null if none."},
            "category": {"type": "string", "enum": ["meal", "travel", "equipment"],
                         "instructions": "What is the claim's category?"},
        }
        for name, (when, _) in OPERATOR_FIELDS.items():
            claim_questions[name] = {"type": "boolean", "instructions": f"Is this claim fine ({name})?",
                               "depends_on": list(when), "when": when}
        row = {"kind": "operators", "claim": claim[:40]}
        try:
            done = client.generate(context=claim, questions=claim_questions)
            answers = {k: done.result[k] for k in ("amount", "tip", "category")}
            should = {name: bool(check(answers)) for name, (_, check) in OPERATOR_FIELDS.items()}
            ran = {name: name in done.result for name in OPERATOR_FIELDS}
            asked = {name: any(f'Field: "{name}"' in text for text in sent) for name in OPERATOR_FIELDS}
            row.update(seconds=round(time.perf_counter() - start, 3), answers=answers, ran=ran,
                       skipped=done.skipped, branch_ok=(ran == should == asked
                                                        and done.skipped == [n for n in OPERATOR_FIELDS if not should[n]]))
        except Exception as exc:  # recorded, not fatal
            row["error"] = f"{type(exc).__name__}: {exc}"
        rows.append(row)
        print(json.dumps(row), flush=True)

    with open(args.out / "results.jsonl", "w") as f:
        f.writelines(json.dumps(r) + "\n" for r in rows)
    summary = {"calls": len(rows), "errors": sum("error" in r for r in rows),
               "branch_ok": f'{sum(r.get("branch_ok", False) for r in rows)}/{len(rows)}',
               "category_ok": f'{sum(r.get("category_ok", False) for r in rows)}/{len(rows)}'}
    (args.out / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
