"""Minimal SGLang reproducer: token_ids_logprob requests mixed with text generation.

No TypeLLM involved. Each condition runs concurrent raw /generate requests for
a few seconds, then checks the server survived. Run against a server you can
afford to crash.
"""
import json
import sys
import threading
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor

URL = sys.argv[1] if len(sys.argv) > 1 else "http://127.0.0.1:30000"
SECONDS = 20
PROMPT = "<|im_start|>user\nA single roll of a fair die. Which face?<|im_end|>\n<|im_start|>assistant\n"


def post(payload, timeout=60):
    request = urllib.request.Request(URL + "/generate", json.dumps(payload).encode(),
                                     {"Content-Type": "application/json"})
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return json.loads(response.read())


def healthy():
    try:
        with urllib.request.urlopen(URL + "/health", timeout=5) as response:
            return response.status == 200
    except Exception:
        return False


def wait_healthy(limit=600):
    start = time.time()
    while time.time() - start < limit:
        if healthy():
            return True
        time.sleep(5)
    return False


def score(n):
    # TypeLLM's candidate scoring: one output token, logprobs of given token ids.
    return post({"text": [f"{PROMPT}{n}", f"{PROMPT}{n} "],
                 "sampling_params": {"max_new_tokens": 1, "temperature": 0},
                 "return_logprob": True, "token_ids_logprob": [[16, 17, 18], [16, 17, 18]],
                 "return_text_in_logprobs": True})


def text(n, temperature):
    return post({"text": [f"{PROMPT}{n}"],
                 "sampling_params": [{"max_new_tokens": 48, "temperature": temperature,
                                      "sampling_seed": n}]})


JSON_CHAR = r'(?:[^"\\\x00-\x1f]|\\["\\/bfnrt]|\\u[0-9a-fA-F]{4})'


def regex_text(n, temperature):
    # TypeLLM's text fields: the prompt ends at '{"name":', a regex closes the string.
    return post({"text": [f'{PROMPT}{{"name":'],
                 "sampling_params": [{"max_new_tokens": 48, "temperature": temperature,
                                      "sampling_seed": n, "regex": ' ?"' + JSON_CHAR + '*"\\}'}]})


CONDITIONS = {
    "score_only": [score],
    "text_only_sampled": [lambda n: text(n, 1.0)],
    "mixed_sampled": [score, lambda n: text(n, 1.0)],
    "mixed_greedy": [score, lambda n: text(n, 0.0)],
    "regex_only": [lambda n: regex_text(n, 1.0)],
    "mixed_regex_sampled": [score, lambda n: regex_text(n, 1.0)],
    "mixed_regex_greedy": [score, lambda n: regex_text(n, 0.0)],
}


def run(name, kinds):
    stop = time.time() + SECONDS
    counts, errors = {"ok": 0}, []
    lock = threading.Lock()

    def worker(w):
        n = w * 1000
        while time.time() < stop:
            n += 1
            try:
                kinds[n % len(kinds)](n)
                with lock:
                    counts["ok"] += 1
            except Exception as exc:
                with lock:
                    errors.append(repr(exc)[:160])
                return

    with ThreadPoolExecutor(16) as pool:
        list(pool.map(worker, range(16)))
    alive = healthy()
    row = {"condition": name, "ok_requests": counts["ok"], "errors": len(errors),
           "first_error": errors[0] if errors else None, "server_alive": alive}
    print(json.dumps(row), flush=True)
    if not alive:
        print(json.dumps({"condition": name, "restarted": wait_healthy()}), flush=True)
    return row


if __name__ == "__main__":
    names = sys.argv[2:] or list(CONDITIONS)
    assert wait_healthy(), "server not ready"
    for name in names:
        run(name, CONDITIONS[name])
