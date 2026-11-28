#!/usr/bin/env python3
"""Quick latency regression benchmark for the Rust /api/chat/stream endpoint.

Posts a deterministic prompt at several generation lengths, reads the NDJSON
`done` event, and writes raw samples plus a per-length mean/min speed summary.

Usage:
    python3 bench/latency_bench.py --runs 3 --tokens 10,32,64 --out bench/latency.json
"""

import argparse
import json
import os
import statistics
import sys
import urllib.error
import urllib.request

DEFAULT_URL = "http://127.0.0.1:8080"
DEFAULT_OUT = "bench/latency.json"
DEFAULT_TOKENS = "10,32,64"
DEFAULT_PROMPT_WORDS = 40
REQUEST_TIMEOUT = 900
DONE_FIELDS = (
    "prompt_tokens",
    "prompt_ms",
    "prompt_tok_s",
    "tokens",
    "gen_ms",
    "tok_s",
)
SENTENCE = (
    "The quick brown fox jumps over the lazy dog. "
    "Быстрая коричневая лиса перепрыгивает через ленивую собаку."
)


class BenchError(Exception):
    """A benchmark failure that should abort with a clear message."""


def positive_int(value):
    try:
        number = int(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(f"expected an integer, got {value!r}") from exc
    if number < 1:
        raise argparse.ArgumentTypeError(f"expected a positive integer, got {number}")
    return number


def token_list(value):
    try:
        numbers = [int(part.strip()) for part in value.split(",")]
    except ValueError as exc:
        raise argparse.ArgumentTypeError(
            f"expected comma-separated integers, got {value!r}"
        ) from exc
    if any(number < 1 for number in numbers):
        raise argparse.ArgumentTypeError("token counts must be positive")
    return numbers


def build_prompt(word_count):
    words = SENTENCE.split()
    return " ".join(words[i % len(words)] for i in range(word_count))


def read_done(response, endpoint):
    done = None
    for raw in response:
        line = raw.decode("utf-8", "replace").strip()
        if not line:
            continue
        try:
            event = json.loads(line)
        except ValueError as exc:
            raise BenchError(f"invalid NDJSON from {endpoint}: {line!r}") from exc
        if not isinstance(event, dict):
            raise BenchError(f"invalid NDJSON event from {endpoint}: {line!r}")
        kind = event.get("type")
        if kind == "error":
            raise BenchError(
                f"server error on {endpoint}: {event.get('error', 'unknown error')}"
            )
        if kind == "done":
            done = event
    if done is None:
        raise BenchError(f"stream from {endpoint} ended without a done event")
    sample = {}
    for field in DONE_FIELDS:
        value = done.get(field)
        if not isinstance(value, (int, float)) or isinstance(value, bool):
            raise BenchError(f"done event from {endpoint} has invalid {field}: {value!r}")
        sample[field] = value
    return sample


def post_stream(url, prompt, n_predict):
    endpoint = url.rstrip("/") + "/api/chat/stream"
    body = json.dumps({
        "prompt": prompt,
        "n_predict": n_predict,
        "temp": 0,
        "top_p": 0.9,
    }).encode("utf-8")
    request = urllib.request.Request(
        endpoint,
        data=body,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=REQUEST_TIMEOUT) as response:
            return read_done(response, endpoint)
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", "replace").strip()
        raise BenchError(f"HTTP {exc.code} from {endpoint}: {detail or exc.reason}") from exc
    except urllib.error.URLError as exc:
        raise BenchError(f"cannot reach {endpoint}: {exc.reason}") from exc
    except OSError as exc:
        raise BenchError(f"request to {endpoint} failed: {exc}") from exc


def summarize(samples):
    return {
        "prompt_tok_s": {
            "mean": round(statistics.mean(s["prompt_tok_s"] for s in samples), 2),
            "min": round(min(s["prompt_tok_s"] for s in samples), 2),
        },
        "tok_s": {
            "mean": round(statistics.mean(s["tok_s"] for s in samples), 2),
            "min": round(min(s["tok_s"] for s in samples), 2),
        },
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--url", default=DEFAULT_URL)
    parser.add_argument("--out", default=DEFAULT_OUT)
    parser.add_argument("--runs", type=positive_int, default=3)
    parser.add_argument("--tokens", type=token_list, default=token_list(DEFAULT_TOKENS))
    parser.add_argument("--prompt-words", type=positive_int, default=DEFAULT_PROMPT_WORDS)
    args = parser.parse_args()

    prompt = build_prompt(args.prompt_words)
    report = {
        "url": args.url,
        "prompt_words": args.prompt_words,
        "prompt_chars": len(prompt),
        "runs": args.runs,
        "samples": {},
        "summary": {},
    }

    try:
        for count in args.tokens:
            samples = []
            key = str(count)
            report["samples"][key] = samples
            for run in range(1, args.runs + 1):
                sample = post_stream(args.url, prompt, count)
                sample["run"] = run
                sample["n_predict"] = count
                samples.append(sample)
                print(
                    f"tokens={count} run={run}/{args.runs} "
                    f"prefill {sample['prompt_tok_s']:.2f} tok/s "
                    f"({sample['prompt_tokens']} tok, {sample['prompt_ms']} ms), "
                    f"gen {sample['tok_s']:.2f} tok/s "
                    f"({sample['tokens']} tok, {sample['gen_ms']} ms)",
                    flush=True,
                )
            report["summary"][key] = summarize(samples)
    except BenchError as exc:
        print("error: " + str(exc), file=sys.stderr)
        return 1

    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    out = os.path.normpath(os.path.join(root, args.out))
    os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(out, "w", encoding="utf-8") as fh:
        json.dump(report, fh, ensure_ascii=False, indent=2)
    print("report: " + out, flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
