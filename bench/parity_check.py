#!/usr/bin/env python3
"""Check that CPU and CUDA generation return the same text for each prompt.

Runs the two binaries one at a time, so only one copy of the model sits in VRAM.
Writes a per-prompt JSON report to bench/parity_report.json.

Usage:
    python3 bench/parity_check.py --tokens 12
"""

import argparse
import json
import os
import subprocess
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BINARY = "target/release/llama-rs"
DEFAULT_MODEL = "models/Qwen3.5-4B-Q4_K_M.gguf"
DEFAULT_PROMPTS = "bench/parity_prompts.txt"
DEFAULT_TOKENS = 12
REPORT = "bench/parity_report.json"
TIMEOUT_SECONDS = 300
GPU_MARKER = "GPU graph: all layers on device (greedy sampling)"

# Written once to DEFAULT_PROMPTS: English, Russian, code, long question,
# short question, numbers, emoji, empty-ish. Escapes keep this file ASCII.
STARTER_PROMPTS = (
    "Explain in one sentence why the sky is blue.",
    "\u041e\u0431\u044a\u044f\u0441\u043d\u0438 \u043e\u0434\u043d\u0438\u043c "
    "\u043f\u0440\u0435\u0434\u043b\u043e\u0436\u0435\u043d\u0438\u0435\u043c, "
    "\u043f\u043e\u0447\u0435\u043c\u0443 \u043d\u0435\u0431\u043e \u0433\u043e\u043b\u0443\u0431\u043e\u0435.",
    "Write Python: def add(a, b): return a + b  # explain the result",
    "A library opened in 1902 with 3000 books, grew to 2.4 million items by "
    "2000, added three branches, and launched a digital archive in 2011 that "
    "now holds 18 million scanned pages. In what year did the archive launch "
    "and how many pages does it hold? Answer in one short sentence.",
    "Why?",
    "What is 17 times 24, and which primes are below 30?",
    "What does this mean: \U0001f600?",
    ".",
)


def load_prompts(path):
    """Create the starter file on first run, then read one prompt per line."""
    if not os.path.exists(path):
        parent = os.path.dirname(path)
        if parent:
            os.makedirs(parent, exist_ok=True)
        with open(path, "w", encoding="utf-8") as handle:
            handle.write("\n".join(STARTER_PROMPTS) + "\n")
    with open(path, encoding="utf-8") as handle:
        return [line.strip() for line in handle if line.strip()]


def run_binary(prompt, tokens, model, gpu):
    """Run one binary; subprocess.run kills the child when timeout fires."""
    command = [
        BINARY,
        "--model", model,
        "--prompt", prompt,
        "-n", str(tokens),
        "--temp", "0",
    ]
    if gpu:
        command.append("--gpu")
    try:
        process = subprocess.run(
            command,
            cwd=ROOT,
            capture_output=True,
            text=True,
            timeout=TIMEOUT_SECONDS,
        )
    except subprocess.TimeoutExpired:
        return {
            "stdout": "",
            "stderr": "timeout after %d seconds" % TIMEOUT_SECONDS,
            "returncode": None,
        }
    except OSError as exc:
        return {"stdout": "", "stderr": str(exc), "returncode": None}
    return {
        "stdout": process.stdout,
        "stderr": process.stderr,
        "returncode": process.returncode,
    }


def generated_text(stdout, gpu):
    """Strip the CLI GPU banner and the trailing newline added by println!."""
    if gpu:
        lines = stdout.splitlines()
        for index, line in enumerate(lines):
            if line.startswith(GPU_MARKER):
                stdout = "\n".join(lines[index + 1:])
                break
    return stdout.rstrip("\n")


def error_detail(result, side):
    detail = result["stderr"].strip()
    if not detail:
        detail = "exit code %s" % result["returncode"]
    return "%s: %s" % (side, detail)


def short(value, limit=80):
    """ascii() escapes non-ASCII, so console lines stay free of emoji."""
    value = ascii(value)
    return value if len(value) <= limit else value[:limit] + "..."


def compare(cpu_text, gpu_text):
    """Return (first differing token index, short snippet)."""
    if cpu_text == gpu_text:
        return -1, ""
    cpu_tokens = cpu_text.split()
    gpu_tokens = gpu_text.split()
    for index, (left, right) in enumerate(zip(cpu_tokens, gpu_tokens)):
        if left != right:
            return index, "%s != %s" % (short(left, 40), short(right, 40))
    if len(cpu_tokens) != len(gpu_tokens):
        index = min(len(cpu_tokens), len(gpu_tokens))
        left = cpu_tokens[index] if index < len(cpu_tokens) else "<end>"
        right = gpu_tokens[index] if index < len(gpu_tokens) else "<end>"
        return index, "%s != %s" % (short(left, 40), short(right, 40))
    return -1, "whitespace-only difference: %s != %s" % (
        short(cpu_text, 30),
        short(gpu_text, 30),
    )


def check_prompt(index, prompt, tokens, model):
    cpu = run_binary(prompt, tokens, model, gpu=False)
    gpu = run_binary(prompt, tokens, model, gpu=True)

    if cpu["returncode"] != 0:
        status, mismatch, snippet = "ERROR", -1, error_detail(cpu, "cpu")
    elif gpu["returncode"] != 0:
        status, mismatch, snippet = "ERROR", -1, error_detail(gpu, "gpu")
    else:
        cpu_text = generated_text(cpu["stdout"], gpu=False)
        gpu_text = generated_text(gpu["stdout"], gpu=True)
        mismatch, snippet = compare(cpu_text, gpu_text)
        status = "MISMATCH" if snippet else "ok"

    prefix = "[%d] " % (index + 1)
    if status == "ok":
        print(prefix + "ok")
    elif status == "ERROR":
        print(prefix + "ERROR " + short(snippet))
    else:
        print(prefix + "MISMATCH token %d: %s" % (mismatch, snippet))

    return {
        "prompt_index": index,
        "prompt": prompt,
        "status": status,
        "cpu_output": generated_text(cpu["stdout"], gpu=False),
        "gpu_output": generated_text(gpu["stdout"], gpu=True),
        "cpu_stderr": cpu["stderr"],
        "gpu_stderr": gpu["stderr"],
        "first_mismatch": mismatch,
        "snippet": snippet,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--tokens", type=int, default=DEFAULT_TOKENS)
    parser.add_argument("--prompts", default=DEFAULT_PROMPTS)
    args = parser.parse_args()

    prompts_path = args.prompts if os.path.isabs(args.prompts) else os.path.join(ROOT, args.prompts)
    try:
        prompts = load_prompts(prompts_path)
    except OSError as exc:
        print("ERROR prompts: " + short(str(exc)))
        return 1
    if not prompts:
        print("ERROR: no prompts in " + args.prompts)
        return 1

    results = [check_prompt(i, p, args.tokens, args.model) for i, p in enumerate(prompts)]
    counts = {
        "ok": sum(r["status"] == "ok" for r in results),
        "mismatch": sum(r["status"] == "MISMATCH" for r in results),
        "error": sum(r["status"] == "ERROR" for r in results),
    }

    report = {
        "model": args.model,
        "tokens": args.tokens,
        "prompts_file": args.prompts,
        "summary": counts,
        "results": results,
    }
    report_path = os.path.join(ROOT, REPORT)
    try:
        with open(report_path, "w", encoding="utf-8") as handle:
            json.dump(report, handle, indent=2, ensure_ascii=True)
            handle.write("\n")
    except OSError as exc:
        print("ERROR report: " + short(str(exc)))
        return 1

    print("summary: %d ok, %d MISMATCH, %d ERROR" % (
        counts["ok"], counts["mismatch"], counts["error"],
    ))
    return 1 if counts["mismatch"] or counts["error"] else 0


if __name__ == "__main__":
    sys.exit(main())
