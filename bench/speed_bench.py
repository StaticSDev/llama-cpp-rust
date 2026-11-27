#!/usr/bin/env python3
"""Compare llama.cpp upstream and this Rust port on the same prompt.

Starts each server in turn so only one copy of the model is resident in VRAM,
then writes a JSON report with prompt and generation speed in tokens/second.

Usage:
    python3 bench/speed_bench.py --runs 3 --max-tokens 200 --out bench/results.json
"""

import argparse
import json
import os
import signal
import subprocess
import sys
import time
import urllib.error
import urllib.request

PROMPT = (
    "You are a careful assistant. Read the passage and answer the question.\n\n"
    "Passage: The city library opened in 1902 with a collection of 3000 books. "
    "Over the next century it grew to 2.4 million items and added three branches. "
    "In 1998 the main building was renovated, and a digital archive was launched in 2011. "
    "The archive now holds 18 million scanned pages. Every year the library runs a summer "
    "reading program for children and a weekly coding club for teenagers. "
    "The librarians use a simple catalog system: each item gets a unique number, "
    "and the number is printed on a label. Readers can search by title, author or subject. "
    "The catalog was moved online in 2016, which made the collection available worldwide.\n\n"
    "Question: In what year was the digital archive launched, and how many scanned pages does it hold? "
    "Answer in one short sentence.\n"
)

UPSTREAM_BIN = "/home/statics/bench/llama-upstream/build-up/bin/llama-server"
RUST_BIN = "target/release/llama-web"
MODEL = "models/Qwen3.5-4B-Q4_K_M.gguf"


def wait_http(url, timeout=180):
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            with urllib.request.urlopen(url, timeout=2) as response:
                response.read()
                return
        except (urllib.error.URLError, TimeoutError, OSError):
            time.sleep(1)
    raise RuntimeError("server did not become ready: " + url)


def stop(proc):
    if proc is None or proc.poll() is not None:
        return
    proc.send_signal(signal.SIGTERM)
    try:
        proc.wait(timeout=20)
    except subprocess.TimeoutExpired:
        proc.kill()
        proc.wait(timeout=10)
    time.sleep(2)


def upstream_run(url, max_tokens):
    body = json.dumps({
        "messages": [{"role": "user", "content": PROMPT}],
        "max_tokens": max_tokens,
        "temperature": 0,
        "seed": 1234,
    }).encode()
    request = urllib.request.Request(
        url.rstrip("/") + "/v1/chat/completions",
        data=body,
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(request, timeout=900) as response:
        data = json.load(response)
    timings = data.get("timings", {})
    return {
        "prompt_n": timings.get("prompt_n"),
        "prompt_tok_s": timings.get("prompt_per_second"),
        "gen_n": timings.get("predicted_n"),
        "gen_tok_s": timings.get("predicted_per_second"),
        "text": data["choices"][0]["message"].get("content", ""),
    }


def rust_run(url, max_tokens):
    body = json.dumps({
        "prompt": PROMPT,
        "n_predict": max_tokens,
        "temp": 0,
        "top_p": 0.9,
    }).encode()
    request = urllib.request.Request(
        url.rstrip("/") + "/api/chat/stream",
        data=body,
        headers={"Content-Type": "application/json"},
    )
    text = ""
    last = None
    with urllib.request.urlopen(request, timeout=900) as response:
        for raw in response:
            line = raw.decode("utf-8", "replace").strip()
            if not line:
                continue
            event = json.loads(line)
            if event.get("type") == "token":
                text += event.get("text", "")
            elif event.get("type") == "done":
                last = event
            elif event.get("type") == "error":
                raise RuntimeError(event.get("error", "generation failed"))
    if last is None:
        raise RuntimeError("stream ended without a done event")
    return {
        "prompt_n": last.get("prompt_tokens"),
        "prompt_tok_s": last.get("prompt_tok_s"),
        "gen_n": last.get("tokens"),
        "gen_tok_s": last.get("tok_s"),
        "text": text,
    }


def benchmark(url, kind, runs, max_tokens):
    samples = []
    for i in range(runs):
        fn = upstream_run if kind == "upstream" else rust_run
        sample = fn(url, max_tokens)
        samples.append(sample)
        print(
            f"[{kind}] run {i + 1}: prompt {sample['prompt_tok_s']:.2f} tok/s "
            f"({sample['prompt_n']} tok), gen {sample['gen_tok_s']:.2f} tok/s "
            f"({sample['gen_n']} tok)",
            flush=True,
        )
    return {
        "kind": kind,
        "url": url,
        "runs": [{k: v for k, v in s.items() if k != "text"} for s in samples],
        "prompt_tok_s": sum(s["prompt_tok_s"] for s in samples) / runs,
        "gen_tok_s": sum(s["gen_tok_s"] for s in samples) / runs,
        "prompt_n": samples[0]["prompt_n"],
        "gen_n": round(sum(s["gen_n"] for s in samples) / runs, 1),
        "same_answer": all(s["text"] == samples[0]["text"] for s in samples),
    }


def average(vals):
    return sum(vals) / len(vals) if vals else 0.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", type=int, default=1)
    ap.add_argument("--max-tokens", type=int, default=6)
    ap.add_argument("--out", default="bench/results.json")
    ap.add_argument("--model", default=MODEL)
    ap.add_argument("--upstream-bin", default=UPSTREAM_BIN)
    ap.add_argument("--rust-bin", default=RUST_BIN)
    ap.add_argument("--upstream-port", type=int, default=8081)
    ap.add_argument("--rust-port", type=int, default=8080)
    ap.add_argument("--upstream-cache-reuse", type=int, default=0)
    args = ap.parse_args()

    here = os.path.dirname(os.path.abspath(__file__))
    root = os.path.dirname(here)
    model = os.path.normpath(os.path.join(root, args.model))
    upstream_bin = os.path.normpath(os.path.join(root, args.upstream_bin)) if not os.path.isabs(args.upstream_bin) else args.upstream_bin
    rust_bin = os.path.normpath(os.path.join(root, args.rust_bin))
    if not os.path.exists(model):
        raise SystemExit("model not found: " + model)
    for binary in (upstream_bin, rust_bin):
        if not os.path.exists(binary):
            raise SystemExit("binary not found: " + binary)

    env = os.environ.copy()
    env["LD_LIBRARY_PATH"] = os.path.dirname(upstream_bin) + ":" + env.get("LD_LIBRARY_PATH", "")
    upstream_url = f"http://127.0.0.1:{args.upstream_port}"
    rust_url = f"http://127.0.0.1:{args.rust_port}"

    results = {
        "model": os.path.basename(model),
        "prompt_chars": len(PROMPT),
        "max_tokens": args.max_tokens,
        "runs": args.runs,
        "gpu": "NVIDIA GeForce GTX 1660 SUPER (sm_75, 6 GB)",
    }
    proc = None
    try:
        cmd = [
            upstream_bin, "-m", model, "--alias", "qwen3.5-4b",
            "--host", "127.0.0.1", "--port", str(args.upstream_port),
            "-ngl", "99", "-c", "32768", "-t", "6", "-tb", "12",
            "-b", "2048", "-ub", "512", "-np", "1", "-fa", "on",
            "-ctk", "q4_0", "-ctv", "q4_0", "--context-shift",
            "--jinja",
            "--reasoning-format", "deepseek-legacy",
        ]
        if args.upstream_cache_reuse > 0:
            cmd.extend(["--cache-reuse", str(args.upstream_cache_reuse)])
        print("[bench] starting upstream llama-server", flush=True)
        proc = subprocess.Popen(cmd, env=env, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        wait_http(upstream_url + "/health")
        results["upstream"] = benchmark(upstream_url, "upstream", args.runs, args.max_tokens)
        stop(proc)
        proc = None

        print("[bench] starting Rust llama-web with CUDA", flush=True)
        cmd = [rust_bin, "--model", model, "--port", str(args.rust_port), "--gpu"]
        proc = subprocess.Popen(cmd, cwd=root, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        wait_http(rust_url + "/api/status")
        results["rust"] = benchmark(rust_url, "rust", args.runs, args.max_tokens)
    finally:
        stop(proc)

    up, rs = results["upstream"], results["rust"]
    results["summary"] = {
        "upstream_prompt_tok_s": round(up["prompt_tok_s"], 2),
        "rust_prompt_tok_s": round(rs["prompt_tok_s"], 2),
        "upstream_gen_tok_s": round(up["gen_tok_s"], 2),
        "rust_gen_tok_s": round(rs["gen_tok_s"], 2),
        "gen_ratio_rust_over_upstream": round(rs["gen_tok_s"] / up["gen_tok_s"], 2),
        "prompt_ratio_rust_over_upstream": round(rs["prompt_tok_s"] / up["prompt_tok_s"], 2),
    }
    out = os.path.normpath(os.path.join(root, args.out))
    with open(out, "w", encoding="utf-8") as fh:
        json.dump(results, fh, ensure_ascii=False, indent=2)
    print(json.dumps(results["summary"], indent=2), flush=True)
    print("report: " + out, flush=True)


if __name__ == "__main__":
    main()
