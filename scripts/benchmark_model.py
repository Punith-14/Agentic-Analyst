"""Measure what the card can actually do.

Produces TOKEN_BUDGET for the loop, and the before/after numbers for the
context-reduction work.

    python scripts/benchmark_model.py --label baseline
    python scripts/benchmark_model.py --label kv-q8

Needs Ollama up with the model pulled.
"""

from __future__ import annotations

import argparse
import json
import platform
import statistics
import subprocess
import sys
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path

try:
    import requests
except ImportError:
    sys.exit("pip install requests")

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "data" / "benchmarks" / "model_baseline.json"

# a 10-step run lands around 6-12k; go well past that to find the ceiling
CONTEXT_STEPS = [500, 1000, 2000, 4000, 6000, 8000, 10000, 12000,
                 16000, 20000, 24000, 32000]


# --------------------------------------------------------------------------
# GPU
# --------------------------------------------------------------------------

def nvidia_smi() -> dict | None:
    """VRAM and GPU name. None if nvidia-smi isn't on PATH."""
    try:
        out = subprocess.run(
            ["nvidia-smi",
             "--query-gpu=name,memory.used,memory.total,temperature.gpu",
             "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=10, check=True).stdout.strip()
        name, used, total, temp = [x.strip() for x in out.split(",")]
        return {"gpu": name, "vram_used_mb": int(used),
                "vram_total_mb": int(total), "temp_c": int(temp)}
    except Exception:
        return None


def vram_used() -> int | None:
    s = nvidia_smi()
    return s["vram_used_mb"] if s else None


# --------------------------------------------------------------------------
# Ollama
# --------------------------------------------------------------------------

def check_ollama(host: str, model: str) -> None:
    try:
        r = requests.get(f"{host}/api/tags", timeout=10)
        r.raise_for_status()
    except Exception:
        sys.exit(f"Cannot reach Ollama at {host}.\n"
                 f"  Install from https://ollama.com, then it should start automatically.\n"
                 f"  Check with:  ollama list")

    names = [m["name"] for m in r.json().get("models", [])]
    if not any(n.split(":")[0] == model.split(":")[0] for n in names):
        sys.exit(f"Model {model!r} not found. Available: {names}\n"
                 f"  Pull it with:  ollama pull {model}")


def generate(host: str, model: str, prompt: str, num_predict: int = 128,
             num_ctx: int | None = None) -> dict:
    """One generation. Ollama's own timings plus wall-clock."""
    opts: dict = {"temperature": 0.0, "num_predict": num_predict}
    if num_ctx:
        opts["num_ctx"] = num_ctx

    t0 = time.perf_counter()
    r = requests.post(f"{host}/api/generate",
                      json={"model": model, "prompt": prompt,
                            "stream": False, "options": opts},
                      timeout=600)
    wall = time.perf_counter() - t0
    r.raise_for_status()
    d = r.json()

    ns = 1e-9
    return {
        "wall_s": round(wall, 3),
        "prompt_tokens": d.get("prompt_eval_count", 0),
        "output_tokens": d.get("eval_count", 0),
        "prompt_eval_s": round(d.get("prompt_eval_duration", 0) * ns, 3),
        "gen_s": round(d.get("eval_duration", 0) * ns, 3),
        "response": d.get("response", "")[:200],
    }


# --------------------------------------------------------------------------
# filler that looks like a real prompt
# --------------------------------------------------------------------------

FILLER_STEP = """
Thought: I should inspect the schema before writing the query.
Action: get_schema
Input: {{"table": "Table{i}"}}
Observation: 42 rows. [{{"name": "ColumnA", "type": "INTEGER"}}, {{"name": "ColumnB", "type": "TEXT"}}, {{"name": "ColumnC", "type": "REAL"}}]
"""


def make_prompt(target_tokens: int, defeat_cache: bool = True) -> str:
    """Prompt of roughly target_tokens, shaped like a real trajectory.

    The nonce is there because Ollama caches common prefixes — without it,
    consecutive sweep steps share most of their tokens and prompt_eval_count
    only reports the new ones, so a 16k prompt looks like 8k.
    """
    nonce = f"[session {uuid.uuid4().hex}]\n" if defeat_cache else ""
    head = (nonce +
            "You are a data analyst agent. Answer using the tools provided.\n"
            "Question: Which genre generated the most revenue in 2013?\n\n")
    body, i = [], 0
    # ~4 chars per token
    while len(head + "".join(body)) // 4 < target_tokens:
        body.append(FILLER_STEP.format(i=i))
        i += 1
    return head + "".join(body) + "\nThought:"


# --------------------------------------------------------------------------
# benchmarks
# --------------------------------------------------------------------------

def bench_speed(host: str, model: str, repeats: int) -> dict:
    print("\n[1/3] Generation speed")
    prompt = make_prompt(500)
    runs = []
    for k in range(repeats):
        r = generate(host, model, prompt, num_predict=128)
        tps = r["output_tokens"] / r["gen_s"] if r["gen_s"] else 0
        runs.append(tps)
        print(f"      run {k+1}: {r['output_tokens']} tokens in "
              f"{r['gen_s']}s = {tps:.1f} tok/s")
    return {
        "tokens_per_sec_mean": round(statistics.mean(runs), 1),
        "tokens_per_sec_min": round(min(runs), 1),
        "runs": [round(x, 1) for x in runs],
    }


def bench_context(host: str, model: str, max_context: int) -> list[dict]:
    """Sweep context sizes. Stops at the first failure — that's the ceiling."""
    print("\n[2/3] Context sweep  (VRAM and latency as the prompt grows)")
    print(f"      {'target':>8} {'actual':>8} {'VRAM MB':>9} {'prefill':>9} "
          f"{'gen':>7} {'tok/s':>7}")

    idle = vram_used()
    rows: list[dict] = []
    for target in [c for c in CONTEXT_STEPS if c <= max_context]:
        prompt = make_prompt(target)
        try:
            r = generate(host, model, prompt, num_predict=64,
                         num_ctx=max(target + 512, 2048))
        except Exception as e:
            print(f"      {target:>8}  FAILED — {type(e).__name__}: {str(e)[:80]}")
            rows.append({"target_tokens": target, "ok": False, "error": str(e)[:200]})
            break

        used = vram_used()
        tps = r["output_tokens"] / r["gen_s"] if r["gen_s"] else 0

        # flag a big shortfall — if the prompt got truncated, any ceiling we
        # report from this sweep is fiction
        shortfall = target - r["prompt_tokens"]
        suspect = " <- truncated?" if shortfall > target * 0.25 else ""

        rows.append({
            "target_tokens": target,
            "ok": True,
            "prompt_tokens": r["prompt_tokens"],
            "prompt_shortfall": shortfall,
            "vram_used_mb": used,
            "vram_over_idle_mb": (used - idle) if (used and idle) else None,
            "prefill_s": r["prompt_eval_s"],
            "gen_s": r["gen_s"],
            "tokens_per_sec": round(tps, 1),
        })
        print(f"      {target:>8} {r['prompt_tokens']:>8} {str(used):>9} "
              f"{r['prompt_eval_s']:>8.2f}s {r['gen_s']:>6.2f}s {tps:>7.1f}"
              f"{suspect}")
    return rows


def bench_realistic(host: str, model: str) -> dict:
    """Closest thing to what the loop actually sends at its worst."""
    print("\n[3/3] Realistic 10-step prompt")
    prompt = make_prompt(9000)
    r = generate(host, model, prompt, num_predict=150)
    print(f"      {r['prompt_tokens']} prompt tokens · prefill {r['prompt_eval_s']}s "
          f"· gen {r['gen_s']}s · total {r['wall_s']}s")
    est_run = r["wall_s"] * 10 * 0.6      # steps are shorter early on
    print(f"      -> a 10-step run costs roughly {est_run:.0f}s")
    print(f"      -> 200 runs ≈ {est_run * 200 / 3600:.1f} hours (overnight job)")
    return {**r, "estimated_10_step_run_s": round(est_run, 1),
            "estimated_200_runs_hours": round(est_run * 200 / 3600, 2)}


# --------------------------------------------------------------------------
# main
# --------------------------------------------------------------------------

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--host", default="http://localhost:11434")
    ap.add_argument("--model", default="qwen2.5-coder:3b")
    ap.add_argument("--repeats", type=int, default=3)
    ap.add_argument("--max-context", type=int, default=32000)
    ap.add_argument("--label", default="baseline",
                    help="e.g. 'baseline' or 'kv-q8' — lets you compare configs")
    args = ap.parse_args()

    print(f"Benchmarking {args.model} at {args.host}")
    check_ollama(args.host, args.model)

    gpu = nvidia_smi()
    if gpu:
        print(f"  GPU: {gpu['gpu']}  ({gpu['vram_total_mb']} MB total, "
              f"{gpu['vram_used_mb']} MB in use)")
    else:
        print("  !! nvidia-smi unavailable — VRAM will not be recorded.")

    results = {
        "label": args.label,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "model": args.model,
        "platform": platform.platform(),
        "gpu": gpu,
        "speed": bench_speed(args.host, args.model, args.repeats),
        "context_sweep": bench_context(args.host, args.model, args.max_context),
        "realistic": bench_realistic(args.host, args.model),
    }

    # the number this is all for
    ok = [r for r in results["context_sweep"] if r.get("ok")]
    ceiling = max((r["target_tokens"] for r in ok), default=0)
    budget = int(ceiling * 0.8)
    results["max_context_ok"] = ceiling
    results["recommended_token_budget"] = budget

    OUT.parent.mkdir(parents=True, exist_ok=True)
    history = []
    if OUT.exists():
        try:
            prev = json.loads(OUT.read_text())
            history = prev if isinstance(prev, list) else [prev]
        except json.JSONDecodeError:
            pass
    history.append(results)
    OUT.write_text(json.dumps(history, indent=2))

    print("\n" + "=" * 62)
    print(f"  Speed             {results['speed']['tokens_per_sec_mean']} tok/s")
    print(f"  Largest context   {ceiling:,} tokens")
    print(f"  TOKEN_BUDGET      {budget:,}    <- put this in your .env")
    if gpu:
        peak = max((r.get("vram_used_mb") or 0) for r in ok) if ok else 0
        print(f"  Peak VRAM         {peak} / {gpu['vram_total_mb']} MB")
    print(f"  Saved to          {OUT.relative_to(ROOT)}")
    print("=" * 62)
    print("\nNext: enable 8-bit KV cache, rerun with --label kv-q8, and compare.")


if __name__ == "__main__":
    main()
