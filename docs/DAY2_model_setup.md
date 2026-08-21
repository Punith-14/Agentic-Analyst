# Day 2 — Get the model running and measure it

**Time: 2–3 hours. Mostly waiting on a download.**

Goal: Qwen2.5-Coder-7B running locally, and four numbers written down that
every later decision depends on.

---

## 1. Install Ollama

Download from [ollama.com](https://ollama.com) and install. It runs as a
background service — no compiler, no CUDA setup, no Python bindings.

Verify:

```powershell
ollama --version
```

## 2. Pull the model

```powershell
```

~4.7 GB. Ollama's default `7b` tag is Q4_K_M — 4-bit, which is what we want.

Quick sanity check:

```powershell
ollama run qwen2.5-coder:7b "Write a SQL query counting rows in a table called Album"
```

Type `/bye` to exit.

**Why this model:** trained heavily on code including SQL, and 4-bit puts the
weights at roughly 4.5 GB — leaving ~3 GB of your 8 GB for the conversation.

---

## 3. Baseline benchmark — before touching any settings

```powershell
python scripts\benchmark_model.py --label baseline
```

Takes 5–10 minutes. It sweeps context sizes until something breaks, so a
failure at the top of the sweep is the point, not a problem.

**Write these down:**

| | Your number |
|---|---|
| Tokens/sec | |
| Largest context that worked | |
| Peak VRAM | |
| Recommended TOKEN_BUDGET | |

---

## 4. Enable 8-bit KV cache, then measure again

This roughly halves the memory the conversation uses, at under 0.1% quality
cost. One setting.

```powershell
setx OLLAMA_FLASH_ATTENTION "1"
setx OLLAMA_KV_CACHE_TYPE "q8_0"
```

Then **fully restart Ollama** — quit it from the system tray, not just close a
window. `setx` only affects new processes.

```powershell
python scripts\benchmark_model.py --label kv-q8
```

**Compare the two runs.** You should see a larger maximum context at similar
speed. That difference is a result you report:

> "Enabling 8-bit KV cache raised the usable context from X to Y tokens on an
> 8 GB card, with no measurable change in generation speed."

Avoid `q4_0` for the KV cache — 4-bit cache does degrade output quality, unlike
4-bit weights.

---

## 5. Put the numbers in `.env`

```powershell
copy .env.example .env
```

Then edit:

```
AGENT_MODEL=qwen2.5-coder:7b
TOKEN_BUDGET=<the recommended number from the benchmark>
MAX_STEPS=10
AGENT_TEMPERATURE=0.7
```

`.env` is gitignored. Never commit it.

---

## 6. Check the Python wrapper talks to it

```powershell
python -c "from agent.llm import OllamaLLM; print(OllamaLLM()('SELECT 1 -- explain briefly:', max_tokens=40))"
```

If that prints text, Day 2 is done.

---

## What these numbers are for

**TOKEN_BUDGET** — the loop stops a run before it exceeds this, instead of
crashing with CUDA out-of-memory at step 9 after two minutes of work.

**Tokens/sec** — tells you how long the Sunday overnight generation job takes.
The benchmark prints an estimate for 200 runs.

**Peak VRAM** — tells you whether you can train the critic while the model is
loaded. Almost certainly not, which is why generation and training are separate
phases in the plan.

**The baseline vs kv-q8 comparison** — your first measured result, obtained on
day two, for about ten minutes of work.

---

## If something goes wrong

**"Cannot reach Ollama"** — check the tray icon; run `ollama list`.

**Very slow (under 5 tok/s)** — it is running on CPU. Check `nvidia-smi` shows
`ollama` using VRAM. Update your NVIDIA driver if not.

**Out of memory at small contexts** — close Chrome. A browser can hold over a
gigabyte of VRAM, and you only have eight.

**Benchmark fails immediately at 500 tokens** — the model did not load. Try
`ollama run qwen2.5-coder:7b` by hand and read the error.
