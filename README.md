# Project 23 — Sub-problem B

ReAct agent core, trajectory logger, and a learned trajectory critic.

Punith KM · Team Mind Matrix

---

## What this is

A data analyst agent. The user asks a question in plain English; the agent
inspects the database schema, writes and runs SQL, recovers when a query
errors, and reports an answer.

This repo is **layer B** of a four-layer project:

| Layer | Owner | Scope |
|---|---|---|
| A | Dhrub | Tool library, sandbox, Pydantic validation |
| **B** | **Punith** | **ReAct loop, trajectory logger, learned critic** |
| C | Krishna | Memory: short-term, episodic, semantic |
| D | Harish | LangGraph orchestration, React UI |

The LLM itself is **not** trained. It reads the schema at runtime. The trained
component here is a step-level critic that predicts whether a run is failing,
so a doomed run can be stopped early instead of burning ten iterations.

---

## Layout

```
contracts.py              shared schemas — A, C and D build against these
CONTRIBUTING.md           git workflow — read before your first push
agent/
  loop.py                 the ReAct loop
  parser.py               model output -> Action
  prompt.py               prompt construction
  llm.py                  Ollama wrapper + scripted fake for testing
  _stubs.py               placeholder tools until layer A lands
  checker.py              scores a run against the gold SQL
  logger.py               append-only JSONL trajectory logging
  labeller.py             0 / 0.5 / 1 step scoring rules
  critic/
    features.py           25 engineered features, leakage-checked
data_prep/
  prepare_spider.py       Spider download -> databases + task suite
scripts/
  benchmark_model.py      measure tok/s, VRAM, context ceiling
  demo_loop.py            run the loop, scripted or real
  generate_trajectories.py  run the suite N times, log every run
  label_dataset.py        trajectories -> labelled_steps.parquet
notebooks/
  01_dataset_exploration.ipynb
  02_critic_models.ipynb  EDA, tuning, evaluation — outputs kept on purpose
data/
  db/                     3 SQLite databases, committed
  tasks/                  the 150-question suite, committed
  trajectories/           1,700 logged runs, committed — see below
  critic/                 labelled table + the frozen split
models/
  critic_lgbm_v1.joblib   the trained critic
tests/                    84 tests, no GPU required
```

---

## Setup

```bash
python -m venv .venv
.venv\Scripts\Activate.ps1        # Windows
pip install -r requirements.txt
pytest -q                          # 84 tests, no GPU needed
```

That is the whole setup. **You do not need the Spider download** — the three
databases and the task suite are committed, so a clone is immediately runnable.

Only rebuild the data if you want to change the question set. That needs the
1.8 GB Spider archive extracted to `data/spider_data`:

```bash
python data_prep/prepare_spider.py --spider-dir data/spider_data --list
python data_prep/prepare_spider.py --spider-dir data/spider_data \
    --dbs formula_1,college_2 --holdout chinook_1 --n 120 --n-holdout 30
```

Optional: `cp .env.example .env` to override the model, host or temperature.
The defaults are what every committed result was produced with.

---

## Trying it

Against the scripted model — no GPU needed:

```bash
python scripts/demo_loop.py                    # error recovery scenario
python scripts/demo_loop.py --all              # every scenario
python scripts/demo_loop.py --list
```

Against the real model — needs Ollama and `ollama pull qwen2.5-coder:3b`:

```bash
python scripts/benchmark_model.py --label baseline
python scripts/demo_loop.py --real --raw
```

---

## Notes

**Model choice.** Qwen2.5-Coder-3B, not 7B. The dev card is a 6 GB RTX 3050
Laptop; a 7B at Q4 is ~4.7 GB of weights and leaves nothing for the KV cache.
The 3B is also about twice as fast, which matters across 200 overnight runs.

**Databases are separate, not merged.** Merging the three would put ~35 tables
in every prompt — measured at ~1500 tokens versus ~500 for one database, and
~32 if only table names are sent and the agent calls `get_schema` on demand.

**No constrained decoding.** Forcing a strict tool-call schema produces valid
JSON but measurably worse decisions on small models. The parser handles free
text instead and records its failures.

**Trajectory logs are training data.** Three of the four ML components in the
project train on them. `data/trajectories/*.jsonl` is append-only and must not
be deleted — the runs were sampled at temperature 0.7 and cannot be reproduced.

**Keep the two trajectory files separate.** ⚠️ Merging them will bite you
silently.

| File | Runs | Steps | Solve rate | Used for training |
|---|---|---|---|---|
| `2026-08-15-train.jsonl` | 1,620 | 5,461 | 30.9% | yes |
| `2026-08-15-superseded.jsonl` | 80 | 526 | **9.2%** | **no** |

The superseded batch predates four fixes — full schema in `get_schema`, a worked
example in the prompt, loop detection, `max_steps` 8→12 — and is kept only as
before/after evidence for them.

It is excluded by `label_dataset.py`, which matches on the **filename**
(`--exclude superseded`). That is not laziness: every run generated before the
provenance fix records the same `context_policy` string, so nothing *inside* the
record distinguishes a good run from a bad one. **Merge the two files, or rename
the second, and those 80 runs rejoin the training set with no way to separate
them again.**

Runs generated from now on carry the real configuration in `context_policy`
(`schema=`, `guards=`, `max_steps=`) and don't have this problem.

---

## Status

**Done.** Contracts, stub tools, scripted model, prompt builder, parser, ReAct
loop with guards, JSONL logging, dataset prep, answer checking via execution
accuracy, trajectory generation (1,700 runs), ternary labelling, feature
engineering, and the LightGBM critic — tuned, thresholded and evaluated once on
a held-out test set.

**Critic, as it stands:** PR-AUC 0.982 on test against a 0.817 base rate. At
threshold 0.694 it stops 646 runs of which 629 were failing (97.4% precision),
catches 74.8% of all failures, and costs 9% of successful runs. 59% of doomed
runs are flagged at step 0. Inference is 3.35 ms/step.

**Next.** Parser repair pass (104/1,200 runs still die on parse failure);
probability calibration so the threshold is readable; the holdout experiment on
`chinook_1`, a database the critic has never seen; ModernBERT on step text to
measure what reading the SQL adds over counting features; an LLM-as-judge
baseline for latency comparison; then wiring the critic into the loop and
measuring steps saved with solve rate held constant.
