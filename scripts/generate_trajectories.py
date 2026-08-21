"""Run the whole task suite N times and log every run.

    python scripts/generate_trajectories.py --attempts 2
    python scripts/generate_trajectories.py --attempts 5 --tag overnight
    python scripts/generate_trajectories.py --attempts 1 --scripted   # no GPU
    python scripts/generate_trajectories.py --resume --tag overnight

Writes to data/trajectories/YYYY-MM-DD[-tag].jsonl, one run per line, flushed
after each run. Safe to Ctrl-C — rerun with --resume to pick up where it
stopped.

Temperature matters here. At 0 every attempt at a question is identical and
you get one trajectory five times. The default 0.7 is what produces the
variety the critic learns from.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
import traceback
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from agent import _stubs
from agent._stubs import TOOL_SPECS, TOOLS
from agent.checker import last_successful_sql, score_run
from agent.llm import OllamaLLM, scripted
from agent.logger import TrajectoryLogger, done_keys, load_all, print_summary
from agent.loop import AgentConfig, run_agent

ROOT = Path(__file__).resolve().parent.parent
TASKS = ROOT / "data" / "tasks" / "task_suite.json"


def load_tasks(split: str | None = None) -> list[dict]:
    tasks = json.load(open(TASKS, encoding="utf-8"))
    if split:
        tasks = [t for t in tasks if t.get("split") == split]
    return tasks


def full_schema(db: str) -> dict:
    """Every table with its columns — see prompt.build_prompt for why."""
    _stubs.set_db(db)
    r = TOOLS["get_schema"]()
    return r.data if r.status == "ok" else {}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--attempts", type=int, default=2,
                    help="runs per question; >1 is what gives the critic variety")
    ap.add_argument("--split", default="main", help="main | holdout | all")
    ap.add_argument("--limit", type=int, default=0, help="first N questions only")
    ap.add_argument("--max-steps", type=int, default=12)
    ap.add_argument("--tag", default="")
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--scripted", action="store_true",
                    help="use the fake model — for testing the pipeline")
    ap.add_argument("--quiet", action="store_true")
    args = ap.parse_args()

    split = None if args.split == "all" else args.split
    tasks = load_tasks(split)
    if args.limit:
        tasks = tasks[: args.limit]
    if not tasks:
        sys.exit(f"No tasks for split={args.split!r}. Has prepare_spider.py run?")

    llm = scripted("clean_success") if args.scripted else OllamaLLM()

    schemas = {db: full_schema(db) for db in {t["db"] for t in tasks}}

    logger = TrajectoryLogger(tag=args.tag)
    already = done_keys(logger.path) if args.resume else set()

    total = len(tasks) * args.attempts
    print(f"{len(tasks)} questions x {args.attempts} attempts = {total} runs")
    print(f"model:  {getattr(llm, 'name', '?')}")
    print(f"output: {logger.path}")
    if already:
        print(f"resume: {len(already)} runs already logged, skipping those")
    print()

    t0 = time.perf_counter()
    done = failed = 0

    with logger as log:
        for attempt in range(args.attempts):
            for task in tasks:
                key = (task["task_id"], attempt)
                if key in already:
                    continue

                _stubs.set_db(task["db"])
                n = done + failed + 1

                try:
                    rec = run_agent(
                        question=task["question"],
                        llm=llm,
                        tools=TOOLS,
                        tool_specs=TOOL_SPECS,
                        schema=schemas[task["db"]],
                        task_id=task["task_id"],
                        config=AgentConfig(max_steps=args.max_steps,
                                           verbose=not args.quiet),
                    )
                except KeyboardInterrupt:
                    print("\ninterrupted — logged runs are safe, "
                          "rerun with --resume")
                    raise
                except Exception:                       # noqa: BLE001
                    # one bad run must not kill an overnight job
                    failed += 1
                    print(f"[{n}/{total}] {task['task_id']} CRASHED")
                    traceback.print_exc(limit=3)
                    continue

                correct, reason, method = score_run(rec, task["gold_sql"], task["db"])
                rec.correct = correct
                rec.correct_method = method
                rec.predicted_sql = last_successful_sql(rec)
                for s in rec.steps:
                    s.run_final_correct = correct

                log.write(rec)
                done += 1

                mark = {True: "OK  ", False: "WRONG", None: "SKIP"}[correct]
                elapsed = time.perf_counter() - t0
                eta = (elapsed / (done + failed)) * (total - done - failed)
                print(f"[{n}/{total}] {task['task_id']} {mark} "
                      f"{rec.termination:<18} {len(rec.steps)}st "
                      f"{rec.total_duration_ms/1000:.1f}s  "
                      f"[{method}] ({reason})  eta {eta/60:.0f}m")

                if args.scripted:
                    llm.reset()

    print(f"\n{done} runs logged, {failed} crashed, "
          f"{(time.perf_counter()-t0)/60:.1f} min")
    print_summary(load_all(pattern=logger.path.name))


if __name__ == "__main__":
    main()
