"""Label the trajectories and write one flat dataset for the notebook.

    python scripts/label_dataset.py
    python scripts/label_dataset.py --dry-run

    # the chinook_1 holdout — a database the critics have never seen.
    # Separate output file: it must never join the training table.
    python scripts/label_dataset.py --pattern "*-holdout.jsonl" \
        --out holdout_steps.parquet

Does two things and nothing else:

  1. Adds a 0 / 0.5 / 1 score to every step, and a run-level failure flag.
  2. Flattens everything to data/critic/labelled_steps.parquet — one row per
     step, all raw fields kept.

No feature engineering here. That happens in the notebook so it can be seen and
argued with; only the labelling is fixed enough to live in a script.

WHY TWO TARGETS ARE STORED
  label_step  (0 / 0.5 / 1)  — from our own rules
  y_run_fails (0 / 1)        — from comparing the answer against gold SQL

The first is partly derivable from the step's own fields, so a model trained on
it mostly relearns the rules — measured at 92.6% reproducible. The second comes
from outside the step entirely and is what early stopping actually needs. Both
are written; the notebook explains the choice.

WHICH FILES GET USED
  data/trajectories/*-train.jsonl        labelled
  everything else in that folder         ignored

This is deliberately opt-in. It used to glob *.jsonl and skip a list of bad
names, and that failed the first time it was tested for real: the UI
walkthrough wrote 33 synthetic runs against a demo database into the same
folder, and they would have been labelled and merged into the training set
without a word. An allow-list can't fail that way.

  -train.jsonl        1,620 runs -> 5,461 steps, the critic's training data
  -superseded.jsonl      80 runs -> pre-fix config, 9.2% solve rate vs 30.9%,
                                    kept as before/after evidence
  everything else        demo output, other layers, scratch

The superseded batch also can't be identified from its contents: every run
generated before the provenance fix records the same context_policy string.
The filename is the only signal, which is the second reason this is an
allow-list rather than a deny-list.
"""

from __future__ import annotations

import argparse
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd

from agent.labeller import label_run
from agent.logger import DEFAULT_DIR, iter_runs
from contracts import RunRecord

OUT = Path(__file__).resolve().parent.parent / "data" / "critic"


def flatten(run: RunRecord) -> list[dict]:
    """One row per step. Raw fields only — no derived features."""
    rows = []
    for s in run.steps:
        a = s.action
        obs = s.observation
        rows.append({
            # keys
            "run_id": run.run_id,
            "task_id": run.task_id,
            "step_index": s.step_index,

            # targets
            "label_step": s.label_step,
            "y_run_fails": None if run.correct is None else int(not run.correct),
            "run_correct": run.correct,
            "label_execution": s.label_execution,

            # what happened
            "question": run.question,
            "thought": s.thought,
            "tool": a.tool if a else None,
            "sql": (a.args.get("query", "") if a and a.tool == "run_sql" else ""),
            "args": str(a.args) if a else "",
            "status": s.status,
            "raw_model_output": s.raw_model_output,

            # observation
            "obs_status": obs.status if obs else None,
            "obs_rows": obs.row_count if obs else None,
            "obs_truncated": obs.truncated if obs else False,
            "obs_error": obs.error if obs else None,

            # counters logged at the time
            "error_category": s.error_category,
            "consecutive_errors": s.consecutive_errors,
            "total_errors_so_far": s.total_errors_so_far,
            "parse_repair_count": s.parse_repair_count,
            "repeat_count": s.repeat_count,
            "is_retry": s.is_retry,
            "schema_inspected_before": s.schema_inspected_before,
            "tokens_in_prompt": s.tokens_in_prompt,
            "duration_ms": s.duration_ms,

            # run context
            "termination": run.termination,
            "run_total_steps": len(run.steps),
            "correct_method": run.correct_method,
            "model_name": run.model_name,
            "temperature": run.temperature,
            "context_policy": run.context_policy,
        })
    return rows


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", type=Path, default=DEFAULT_DIR)
    # OPT-IN, not opt-out. This used to be "*.jsonl" with an exclude list,
    # which meant anything dropped into data/trajectories/ silently joined the
    # training set — and it happened: 33 synthetic demo runs from the UI
    # walkthrough landed there. Name the files you want; ignore everything else.
    ap.add_argument("--pattern", default="*-train.jsonl",
                    help="only files matching this are labelled")
    ap.add_argument("--exclude", nargs="*", default=["superseded"],
                    help="filename fragments to skip even if they match "
                         "--pattern; belt and braces")
    # The holdout must never land in the training table. Separate file, always.
    ap.add_argument("--out", default="labelled_steps.parquet",
                    help="output filename inside data/critic/")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    all_rows, files = [], []
    for p in sorted(args.dir.glob(args.pattern)):
        if any(x in p.name for x in args.exclude):
            print(f"  skipping {p.name} (superseded config)")
            continue
        runs = [label_run(r) for r in iter_runs(p)]
        rows = [row for r in runs for row in flatten(r)]
        all_rows.extend(rows)
        files.append((p.name, len(runs), len(rows)))

    if not all_rows:
        sys.exit("No runs found. Has generate_trajectories.py been run?")

    df = pd.DataFrame(all_rows)

    print(f"\n{'file':<32}{'runs':>7}{'steps':>8}")
    print("-" * 47)
    for name, nr, ns in files:
        print(f"{name:<32}{nr:>7}{ns:>8}")

    print(f"\n{'=' * 56}")
    print(f"  {len(df):,} steps from {df.run_id.nunique():,} runs, "
          f"{df.task_id.nunique()} questions")

    print(f"\n  label_step (our rules):")
    for k in [0.0, 0.5, 1.0]:
        n = (df.label_step == k).sum()
        print(f"    {k}   {n:>6}  {n/len(df):>6.1%}")
    n_null = df.label_step.isna().sum()
    print(f"    none {n_null:>6}   (unscoreable runs)")

    print(f"\n  y_run_fails (from gold SQL):")
    vc = df.y_run_fails.value_counts(dropna=False)
    for k in [0.0, 1.0]:
        n = int(vc.get(k, 0))
        print(f"    {'run failed' if k else 'run correct':<12} {n:>6}  {n/len(df):>6.1%}")
    print(f"    {'unscoreable':<12} {int(df.y_run_fails.isna().sum()):>6}")

    configs = df.context_policy.nunique()
    if configs > 1:
        print(f"\n  !! {configs} agent configurations present — the notebook "
              f"should filter, or the failure distributions are mixed")

    if args.dry_run:
        print("\n  dry run — nothing written")
        return

    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / args.out
    df.to_parquet(path, index=False)
    print(f"\n  -> {path}")
    print(f"  {len(df.columns)} columns, ready for the notebook")
    print("=" * 56)


if __name__ == "__main__":
    main()
