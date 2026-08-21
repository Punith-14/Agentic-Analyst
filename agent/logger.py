"""Writes runs to JSONL and reads them back.

Append-only, flushed after every run. A crash three hours into an overnight
job should cost the current run, not the whole night.

These files are training data for all four ML components — never overwrite or
delete them.
"""

from __future__ import annotations

import json
from collections import Counter
from datetime import date
from pathlib import Path
from typing import Iterator

from contracts import RunRecord

DEFAULT_DIR = Path(__file__).resolve().parent.parent / "data" / "trajectories"


def log_path(out_dir: Path | None = None, tag: str = "") -> Path:
    """One file per day, optionally tagged so separate experiments don't mix."""
    d = out_dir or DEFAULT_DIR
    d.mkdir(parents=True, exist_ok=True)
    name = f"{date.today().isoformat()}{('-' + tag) if tag else ''}.jsonl"
    return d / name


class TrajectoryLogger:
    """Append runs to a JSONL file.

        with TrajectoryLogger(tag="smoke") as log:
            log.write(record)
    """

    def __init__(self, path: Path | None = None, out_dir: Path | None = None,
                 tag: str = ""):
        self.path = path or log_path(out_dir, tag)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._fh = None
        self.count = 0

    def __enter__(self) -> "TrajectoryLogger":
        self._fh = open(self.path, "a", encoding="utf-8")
        return self

    def __exit__(self, *exc) -> None:
        if self._fh:
            self._fh.close()
            self._fh = None

    def write(self, record: RunRecord) -> None:
        line = record.model_dump_json()
        if self._fh:
            self._fh.write(line + "\n")
            self._fh.flush()          # survive a crash mid-job
        else:
            with open(self.path, "a", encoding="utf-8") as f:
                f.write(line + "\n")
        self.count += 1


def iter_runs(path: Path) -> Iterator[RunRecord]:
    """Stream runs. Skips corrupt lines rather than dying on them — a truncated
    last line after a hard kill shouldn't cost you the file."""
    with open(path, encoding="utf-8") as f:
        for n, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                yield RunRecord.model_validate_json(line)
            except Exception:                     # noqa: BLE001
                print(f"  ! skipping malformed line {n} in {path.name}")


def load_all(directory: Path | None = None, pattern: str = "*.jsonl") -> list[RunRecord]:
    d = directory or DEFAULT_DIR
    runs: list[RunRecord] = []
    for p in sorted(d.glob(pattern)):
        runs.extend(iter_runs(p))
    return runs


def done_keys(path: Path) -> set[tuple[str, int]]:
    """(task_id, attempt) pairs already logged, so a rerun can resume.

    Attempt number is recovered from position, since RunRecord doesn't carry
    one — runs of the same task are counted in file order.
    """
    seen: Counter[str] = Counter()
    out: set[tuple[str, int]] = set()
    if not path.exists():
        return out
    for r in iter_runs(path):
        out.add((r.task_id, seen[r.task_id]))
        seen[r.task_id] += 1
    return out


def summarise(runs: list[RunRecord]) -> dict:
    """Quick health check on a batch — call this before training on it."""
    if not runs:
        return {"runs": 0}

    steps = sum(len(r.steps) for r in runs)
    correct = [r for r in runs if r.correct is True]
    errored = sum(1 for r in runs for s in r.steps if s.status == "error")

    return {
        "runs": len(runs),
        "steps": steps,
        "mean_steps": round(steps / len(runs), 2),
        "solved": len(correct),
        "solve_rate": round(len(correct) / len(runs), 3),
        "error_steps": errored,
        "error_step_rate": round(errored / steps, 3) if steps else 0,
        "termination": dict(Counter(r.termination for r in runs).most_common()),
        "scored_by": dict(Counter(r.correct_method or "none" for r in runs).most_common()),
        "error_categories": dict(Counter(
            s.error_category for r in runs for s in r.steps
            if s.error_category != "none").most_common()),
        "mean_tokens": round(sum(r.total_tokens for r in runs) / len(runs)),
        "total_minutes": round(sum(r.total_duration_ms for r in runs) / 60000, 1),
    }


def print_summary(runs: list[RunRecord]) -> None:
    s = summarise(runs)
    if not s.get("runs"):
        print("no runs")
        return
    print(f"\n{'=' * 58}")
    print(f"  runs            {s['runs']}")
    print(f"  steps           {s['steps']}  (mean {s['mean_steps']}/run)")
    print(f"  solved          {s['solved']}  ({s['solve_rate']:.1%})")
    print(f"  error steps     {s['error_steps']}  ({s['error_step_rate']:.1%})")
    print(f"  mean tokens     {s['mean_tokens']}")
    print(f"  wall time       {s['total_minutes']} min")
    print(f"\n  scored by:")
    for k, v in s["scored_by"].items():
        print(f"    {k:<22} {v}")
    print(f"\n  termination:")
    for k, v in s["termination"].items():
        print(f"    {k:<22} {v}")
    if s["error_categories"]:
        print(f"\n  error categories:")
        for k, v in s["error_categories"].items():
            print(f"    {k:<22} {v}")
    print("=" * 58)


if __name__ == "__main__":
    import sys
    d = Path(sys.argv[1]) if len(sys.argv) > 1 else DEFAULT_DIR
    print_summary(load_all(d))
