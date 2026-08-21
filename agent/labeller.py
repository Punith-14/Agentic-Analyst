"""Assigns a 0 / 0.5 / 1 score to every step.

Ternary rather than binary so that recoverable exploration isn't punished as
failure. Training collapses to binary — see critic/features.py — but the three
levels are what get stored.

    python -m agent.labeller                    # label everything
    python -m agent.labeller --dry-run          # just show the distribution

Known limitation, worth stating in the write-up: this scheme has no credit
assignment. When a run fails, every step in it is penalised, including the good
ones. The LLM annotator and rollout labelling are what fix that; this is the
free version.
"""

from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path

from contracts import RunRecord, TrajectoryStep
from agent.logger import DEFAULT_DIR, iter_runs

GOOD, EXPLORE, BAD = 1.0, 0.5, 0.0


def label_step(step: TrajectoryStep, run: RunRecord) -> float | None:
    """Score one step. None means unscoreable — drop it from training.

    Order matters: the overrides below don't depend on the run outcome, so they
    come first.
    """
    # Repeating a call is never useful, whether it worked before or not.
    if step.repeat_count >= 2:
        return BAD

    # Nothing usable came out of the model.
    if step.action is None:
        return BAD

    # The run was never scored — usually a broken gold query. Not the agent's
    # fault and not evidence either way, so it can't be a training example.
    if run.correct is None:
        return None

    # The answer step is judged on the answer.
    if step.status == "final":
        return GOOD if run.correct else BAD

    if run.correct:
        # It failed but the run recovered — this is the case the 0.5 exists for.
        return EXPLORE if step.status == "error" else GOOD

    # Run failed. An errored step is bad; a step that ran fine might have been
    # fine — we can't tell which step caused the failure, so it sits in between.
    return BAD if step.status == "error" else EXPLORE


def label_run(run: RunRecord) -> RunRecord:
    for s in run.steps:
        s.label_step = label_step(s, run)
        if s.observation is None:
            s.label_execution = None
        elif s.observation.status == "error":
            s.label_execution = "error"
        elif (s.observation.row_count or 0) == 0:
            s.label_execution = "empty"
        else:
            s.label_execution = "ok"
    return run


def label_file(path: Path, dry_run: bool = False) -> Counter:
    runs = [label_run(r) for r in iter_runs(path)]
    dist = Counter(s.label_step for r in runs for s in r.steps)

    if not dry_run:
        tmp = path.with_suffix(".jsonl.tmp")
        with open(tmp, "w", encoding="utf-8") as f:
            for r in runs:
                f.write(r.model_dump_json() + "\n")
        tmp.replace(path)          # atomic, so a crash can't truncate the file

    return dist


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", type=Path, default=DEFAULT_DIR)
    ap.add_argument("--pattern", default="*.jsonl")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    total = Counter()
    for p in sorted(args.dir.glob(args.pattern)):
        dist = label_file(p, args.dry_run)
        total.update(dist)
        n = sum(v for k, v in dist.items() if k is not None)
        print(f"{p.name:<32} {n:>6} labelled, {dist[None]:>4} dropped")

    scored = sum(v for k, v in total.items() if k is not None)
    print(f"\n{'=' * 46}")
    for k, name in [(BAD, "0.0  bad"), (EXPLORE, "0.5  exploratory"), (GOOD, "1.0  good")]:
        print(f"  {name:<20} {total[k]:>6}  {total[k]/scored:>6.1%}")
    print(f"  {'dropped':<20} {total[None]:>6}")
    print(f"  {'usable':<20} {scored:>6}")
    print(f"\n  binary (bad vs not): {total[BAD]/scored:.1%} / "
          f"{(total[EXPLORE]+total[GOOD])/scored:.1%}")
    print("=" * 46)
    if args.dry_run:
        print("\ndry run — nothing written")


if __name__ == "__main__":
    main()
