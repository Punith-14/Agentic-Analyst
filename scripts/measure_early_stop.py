"""What does early stopping actually buy, and what does it cost?

    python scripts/measure_early_stop.py                    # test runs only
    python scripts/measure_early_stop.py --all              # every logged run
    python scripts/measure_early_stop.py --sweep            # threshold curve
    python scripts/measure_early_stop.py --no-text          # LightGBM alone

WHY REPLAY RATHER THAN RE-RUN THE AGENT
The obvious experiment is to run the suite twice, with and without the critic,
and compare. That costs hours of GPU time and — worse — the agent samples at
temperature 0.7, so the two arms would differ for reasons that have nothing to
do with the critic. Separating the critic's effect from sampling noise would
need many repetitions.

Replay avoids both. Every logged run already contains the full step sequence
and the verified outcome. Scoring each step in order answers "at which step
would the critic have stopped this run" exactly, with no randomness, in
seconds. What it cannot capture is any change in the agent's behaviour caused
by stopping — but stopping only ever truncates a run, it never alters the steps
before it, so the counterfactual is sound.

WHAT IS MEASURED
  steps saved     the compute that early stopping removes
  answers lost    correct runs stopped before they answered — the real cost
  failures caught doomed runs ended early

The last one is the flattering number and the least important. A critic that
stops everything catches 100% of failures.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from agent.critic.infer import DEFAULT_THRESHOLD, CriticScorer

DATA = ROOT / "data" / "critic"


def load_steps(test_only: bool = True) -> pd.DataFrame:
    df = pd.read_parquet(DATA / "labelled_steps.parquet")
    df = df[df.y_run_fails.notna()].copy()
    df["y"] = df.y_run_fails.astype(int)

    if test_only:
        split = json.loads((DATA / "split.json").read_text())
        df = df[df.run_id.isin(set(split["test_runs"]))].copy()

    return df.sort_values(["run_id", "step_index"]).reset_index(drop=True)


def simulate(df: pd.DataFrame, threshold: float, min_step: int = 0) -> pd.DataFrame:
    """One row per run: where would the critic have stopped it, and at what cost.

    `p_fail` must already be on df. Steps are walked in order and the first
    one over the threshold ends the run — exactly what agent/loop.py does.
    """
    out = []
    for run_id, g in df.groupby("run_id", sort=False):
        g = g.sort_values("step_index")
        n_steps = len(g)
        failed = bool(g.y.iloc[0])          # run-level, same for every step

        over = g[(g.p_fail > threshold) & (g.step_index >= min_step)]
        stopped = len(over) > 0
        stop_at = int(over.step_index.iloc[0]) if stopped else None

        # Steps after the stop point never happen. +1 because step_index is
        # 0-based and the stopping step itself still runs.
        steps_run = (stop_at + 1) if stopped else n_steps

        out.append({
            "run_id": run_id,
            "failed": failed,
            "n_steps": n_steps,
            "stopped": stopped,
            "stop_at": stop_at,
            "steps_run": steps_run,
            "steps_saved": n_steps - steps_run,
            # A correct run stopped before its final step loses its answer.
            "answer_lost": bool(stopped and not failed),
        })
    return pd.DataFrame(out)


def report(sim: pd.DataFrame, threshold: float, label: str = "") -> dict:
    total_steps = int(sim.n_steps.sum())
    steps_run = int(sim.steps_run.sum())
    saved = total_steps - steps_run

    n_runs = len(sim)
    n_ok = int((~sim.failed).sum())
    n_fail = int(sim.failed.sum())

    lost = int(sim.answer_lost.sum())
    caught = int((sim.stopped & sim.failed).sum())

    head = f"  early stopping at threshold {threshold}"
    if label:
        head += f"   [{label}]"
    print("\n" + "=" * 62)
    print(head)
    print("=" * 62)
    print(f"  {n_runs:,} runs   {n_fail:,} failed   {n_ok:,} succeeded")
    print()
    print(f"  steps without critic   {total_steps:>7,}")
    print(f"  steps with critic      {steps_run:>7,}")
    print(f"  STEPS SAVED            {saved:>7,}   ({saved/total_steps:.1%})")
    print()
    print(f"  failing runs caught    {caught:>7,} / {n_fail:,}  ({caught/max(n_fail,1):.1%})")
    print(f"  ANSWERS LOST           {lost:>7,} / {n_ok:,}  ({lost/max(n_ok,1):.1%})")
    print()

    kept = n_ok - lost
    print(f"  solve rate before      {n_ok/n_runs:.1%}")
    print(f"  solve rate after       {kept/n_runs:.1%}")

    if caught:
        med = sim[sim.stopped & sim.failed].stop_at.median()
        print(f"\n  median stop step on caught failures: {med:.0f}")
    print("=" * 62)

    return {
        "threshold": threshold,
        "runs": n_runs,
        "steps_before": total_steps,
        "steps_after": steps_run,
        "steps_saved": saved,
        "steps_saved_pct": round(saved / total_steps, 4),
        "failures_caught": caught,
        "failures_caught_pct": round(caught / max(n_fail, 1), 4),
        "answers_lost": lost,
        "answers_lost_pct": round(lost / max(n_ok, 1), 4),
        "solve_rate_before": round(n_ok / n_runs, 4),
        "solve_rate_after": round(kept / n_runs, 4),
    }


def sweep(df: pd.DataFrame, min_step: int = 0) -> pd.DataFrame:
    rows = []
    for t in [0.3, 0.4, 0.5, 0.6, 0.694, 0.75, 0.8, 0.85, 0.9, 0.95]:
        sim = simulate(df, t, min_step)
        n_ok = int((~sim.failed).sum())
        n_fail = int(sim.failed.sum())
        total = int(sim.n_steps.sum())
        rows.append({
            "threshold": t,
            "steps saved %": 100 * (total - sim.steps_run.sum()) / total,
            "answers lost %": 100 * sim.answer_lost.sum() / max(n_ok, 1),
            "failures caught %": 100 * (sim.stopped & sim.failed).sum() / max(n_fail, 1),
            "answers lost": int(sim.answer_lost.sum()),
        })
    return pd.DataFrame(rows)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--threshold", type=float, default=DEFAULT_THRESHOLD)
    ap.add_argument("--min-step", type=int, default=0)
    ap.add_argument("--all", action="store_true",
                    help="every logged run, not just the held-out ones")
    ap.add_argument("--no-text", action="store_true",
                    help="LightGBM only — no 600 MB text model")
    ap.add_argument("--sweep", action="store_true",
                    help="show the whole threshold curve")
    args = ap.parse_args()

    print("loading critic ...")
    critic = CriticScorer.load(use_text=not args.no_text)
    print(f"  {critic.version}")

    df = load_steps(test_only=not args.all)
    scope = "all logged runs" if args.all else "held-out test runs"
    print(f"\nscoring {len(df):,} steps from {df.run_id.nunique():,} {scope} ...")
    df["p_fail"] = critic.score_frame(df).values

    if args.sweep:
        print("\n" + "=" * 74)
        print("  threshold sweep — pick the point, don't inherit it")
        print("=" * 74)
        print(sweep(df, args.min_step).round(2).to_string(index=False))
        print("=" * 74)
        print("\n  Read the two middle columns together. Saving 40% of steps for")
        print("  1% of answers is a good trade; saving 60% for 15% is not.")
        return

    sim = simulate(df, args.threshold, args.min_step)
    summary = report(sim, args.threshold, critic.version)

    out = DATA / "early_stop_measurement.json"
    summary["critic"] = critic.version
    summary["scope"] = scope
    summary["min_step"] = args.min_step
    out.write_text(json.dumps(summary, indent=2))
    print(f"\n-> {out}")


if __name__ == "__main__":
    main()
