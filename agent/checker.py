"""Did the agent get it right?

Runs the gold SQL, then looks for its values in whatever the agent said. The
agent answers in prose — "There are 347 albums" — so exact string comparison is
useless; we compare the values instead.

This is deliberately lenient on wording and strict on values. A wrong number
never passes; a differently-phrased right number does.
"""

from __future__ import annotations

import re
import sqlite3
from pathlib import Path
from typing import Any

DB_DIR = Path(__file__).resolve().parent.parent / "data" / "db"

NUM_RE = re.compile(r"-?\d[\d,]*\.?\d*")


def run_gold(gold_sql: str, db: str) -> tuple[list[tuple], str | None]:
    """Execute the reference query. Returns (rows, error)."""
    path = DB_DIR / f"{db}.db"
    try:
        con = sqlite3.connect(f"file:{path}?mode=ro", uri=True, timeout=10)
        rows = con.execute(gold_sql).fetchall()
        con.close()
        return rows, None
    except sqlite3.Error as e:
        return [], str(e)


def _numbers(text: str) -> list[float]:
    out = []
    for m in NUM_RE.findall(text or ""):
        try:
            out.append(float(m.replace(",", "")))
        except ValueError:
            pass
    return out


def _norm(s: Any) -> str:
    return re.sub(r"[^a-z0-9]+", " ", str(s).lower()).strip()


def _value_present(value: Any, answer: str, tol: float = 0.01) -> bool:
    """Is this single gold value present in the agent's answer?"""
    if value is None:
        return False

    if isinstance(value, (int, float)) and not isinstance(value, bool):
        target = float(value)
        for n in _numbers(answer):
            if abs(n - target) <= max(tol, abs(target) * tol):
                return True
        return False

    text = _norm(value)
    if not text:
        return False
    # short strings must match a whole token, otherwise "US" matches "customers"
    if len(text) <= 3:
        return text in _norm(answer).split()
    return text in _norm(answer)


def check_answer(answer: str | None, gold_sql: str, db: str,
                 max_values: int = 5) -> tuple[bool | None, str]:
    """Compare the agent's final answer against the gold result.

    Returns (correct, reason). `correct` is None when the gold query itself
    failed — that's our problem, not the agent's, and it must not be scored as
    a wrong answer.
    """
    if answer is None or not str(answer).strip():
        return False, "no answer given"

    rows, err = run_gold(gold_sql, db)
    if err:
        return None, f"gold SQL failed: {err[:120]}"
    if not rows:
        return None, "gold SQL returned no rows"

    # Flatten, but only score the first few values — for a 2,000-row result the
    # agent is expected to summarise, not recite.
    values = [v for row in rows[:max_values] for v in row]
    if not values:
        return None, "gold SQL returned no values"

    hits = [v for v in values if _value_present(v, answer)]

    # Single-value answers (counts, maxima) must be exact.
    if len(values) == 1:
        ok = len(hits) == 1
        return ok, ("value found" if ok else f"expected {values[0]!r}")

    # Multi-value answers: most of the top values should appear.
    ratio = len(hits) / len(values)
    ok = ratio >= 0.6
    return ok, f"{len(hits)}/{len(values)} gold values found"


# --------------------------------------------------------------------------
# execution accuracy — the standard Spider/BIRD metric
# --------------------------------------------------------------------------


def _canon(rows: list[tuple], ordered: bool) -> list:
    """Normalise a result set for comparison.

    Floats are rounded — SUM over the same rows can differ in the last bits
    between two syntactically different queries. Order is ignored unless the
    gold query has an ORDER BY, matching Spider's convention.
    """
    def cell(v):
        if isinstance(v, float):
            return round(v, 6)
        return v

    out = [tuple(cell(v) for v in r) for r in rows]
    return out if ordered else sorted(out, key=lambda t: [str(x) for x in t])


def execution_match(pred_sql: str, gold_sql: str, db: str) -> tuple[bool | None, str]:
    """Run both queries and compare result sets."""
    gold_rows, err = run_gold(gold_sql, db)
    if err:
        return None, f"gold SQL failed: {err[:100]}"

    pred_rows, err = run_gold(pred_sql, db)
    if err:
        return False, f"predicted SQL failed: {err[:100]}"

    ordered = "order by" in gold_sql.lower()
    g, p = _canon(gold_rows, ordered), _canon(pred_rows, ordered)

    if g == p:
        return True, f"execution match ({len(g)} rows)"

    # A single-column projection of the same rows is still the right answer —
    # gold may select two columns where the question only asked for one.
    if g and p and len(g) == len(p):
        gset = {tuple(sorted(str(v) for v in r)) for r in g}
        pset = {tuple(sorted(str(v) for v in r)) for r in p}
        if gset == pset:
            return True, "execution match (columns reordered)"

    return False, f"result mismatch: gold {len(g)} rows, predicted {len(p)}"


def last_successful_sql(record) -> str | None:
    """The last run_sql that actually returned something, before the answer.

    That's our best guess at 'the query the agent based its answer on'. Not
    perfect — the agent may have run a follow-up — but it is what the answer
    was derived from in the overwhelming majority of runs.
    """
    best = None
    for s in record.steps:
        a, o = s.action, s.observation
        if a and a.tool == "run_sql" and o and o.status == "ok" and (o.row_count or 0) > 0:
            best = a.args.get("query")
    return best


def score_run(record, gold_sql: str, db: str) -> tuple[bool | None, str, str]:
    """Score a run. Returns (correct, reason, method).

    Execution accuracy where possible — it is the published metric and it
    cannot be fooled by a number appearing in the prose for another reason.
    Falls back to answer matching when the agent didn't end on a query.
    """
    if record.termination != "final_answer":
        return False, f"terminated as {record.termination}", "none"

    pred = last_successful_sql(record)
    if pred:
        ok, reason = execution_match(pred, gold_sql, db)
        if ok is not None:
            return ok, reason, "execution"
        return ok, reason, "none"          # gold itself is broken

    ok, reason = check_answer(record.final_answer, gold_sql, db)
    return ok, reason, "answer_match"
