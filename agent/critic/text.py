"""Turns a trajectory step into the string the text critic reads.

The feature critic (critic/features.py) counts things about a query: how long
it is, how many joins, whether it errored. It cannot tell whether the query
joins the RIGHT two tables. That is the gap this is meant to close, and the
whole experiment rests on which text we hand over.

Same leakage rule as the feature critic: nothing here may come from a later
step or from the run outcome. Every field is knowable at the moment the step
was taken.

Four candidate formats, deliberately nested so the comparison is clean —
each adds exactly one thing to the one before it:

    sql        the query alone
    qsql       + what the user actually asked
    full       + the model's stated reasoning
    full_err   + the error message, when there is one

Notebook 04 measures which of these carries signal before any GPU time is
spent. Do not add a fifth without measuring it the same way.
"""

from __future__ import annotations

import pandas as pd

FORMATS = ["sql", "qsql", "full", "full_err"]

# Field markers rather than bare concatenation. BERT has no idea where the
# question stops and the SQL starts unless we tell it, and these tokens give
# it something consistent to key on.
SEP = " [SEP] "


def _clean(x) -> str:
    """Missing values arrive as None, NaN or the string 'None' depending on
    which column and which pandas version. All three mean 'nothing here'."""
    if x is None:
        return ""
    s = str(x).strip()
    if s.lower() in {"none", "nan", "null"}:
        return ""
    return " ".join(s.split())          # collapse newlines and runs of spaces


def build_text(row, fmt: str = "full") -> str:
    """One row of labelled_steps.parquet -> one string."""
    q = _clean(row.get("question"))
    t = _clean(row.get("thought"))
    sql = _clean(row.get("sql"))
    tool = _clean(row.get("tool")) or "none"
    err = _clean(row.get("obs_error"))

    # 27% of steps have no SQL — get_schema calls, final answers, parse
    # failures. Saying so explicitly beats handing the model an empty string,
    # which it would have to interpret.
    body = sql if sql else f"(no query, tool={tool})"

    if fmt == "sql":
        return body
    if fmt == "qsql":
        return f"question: {q}{SEP}sql: {body}"
    if fmt == "full":
        return f"question: {q}{SEP}thought: {t}{SEP}sql: {body}"
    if fmt == "full_err":
        tail = f"{SEP}error: {err}" if err else ""
        return f"question: {q}{SEP}thought: {t}{SEP}sql: {body}{tail}"

    raise ValueError(f"unknown format {fmt!r}, expected one of {FORMATS}")


def add_text_column(df: pd.DataFrame, fmt: str = "full") -> pd.DataFrame:
    out = df.copy()
    out["text"] = [build_text(r, fmt) for _, r in out.iterrows()]
    return out
