"""Placeholder tools until layer A is ready.

Same ToolResult contract, so swapping over is one import:

    from agent._stubs import TOOLS, TOOL_SPECS     # now
    from tools import TOOLS, TOOL_SPECS            # later

These hit the real SQLite files, so results are genuine — only the sandboxing
and validation are simplified. The failure paths are deliberate; a stub that
always succeeds hides every bug in the error handling.
"""

from __future__ import annotations

import re
import sqlite3
import time
from pathlib import Path
from typing import Callable

from contracts import MAX_ROWS_IN_DATA, ToolResult

DB_DIR = Path(__file__).resolve().parent.parent / "data" / "db"
TIMEOUT_S = 10.0

# the runner sets this per task
CURRENT_DB = "chinook_1"


def set_db(db_name: str) -> None:
    global CURRENT_DB
    CURRENT_DB = db_name


def _connect() -> sqlite3.Connection:
    path = DB_DIR / f"{CURRENT_DB}.db"
    con = sqlite3.connect(f"file:{path}?mode=ro", uri=True, timeout=TIMEOUT_S)
    con.row_factory = sqlite3.Row
    return con


def _classify(msg: str) -> tuple[str, str | None]:
    m = msg.lower()
    if "no such column" in m:
        return "schema_missing_column", "columns"
    if "no such table" in m:
        return "schema_missing_table", "tables"
    if "syntax error" in m or "incomplete input" in m:
        return "syntax", None
    if "timeout" in m or "locked" in m:
        return "timeout", None
    return "runtime", None


def _columns_hint(query: str) -> str | None:
    """List the real columns of whatever tables the query mentions.

    Worth the extra round-trip: without it the model just guesses again and
    burns another iteration.
    """
    names = set(re.findall(r"\bfrom\s+([a-zA-Z_]\w*)", query, re.I)) | \
            set(re.findall(r"\bjoin\s+([a-zA-Z_]\w*)", query, re.I))
    if not names:
        return None
    try:
        con = _connect()
        parts = []
        for t in list(names)[:3]:
            cols = [r[1] for r in con.execute(f'PRAGMA table_info("{t}")')]
            if cols:
                parts.append(f"{t}: {', '.join(cols)}")
        con.close()
        return "available columns — " + " | ".join(parts) if parts else None
    except sqlite3.Error:
        return None


def stub_run_sql(query: str = "") -> ToolResult:
    t0 = time.perf_counter()
    ms = lambda: int((time.perf_counter() - t0) * 1000)

    if not query.strip():
        return ToolResult(status="error", tool="run_sql",
                          error="empty query", error_category="invalid_args",
                          hint="pass a SELECT statement as `query`", duration_ms=ms())

    if re.match(r"\s*(insert|update|delete|drop|alter|create)\b", query, re.I):
        return ToolResult(status="error", tool="run_sql",
                          error="database is read-only; only SELECT is allowed",
                          error_category="permission", duration_ms=ms())

    try:
        con = _connect()
        cur = con.execute(query)
        rows = cur.fetchall()
        cols = [d[0] for d in cur.description] if cur.description else []
        con.close()
    except sqlite3.Error as e:
        cat, want = _classify(str(e))
        hint = _columns_hint(query) if want == "columns" else None
        if want == "tables":
            r = stub_get_schema()
            hint = ("available tables: " + ", ".join(r.data)) if r.status == "ok" else None
        return ToolResult(status="error", tool="run_sql",
                          error=str(e)[:200], error_full=repr(e),
                          error_category=cat, hint=hint, duration_ms=ms())

    if not rows:
        return ToolResult(status="ok", tool="run_sql", data=[], row_count=0,
                          error_category="empty_result",
                          hint="query ran but matched no rows — check your filters",
                          duration_ms=ms())

    data = [dict(zip(cols, r)) for r in rows[:MAX_ROWS_IN_DATA]]
    return ToolResult(status="ok", tool="run_sql", data=data,
                      row_count=len(rows),
                      truncated=len(rows) > MAX_ROWS_IN_DATA,
                      duration_ms=ms())


def stub_get_schema(table: str | None = None) -> ToolResult:
    t0 = time.perf_counter()
    ms = lambda: int((time.perf_counter() - t0) * 1000)
    try:
        con = _connect()
        tables = [r[0] for r in con.execute(
            "SELECT name FROM sqlite_master WHERE type='table' "
            "AND name NOT LIKE 'sqlite_%' ORDER BY name")]

        if table is None:
            # Return EVERY table with its columns, not just the names.
            # Measured on the first batch: 76% of queries errored before any
            # get_schema call, and still 65% after one — because a join needs
            # several tables and the agent only ever looked up one. The whole
            # schema is ~500 tokens, which we can afford.
            full = {}
            for t in tables:
                cols = [r[1] for r in con.execute(f'PRAGMA table_info("{t}")')]
                fks = [f'{r[3]} -> {r[2]}.{r[4]}'
                       for r in con.execute(f'PRAGMA foreign_key_list("{t}")')]
                full[t] = {"columns": cols}
                if fks:
                    full[t]["foreign_keys"] = fks
            con.close()
            return ToolResult(status="ok", tool="get_schema",
                              data=full, row_count=len(tables),
                              duration_ms=ms())

        if table not in tables:
            con.close()
            return ToolResult(status="error", tool="get_schema",
                              error=f"no such table: {table}",
                              error_category="schema_missing_table",
                              hint=f"available tables: {', '.join(tables)}",
                              duration_ms=ms())

        cols = [{"name": r[1], "type": r[2] or "TEXT", "pk": bool(r[5])}
                for r in con.execute(f'PRAGMA table_info("{table}")')]
        fks = [{"column": r[3], "references": f"{r[2]}.{r[4]}"}
               for r in con.execute(f'PRAGMA foreign_key_list("{table}")')]
        con.close()
        return ToolResult(status="ok", tool="get_schema",
                          data={"table": table, "columns": cols, "foreign_keys": fks},
                          row_count=len(cols), duration_ms=ms())
    except sqlite3.Error as e:
        return ToolResult(status="error", tool="get_schema", error=str(e)[:200],
                          error_full=repr(e), error_category="runtime", duration_ms=ms())


def stub_python_repl(code: str = "") -> ToolResult:
    """Minimal on purpose — the real sandbox is layer A's job. This exists so
    the loop has a second tool and a blocked-import path to exercise."""
    t0 = time.perf_counter()
    ms = lambda: int((time.perf_counter() - t0) * 1000)

    blocked = ("os", "sys", "subprocess", "socket", "shutil",
               "requests", "importlib", "open(", "__import__", "eval(", "exec(")
    for b in blocked:
        if b in code:
            return ToolResult(status="error", tool="python_repl",
                              error=f"blocked in sandbox: {b}",
                              error_category="permission",
                              hint="only arithmetic and simple expressions are allowed",
                              duration_ms=ms())
    try:
        result = eval(code, {"__builtins__": {}}, {"round": round, "abs": abs,
                                                   "min": min, "max": max, "sum": sum})
        return ToolResult(status="ok", tool="python_repl", data=result, duration_ms=ms())
    except Exception as e:                       # noqa: BLE001 — tools don't raise
        return ToolResult(status="error", tool="python_repl", error=str(e)[:200],
                          error_full=repr(e), error_category="runtime", duration_ms=ms())


TOOLS: dict[str, Callable[..., ToolResult]] = {
    "run_sql": stub_run_sql,
    "get_schema": stub_get_schema,
    "python_repl": stub_python_repl,
}

TOOL_SPECS: list[dict] = [
    {
        "name": "run_sql",
        "description": "Run a read-only SQL query against the database.",
        "args": {"query": "string — a valid SQLite SELECT statement"},
        "returns": f"up to {MAX_ROWS_IN_DATA} rows, plus the true row count",
        "example": {"query": "SELECT region, SUM(sales) FROM orders GROUP BY region"},
    },
    {
        "name": "get_schema",
        "description": "List tables, or the columns of one table.",
        "args": {"table": "string or omitted — omit to list all table names"},
        "returns": "table names, or columns and foreign keys for one table",
        "example": {"table": "albums"},
    },
    {
        "name": "python_repl",
        "description": "Evaluate a simple arithmetic expression.",
        "args": {"code": "string — a single Python expression"},
        "returns": "the value of the expression",
        "example": {"code": "round(45000 / 3, 2)"},
    },
]
