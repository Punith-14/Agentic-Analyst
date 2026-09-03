# tools/sql_tools.py
"""Dhrub (Layer A) - Real SQL & Schema Tools.
Adheres strictly to Contract 1 (ToolResult).
Never raises exceptions; wraps all failures into ToolResult(status="error").
"""
import os
import time
import sqlite3
import traceback
from typing import Optional, List, Dict, Any
from contracts import ToolResult, ErrorCategory
from tools.db import no_database, resolve, set_db   # noqa: F401  (set_db re-exported)

# No DEFAULT_DB_PATH any more. It used to be data/db/analytics.db, with
# _ensure_db_exists() creating that file when it was missing — so a query
# against a database nobody had selected would quietly succeed against
# generated demo data and the agent would report the numbers as fact.
#
# Now: nothing selected -> an error the UI can show the user. Call
# tools.db.set_db(path) with an uploaded file, or pass db_path explicitly.

def _extract_available_columns(db_path: str, table_name: Optional[str] = None) -> List[str]:
    """Helper to extract real column names for error recovery hints."""
    try:
        conn = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True, timeout=5.0)
        cursor = conn.cursor()
        cols = []
        if table_name:
            cursor.execute(f"PRAGMA table_info({table_name});")
            cols = [row[1] for row in cursor.fetchall()]
        else:
            cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%';")
            tables = [row[0] for row in cursor.fetchall()]
            for t in tables:
                cursor.execute(f"PRAGMA table_info({t});")
                for r in cursor.fetchall():
                    cols.append(f"{t}.{r[1]}")
        conn.close()
        return cols
    except Exception:
        return []

def _extract_available_tables(db_path: str) -> List[str]:
    """Helper to extract real table names for error recovery hints."""
    try:
        conn = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True, timeout=5.0)
        cursor = conn.cursor()
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%';")
        tables = [row[0] for row in cursor.fetchall()]
        conn.close()
        return tables
    except Exception:
        return []

def run_sql(query: str, db_path: Optional[str] = None) -> ToolResult:
    """Run a read-only SQL query against the selected database.
    - Read-only connection: mode=ro
    - 10-second timeout
    - Max 20 rows in data, with row_count and truncated flag
    - Structured error categories and actionable hints

    db_path=None uses whatever tools.db.set_db() selected. If nothing has
    been selected, this returns an error asking for a dataset rather than
    guessing at one.
    """
    start_time = time.time()

    db_path = resolve(db_path)
    if db_path is None:
        return no_database("run_sql")

    # Security check: Block write operations
    forbidden = ["INSERT", "UPDATE", "DELETE", "DROP", "ALTER", "CREATE", "TRUNCATE", "REPLACE", "ATTACH", "DETACH"]
    query_upper = query.strip().upper()
    for kw in forbidden:
        # Check as whole word or statement start
        if kw in query_upper.split() or query_upper.startswith(kw):
            duration_ms = int((time.time() - start_time) * 1000)
            return ToolResult(
                status="error",
                tool="run_sql",
                error="Security Violation: Only read-only SELECT queries are allowed.",
                error_full=f"Attempted forbidden keyword '{kw}' in query: {query}",
                error_category="permission",
                duration_ms=duration_ms,
                hint="Use read-only SELECT queries only."
            )

    try:
        # Connect using read-only URI mode as strictly required
        conn = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True, timeout=10.0)
        cursor = conn.cursor()
        
        cursor.execute(query)
        columns = [d[0] for d in cursor.description] if cursor.description else []
        rows = cursor.fetchall()
        conn.close()

        total_row_count = len(rows)
        truncated = total_row_count > 20
        limited_rows = rows[:20]

        result_data = [dict(zip(columns, row)) for row in limited_rows]
        duration_ms = int((time.time() - start_time) * 1000)

        hint = f"Returned {total_row_count} rows (truncated to 20)." if truncated else None
        return ToolResult(
            status="ok",
            tool="run_sql",
            data=result_data,
            row_count=total_row_count,
            truncated=truncated,
            duration_ms=duration_ms,
            hint=hint
        )

    except sqlite3.OperationalError as e:
        err_msg = str(e)
        full_tb = traceback.format_exc()
        duration_ms = int((time.time() - start_time) * 1000)

        # Categorize missing column vs missing table vs syntax
        if "no such column" in err_msg.lower():
            missing_col = err_msg.split(":")[-1].strip() if ":" in err_msg else ""
            avail_cols = _extract_available_columns(db_path)
            hint_str = f"available columns: {', '.join(avail_cols[:10])}" if avail_cols else "Check schema using get_schema."
            return ToolResult(
                status="error",
                tool="run_sql",
                error=f"no such column: {missing_col}"[:200],
                error_full=full_tb,
                error_category="schema_missing_column",
                duration_ms=duration_ms,
                hint=hint_str
            )
        elif "no such table" in err_msg.lower():
            avail_tables = _extract_available_tables(db_path)
            hint_str = f"available tables: {', '.join(avail_tables)}" if avail_tables else "Call get_schema() to see valid tables."
            return ToolResult(
                status="error",
                tool="run_sql",
                error=f"SQLite Error: {err_msg}"[:200],
                error_full=full_tb,
                error_category="schema_missing_table",
                duration_ms=duration_ms,
                hint=hint_str
            )
        else:
            return ToolResult(
                status="error",
                tool="run_sql",
                error=f"SQL Syntax/Operational Error: {err_msg}"[:200],
                error_full=full_tb,
                error_category="syntax",
                duration_ms=duration_ms,
                hint="Verify SQL syntax and table/column names with get_schema."
            )

    except Exception as e:
        duration_ms = int((time.time() - start_time) * 1000)
        return ToolResult(
            status="error",
            tool="run_sql",
            error=f"Execution Error: {str(e)}"[:200],
            error_full=traceback.format_exc(),
            error_category="runtime",
            duration_ms=duration_ms,
            hint="Verify database path and SQL query format."
        )

def get_schema(table: Optional[str] = None, db_path: Optional[str] = None) -> ToolResult:
    """Inspect database tables and column definitions.

    - table=None: EVERY table, with its columns and foreign keys
    - table='orders': the same detail for one table

    WHY table=None RETURNS EVERYTHING
    It used to return table names only, to save prompt tokens. Measured, that
    was a bad trade: the agent could see `orders` existed but not what was in
    it, so it guessed column names and 75% of its queries errored. Sending the
    full schema dropped that to 13%.

    The token cost is ~500 for a database of this size, not the ~1,500 the old
    docstring claimed — that figure was for three databases merged together,
    which is not something we do.
    """
    start_time = time.time()

    db_path = resolve(db_path)
    if db_path is None:
        return no_database("get_schema")

    try:
        conn = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True, timeout=10.0)
        cursor = conn.cursor()

        # Case 1: no table named -> the whole schema
        if not table:
            cursor.execute("SELECT name FROM sqlite_master WHERE type='table' "
                           "AND name NOT LIKE 'sqlite_%';")
            # Sorted, not sqlite_master order. The schema goes into the prompt
            # verbatim, so table order is part of the model's input — leaving it
            # to creation order would make prompts differ between databases that
            # happen to have been built in a different sequence, and stop runs
            # being byte-comparable with the ones the critic was trained on.
            names = sorted(row[0] for row in cursor.fetchall())

            schema: Dict[str, Any] = {}
            for name in names:
                cursor.execute(f"PRAGMA table_info({name});")
                cols = [r[1] for r in cursor.fetchall()]

                cursor.execute(f"PRAGMA foreign_key_list({name});")
                fks = [f"{r[3]} -> {r[2]}.{r[4]}" for r in cursor.fetchall()]

                schema[name] = {"columns": cols, "foreign_keys": fks}

            conn.close()
            duration_ms = int((time.time() - start_time) * 1000)
            return ToolResult(
                status="ok",
                tool="get_schema",
                data=schema,
                row_count=len(schema),
                truncated=False,
                duration_ms=duration_ms,
                hint="Pass table='<name>' for column types and primary keys.",
            )

        # Case 2: Inspect specific table
        cursor.execute(f"PRAGMA table_info({table});")
        columns_info = cursor.fetchall()

        if not columns_info:
            # Table does not exist
            cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%';")
            avail_tables = [row[0] for row in cursor.fetchall()]
            conn.close()
            duration_ms = int((time.time() - start_time) * 1000)
            return ToolResult(
                status="error",
                tool="get_schema",
                error=f"Table '{table}' does not exist.",
                error_full=f"Table '{table}' not found in SQLite schema.",
                error_category="schema_missing_table",
                duration_ms=duration_ms,
                hint=f"available tables: {', '.join(avail_tables)}"
            )

        cursor.execute(f"PRAGMA foreign_key_list({table});")
        fk_info = cursor.fetchall()
        conn.close()

        columns = []
        for col in columns_info:
            columns.append({
                "column_id": col[0],
                "name": col[1],
                "type": col[2],
                "not_null": bool(col[3]),
                "primary_key": bool(col[5])
            })

        foreign_keys = []
        for fk in fk_info:
            foreign_keys.append({
                "from_column": fk[3],
                "to_table": fk[2],
                "to_column": fk[4]
            })

        duration_ms = int((time.time() - start_time) * 1000)
        return ToolResult(
            status="ok",
            tool="get_schema",
            data={"table": table, "columns": columns, "foreign_keys": foreign_keys},
            row_count=len(columns),
            truncated=False,
            duration_ms=duration_ms
        )

    except Exception as e:
        duration_ms = int((time.time() - start_time) * 1000)
        return ToolResult(
            status="error",
            tool="get_schema",
            error=f"Schema Retrieval Error: {str(e)}"[:200],
            error_full=traceback.format_exc(),
            error_category="runtime",
            duration_ms=duration_ms,
            hint="Check if the SQLite database file path is valid."
        )