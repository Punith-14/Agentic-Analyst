# tools/stubs.py
"""Harish (Layer A) - Contract 7: Stub Tools.
Ships early so B, C, and D can build and test against fixed ToolResult shapes.
Includes forced-failure paths for testing error recovery.
"""
from typing import Any, Dict, Optional, Callable
from contracts import ToolResult

def stub_run_sql(query: str, db_path: Optional[str] = None, **kwargs) -> ToolResult:
    """Fake SQL executor with forced failure on 'sale_amount'."""
    if "sale_amount" in query:  # Forced failure path from Contract 7
        return ToolResult(
            status="error",
            tool="run_sql",
            error="no such column: sale_amount",
            error_category="schema_missing_column",
            hint="available columns: order_id, region, sales, date",
            duration_ms=12,
            truncated=False,
            row_count=None,
        )
    return ToolResult(
        status="ok",
        tool="run_sql",
        data=[
            {"region": "North", "total": 45000},
            {"region": "South", "total": 38000},
        ],
        row_count=2,
        truncated=False,
        duration_ms=34,
    )

def stub_get_schema(table: Optional[str] = None, db_path: Optional[str] = None, **kwargs) -> ToolResult:
    """Fake schema inspector."""
    if table is None:
        return ToolResult(
            status="ok",
            tool="get_schema",
            data={"tables": ["orders", "sales", "customers", "products", "regions"]},
            row_count=5,
            duration_ms=8,
        )
    if table.lower() in ["orders", "sales"]:
        return ToolResult(
            status="ok",
            tool="get_schema",
            data={
                "table": table,
                "columns": [
                    {"name": "order_id", "type": "INTEGER", "primary_key": True},
                    {"name": "region", "type": "TEXT", "primary_key": False},
                    {"name": "sales", "type": "REAL", "primary_key": False},
                    {"name": "date", "type": "TEXT", "primary_key": False},
                ],
                "foreign_keys": []
            },
            row_count=4,
            duration_ms=10,
        )
    return ToolResult(
        status="error",
        tool="get_schema",
        error=f"Table '{table}' does not exist.",
        error_category="schema_missing_table",
        hint="available tables: orders, sales, customers, products, regions",
        duration_ms=9,
    )

def stub_python_repl(code: str, **kwargs) -> ToolResult:
    """Fake Python REPL with security check."""
    for mod in ["os", "sys", "subprocess", "socket", "shutil", "requests", "importlib", "builtins"]:
        if f"import {mod}" in code or f"from {mod}" in code:
            return ToolResult(
                status="error",
                tool="python_repl",
                error=f"Security Violation: Importing '{mod}' is blocked.",
                error_category="permission",
                hint="Use pure Python math and data processing only.",
                duration_ms=5,
            )
    return ToolResult(
        status="ok",
        tool="python_repl",
        data="45000",
        duration_ms=25,
    )

def stub_make_chart(spec: dict, **kwargs) -> ToolResult:
    """Fake chart generator."""
    return ToolResult(
        status="ok",
        tool="make_chart",
        data={
            "chart_path": "data/charts/stub_chart.png",
            "type": spec.get("type", "bar"),
            "title": spec.get("title", "Data Chart"),
            "spec": spec
        },
        duration_ms=45,
        hint=f"Chart rendered successfully for '{spec.get('title', 'Data Chart')}'."
    )

def stub_stats_test(kind: str = "t_test", **kwargs) -> ToolResult:
    """Fake stats test."""
    return ToolResult(
        status="ok",
        tool="stats_test",
        data={"kind": kind, "statistic": 2.45, "p_value": 0.016, "significant": True},
        duration_ms=18,
    )

def stub_calculator(expression: str, **kwargs) -> ToolResult:
    """Fake calculator."""
    return ToolResult(
        status="ok",
        tool="calculator",
        data=83000.0,
        duration_ms=2,
    )

def stub_ml_regress(table: str = "sales", target: str = "sales", **kwargs) -> ToolResult:
    """Fake ML regression."""
    return ToolResult(
        status="ok",
        tool="ml_regress",
        data={"model": "LinearRegression", "r2_score": 0.87, "rmse": 124.5},
        duration_ms=65,
    )

def stub_ml_cluster(table: str = "sales", k: int = 3, **kwargs) -> ToolResult:
    """Fake ML clustering."""
    return ToolResult(
        status="ok",
        tool="ml_cluster",
        data={"k": k, "inertia": 450.2, "cluster_counts": [12, 18, 10]},
        duration_ms=55,
    )

STUB_TOOLS: Dict[str, Callable[..., ToolResult]] = {
    "run_sql": stub_run_sql,
    "get_schema": stub_get_schema,
    "python_repl": stub_python_repl,
    "make_chart": stub_make_chart,
    "stats_test": stub_stats_test,
    "calculator": stub_calculator,
    "ml_regress": stub_ml_regress,
    "ml_cluster": stub_ml_cluster,
}

TOOLS = STUB_TOOLS
