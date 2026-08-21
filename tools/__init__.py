# tools/__init__.py
"""Dhrub (Layer A) - Central Tool Registry and Prompt Specifications.
Provides:
- TOOLS: dict[str, Callable[..., ToolResult]]
- TOOL_SPECS: list[dict]
"""
from typing import Dict, Callable, List, Any
from contracts import ToolResult

from tools.sql_tools import run_sql, get_schema
from tools.python_tools import python_repl
from tools.charts import make_chart
from tools.stats_tools import stats_test, calculator
from tools.ml_tools import ml_regress, ml_cluster
from tools.stubs import STUB_TOOLS

# Real Tools Registry
TOOLS: Dict[str, Callable[..., ToolResult]] = {
    "run_sql": run_sql,
    "get_schema": get_schema,
    "python_repl": python_repl,
    "make_chart": make_chart,
    "stats_test": stats_test,
    "calculator": calculator,
    "ml_regress": ml_regress,
    "ml_cluster": ml_cluster,
}

# Prompt-ready Tool Specifications (Contract 1.2 / Page 8)
TOOL_SPECS: List[Dict[str, Any]] = [
    {
        "name": "run_sql",
        "description": "Run a read-only SQL query against the analytics database.",
        "args": {
            "query": "string - a valid SQLite SELECT statement",
            "db_path": "optional string - database path (defaults to data/db/analytics.db)"
        },
        "returns": "up to 20 rows, plus the true row count and truncation status",
        "example": {"query": "SELECT region, SUM(sales) FROM orders GROUP BY region;"}
    },
    {
        "name": "get_schema",
        "description": "Inspect database schema. If table is None, returns table names. If table name given, returns column names, types, primary keys, and foreign keys.",
        "args": {
            "table": "optional string - table name to inspect (None to list all tables)",
            "db_path": "optional string - database path"
        },
        "returns": "table list or column schema definitions",
        "example": {"table": "orders"}
    },
    {
        "name": "python_repl",
        "description": "Execute sandboxed Python code to calculate metrics, transform data, or process logic.",
        "args": {"code": "string - valid Python code snippet"},
        "returns": "stdout output or value of 'result' variable",
        "example": {"code": "result = sum([15000, 24000, 18000])"}
    },
    {
        "name": "make_chart",
        "description": "Generate visual charts (bar, line, scatter, pie, histogram) and save as PNG.",
        "args": {
            "spec": "dictionary with 'type', 'title', 'x', 'y', 'x_label', 'y_label'"
        },
        "returns": "path to the generated chart PNG file in data/charts/",
        "example": {
            "spec": {
                "type": "bar",
                "title": "Sales by Region",
                "x": ["North America", "Europe", "Asia"],
                "y": [180500, 125000, 135000]
            }
        }
    },
    {
        "name": "stats_test",
        "description": "Run statistical tests including t-test, correlation, chi-square, and descriptive stats.",
        "args": {
            "kind": "string - 't_test', 'correlation', 'chi_square', or 'descriptive'",
            "data": "optional list or dict of numbers",
            "table": "optional string table name",
            "col1": "optional column 1 name",
            "col2": "optional column 2 name"
        },
        "returns": "test statistics, p-values, significance, and effect size",
        "example": {"kind": "correlation", "table": "orders", "col1": "sales", "col2": "profit"}
    },
    {
        "name": "calculator",
        "description": "Evaluate mathematical expressions safely.",
        "args": {"expression": "string - mathematical expression"},
        "returns": "numerical result of the expression",
        "example": {"expression": "(48000 + 50000 + 45000) / 3"}
    },
    {
        "name": "ml_regress",
        "description": "Train scikit-learn regression model (linear, ridge, random_forest) on table columns.",
        "args": {
            "table": "string table name",
            "target": "string target column name",
            "features": "list of feature column names",
            "model_type": "string - 'linear', 'ridge', 'random_forest'"
        },
        "returns": "R2 score, RMSE, feature coefficients/importance",
        "example": {"table": "orders", "target": "sales", "features": ["quantity", "discount"]}
    },
    {
        "name": "ml_cluster",
        "description": "Perform KMeans clustering on database features.",
        "args": {
            "table": "string table name",
            "features": "list of feature column names",
            "k": "integer number of clusters (default 3)"
        },
        "returns": "cluster centers, silhouette score, and counts",
        "example": {"table": "orders", "features": ["sales", "profit"], "k": 3}
    }
]

TOOL_LIST = list(TOOLS.values())