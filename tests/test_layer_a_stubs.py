# tests/test_layer_a_stubs.py
"""Tests for Layer A Stub Tools (Contract 7)."""
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

from contracts import ToolResult
from tools.stubs import (
    stub_run_sql, stub_get_schema, stub_python_repl, stub_make_chart,
    stub_stats_test, stub_calculator, stub_ml_regress, stub_ml_cluster, STUB_TOOLS
)

def test_stub_run_sql_success():
    res = stub_run_sql("SELECT region, SUM(sales) FROM orders GROUP BY region;")
    assert isinstance(res, ToolResult)
    assert res.status == "ok"
    assert res.tool == "run_sql"
    assert len(res.data) == 2
    assert res.row_count == 2
    assert res.duration_ms > 0

def test_stub_run_sql_forced_failure():
    """Contract 7 forced-failure path on 'sale_amount'."""
    res = stub_run_sql("SELECT sale_amount FROM orders;")
    assert isinstance(res, ToolResult)
    assert res.status == "error"
    assert res.error == "no such column: sale_amount"
    assert res.error_category == "schema_missing_column"
    assert "available columns" in res.hint

def test_stub_get_schema():
    res = stub_get_schema()
    assert res.status == "ok"
    assert "tables" in res.data
    assert "orders" in res.data["tables"]

    res_tbl = stub_get_schema("orders")
    assert res_tbl.status == "ok"
    assert res_tbl.data["table"] == "orders"

    res_err = stub_get_schema("unknown_table_xyz")
    assert res_err.status == "error"
    assert res_err.error_category == "schema_missing_table"

def test_stub_python_repl():
    res = stub_python_repl("result = 45000")
    assert res.status == "ok"
    assert res.data == "45000"

    res_sec = stub_python_repl("import os; os.system('dir')")
    assert res_sec.status == "error"
    assert res_sec.error_category == "permission"

def test_all_stubs_present_in_registry():
    expected_tools = ["run_sql", "get_schema", "python_repl", "make_chart", "stats_test", "calculator", "ml_regress", "ml_cluster"]
    for t in expected_tools:
        assert t in STUB_TOOLS
        assert callable(STUB_TOOLS[t])
