# tests/test_tools.py
"""Comprehensive tests for Dhrub's Real Tool Library (Layer A).
Covers success and error paths for all 8 tools, truncation contract, read-only DB mode,
REPL security sandboxing, and ToolResult schema validation.
"""
import os
import sqlite3
import tempfile
import pytest
import sys
from pathlib import Path

# Add project root to sys.path
sys.path.append(str(Path(__file__).parent.parent))

from contracts import ToolResult
from tools.sql_tools import run_sql, get_schema
from tools.python_tools import python_repl
from tools.charts import make_chart
from tools.stats_tools import stats_test, calculator
from tools.ml_tools import ml_regress, ml_cluster
from tools import TOOLS, TOOL_SPECS


@pytest.fixture(scope="module")
def test_db():
    """Create a temporary SQLite database for testing."""
    with tempfile.TemporaryDirectory() as temp_dir:
        db_file = os.path.join(temp_dir, "test_analytics.db")
        conn = sqlite3.connect(db_file)
        cur = conn.cursor()
        
        cur.execute("CREATE TABLE orders (order_id INTEGER PRIMARY KEY, region TEXT, sales REAL, discount REAL, quantity INT, profit REAL, date TEXT);")
        # Insert 30 rows to test 20-row truncation contract
        rows = [
            (i, "North America" if i % 2 == 0 else "Europe", 1000.0 * i, 0.05, 5, 200.0 * i, f"2023-01-{i:02d}")
            for i in range(1, 31)
        ]
        cur.executemany("INSERT INTO orders VALUES (?, ?, ?, ?, ?, ?, ?);", rows)
        conn.commit()
        conn.close()

        yield db_file


# 1. SQL Tool Tests
def test_run_sql_success_and_truncation(test_db):
    """Test standard SELECT query and 20-row truncation contract."""
    res = run_sql("SELECT * FROM orders;", db_path=test_db)
    assert isinstance(res, ToolResult)
    assert res.status == "ok"
    assert res.tool == "run_sql"
    assert len(res.data) == 20  # Max 20 rows
    assert res.row_count == 30   # True total row count
    assert res.truncated is True
    assert res.duration_ms >= 0

def test_run_sql_security_block(test_db):
    """Test that destructive / write queries are blocked."""
    res = run_sql("DROP TABLE orders;", db_path=test_db)
    assert res.status == "error"
    assert res.error_category == "permission"
    assert "Security Violation" in res.error

def test_run_sql_missing_column_hint(test_db):
    """Test that missing column errors produce actionable recovery hints."""
    res = run_sql("SELECT invalid_col FROM orders;", db_path=test_db)
    assert res.status == "error"
    assert res.error_category == "schema_missing_column"
    assert res.hint is not None
    assert "available columns" in res.hint

def test_get_schema_table_list(test_db):
    """Test get_schema(table=None) returns list of tables."""
    res = get_schema(table=None, db_path=test_db)
    assert res.status == "ok"
    assert "tables" in res.data
    assert "orders" in res.data["tables"]

def test_get_schema_specific_table(test_db):
    """Test get_schema(table='orders') returns detailed column definitions."""
    res = get_schema(table="orders", db_path=test_db)
    assert res.status == "ok"
    assert res.data["table"] == "orders"
    assert len(res.data["columns"]) == 7

def test_get_schema_missing_table(test_db):
    """Test get_schema with nonexistent table."""
    res = get_schema(table="non_existing_table", db_path=test_db)
    assert res.status == "error"
    assert res.error_category == "schema_missing_table"


# 2. Python REPL Tests
def test_python_repl_success():
    res = python_repl("x = [10, 20, 30]\nprint(sum(x))")
    assert res.status == "ok"
    assert res.data == "60"
    assert res.duration_ms >= 0

def test_python_repl_result_variable():
    res = python_repl("result = math.sqrt(144)")
    assert res.status == "ok"
    assert res.data == "12.0"

def test_python_repl_security_sandbox():
    """Test blocked imports: os, sys, subprocess, socket, etc."""
    res_os = python_repl("import os\nprint(os.listdir())")
    assert res_os.status == "error"
    assert res_os.error_category == "permission"

    res_sub = python_repl("import subprocess\nsubprocess.run(['ls'])")
    assert res_sub.status == "error"
    assert res_sub.error_category == "permission"


# 3. Chart Tool Tests
def test_make_chart_success():
    spec = {
        "type": "bar",
        "title": "Test Sales Chart",
        "x": ["North", "South", "East", "West"],
        "y": [100, 200, 150, 300],
        "x_label": "Region",
        "y_label": "Sales"
    }
    res = make_chart(spec)
    assert res.status == "ok"
    assert "chart_path" in res.data
    assert os.path.exists(res.data["chart_path"])

def test_make_chart_error():
    res = make_chart("invalid_spec_string")
    assert res.status == "error"
    assert res.error_category == "invalid_args"


# 4. Stats & Calculator Tool Tests
def test_calculator_success():
    res = calculator("(45000 + 38000) / 2")
    assert res.status == "ok"
    assert res.data == 41500.0

def test_calculator_zero_division():
    res = calculator("100 / 0")
    assert res.status == "error"
    assert res.error_category == "runtime"

def test_stats_test_correlation(test_db):
    res = stats_test(kind="correlation", table="orders", col1="sales", col2="profit", db_path=test_db)
    assert res.status == "ok"
    assert "r" in res.data

def test_stats_test_descriptive():
    res = stats_test(kind="descriptive", data=[10, 20, 30, 40, 50])
    assert res.status == "ok"
    assert res.data["mean"] == 30.0


# 5. ML Wrappers Tests
def test_ml_regress(test_db):
    res = ml_regress(table="orders", target="sales", features=["discount", "quantity"], db_path=test_db)
    assert res.status == "ok"
    assert "r2_score" in res.data

def test_ml_cluster(test_db):
    res = ml_cluster(table="orders", features=["sales", "profit"], k=2, db_path=test_db)
    assert res.status == "ok"
    assert "cluster_centers" in res.data


# 6. Registry Validation
def test_tools_registry_complete():
    assert len(TOOLS) == 8
    assert len(TOOL_SPECS) == 8
    for spec in TOOL_SPECS:
        assert "name" in spec
        assert "description" in spec
        assert "args" in spec
        assert spec["name"] in TOOLS