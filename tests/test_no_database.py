"""No dataset selected -> a clear error, never an invented answer.

This is the behaviour these tests exist to lock down:

    the tools used to default to data/db/analytics.db, and create it if it
    was missing. So a query against a database nobody had chosen would
    succeed against generated demo data, and the agent would answer

        "North America had the highest sales at $180,500."

    That looks like a real answer. If a user's upload failed, or the frontend
    forgot to connect their file, nothing anywhere would say so.

Anyone reinstating a default will fail these tests, which is the point.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from tools.db import clear_db, current_db, set_db
from tools.ml_tools import ml_regress
from tools.sql_tools import get_schema, run_sql


@pytest.fixture(autouse=True)
def no_db_selected():
    clear_db()
    yield
    clear_db()


# ==========================================================================
# nothing selected
# ==========================================================================


def test_run_sql_refuses_without_a_database():
    r = run_sql("SELECT 1")
    assert r.status == "error"
    assert r.data is None, "returned data with no database connected"
    assert "no dataset" in r.error.lower()


def test_get_schema_refuses_without_a_database():
    r = get_schema()
    assert r.status == "error"
    assert r.data is None


def test_ml_tools_refuse_without_a_database():
    r = ml_regress(table="orders", target="sales", features=["quantity"])
    assert r.status == "error"
    assert "no dataset" in r.error.lower()


def test_the_error_tells_the_user_what_to_do():
    """It is shown in the UI, so it has to read like a message to a person,
    not a stack trace."""
    r = run_sql("SELECT 1")
    assert "upload" in (r.error + (r.hint or "")).lower()
    assert "Traceback" not in r.error


def test_tools_never_raise_even_with_no_database():
    """Contract 1: tools return status='error', they do not raise. The loop
    depends on it."""
    for call in [lambda: run_sql("SELECT 1"), lambda: get_schema(),
                 lambda: get_schema(table="orders")]:
        assert call().status == "error"


def test_no_database_is_auto_created():
    """_ensure_db_exists() used to generate analytics.db on demand. A tool
    asked to query a missing database must not invent one."""
    run_sql("SELECT 1")
    get_schema()
    assert current_db() is None, "a database appeared without being asked for"


# ==========================================================================
# once one is selected
# ==========================================================================


def test_set_db_by_name():
    set_db("chinook_1")
    assert current_db().endswith("chinook_1.db")
    assert run_sql("SELECT COUNT(*) FROM Album").status == "ok"


def test_set_db_by_path():
    """The form the frontend uses for an uploaded file."""
    p = Path(__file__).resolve().parent.parent / "data" / "db" / "chinook_1.db"
    set_db(str(p))
    assert run_sql("SELECT COUNT(*) FROM Album").status == "ok"


def test_set_db_rejects_a_missing_file():
    with pytest.raises(FileNotFoundError):
        set_db("/nowhere/does_not_exist.db")


def test_explicit_db_path_still_wins():
    """Layer D passes db_path directly in places; that must keep working."""
    p = Path(__file__).resolve().parent.parent / "data" / "db" / "chinook_1.db"
    assert run_sql("SELECT COUNT(*) FROM Album", db_path=str(p)).status == "ok"
    assert current_db() is None, "an explicit path should not change the selection"


def test_switching_databases_switches_results():
    set_db("chinook_1")
    assert run_sql("SELECT COUNT(*) FROM Album").status == "ok"

    set_db("college_2")
    assert run_sql("SELECT COUNT(*) FROM Album").status == "error", \
        "still querying the previous database"
    assert run_sql("SELECT COUNT(*) FROM student").status == "ok"
