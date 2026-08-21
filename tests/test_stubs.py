"""Stub tools and the scripted LLM.

These enforce the ToolResult contract, so when layer A lands we point this at
the real tools and it should pass unchanged.

Spider's chinook_1 uses singular capitalised table names (Album, Track), not
the plural lowercase of the original Chinook schema — hence the constants.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from agent._stubs import TOOL_SPECS, TOOLS, set_db
from agent.llm import SCENARIOS, ScriptedLLM, scripted
from contracts import MAX_ROWS_IN_DATA, ToolResult

# chinook_1 fixtures
T_SMALL = "Album"          # 347 rows
T_BIG = "Track"            # 3,503 rows — exercises truncation
COL_REAL = "Title"         # a real column on Album
COL_FAKE = "album_title"   # plausible but wrong — the recovery case


@pytest.fixture(autouse=True)
def use_chinook():
    set_db("chinook_1")


# --------------------------------------------------------------------------
# the contract
# --------------------------------------------------------------------------


def test_every_tool_returns_toolresult():
    """Tools never raise, whatever you throw at them."""
    bad_inputs = [
        ("run_sql", {"query": "SELECT * FROM does_not_exist"}),
        ("run_sql", {"query": "SELCT nonsense ((("}),
        ("run_sql", {"query": ""}),
        ("run_sql", {"query": f"DROP TABLE {T_SMALL}"}),
        ("get_schema", {"table": "not_a_table"}),
        ("get_schema", {}),
        ("python_repl", {"code": "import os"}),
        ("python_repl", {"code": "1/0"}),
        ("python_repl", {"code": "@@@"}),
    ]
    for tool, args in bad_inputs:
        result = TOOLS[tool](**args)
        assert isinstance(result, ToolResult), f"{tool}{args} did not return ToolResult"
        assert result.tool == tool
        assert result.duration_ms >= 0


def test_errors_are_categorised():
    r = TOOLS["run_sql"](query=f"SELECT nope FROM {T_SMALL}")
    assert r.status == "error"
    assert r.error_category == "schema_missing_column"
    assert r.error is not None and len(r.error) <= 200

    r = TOOLS["run_sql"](query="SELECT * FROM nope")
    assert r.error_category == "schema_missing_table"

    r = TOOLS["run_sql"](query="SELCT (((")
    assert r.error_category == "syntax"


def test_missing_column_gives_a_hint():
    """Without the hint the model just guesses again and burns an iteration."""
    r = TOOLS["run_sql"](query=f"SELECT {COL_FAKE} FROM {T_SMALL}")
    assert r.status == "error"
    assert r.hint is not None
    assert COL_REAL.lower() in r.hint.lower(), \
        f"hint should name the real column: {r.hint}"


def test_writes_are_refused():
    for q in [f"INSERT INTO {T_SMALL} VALUES (1)",
              f"DROP TABLE {T_SMALL}",
              f"UPDATE {T_SMALL} SET {COL_REAL}='x'",
              f"DELETE FROM {T_SMALL}"]:
        r = TOOLS["run_sql"](query=q)
        assert r.status == "error"
        assert r.error_category == "permission"


def test_results_are_truncated():
    r = TOOLS["run_sql"](query=f"SELECT * FROM {T_BIG}")
    assert r.status == "ok"
    assert len(r.data) <= MAX_ROWS_IN_DATA
    assert r.truncated is True
    assert r.row_count > MAX_ROWS_IN_DATA, "row_count must be the TRUE count"


def test_empty_result_is_flagged_not_errored():
    r = TOOLS["run_sql"](query=f"SELECT * FROM {T_SMALL} WHERE {COL_REAL}='___nope___'")
    assert r.status == "ok"
    assert r.row_count == 0
    assert r.error_category == "empty_result"


def test_traceback_never_reaches_the_prompt():
    r = TOOLS["run_sql"](query=f"SELECT nope FROM {T_SMALL}")
    obs = r.short_observation()
    assert r.error_full is not None          # logged
    assert r.error_full not in obs           # but not shown to the model
    assert len(obs) < 500


# --------------------------------------------------------------------------
# schema tool
# --------------------------------------------------------------------------


def test_get_schema_returns_every_table_with_columns():
    """Changed after the first batch: 65% of queries still errored after a
    single-table lookup, because joins need several tables."""
    r = TOOLS["get_schema"]()
    assert r.status == "ok"
    assert T_SMALL in r.data
    assert T_BIG in r.data
    assert COL_REAL in r.data[T_SMALL]["columns"]
    assert any("foreign_keys" in v for v in r.data.values())


def test_get_schema_returns_columns():
    r = TOOLS["get_schema"](table=T_SMALL)
    assert r.status == "ok"
    names = [c["name"] for c in r.data["columns"]]
    assert COL_REAL in names


def test_get_schema_unknown_table_lists_real_ones():
    r = TOOLS["get_schema"](table="Albumz")
    assert r.status == "error"
    assert T_SMALL in r.hint


def test_get_schema_reports_foreign_keys():
    r = TOOLS["get_schema"](table=T_BIG)
    assert r.status == "ok"
    assert len(r.data["foreign_keys"]) > 0, "Track should have FKs to Album/Genre"


# --------------------------------------------------------------------------
# sandbox
# --------------------------------------------------------------------------


@pytest.mark.parametrize("code", [
    "import os", "__import__('os')", "open('/etc/passwd')",
    "eval('1+1')", "exec('x=1')", "import subprocess",
])
def test_blocked_imports(code):
    r = TOOLS["python_repl"](code=code)
    assert r.status == "error"
    assert r.error_category == "permission"


def test_repl_arithmetic_works():
    r = TOOLS["python_repl"](code="round(45000 / 3, 2)")
    assert r.status == "ok"
    assert r.data == 15000.0


# --------------------------------------------------------------------------
# tool specs
# --------------------------------------------------------------------------


def test_specs_match_the_registry():
    assert {s["name"] for s in TOOL_SPECS} == set(TOOLS)
    for s in TOOL_SPECS:
        assert s["description"] and s["args"] and s["example"]


# --------------------------------------------------------------------------
# scripted LLM
# --------------------------------------------------------------------------


def test_every_scenario_is_replayable():
    for name in SCENARIOS:
        llm = scripted(name)
        outs = [llm("prompt") for _ in range(len(SCENARIOS[name]))]
        assert outs == SCENARIOS[name]


def test_script_exhaustion_returns_final_answer():
    llm = ScriptedLLM(["Thought: one\nAction: get_schema\nInput: {}"])
    llm("p")
    assert "final_answer" in llm("p")         # never hangs


def test_reset():
    llm = scripted("clean_success")
    llm("p"); llm("p")
    llm.reset()
    assert llm.calls == 0


def test_scenarios_cover_the_failure_paths():
    """Don't delete these — the loop tests depend on them."""
    for required in ["recovers_from_bad_column", "malformed_then_ok",
                     "repeats_itself", "hallucinates_tool",
                     "keeps_failing", "never_finishes", "large_result"]:
        assert required in SCENARIOS


def test_scenario_sql_actually_runs_against_the_database():
    """Catches a success scenario rotting after a schema change."""
    import json
    import re

    for name in ["clean_success", "large_result"]:
        for line in SCENARIOS[name]:
            m = re.search(r"Action: (\w+)\nInput: (\{.*\})", line, re.S)
            if not m or m.group(1) not in TOOLS:
                continue
            r = TOOLS[m.group(1)](**json.loads(m.group(2)))
            assert r.status == "ok", f"{name}: {m.group(2)} -> {r.error}"
