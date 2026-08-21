"""Logger, answer checker and batch generation."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from agent._stubs import TOOL_SPECS, TOOLS, set_db
from agent.checker import check_answer, run_gold
from agent.llm import scripted
from agent.logger import (TrajectoryLogger, done_keys, iter_runs, load_all,
                          summarise)
from agent.loop import AgentConfig, run_agent
from contracts import RunRecord


@pytest.fixture(autouse=True)
def use_chinook():
    set_db("chinook_1")


def make_run(scenario="clean_success", task_id="t001") -> RunRecord:
    return run_agent(
        question="How many albums are there?",
        llm=scripted(scenario),
        tools=TOOLS,
        tool_specs=TOOL_SPECS,
        table_names=["Album", "Artist", "Track"],
        task_id=task_id,
        config=AgentConfig(max_steps=5, verbose=False),
    )


# ==========================================================================
# logger
# ==========================================================================


def test_write_and_read_back(tmp_path):
    p = tmp_path / "t.jsonl"
    rec = make_run()
    with TrajectoryLogger(path=p) as log:
        log.write(rec)

    back = list(iter_runs(p))
    assert len(back) == 1
    assert back[0].run_id == rec.run_id
    assert len(back[0].steps) == len(rec.steps)
    assert back[0].steps[0].action.tool == rec.steps[0].action.tool


def test_appends_never_overwrites(tmp_path):
    p = tmp_path / "t.jsonl"
    with TrajectoryLogger(path=p) as log:
        log.write(make_run())
    with TrajectoryLogger(path=p) as log:       # reopened
        log.write(make_run())
    assert len(list(iter_runs(p))) == 2


def test_flushed_after_every_write(tmp_path):
    """An overnight job that dies at hour three must keep everything before it."""
    p = tmp_path / "t.jsonl"
    log = TrajectoryLogger(path=p)
    log.__enter__()
    log.write(make_run())
    assert len(list(iter_runs(p))) == 1, "not readable until close — data at risk"
    log.__exit__()


def test_malformed_line_is_skipped_not_fatal(tmp_path):
    p = tmp_path / "t.jsonl"
    with TrajectoryLogger(path=p) as log:
        log.write(make_run())
    with open(p, "a") as f:
        f.write('{"truncated": tru\n')          # simulates a hard kill
    with TrajectoryLogger(path=p) as log:
        log.write(make_run())

    assert len(list(iter_runs(p))) == 2


def test_resume_keys(tmp_path):
    p = tmp_path / "t.jsonl"
    with TrajectoryLogger(path=p) as log:
        log.write(make_run(task_id="t001"))
        log.write(make_run(task_id="t001"))
        log.write(make_run(task_id="t002"))

    keys = done_keys(p)
    assert ("t001", 0) in keys
    assert ("t001", 1) in keys
    assert ("t002", 0) in keys
    assert ("t002", 1) not in keys


def test_summary(tmp_path):
    p = tmp_path / "t.jsonl"
    with TrajectoryLogger(path=p) as log:
        for s in ["clean_success", "recovers_from_bad_column", "keeps_failing"]:
            log.write(make_run(s))

    s = summarise(list(iter_runs(p)))
    assert s["runs"] == 3
    assert s["steps"] > 0
    assert s["mean_steps"] > 0
    assert "final_answer" in s["termination"]
    assert s["error_steps"] > 0


def test_summary_handles_empty():
    assert summarise([])["runs"] == 0


# ==========================================================================
# checker
# ==========================================================================


def test_gold_sql_runs():
    rows, err = run_gold("SELECT COUNT(*) FROM Album", "chinook_1")
    assert err is None
    assert rows[0][0] == 347


def test_number_found_in_prose():
    ok, _ = check_answer("There are 347 albums in the database.",
                         "SELECT COUNT(*) FROM Album", "chinook_1")
    assert ok is True


def test_formatted_number_found():
    ok, _ = check_answer("The total is 3,503 tracks.",
                         "SELECT COUNT(*) FROM Track", "chinook_1")
    assert ok is True


def test_wrong_number_rejected():
    """The fabricated-answer case seen live — must not pass."""
    ok, _ = check_answer("There are 354 albums in the database.",
                         "SELECT COUNT(*) FROM Album", "chinook_1")
    assert ok is False


def test_missing_answer_rejected():
    for ans in [None, "", "   "]:
        ok, _ = check_answer(ans, "SELECT COUNT(*) FROM Album", "chinook_1")
        assert ok is False


def test_string_value_found():
    ok, _ = check_answer(
        "The artist is AC/DC.",
        "SELECT Name FROM Artist WHERE ArtistId = 1", "chinook_1")
    assert ok is True


def test_broken_gold_sql_returns_none_not_false():
    """A broken reference query is our bug — it must not be scored as the
    agent getting it wrong, or the solve rate is quietly understated."""
    ok, reason = check_answer("anything", "SELECT * FROM NoSuchTable", "chinook_1")
    assert ok is None
    assert "gold SQL failed" in reason


def test_multi_value_partial_match():
    gold = "SELECT Title FROM Album ORDER BY AlbumId LIMIT 3"
    rows, _ = run_gold(gold, "chinook_1")
    titles = [r[0] for r in rows]

    ok, _ = check_answer(", ".join(titles), gold, "chinook_1")
    assert ok is True

    ok, _ = check_answer("Something else entirely", gold, "chinook_1")
    assert ok is False


# ==========================================================================
# scoring a whole run
# ==========================================================================


def test_score_run_rejects_non_final_termination():
    from agent.checker import score_run
    rec = make_run("never_finishes")
    ok, reason, method = score_run(rec, "SELECT COUNT(*) FROM Album", "chinook_1")
    assert ok is False
    assert "terminated as" in reason
    assert method == "none"


def test_execution_accuracy_used_when_the_agent_ends_on_a_query():
    """Preferred over prose matching: it is the published Spider metric and a
    number can't match coincidentally."""
    from agent.checker import execution_match, last_successful_sql
    rec = make_run("clean_success")
    assert last_successful_sql(rec) is not None

    ok, why = execution_match("SELECT COUNT(*) FROM Album",
                              "SELECT COUNT(*) AS n FROM Album", "chinook_1")
    assert ok is True and "execution match" in why


def test_execution_ignores_row_order_unless_gold_orders():
    from agent.checker import execution_match
    ok, _ = execution_match("SELECT Title FROM Album LIMIT 5",
                            "SELECT Title FROM Album LIMIT 5", "chinook_1")
    assert ok is True

    ok, _ = execution_match("SELECT Name FROM Artist ORDER BY Name DESC LIMIT 3",
                            "SELECT Name FROM Artist ORDER BY Name ASC LIMIT 3",
                            "chinook_1")
    assert ok is False, "gold has ORDER BY, so order must match"


def test_execution_catches_a_wrong_but_valid_query():
    """The silent-error case — runs fine, answers the wrong question."""
    from agent.checker import execution_match
    ok, _ = execution_match("SELECT AVG(UnitPrice) FROM Track",
                            "SELECT SUM(UnitPrice) FROM Track", "chinook_1")
    assert ok is False


def test_broken_predicted_sql_is_wrong_not_skipped():
    from agent.checker import execution_match
    ok, why = execution_match("SELECT nope FROM Album",
                              "SELECT COUNT(*) FROM Album", "chinook_1")
    assert ok is False
    assert "predicted SQL failed" in why


def test_correct_flag_propagates_to_steps(tmp_path):
    """The critic reads run_final_correct off each step, so it has to be set
    on every one, not just the record."""
    rec = make_run()
    rec.correct = True
    for s in rec.steps:
        s.run_final_correct = True

    p = tmp_path / "t.jsonl"
    with TrajectoryLogger(path=p) as log:
        log.write(rec)

    back = list(iter_runs(p))[0]
    assert all(s.run_final_correct is True for s in back.steps)
