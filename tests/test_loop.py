"""Parser, prompt builder and the ReAct loop."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from agent._stubs import TOOL_SPECS, TOOLS, set_db
from agent.llm import ScriptedLLM, scripted
from agent.loop import AgentConfig, run_agent
from agent.parser import parse_action
from agent.prompt import build_prompt, estimate_tokens
from contracts import RunRecord, TrajectoryStep


@pytest.fixture(autouse=True)
def use_chinook():
    set_db("chinook_1")


def go(scenario: str, max_steps: int = 5) -> RunRecord:
    return run_agent(
        question="How many albums are there?",
        llm=scripted(scenario),
        tools=TOOLS,
        tool_specs=TOOL_SPECS,
        table_names=["Album", "Artist", "Track"],
        config=AgentConfig(max_steps=max_steps, verbose=False),
    )


# ==========================================================================
# parser
# ==========================================================================


def test_parses_the_expected_format():
    r = parse_action('Thought: I need the schema.\n'
                     'Action: get_schema\n'
                     'Input: {"table": "Album"}')
    assert r.ok
    assert r.thought == "I need the schema."
    assert r.action.tool == "get_schema"
    assert r.action.args == {"table": "Album"}


def test_thought_without_the_label():
    """Qwen dropped the label entirely at one point and every thought came
    back empty. Handle both forms."""
    r = parse_action(' I need to count the albums first.\n'
                     'Action: run_sql\n'
                     'Input: {"query": "SELECT count(*) FROM Album"}')
    assert r.ok
    assert r.thought == "I need to count the albums first."


def test_thought_label_still_works():
    r = parse_action('Thought: explicit label\nAction: get_schema\nInput: {}')
    assert r.thought == "explicit label"


def test_multiline_unlabelled_thought():
    r = parse_action('The question asks for a count.\n'
                     'I should query the Album table.\n'
                     'Action: run_sql\nInput: {"query": "SELECT 1"}')
    assert r.ok
    assert "count" in r.thought and "Album" in r.thought


def test_parses_empty_input():
    r = parse_action("Thought: List tables.\nAction: get_schema\nInput: {}")
    assert r.ok and r.action.args == {}


def test_strips_markdown_fences():
    r = parse_action('Thought: go\nAction: run_sql\n'
                     'Input: ```json\n{"query": "SELECT 1"}\n```')
    assert r.ok and r.action.args["query"] == "SELECT 1"


def test_ignores_prose_before_the_json():
    r = parse_action('Thought: hmm {not json}\nAction: run_sql\n'
                     'Input: Here you go: {"query": "SELECT 1"}')
    assert r.ok and r.action.args["query"] == "SELECT 1"


def test_handles_multiline_sql():
    sql = "SELECT a,\n b\nFROM Album\nWHERE x = 1"
    r = parse_action('Thought: q\nAction: run_sql\n'
                     'Input: {"query": ' + repr(sql).replace("'", '"') + '}')
    assert r.ok
    assert "FROM Album" in r.action.args["query"]


def test_nested_json_args():
    r = parse_action('Thought: t\nAction: make_chart\n'
                     'Input: {"spec": {"kind": "bar", "x": "region"}}')
    assert r.ok and r.action.args["spec"]["kind"] == "bar"


def test_bare_json_object_form():
    r = parse_action('{"tool": "get_schema", "args": {"table": "Album"}}')
    assert r.ok and r.action.tool == "get_schema"


def test_final_answer_is_flagged():
    r = parse_action('Thought: done\nAction: final_answer\n'
                     'Input: {"answer": "347 albums"}')
    assert r.ok and r.action.is_final
    assert r.action.final_answer == "347 albums"


def test_prose_only_fails_cleanly():
    r = parse_action("I think we should look at the albums table first.")
    assert not r.ok
    assert r.error and "no Action" in r.error


def test_empty_output_fails_cleanly():
    assert not parse_action("").ok
    assert not parse_action("   \n  ").ok


def test_broken_json_reports_why():
    r = parse_action('Thought: t\nAction: run_sql\nInput: {"query": "SELECT 1"')
    assert not r.ok
    assert "JSON" in r.error


def test_parser_never_raises():
    for junk in ["", "}{", "Action:", "Action: x\nInput:", "\x00\x01",
                 "Thought: " + "a" * 10000, '{"tool":}', "Action: 123"]:
        parse_action(junk)          # must not raise


# ==========================================================================
# prompt
# ==========================================================================


def test_prompt_contains_the_essentials():
    p = build_prompt("How many albums?", TOOL_SPECS, [], ["Album", "Track"])
    assert "How many albums?" in p
    assert "run_sql" in p
    assert "final_answer" in p
    assert "Album" in p
    # Must NOT end with "Thought:" — priming that label made Qwen3B treat it as
    # already satisfied and jump straight to "Action:", losing every thought.
    assert not p.rstrip().endswith("Thought:")
    assert p.rstrip().endswith("How many albums?")


def test_prompt_includes_a_worked_example():
    """Dropping the example regressed thought capture on Qwen3B."""
    p = build_prompt("q", TOOL_SPECS, [], ["Album"])
    assert "Thought:" in p
    assert "Observation:" in p
    assert p.count("Action:") >= 3, "the example should show several turns"


def test_prompt_grows_with_history():
    empty = build_prompt("q", TOOL_SPECS, [], ["Album"])
    rec = go("clean_success")
    withhist = build_prompt("q", TOOL_SPECS, rec.steps, ["Album"])
    assert estimate_tokens(withhist) > estimate_tokens(empty)


def test_prompt_never_leaks_the_traceback():
    """Tracebacks are logged, not prompted."""
    rec = go("keeps_failing")
    p = build_prompt("q", TOOL_SPECS, rec.steps, ["Album"])
    for s in rec.steps:
        if s.observation and s.observation.error_full:
            assert s.observation.error_full not in p


# ==========================================================================
# the loop
# ==========================================================================


def test_clean_run_reaches_final_answer():
    rec = go("clean_success")
    assert rec.termination == "final_answer"
    assert rec.final_answer
    assert rec.steps[-1].status == "final"


def test_error_does_not_stop_the_loop():
    """Tool errors are observations, not exceptions."""
    rec = go("recovers_from_bad_column")
    assert rec.termination == "final_answer"
    assert any(s.status == "error" for s in rec.steps), "should have hit an error"
    assert len(rec.steps) > 2, "should have continued past the error"


def test_the_hint_is_shown_to_the_model():
    rec = go("recovers_from_bad_column")
    failed = next(s for s in rec.steps if s.status == "error")
    assert failed.observation.hint
    assert "title" in failed.observation.short_observation().lower()


def test_unknown_tool_becomes_an_observation():
    rec = go("hallucinates_tool")
    bad = next(s for s in rec.steps if s.error_category == "unknown_tool")
    assert bad.observation.status == "error"
    assert "run_sql" in bad.observation.hint
    assert rec.termination == "final_answer", "should recover afterwards"


def test_max_steps_is_enforced():
    rec = go("never_finishes", max_steps=5)
    assert rec.termination == "max_iterations"
    assert len(rec.steps) == 5


def test_repeated_action_stops_the_run():
    """First batch: 58/80 runs repeated an identical action, 213 wasted steps."""
    rec = go("repeats_itself", max_steps=10)
    assert rec.termination == "repeated_action"
    assert rec.steps[-1].repeat_count > 2


def test_consecutive_errors_stops_the_run():
    from agent.llm import ScriptedLLM
    llm = ScriptedLLM([
        f'Thought: t{i}\nAction: run_sql\nInput: {{"query": "SELECT bad{i} FROM Album"}}'
        for i in range(8)
    ])
    rec = run_agent("q", llm, TOOLS, TOOL_SPECS,
                    config=AgentConfig(max_steps=10, verbose=False))
    assert rec.termination == "consecutive_errors"


def test_full_schema_goes_into_the_prompt():
    """get_schema() now returns every table with columns — 65% of queries were
    still erroring after a single-table lookup."""
    r = TOOLS["get_schema"]()
    assert r.status == "ok"
    assert "Album" in r.data
    assert "Title" in r.data["Album"]["columns"]
    assert "Track" in r.data

    p = build_prompt("q", TOOL_SPECS, [], schema=r.data)
    assert "Album(" in p
    assert "Title" in p
    assert "case-sensitive" in p


def test_parse_failure_is_recorded_not_raised():
    llm = ScriptedLLM(["just some prose with no action in it at all"])
    rec = run_agent("q", llm, TOOLS, TOOL_SPECS, ["Album"],
                    config=AgentConfig(max_steps=3, verbose=False))
    assert rec.termination == "parse_failure"
    assert rec.steps[0].raw_model_output


def test_truncation_is_flagged():
    rec = go("large_result")
    big = next(s for s in rec.steps if s.observation and s.observation.truncated)
    assert big.observation.row_count > 20
    assert len(big.observation.data) <= 20
    assert big.observation_truncated is True


def test_a_raising_tool_does_not_kill_the_run():
    def explodes(**kwargs):
        raise RuntimeError("layer A bug")

    llm = ScriptedLLM([
        'Thought: t\nAction: boom\nInput: {}',
        'Thought: recovered\nAction: final_answer\nInput: {"answer": "ok"}',
    ])
    rec = run_agent("q", llm, {**TOOLS, "boom": explodes},
                    TOOL_SPECS + [{"name": "boom", "description": "d",
                                   "args": {}, "example": {}}],
                    ["Album"], config=AgentConfig(max_steps=3, verbose=False))
    assert rec.termination == "final_answer"
    assert rec.steps[0].observation.status == "error"


def test_wrong_kwargs_become_an_observation():
    llm = ScriptedLLM([
        'Thought: t\nAction: get_schema\nInput: {"wrong_arg": 1}',
        'Thought: fixed\nAction: final_answer\nInput: {"answer": "ok"}',
    ])
    rec = run_agent("q", llm, TOOLS, TOOL_SPECS, ["Album"],
                    config=AgentConfig(max_steps=3, verbose=False))
    assert rec.steps[0].observation.error_category == "invalid_args"
    assert rec.termination == "final_answer"


# ==========================================================================
# what the loop records
# ==========================================================================


def test_every_step_is_a_valid_trajectory_step():
    for scenario in ["clean_success", "recovers_from_bad_column",
                     "keeps_failing", "large_result", "hallucinates_tool"]:
        rec = go(scenario)
        for i, s in enumerate(rec.steps):
            assert isinstance(s, TrajectoryStep)
            assert s.step_index == i
            assert s.run_id == rec.run_id
            assert s.tokens_in_prompt > 0
            assert s.duration_ms >= 0
            assert s.run_total_steps == len(rec.steps)


def test_error_counters_accumulate():
    rec = go("keeps_failing")
    errs = [s for s in rec.steps if s.status == "error"]
    assert len(errs) >= 3
    assert errs[-1].consecutive_errors >= 3
    assert errs[-1].total_errors_so_far >= 3


def test_counter_resets_after_a_success():
    rec = go("recovers_from_bad_column")
    assert rec.steps[0].consecutive_errors == 1
    assert rec.steps[-1].consecutive_errors == 0


def test_schema_inspection_is_tracked():
    """Flag describes state before the step. If the get_schema step recorded
    True it would leak its own outcome into the feature."""
    rec = go("clean_success")
    assert rec.steps[0].schema_inspected_before is False   # the get_schema step
    assert rec.steps[1].schema_inspected_before is True    # everything after
    assert rec.steps[-1].schema_inspected_before is True


def test_action_hash_normalises_whitespace():
    """Same query four times, one with different spacing. Day 7 dedup needs
    these to collide."""
    rec = go("repeats_itself", max_steps=10)
    sql_hashes = [s.action_hash for s in rec.steps
                  if s.action and s.action.tool == "run_sql"]
    # the guard now stops the run at the third repeat
    assert len(sql_hashes) == 3
    assert len(set(sql_hashes)) == 1, "whitespace-only differences must hash the same"


def test_provenance_is_recorded():
    rec = go("clean_success")
    assert rec.model_name == "scripted-fake"
    assert rec.total_tokens > 0
    assert rec.total_duration_ms >= 0


def test_context_policy_encodes_the_actual_config():
    """Not just the constant "full_history".

    We were filtering training runs on this string and it silently matched
    everything, because two genuinely different agent configurations both
    recorded the same value. The guard settings and step limit have to be in
    there or the filter can't discriminate.
    """
    rec = go("clean_success")
    assert rec.context_policy.startswith("full_history")
    for part in ("schema=", "guards=", "max_steps="):
        assert part in rec.context_policy, f"{part} missing from {rec.context_policy!r}"


def test_record_serialises_and_round_trips():
    rec = go("recovers_from_bad_column")
    restored = RunRecord.model_validate_json(rec.model_dump_json())
    assert restored.run_id == rec.run_id
    assert len(restored.steps) == len(rec.steps)
