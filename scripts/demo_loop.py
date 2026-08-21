"""Run the loop against a scripted model, or the real one.

    python scripts/demo_loop.py                      # recovery scenario
    python scripts/demo_loop.py --list
    python scripts/demo_loop.py --all
    python scripts/demo_loop.py --real --raw         # needs Ollama
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from agent._stubs import TOOL_SPECS, TOOLS, set_db
from agent.llm import SCENARIOS, OllamaLLM, scripted
from agent.loop import AgentConfig, run_agent

QUESTIONS = {
    "clean_success": "How many albums are in the database?",
    "recovers_from_bad_column": "Show me five album titles.",
    "malformed_then_ok": "How many albums are there?",
    "repeats_itself": "Query a table that does not exist.",
    "hallucinates_tool": "How many albums are there?",
    "keeps_failing": "Select a column that does not exist.",
    "never_finishes": "Describe every table.",
    "large_result": "Show me all the tracks.",
}


def full_schema() -> dict:
    r = TOOLS["get_schema"]()
    return r.data if r.status == "ok" else {}


def run_one(scenario: str, verbose: bool = True):
    set_db("chinook_1")
    question = QUESTIONS.get(scenario, "How many albums are there?")
    record = run_agent(
        question=question,
        llm=scripted(scenario),
        tools=TOOLS,
        tool_specs=TOOL_SPECS,
        schema=full_schema(),
        task_id=scenario,
        config=AgentConfig(max_steps=5, verbose=verbose),
    )
    return record


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--scenario", default="recovers_from_bad_column")
    ap.add_argument("--list", action="store_true")
    ap.add_argument("--all", action="store_true")
    ap.add_argument("--real", action="store_true",
                    help="use the actual model via Ollama instead of a script")
    ap.add_argument("--raw", action="store_true",
                    help="dump the model's unparsed output for each step")
    args = ap.parse_args()

    if args.list:
        print("Scenarios:")
        for s in SCENARIOS:
            print(f"  {s:<28} {QUESTIONS.get(s, '')}")
        return

    if args.real:
        set_db("chinook_1")
        rec = run_agent(
            question="How many albums are in the database?",
            llm=OllamaLLM(),
            tools=TOOLS,
            tool_specs=TOOL_SPECS,
            schema=full_schema(),
            config=AgentConfig(max_steps=8, verbose=True),
        )
        if args.raw:
            print("\n" + "#" * 66)
            print("RAW MODEL OUTPUT (what the parser had to work with)")
            print("#" * 66)
            for s in rec.steps:
                print(f"\n--- step {s.step_index} "
                      f"(parsed thought: {s.thought!r}) ---")
                print(repr(s.raw_model_output))
        return

    if args.all:
        rows = []
        for s in SCENARIOS:
            rec = run_one(s, verbose=False)
            rows.append((s, rec.termination, len(rec.steps),
                         sum(1 for x in rec.steps if x.status == "error")))
        print(f"\n{'scenario':<28}{'terminated':<18}{'steps':>6}{'errors':>8}")
        print("-" * 60)
        for s, t, n, e in rows:
            print(f"{s:<28}{t:<18}{n:>6}{e:>8}")
        return

    run_one(args.scenario)


if __name__ == "__main__":
    main()
