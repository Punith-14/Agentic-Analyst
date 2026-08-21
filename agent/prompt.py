"""Builds the prompt sent to the model each step.

The model is stateless so the whole history goes out every time. That makes
this the main lever on context size and, more importantly on this hardware, on
prefill latency — it grows roughly 10x between a short prompt and a long one.

TODO(day7): observation masking — keep the last 3 observations, stub the rest.
"""

from __future__ import annotations

import json
from typing import Sequence

from contracts import TrajectoryStep

SYSTEM = """You are a data analyst agent. You answer questions about a SQL database.

You work in a loop. Each turn you write exactly one Thought, one Action, and one Input.
Then you STOP and wait — the system runs your action and gives you an Observation.

Format, exactly:

Thought: <one short sentence — what you are doing and why>
Action: <tool name>
Input: <a single JSON object>

Rules:
- ALWAYS begin with "Thought:". Never skip it.
- Emit ONE action per turn. Never write your own Observation.
- Keep the Thought to one sentence.
- Table and column names are case-sensitive. Use exactly what the schema says.
- If an Observation contains a HINT, read it and use it.
- NEVER invent a number. Only use final_answer when a tool has actually
  returned the value you are reporting. If you cannot get it, say so.
- As soon as an Observation contains the information the question asked for,
  use final_answer immediately. Do not re-run a query to double-check it.
- Never repeat a query you have already run. If it errored, change it. If it
  worked, use the result.
"""

# Qwen3B ignored the "always write a Thought" rule until it was shown this.
# Small models copy examples much more reliably than they follow instructions.
# Also fixed a case where it fabricated an answer after two failed queries.
EXAMPLE = """
Here is a complete example of one turn, in a different database:

Question: How many customers are there?

Thought: I do not know the table names yet, so I will list them.
Action: get_schema
Input: {}
Observation: 3 rows. {"tables": ["Users", "Orders", "Items"]}

Thought: Customers are probably in Users; I will check its columns.
Action: get_schema
Input: {"table": "Users"}
Observation: 4 rows. {"table": "Users", "columns": [{"name": "UserId"}, {"name": "Name"}]}

Thought: I can count the rows in Users now.
Action: run_sql
Input: {"query": "SELECT COUNT(*) AS n FROM Users"}
Observation: 1 rows. [{"n": 128}]

Thought: The tool returned 128, so that is the answer.
Action: final_answer
Input: {"answer": "There are 128 customers."}

Now solve the real question below. Follow exactly that format.
"""

FINAL_TOOL = {
    "name": "final_answer",
    "description": "Give the final answer and stop.",
    "args": {"answer": "string — the answer, in plain language"},
    "returns": "ends the run",
    "example": {"answer": "North, with 45,000 in sales"},
}


def render_tools(specs: Sequence[dict]) -> str:
    lines = ["Available actions:"]
    for s in list(specs) + [FINAL_TOOL]:
        args = ", ".join(f"{k} ({v})" for k, v in s["args"].items()) or "none"
        lines.append(f"\n- {s['name']}")
        lines.append(f"    {s['description']}")
        lines.append(f"    Input keys: {args}")
        lines.append(f"    Example Input: {json.dumps(s['example'])}")
    return "\n".join(lines)


def render_step(step: TrajectoryStep) -> str:
    out = [f"Thought: {step.thought}"]
    if step.action:
        out.append(f"Action: {step.action.tool}")
        out.append(f"Input: {json.dumps(step.action.args)}")
    if step.observation:
        out.append(f"Observation: {step.observation.short_observation()}")
    return "\n".join(out)


def render_history(steps: Sequence[TrajectoryStep]) -> str:
    if not steps:
        return ""
    return "\n\n".join(render_step(s) for s in steps)


def render_schema(schema: dict) -> str:
    """Compact full schema — table(col, col, ...) plus foreign keys."""
    lines = []
    for t, info in schema.items():
        cols = info["columns"] if isinstance(info, dict) else info
        lines.append(f"  {t}({', '.join(cols)})")
        for fk in (info.get("foreign_keys", []) if isinstance(info, dict) else []):
            lines.append(f"      FK {fk}")
    return "\n".join(lines)


def build_prompt(question: str,
                 tool_specs: Sequence[dict],
                 history: Sequence[TrajectoryStep],
                 table_names: Sequence[str] | None = None,
                 schema: dict | None = None) -> str:
    """Assemble the prompt.

    Originally sent table names only (~32 tokens vs ~500 for the full schema).
    Changed after the first batch: 76% of queries errored before any get_schema
    call and 65% after one, because joins need several tables and the agent only
    looked up one. Context turned out not to be the binding constraint, so the
    whole schema goes in.
    """
    parts = [SYSTEM, render_tools(tool_specs), EXAMPLE]

    if schema:
        parts.append("\nDatabase schema (names are case-sensitive):\n"
                     + render_schema(schema))
    elif table_names:
        parts.append(f"\nTables in this database: {', '.join(table_names)}")

    parts.append(f"\nQuestion: {question}")

    hist = render_history(history)
    if hist:
        parts.append("\n" + hist)

    # Don't end on "Thought:" — Qwen treated it as already satisfied and jumped
    # straight to Action:, so every thought came back empty.
    return "\n".join(parts) + "\n"


def estimate_tokens(text: str) -> int:
    """Rough, ~4 chars per token. Swap for tiktoken if we need exact numbers."""
    return len(text) // 4
