"""Shared schemas for Project 23.

Everything that crosses a layer boundary is defined here. Don't change a field
without telling the group — A, C and D all build against this.
"""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from typing import Any, Literal, Optional

from pydantic import BaseModel, Field

ErrorCategory = Literal[
    "none",
    "syntax",
    "schema_missing_column",
    "schema_missing_table",
    "type_error",
    "timeout",
    "empty_result",
    "runtime",
    "permission",
    "unknown_tool",
    "invalid_args",
    "other",
]

Termination = Literal[
    "final_answer",
    "max_iterations",
    "token_budget",
    "repeated_action",
    "no_progress",
    "consecutive_errors",
    "parse_failure",
    "tool_crash",
    "critic_stop",       # not wired up until the critic lands
]

VALID_TERMINATIONS = {
    "final_answer", "max_iterations", "token_budget", "repeated_action",
    "no_progress", "consecutive_errors", "parse_failure", "tool_crash",
    "critic_stop",
}

StepStatus = Literal["continue", "final", "error", "terminated"]

MAX_ROWS_IN_DATA = 20
MAX_ERROR_CHARS = 200


class ToolResult(BaseModel):
    """What every tool returns. Tools don't raise — they return status="error"."""

    status: Literal["ok", "error"]
    tool: str

    data: Any = None                    # capped at MAX_ROWS_IN_DATA rows

    error: Optional[str] = None         # short, goes in the prompt
    error_full: Optional[str] = None    # traceback, log only
    error_category: ErrorCategory = "none"

    row_count: Optional[int] = None     # true count, pre-truncation
    truncated: bool = False

    duration_ms: int = 0

    artifact_id: Optional[str] = None   # handle for large results, e.g. "df_1"
    hint: Optional[str] = None          # recovery info, e.g. the real column names

    def __getitem__(self, item: str) -> Any:
        """Allow subscript access for backward compatibility."""
        return getattr(self, item)

    def short_observation(self) -> str:
        """What the model sees. Deliberately excludes error_full."""
        if self.status == "error":
            parts = [f"ERROR ({self.error_category}): {self.error}"]
            if self.hint:
                parts.append(f"HINT: {self.hint}")
            return " | ".join(parts)

        body = json.dumps(self.data, default=str)[:1500]
        head = ""
        if self.row_count is not None:
            head = f"{self.row_count} rows"
            if self.truncated:
                head += f" (showing first {MAX_ROWS_IN_DATA})"
            head += ". "
        return head + body


class Action(BaseModel):
    tool: str
    args: dict = Field(default_factory=dict)
    is_final: bool = False
    final_answer: Optional[str] = None

    def hash(self) -> str:
        """Stable hash for loop detection.

        Normalises whitespace and case first so two spellings of the same query
        collide.
        """
        norm = {k: " ".join(str(v).lower().split()) for k, v in sorted(self.args.items())}
        payload = self.tool + "|" + json.dumps(norm, sort_keys=True)
        return hashlib.sha1(payload.encode()).hexdigest()[:16]


class TrajectoryStep(BaseModel):
    """One iteration of the agent loop.

    C and D read these, and all four ML components train on them, so the shape
    is frozen once we start collecting.
    """

    run_id: str
    step_index: int
    timestamp: str = Field(default_factory=lambda: datetime.now(timezone.utc).isoformat())

    thought: str = ""
    action: Optional[Action] = None
    observation: Optional[ToolResult] = None
    raw_model_output: str = ""          # kept for debugging the parser

    status: StepStatus = "continue"
    duration_ms: int = 0

    is_retry: bool = False
    parent_step_index: Optional[int] = None
    action_hash: Optional[str] = None
    repeat_count: int = 0

    error_category: ErrorCategory = "none"
    consecutive_errors: int = 0
    total_errors_so_far: int = 0
    parse_repair_count: int = 0

    # critic / router features
    tokens_in_prompt: int = 0
    observation_truncated: bool = False
    observation_rows: Optional[int] = None
    sql_table_count: Optional[int] = None
    sql_join_count: Optional[int] = None
    schema_inspected_before: bool = False

    # filled in by the labelling pipeline after the run
    label_execution: Optional[Literal["ok", "error", "empty"]] = None
    label_step: Optional[float] = None      # 0.0 | 0.5 | 1.0, from our rules
    label_llm: Optional[float] = None       # 0.0 | 0.5 | 1.0, from the annotator
    label_rollout: Optional[float] = None   # subset only, expensive
    label_error_type: Optional[str] = None

    # denormalised so building the training set is a single pass
    run_final_correct: Optional[bool] = None
    run_total_steps: Optional[int] = None


class RunRecord(BaseModel):
    run_id: str
    task_id: str = ""
    question: str
    steps: list[TrajectoryStep] = Field(default_factory=list)

    termination: Termination
    final_answer: Optional[str] = None
    correct: Optional[bool] = None
    # how `correct` was decided — "execution" (result sets compared, the
    # standard Spider metric) or "answer_match" (values found in the prose).
    # Report the split; they are not equally strong.
    correct_method: Optional[Literal["execution", "answer_match", "none"]] = None
    predicted_sql: Optional[str] = None

    total_duration_ms: int = 0
    total_tokens: int = 0

    # provenance — without this we can't tell whether a result changed because
    # of the code or because someone nudged the temperature
    model_name: str = ""
    quantisation: str = ""
    temperature: float = 0.0
    context_policy: str = ""
    critic_version: Optional[str] = None


class Task(BaseModel):
    task_id: str
    question: str
    gold_sql: str
    difficulty: Literal["easy", "medium", "hard"] = "medium"
    db: str = "analytics"
    tags: list[str] = Field(default_factory=list)
    source: str = "spider"


def append_run(record: RunRecord, path: str) -> None:
    """Append one run. Append-only — these files are training data."""
    with open(path, "a", encoding="utf-8") as f:
        f.write(record.model_dump_json() + "\n")


def load_runs(path: str) -> list[RunRecord]:
    out: list[RunRecord] = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                out.append(RunRecord.model_validate_json(line))
    return out
