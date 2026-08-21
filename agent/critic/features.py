"""Turns trajectory steps into a numeric table for the critic.

LEAKAGE IS THE THING TO WATCH HERE. The critic has to score a step using only
what was knowable at that moment. Anything computed from later steps, or from
the run outcome, would make the model look brilliant in evaluation and useless
in the loop.

Rules:
  - no field derived from a future step
  - no field derived from run.correct
  - run_total_steps is EXCLUDED: at step 3 you don't know the run will be 7
    steps long, and it correlates strongly with failure

Every feature below is computable at the moment the step is taken.
"""

from __future__ import annotations

import re
from typing import Iterable

import pandas as pd

from contracts import RunRecord, TrajectoryStep

# Categorical values are mapped to integers rather than one-hot encoded —
# LightGBM handles ordinal categoricals natively and it keeps the table narrow.
ERROR_CATEGORIES = [
    "none", "syntax", "schema_missing_column", "schema_missing_table",
    "type_error", "timeout", "empty_result", "runtime", "permission",
    "unknown_tool", "invalid_args", "other",
]
ERROR_CODE = {c: i for i, c in enumerate(ERROR_CATEGORIES)}

TOOLS = ["run_sql", "get_schema", "python_repl", "final_answer", "other"]
TOOL_CODE = {t: i for i, t in enumerate(TOOLS)}

FEATURES = [
    # where we are
    "step_index",
    "tool",
    # error history so far
    "consecutive_errors",
    "total_errors_so_far",
    "error_category",
    "is_error",
    "parse_repair_count",
    # repetition
    "repeat_count",
    "is_retry",
    # what came back
    "observation_rows",
    "observation_truncated",
    "returned_nothing",
    # query shape
    "sql_length",
    "sql_joins",
    "sql_tables",
    "sql_subqueries",
    "sql_aggregates",
    "has_where",
    "has_group_by",
    # behaviour
    "schema_inspected_before",
    "thought_length",
    "tokens_in_prompt",
    # rates rather than raw counts — these transfer across run lengths
    "error_rate_so_far",
]

CATEGORICAL = ["tool", "error_category"]


def _sql_of(step: TrajectoryStep) -> str:
    if step.action and step.action.tool == "run_sql":
        return str(step.action.args.get("query", ""))
    return ""


def step_features(step: TrajectoryStep, prior_steps: list[TrajectoryStep]) -> dict:
    """Features for one step, given only the steps before it."""
    sql = _sql_of(step).lower()
    tool = step.action.tool if step.action else "other"

    n_prior = len(prior_steps)
    prior_errors = sum(1 for s in prior_steps if s.status == "error")

    return {
        "step_index": step.step_index,
        "tool": TOOL_CODE.get(tool, TOOL_CODE["other"]),

        "consecutive_errors": step.consecutive_errors,
        "total_errors_so_far": step.total_errors_so_far,
        "error_category": ERROR_CODE.get(step.error_category, ERROR_CODE["other"]),
        "is_error": int(step.status == "error"),
        "parse_repair_count": step.parse_repair_count,

        "repeat_count": step.repeat_count,
        "is_retry": int(step.is_retry),

        "observation_rows": step.observation_rows if step.observation_rows is not None else -1,
        "observation_truncated": int(step.observation_truncated),
        "returned_nothing": int((step.observation_rows or 0) == 0
                                and step.observation is not None),

        "sql_length": len(sql),
        "sql_joins": len(re.findall(r"\bjoin\b", sql)),
        "sql_tables": len(set(re.findall(r"\bfrom\s+([a-z_]\w*)", sql)
                              + re.findall(r"\bjoin\s+([a-z_]\w*)", sql))),
        "sql_subqueries": max(sql.count("select") - 1, 0),
        "sql_aggregates": len(re.findall(r"\b(count|sum|avg|min|max)\s*\(", sql)),
        "has_where": int("where" in sql),
        "has_group_by": int("group by" in sql),

        "schema_inspected_before": int(step.schema_inspected_before),
        "thought_length": len(step.thought),
        "tokens_in_prompt": step.tokens_in_prompt,

        "error_rate_so_far": prior_errors / n_prior if n_prior else 0.0,
    }


def run_to_rows(run: RunRecord) -> list[dict]:
    rows = []
    for i, step in enumerate(run.steps):
        if step.label_step is None:          # unscoreable, skip
            continue
        row = step_features(step, run.steps[:i])
        row["label"] = step.label_step
        row["y"] = int(step.label_step == 0.0)      # binary target: is it bad?
        row["run_id"] = run.run_id                  # for grouped splitting
        row["task_id"] = run.task_id
        row["db"] = run.steps[0].action.args.get("db", "") if run.steps and run.steps[0].action else ""
        rows.append(row)
    return rows


def build_frame(runs: Iterable[RunRecord]) -> pd.DataFrame:
    rows = [r for run in runs for r in run_to_rows(run)]
    df = pd.DataFrame(rows)
    for c in CATEGORICAL:
        df[c] = df[c].astype("category")
    return df


def feature_matrix(df: pd.DataFrame) -> pd.DataFrame:
    """Just the model inputs — drops labels and grouping keys."""
    return df[FEATURES]


def check_no_leakage(df: pd.DataFrame) -> None:
    """Fail loudly if an outcome column made it into the feature set.

    Cheap insurance: a leaked target produces a suspiciously good score that is
    hard to explain and easy to miss.
    """
    banned = {"label", "y", "run_id", "task_id", "run_final_correct",
              "run_total_steps", "correct"}
    leaked = banned & set(FEATURES)
    if leaked:
        raise ValueError(f"outcome columns present in FEATURES: {leaked}")

    missing = set(FEATURES) - set(df.columns)
    if missing:
        raise ValueError(f"missing feature columns: {missing}")
