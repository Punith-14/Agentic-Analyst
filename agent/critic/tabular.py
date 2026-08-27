"""The 25 features the LightGBM critic reads, in one place.

Lifted out of notebooks/02_critic_models.ipynb after notebook 05 needed to
score the saved model and had no way to rebuild its inputs. Two copies of a
feature function is the classic train/serve skew bug: they drift, nothing
errors, and the numbers quietly stop meaning the same thing.

LEAKAGE RULES, same as critic/features.py:
  - nothing derived from a later step
  - nothing derived from the run outcome
  - repeat_count is EXCLUDED. The loop guard terminates a run at three repeats,
    so a high repeat_count doesn't predict failure, it CAUSES the termination
    that becomes the failure. Dropping it costs 0.001 PR-AUC.
  - run_total_steps and termination are EXCLUDED: only knowable at the end.
"""

from __future__ import annotations

import pandas as pd
from pandas.api.types import CategoricalDtype

# Categories are pinned rather than inferred. If a test split happens to
# contain no timeout errors, `astype("category")` would build a shorter
# category list, every code would shift by one, and LightGBM would read
# "syntax" where the data said "runtime" — silently, with no error.
TOOLS = ["run_sql", "get_schema", "python_repl", "final_answer", "none"]
ERROR_CATEGORIES = [
    "none", "syntax", "schema_missing_column", "schema_missing_table",
    "type_error", "timeout", "empty_result", "runtime", "permission",
    "unknown_tool", "invalid_args", "other",
]
TOOL_DTYPE = CategoricalDtype(categories=TOOLS)
ERROR_DTYPE = CategoricalDtype(categories=ERROR_CATEGORIES)

CATEGORICAL = ["tool", "error_category"]

FEATURES = [
    "step_index",
    "consecutive_errors", "total_errors_so_far", "is_error",
    "error_rate_so_far", "parse_repair_count",
    "obs_rows", "returned_nothing", "obs_truncated",
    "sql_length", "sql_joins", "sql_subqueries", "sql_aggregates",
    "has_where", "has_group_by", "has_order_by", "has_distinct",
    "schema_inspected_before", "thought_length", "tokens_in_prompt",
    "duration_ms",
    "question_words", "question_has_how_many",
    "tool", "error_category",
]


def engineer(df: pd.DataFrame) -> pd.DataFrame:
    """labelled_steps.parquet -> the model's input matrix."""
    f = pd.DataFrame(index=df.index)

    # position
    f["step_index"] = df.step_index

    # error history as at this step
    f["consecutive_errors"] = df.consecutive_errors
    f["total_errors_so_far"] = df.total_errors_so_far
    f["is_error"] = (df.status == "error").astype(int)
    f["error_rate_so_far"] = df.total_errors_so_far / (df.step_index + 1)
    f["parse_repair_count"] = df.parse_repair_count

    # what came back. -1 is a deliberate sentinel for "not a query", distinct
    # from 0 meaning "a query that returned nothing" — the two say different
    # things about the step and the tree can split on them separately.
    # to_numeric first: obs_rows arrives as object dtype when it holds Nones,
    # and .fillna() on object dtype is deprecated in pandas 2.x.
    obs_rows = pd.to_numeric(df.obs_rows, errors="coerce")
    f["obs_rows"] = obs_rows.fillna(-1)
    f["returned_nothing"] = ((obs_rows == 0) & obs_rows.notna()).astype(int)
    f["obs_truncated"] = df.obs_truncated.fillna(False).astype(bool).astype(int)

    # query shape — describes the SQL, never reads it. This is the gap the
    # text critic is meant to close.
    sql = df.sql.fillna("").str.lower()
    f["sql_length"] = sql.str.len()
    f["sql_joins"] = sql.str.count(r"\bjoin\b")
    f["sql_subqueries"] = (sql.str.count("select") - 1).clip(lower=0)
    f["sql_aggregates"] = sql.str.count(r"\b(count|sum|avg|min|max)\s*\(")
    f["has_where"] = sql.str.contains("where").astype(int)
    f["has_group_by"] = sql.str.contains("group by").astype(int)
    f["has_order_by"] = sql.str.contains("order by").astype(int)
    f["has_distinct"] = sql.str.contains("distinct").astype(int)

    # behaviour
    f["schema_inspected_before"] = df.schema_inspected_before.astype(int)
    f["thought_length"] = df.thought.fillna("").str.len()
    f["tokens_in_prompt"] = df.tokens_in_prompt
    f["duration_ms"] = df.duration_ms

    # question shape — constant within a run, varies across runs
    q = df.question.fillna("")
    f["question_words"] = q.str.split().str.len()
    f["question_has_how_many"] = q.str.lower().str.contains("how many").astype(int)

    # categoricals, with pinned categories
    f["tool"] = df.tool.fillna("none").astype(TOOL_DTYPE)
    f["error_category"] = df.error_category.fillna("none").astype(ERROR_DTYPE)

    return f[FEATURES]
