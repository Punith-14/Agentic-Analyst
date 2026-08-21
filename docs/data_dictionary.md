# Data dictionary — `labelled_steps.parquet`

One row per agent step. 5,461 rows, 33 columns, produced by
`scripts/label_dataset.py`.

Each column is marked:

- **KEY** — identifier, never a feature
- **TARGET** — what we predict, never a feature
- **FEATURE** — safe to use as model input
- **TEXT** — raw text, for the ModernBERT critic later
- **META** — provenance and diagnostics, not a feature
- **LEAK** — looks usable but must not be used, with the reason

---

## Identifiers

| Column | Type | Contains | Use |
|---|---|---|---|
| `run_id` | str | Unique id for one attempt at one question, e.g. `r-4a2f91c3`. All steps of a run share it. | **KEY** — used to group the train/test split |
| `task_id` | str | Which question, e.g. `t047`. Maps to `data/tasks/task_suite.json`. | **KEY** |
| `step_index` | int | Position in the run, starting at 0. | **FEATURE** |

`run_id` is what makes the split honest — every step of a run goes to the same
side, otherwise the model sees near-duplicate rows across the boundary.

---

## Targets

| Column | Type | Contains | Use |
|---|---|---|---|
| `label_step` | float | 0.0 bad · 0.5 exploratory · 1.0 good. From our rules in `agent/labeller.py`. `None` if the run was unscoreable. | **TARGET** — but see the warning below |
| `y_run_fails` | int | 1 if this run ended with a wrong answer, 0 if correct, `None` if unscoreable. From comparing against gold SQL. | **TARGET** — the one we use |
| `run_correct` | bool | Same information as `y_run_fails`, inverted. | **TARGET** |
| `label_execution` | str | `ok` / `error` / `empty` — what the tool did. | **TARGET-ish**, also a diagnostic |

### Why `y_run_fails` and not `label_step`

`label_step` is assigned by rules that read the step's own fields:

```python
if step.repeat_count >= 2:  return BAD
if run failed and step errored:  return BAD
```

Measured: **93% of `label_step` is reproducible from fields the model would also
receive**. Training on it mostly relearns the rule, and the score is meaningless.

`y_run_fails` comes from running the gold SQL and comparing result sets. Nothing
in the step features can produce it.

---

## What the agent did — text

| Column | Type | Contains | Use |
|---|---|---|---|
| `question` | str | The user's question in natural language. Identical across all steps of a run. | **TEXT** |
| `thought` | str | The model's stated reasoning for this step. | **TEXT** — length is a FEATURE |
| `sql` | str | The SQL query, if this step called `run_sql`. Empty otherwise. | **TEXT** — derived shape features below |
| `args` | str | The full action arguments as a string, for any tool. | **TEXT** |
| `raw_model_output` | str | The model's unparsed reply. Kept for debugging the parser. | **META** |
| `tool` | str | Which action was taken: `run_sql`, `get_schema`, `python_repl`, `final_answer`, or `None` on a parse failure. | **FEATURE** (categorical) |
| `status` | str | `continue`, `error`, `final`, or `terminated`. | **FEATURE** (as `is_error`) |

---

## What came back

| Column | Type | Contains | Use |
|---|---|---|---|
| `obs_status` | str | `ok` or `error` from the tool. `None` if no tool ran. | **FEATURE** |
| `obs_rows` | float | True row count *before* truncation. `None` if not a query. | **FEATURE** |
| `obs_truncated` | bool | Whether the tool cut the result to 20 rows. | **FEATURE** |
| `obs_error` | str | Short error message. `None` on success. | **TEXT** |
| `error_category` | str | Structured error type: `none`, `schema_missing_column`, `syntax`, `empty_result`, `runtime`, `timeout`, `permission`, `unknown_tool`, `invalid_args`, `other`. | **FEATURE** (categorical) |

`obs_rows` is genuinely informative: a query returning 0 rows often means a
filter is wrong, and one returning 3,000 often means the aggregation is missing.

---

## Counters, as at this step

These were computed by the loop *at the moment the step ran*, so they contain no
information from later steps.

| Column | Type | Contains | Use |
|---|---|---|---|
| `consecutive_errors` | int | Errors in an unbroken sequence up to here. Resets on success. | **FEATURE** |
| `total_errors_so_far` | int | All errors in this run up to and including this step. | **FEATURE** |
| `parse_repair_count` | int | Times the parser had to ask the model to fix its output. | **FEATURE** |
| `repeat_count` | int | How many times this exact action (tool + normalised args) has been issued in this run. | **LEAK** — see below |
| `is_retry` | bool | Whether this step retries an earlier one. | **FEATURE** |
| `schema_inspected_before` | bool | Whether `get_schema` had succeeded *before* this step. False on the step that does the inspecting. | **FEATURE** |
| `tokens_in_prompt` | int | Prompt size at this step. Grows through the run. | **FEATURE** |
| `duration_ms` | int | Wall-clock time for this step. | **FEATURE** |

### Why `repeat_count` is excluded

The loop guard **terminates a run** when an action repeats three times. So a
high `repeat_count` doesn't predict failure — it *causes* the termination that
becomes the failure. Self-fulfilling.

Dropping it costs 0.001 PR-AUC, so there's no reason to keep it.

`schema_inspected_before` is worth noting: it describes state *before* the step,
so the `get_schema` step itself records `False`. Recording `True` would leak the
step's own outcome into its features.

---

## Run context

| Column | Type | Contains | Use |
|---|---|---|---|
| `termination` | str | How the run ended: `final_answer`, `repeated_action`, `parse_failure`, `consecutive_errors`, `max_iterations`, `token_budget`, `no_progress`, `tool_crash`, `critic_stop`. | **LEAK** — known only at the end |
| `run_total_steps` | int | How many steps the run had in total. | **LEAK** — at step 3 you don't know the run will last 7 |
| `correct_method` | str | How correctness was decided: `execution` (result sets compared, the Spider metric), `answer_match` (values found in the prose), or `none`. | **META** |

`termination` and `run_total_steps` are the two most tempting leaks in the
table. Both are only knowable after the run is over, and both correlate strongly
with failure — which is exactly what makes them dangerous.

---

## Provenance

Recorded so a change in results can be attributed to a cause rather than
guessed at.

| Column | Type | Contains | Use |
|---|---|---|---|
| `model_name` | str | `qwen2.5-coder:3b` | **META** — filter before training |
| `temperature` | float | 0.7 | **META** |
| `context_policy` | str | Agent configuration: schema mode, guard settings, step limit. | **META** — filter before training |

**Always check `context_policy.nunique() == 1` before training.** Runs from
different agent configurations have different failure distributions; mixing them
teaches the critic patterns that no longer occur.

---

## Summary

| Group | Columns | Model input? |
|---|---|---|
| Identifiers | 3 | only for grouping |
| Targets | 4 | no |
| Text | 5 | ModernBERT later; derived lengths now |
| Observation | 5 | yes |
| Counters | 8 | yes, except `repeat_count` |
| Run context | 3 | **no — leakage** |
| Provenance | 3 | no, but filter on them |

**25 engineered features** are built from these in the notebook. Three columns
are deliberately excluded despite looking useful: `repeat_count`,
`run_total_steps` and `termination`.
