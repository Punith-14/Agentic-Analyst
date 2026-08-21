# contracts/__init__.py
from pydantic import BaseModel, Field
from typing import Any, Literal, Optional, List, Dict

ErrorCategory = Literal[
    "none", "syntax", "schema_missing_column", "schema_missing_table",
    "type_error", "timeout", "empty_result", "runtime",
    "permission", "unknown_tool", "invalid_args", "other",
]

class ToolResult(BaseModel):
    status: Literal["ok", "error"]
    tool: str  # tool name that produced this
    data: Any = None  # serialisable. Max 20 rows.
    error: Optional[str] = None  # SHORT message for the prompt (max ~200 chars)
    error_full: Optional[str] = None  # full traceback. LOG ONLY, never prompt.
    error_category: ErrorCategory = "none"
    row_count: Optional[int] = None  # true row count BEFORE truncation
    truncated: bool = False
    duration_ms: int = 0
    artifact_id: Optional[str] = None  # handle for large results, e.g. "df_1"
    hint: Optional[str] = None  # actionable recovery info, e.g. "available columns: id, region, sales"

    def __getitem__(self, item: str) -> Any:
        """Allow subscript access for backward compatibility."""
        return getattr(self, item)

class Action(BaseModel):
    tool: str  # must exist in TOOLS registry
    args: dict = Field(default_factory=dict)  # keyword arguments for the tool
    is_final: bool = False  # True when the agent is answering
    final_answer: Optional[str] = None

class TrajectoryStep(BaseModel):
    # identity
    run_id: str
    step_index: int
    timestamp: str  # ISO 8601
    # the step itself
    thought: str
    action: Optional[Action] = None
    observation: Optional[ToolResult] = None
    raw_model_output: str = ""  # unparsed text, for debugging the parser
    # control
    status: Literal["continue", "final", "error", "terminated"] = "continue"
    duration_ms: int = 0
    # retry / loop structure
    is_retry: bool = False
    parent_step_index: Optional[int] = None
    action_hash: Optional[str] = None  # hash(tool + normalised args)
    repeat_count: int = 0  # times this action_hash was seen
    # error tracking
    error_category: ErrorCategory = "none"
    consecutive_errors: int = 0
    total_errors_so_far: int = 0
    parse_repair_count: int = 0
    # cheap numeric features (for the critic and the router)
    tokens_in_prompt: int = 0
    observation_truncated: bool = False
    observation_rows: Optional[int] = None
    sql_table_count: Optional[int] = None
    sql_join_count: Optional[int] = None
    schema_inspected_before: bool = False
    # labels (filled AFTER the run, by the labelling pipeline)
    label_execution: Optional[Literal["ok", "error", "empty"]] = None
    label_llm: Optional[float] = None  # ternary: 0.0 | 0.5 | 1.0
    label_rollout: Optional[float] = None  # 0.0 - 1.0, subset only
    label_error_type: Optional[str] = None  # from the shared error taxonomy
    # run-level outcome, denormalised onto every step
    run_final_correct: Optional[bool] = None
    run_total_steps: Optional[int] = None

Termination = Literal[
    "final_answer",        # succeeded
    "max_iterations",      # ran out of steps
    "token_budget",        # ran out of context window
    "repeated_action",     # loop detected
    "no_progress",         # state unchanged for k steps
    "consecutive_errors",  # 3 failures in a row
    "parse_failure",       # could not extract an action
    "tool_crash",          # unrecoverable tool error
    "critic_stop",         # RESERVED - critic ended the run early
]

VALID_TERMINATIONS: list[Termination] = [
    "final_answer", "max_iterations", "token_budget", "repeated_action",
    "no_progress", "consecutive_errors", "parse_failure", "tool_crash", "critic_stop"
]

class RunRecord(BaseModel):
    run_id: str
    task_id: str  # links to task_suite.json
    question: str
    steps: list[TrajectoryStep] = Field(default_factory=list)
    termination: Termination
    final_answer: Optional[str] = None
    correct: Optional[bool] = None  # vs gold answer
    total_duration_ms: int = 0
    total_tokens: int = 0
    # provenance - REQUIRED. Without these we cannot explain result changes.
    model_name: str = "Qwen2.5-Coder-7B-Instruct"
    quantisation: str = "Q4_K_M"
    temperature: float = 0.0
    context_policy: str = "mask_last_3"
    critic_version: Optional[str] = None
