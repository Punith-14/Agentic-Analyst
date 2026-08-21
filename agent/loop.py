"""The ReAct loop.

Tool errors come back as observations, not exceptions — the model needs to see
what went wrong to fix it.

Done: JSONL logging (agent/logger.py), action dedup and the no-progress guard,
token accounting.

TODO: parse repair pass — 104 of 1,200 runs still terminate as parse_failure,
      which is the single largest remaining source of lost runs.
TODO: observation masking, so long result sets stop inflating the prompt.
TODO: critic_stop — call the trained critic each step and terminate early.
      The termination value already exists in contracts.py, unused.
"""

from __future__ import annotations

import time
import uuid
from dataclasses import dataclass, field
from typing import Callable, Sequence

from agent.parser import parse_action
from agent.prompt import build_prompt, estimate_tokens
from contracts import Action, RunRecord, Termination, ToolResult, TrajectoryStep


@dataclass
class AgentConfig:
    max_steps: int = 12         # 8 was too few: 54/80 runs never reached an answer
    max_tokens: int = 150       # long thoughts get resent every step
    verbose: bool = True

    # Loop guards. Without these, 58/80 runs in the first batch repeated an
    # identical action — 213 wasted steps out of 526.
    max_repeats: int = 2        # same action_hash this many times -> stop
    max_consecutive_errors: int = 4


def _unknown_tool(tool: str, available: Sequence[str]) -> ToolResult:
    return ToolResult(
        status="error",
        tool=tool,
        error=f"no such action: {tool}",
        error_category="unknown_tool",
        hint=f"available actions: {', '.join(list(available) + ['final_answer'])}",
        duration_ms=0,
    )


def _bad_args(tool: str, exc: Exception) -> ToolResult:
    return ToolResult(
        status="error",
        tool=tool,
        error=f"bad arguments: {exc}"[:200],
        error_full=repr(exc),
        error_category="invalid_args",
        duration_ms=0,
    )


@dataclass
class RunResult:
    record: RunRecord
    steps: list[TrajectoryStep] = field(default_factory=list)


def run_agent(question: str,
              llm: Callable[..., str],
              tools: dict[str, Callable[..., ToolResult]],
              tool_specs: Sequence[dict],
              table_names: Sequence[str] | None = None,
              schema: dict | None = None,
              task_id: str = "",
              config: AgentConfig | None = None) -> RunRecord:
    """Run one question to completion."""
    cfg = config or AgentConfig()
    run_id = f"r-{uuid.uuid4().hex[:8]}"
    steps: list[TrajectoryStep] = []
    t_start = time.perf_counter()

    total_tokens = 0
    consecutive_errors = 0
    total_errors = 0
    schema_seen = False
    final_answer: str | None = None
    termination: Termination = "max_iterations"
    seen_actions: dict[str, int] = {}

    if cfg.verbose:
        print(f"\n{'=' * 66}\nQ: {question}\n{'=' * 66}")

    for i in range(cfg.max_steps):
        t_step = time.perf_counter()

        prompt = build_prompt(question, tool_specs, steps, table_names, schema)
        n_prompt = estimate_tokens(prompt)
        total_tokens += n_prompt

        raw = llm(prompt, max_tokens=cfg.max_tokens)
        parsed = parse_action(raw)

        if not parsed.ok:
            if cfg.verbose:
                print(f"\n[{i}] PARSE FAILED — {parsed.error}")
                print(f"    raw: {raw[:120]!r}")
            steps.append(TrajectoryStep(
                run_id=run_id, step_index=i, thought=parsed.thought,
                action=None, observation=None, raw_model_output=raw,
                status="error", error_category="other",
                consecutive_errors=consecutive_errors + 1,
                total_errors_so_far=total_errors + 1,
                tokens_in_prompt=n_prompt,
                duration_ms=int((time.perf_counter() - t_step) * 1000),
                schema_inspected_before=schema_seen,
            ))
            termination = "parse_failure"
            break

        action: Action = parsed.action

        if cfg.verbose:
            print(f"\n[{i}] Thought: {parsed.thought}")
            print(f"    Action:  {action.tool}({action.args})")

        if action.is_final:
            final_answer = action.final_answer
            steps.append(TrajectoryStep(
                run_id=run_id, step_index=i, thought=parsed.thought,
                action=action, observation=None, raw_model_output=raw,
                status="final", tokens_in_prompt=n_prompt,
                action_hash=action.hash(),
                consecutive_errors=consecutive_errors,
                total_errors_so_far=total_errors,
                schema_inspected_before=schema_seen,
                duration_ms=int((time.perf_counter() - t_step) * 1000),
            ))
            termination = "final_answer"
            if cfg.verbose:
                print(f"    ANSWER:  {final_answer}")
            break

        # --- loop detection ------------------------------------------
        # Counted before the tool runs: re-issuing the same call is pointless
        # whether it previously errored or succeeded. In the first batch the
        # agent re-ran queries it had already answered correctly.
        h = action.hash()
        seen_actions[h] = seen_actions.get(h, 0) + 1
        if seen_actions[h] > cfg.max_repeats:
            steps.append(TrajectoryStep(
                run_id=run_id, step_index=i, thought=parsed.thought,
                action=action, observation=None, raw_model_output=raw,
                status="terminated", action_hash=h,
                repeat_count=seen_actions[h],
                consecutive_errors=consecutive_errors,
                total_errors_so_far=total_errors,
                tokens_in_prompt=n_prompt,
                schema_inspected_before=schema_seen,
                duration_ms=int((time.perf_counter() - t_step) * 1000),
            ))
            termination = "repeated_action"
            if cfg.verbose:
                print(f"    STOP: action repeated {seen_actions[h]} times")
            break

        if action.tool not in tools:
            obs = _unknown_tool(action.tool, list(tools))
        else:
            try:
                obs = tools[action.tool](**action.args)
            except TypeError as e:
                obs = _bad_args(action.tool, e)
            except Exception as e:                       # noqa: BLE001
                # a raising tool is a layer A bug, but don't take the run down
                obs = ToolResult(status="error", tool=action.tool,
                                 error=str(e)[:200], error_full=repr(e),
                                 error_category="runtime", duration_ms=0)

        # snapshot before updating — the flag means "had it looked at the schema
        # BEFORE this step", otherwise the step leaks its own outcome
        schema_seen_before_this_step = schema_seen
        if action.tool == "get_schema" and obs.status == "ok":
            schema_seen = True

        if obs.status == "error":
            consecutive_errors += 1
            total_errors += 1
        else:
            consecutive_errors = 0

        if cfg.verbose:
            print(f"    Obs:     {obs.short_observation()[:160]}")

        steps.append(TrajectoryStep(
            run_id=run_id, step_index=i, thought=parsed.thought,
            action=action, observation=obs, raw_model_output=raw,
            status="continue" if obs.status == "ok" else "error",
            action_hash=h,
            repeat_count=seen_actions[h],
            error_category=obs.error_category,
            consecutive_errors=consecutive_errors,
            total_errors_so_far=total_errors,
            tokens_in_prompt=n_prompt,
            observation_truncated=obs.truncated,
            observation_rows=obs.row_count,
            schema_inspected_before=schema_seen_before_this_step,
            duration_ms=int((time.perf_counter() - t_step) * 1000),
        ))

        if consecutive_errors >= cfg.max_consecutive_errors:
            termination = "consecutive_errors"
            if cfg.verbose:
                print(f"    STOP: {consecutive_errors} errors in a row")
            break

    for s in steps:
        s.run_total_steps = len(steps)

    record = RunRecord(
        run_id=run_id,
        task_id=task_id,
        question=question,
        steps=steps,
        termination=termination,
        final_answer=final_answer,
        total_duration_ms=int((time.perf_counter() - t_start) * 1000),
        total_tokens=total_tokens,
        model_name=getattr(llm, "name", "unknown"),
        quantisation=getattr(llm, "quantisation", ""),
        temperature=getattr(llm, "temperature", 0.0),
        # Describes the actual configuration, not just the history policy.
        # The first version of this was the constant "full_history", which made
        # runs from before and after the loop-guard change indistinguishable —
        # so the provenance filter silently kept both. Runs generated before
        # this fix still carry the bare string, which is why label_dataset.py
        # excludes the superseded batch by FILENAME. See README.
        context_policy=(f"full_history|schema={'full' if schema else 'names'}"
                        f"|guards=repeat{cfg.max_repeats},err{cfg.max_consecutive_errors}"
                        f"|max_steps={cfg.max_steps}"),
    )

    if cfg.verbose:
        print(f"\n{'-' * 66}")
        print(f"terminated: {termination} · {len(steps)} steps · "
              f"{record.total_duration_ms} ms · ~{total_tokens} tokens")
        print("-" * 66)

    return record
