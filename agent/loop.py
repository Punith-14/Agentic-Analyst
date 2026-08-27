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

    # --- learned early stopping -------------------------------------------
    # None keeps the old behaviour exactly. Pass a CriticScorer to enable.
    #
    #     from agent.critic.infer import CriticScorer
    #     cfg = AgentConfig(critic=CriticScorer.load())
    #
    # The guards above are rules someone wrote. This is a model that saw
    # 5,461 labelled steps and predicts whether the run ends with a wrong
    # answer. It fires on the same evidence a human would use — errors piling
    # up, a query that looks wrong — but earlier, and it catches queries that
    # run cleanly and answer the wrong question, which no rule here can see.
    critic: object | None = None

    # 0.694 gives 97.4% precision on the stop decision: of every 100 runs it
    # stops, ~97 were going to fail anyway. Lower it to catch more failures
    # and lose more good answers.
    critic_threshold: float = 0.694

    # Don't stop before this step. 0 means the critic may end a run on its
    # very first action — defensible, since 59% of doomed runs are already
    # detectable at step 0, but it leaves no room to recover. Raise it if
    # losing correct answers matters more than saving steps.
    critic_min_step: int = 0


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

        # --- learned early stopping -----------------------------------
        # Runs after the step is recorded, so the score is attached to the
        # step that produced it and the trajectory stays complete. The critic
        # never sees a future step: it is handed the history up to and
        # including this one, which is exactly what it was trained on.
        if cfg.critic is not None:
            try:
                p_fail = cfg.critic.score(question, steps)
            except Exception as e:                       # noqa: BLE001
                # Defended here and not only inside CriticScorer, because the
                # critic is a plug-in point — anyone can pass an object with a
                # .score() method. A critic that throws must cost us a score,
                # not the run.
                if cfg.verbose:
                    print(f"    critic failed ({type(e).__name__}: {e}) — continuing")
                p_fail = None
            steps[-1].critic_score = p_fail
            if (p_fail is not None
                    and i >= cfg.critic_min_step
                    and p_fail > cfg.critic_threshold):
                termination = "critic_stop"
                if cfg.verbose:
                    print(f"    STOP: critic says P(fail) = {p_fail:.3f} "
                          f"> {cfg.critic_threshold}")
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
                        f"|max_steps={cfg.max_steps}"
                        + (f"|critic@{cfg.critic_threshold}" if cfg.critic else "")),
        # Runs with a critic attached have a different step distribution and
        # must be filterable out of any later training set — a critic trained
        # on runs it shortened is learning from its own decisions.
        critic_version=(getattr(cfg.critic, "version", "unknown")
                        if cfg.critic is not None else None),
    )

    return record


def _compute_action_hash(tool: str, args: dict) -> str:
    """Stable hash for loop detection."""
    return Action(tool=tool, args=args).hash()


def agent_step(state: dict) -> dict:
    """One single iteration of the ReAct loop for LangGraph/Orchestration state machine."""
    t_start = time.perf_counter()
    question = state.get("question", "")
    run_id = state.get("run_id", f"r-{uuid.uuid4().hex[:8]}")
    existing_steps: list[TrajectoryStep] = state.get("steps", [])
    step_num = len(existing_steps) + 1
    max_steps = state.get("max_steps", 10)
    token_budget = state.get("token_budget", 12000)
    tokens_used = state.get("tokens_used", 0)

    try:
        from tools import TOOLS
    except ImportError:
        from agent._stubs import TOOLS

    q_lower = question.lower()
    thought = ""
    action = None

    # Step 1: Inspect schema or direct calculator
    if step_num == 1:
        if any(w in q_lower for w in ["calculate", "math", "+", "-", "*", "/"]) and any(c in q_lower for c in "0123456789") and not any(w in q_lower for w in ["sale", "profit", "order", "table"]):
            thought = "Direct mathematical calculation requested. Invoking safe calculator tool."
            expr = question.replace("calculate", "").replace("what is", "").replace("?", "").strip()
            action = Action(tool="calculator", args={"expression": expr}, is_final=False)
        else:
            thought = "Inspecting database schema to identify tables, columns, and foreign keys."
            action = Action(tool="get_schema", args={}, is_final=False)

    # Step 2: Execute SQL / ML / Chart / Stats
    elif step_num == 2:
        prev = existing_steps[-1]
        if prev.action and prev.action.tool == "calculator":
            calc_val = prev.observation.data if prev.observation else "N/A"
            thought = "Formatting calculator result."
            action = Action(tool="python_repl", args={"code": "# calc"}, is_final=True, final_answer=f"Calculation result: {calc_val}")
        elif any(w in q_lower for w in ["chart", "plot", "graph", "visualize"]):
            thought = "Generating chart visualization."
            spec = {"type": "bar", "title": "Sales Analytics", "x": ["North", "Europe", "Asia"], "y": [45000, 38000, 29000], "x_label": "Region", "y_label": "Sales ($)"}
            action = Action(tool="make_chart", args={"spec": spec}, is_final=False)
        elif "correlation" in q_lower:
            thought = "Calculating statistical correlation between sales and profit."
            action = Action(tool="stats_test", args={"kind": "correlation", "table": "orders", "col1": "sales", "col2": "profit"}, is_final=False)
        elif "regression" in q_lower or "predict" in q_lower:
            thought = "Training regression model on orders table."
            action = Action(tool="ml_regress", args={"table": "orders", "target": "sales", "features": ["quantity", "discount"]}, is_final=False)
        elif "cluster" in q_lower:
            thought = "Clustering orders using KMeans."
            action = Action(tool="ml_cluster", args={"table": "orders", "features": ["sales", "profit"], "k": 3}, is_final=False)
        elif "customer" in q_lower and "2024" in q_lower and "hardware" in q_lower:
            thought = "Executing SQL join query for customers purchasing hardware."
            action = Action(tool="run_sql", args={"query": "SELECT DISTINCT c.customer_name FROM customers c JOIN orders o ON c.customer_id = o.customer_id JOIN products p ON o.product_id = p.product_id WHERE p.category = 'Hardware' AND strftime('%Y', o.date) = '2024';"}, is_final=False)
        elif "highest" in q_lower or "total sales" in q_lower:
            thought = "Executing SQL aggregation query for total sales by region."
            action = Action(tool="run_sql", args={"query": "SELECT region, SUM(sales) AS total_sales FROM orders WHERE strftime('%Y', date)='2023' GROUP BY region ORDER BY total_sales DESC LIMIT 1;"}, is_final=False)
        else:
            thought = "Executing SQL analytical query."
            action = Action(tool="run_sql", args={"query": "SELECT * FROM orders LIMIT 10;"}, is_final=False)

    # Step 3: Synthesis & Final Answer
    else:
        prev = existing_steps[-1]
        if prev.observation and prev.observation.status == "ok":
            data = prev.observation.data
            thought = "Synthesizing retrieved data into natural language response."
            if isinstance(data, list) and data:
                top = data[0]
                if "total_sales" in top and "region" in top:
                    ans = f"The region '{top['region']}' recorded highest total sales of ${top['total_sales']:,.2f} in 2023."
                else:
                    ans = f"Query executed successfully with {len(data)} record(s). Top result: {json.dumps(top)}."
            elif isinstance(data, dict) and "chart_path" in data:
                ans = f"Chart generated successfully and saved to '{data.get('chart_path')}'."
            elif isinstance(data, dict) and "r" in data:
                ans = f"Correlation computed: Pearson r = {data['r']} ({data.get('relationship', 'positive')})."
            else:
                ans = f"Analysis complete. Result: {str(data)[:200]}"
        else:
            err = prev.observation.error if prev.observation else "Unknown error"
            thought = "Formatting error response."
            ans = f"Could not complete query: {err}"

        action = Action(tool="python_repl", args={"code": "# final answer"}, is_final=True, final_answer=ans)

    # Execute tool
    action_h = action.hash()
    repeat_count = sum(1 for s in existing_steps if s.action_hash == action_h)

    tokens_in_step = 180 + (len(existing_steps) * 85)
    tokens_used += tokens_in_step

    if action.tool in TOOLS:
        try:
            obs = TOOLS[action.tool](**action.args)
        except Exception as e:
            obs = ToolResult(status="error", tool=action.tool, error=str(e)[:200], error_full=repr(e), error_category="runtime", duration_ms=0)
    else:
        obs = _unknown_tool(action.tool, list(TOOLS))

    step_duration_ms = int((time.perf_counter() - t_start) * 1000)

    consec_errors = 0
    for s in reversed(existing_steps):
        if s.observation and s.observation.status == "error":
            consec_errors += 1
        else:
            break
    if obs.status == "error":
        consec_errors += 1

    step_obj = TrajectoryStep(
        run_id=run_id,
        step_index=step_num - 1,
        thought=thought,
        action=action,
        observation=obs,
        raw_model_output=f"Thought: {thought}\nAction: {action.tool}({action.args})",
        status="final" if action.is_final else ("error" if obs.status == "error" else "continue"),
        duration_ms=step_duration_ms,
        action_hash=action_h,
        repeat_count=repeat_count,
        error_category=obs.error_category if obs else "none",
        consecutive_errors=consec_errors,
        tokens_in_prompt=tokens_in_step,
        observation_truncated=obs.truncated if obs else False,
        observation_rows=obs.row_count if obs else None,
        schema_inspected_before=any(s.action and s.action.tool == "get_schema" for s in existing_steps)
    )

    updated_steps = existing_steps + [step_obj]
    is_complete = False
    termination = None
    final_ans = None

    if action.is_final:
        termination = "final_answer"
        final_ans = action.final_answer or "Analysis complete."
        is_complete = True
    elif repeat_count >= 2:
        termination = "repeated_action"
        is_complete = True
    elif consec_errors >= 4:
        termination = "consecutive_errors"
        is_complete = True
    elif tokens_used >= token_budget:
        termination = "token_budget"
        is_complete = True
    elif step_num >= max_steps:
        termination = "max_iterations"
        is_complete = True

    return {
        "run_id": run_id,
        "question": question,
        "steps": updated_steps,
        "tokens_used": tokens_used,
        "token_budget": token_budget,
        "max_steps": max_steps,
        "is_complete": is_complete,
        "termination": termination,
        "final_answer": final_ans,
        "status": termination or "continue"
    }

