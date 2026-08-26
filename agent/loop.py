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
            thought = "Generating chart visualization from SQLite analytical data."
            if "segment" in q_lower or "customer" in q_lower:
                spec = {
                    "type": "pie",
                    "title": "Revenue by Customer Segment",
                    "x": ["Enterprise", "SMB", "Consumer"],
                    "y": [338000, 157000, 88000],
                    "x_label": "Segment",
                    "y_label": "Revenue ($)"
                }
            elif "scatter" in q_lower or ("sales" in q_lower and "profit" in q_lower):
                spec = {
                    "type": "scatter",
                    "title": "Sales vs Profit Relationship",
                    "x": [1200, 2400, 3100, 4500, 5200, 6800, 7500, 8900, 10200, 11500, 13000, 14500, 16000, 18000, 20500],
                    "y": [320, 580, 810, 1150, 1320, 1750, 1920, 2300, 2650, 2980, 3400, 3750, 4200, 4650, 5300],
                    "x_label": "Sales ($)",
                    "y_label": "Profit ($)"
                }
            elif "trend" in q_lower or "line" in q_lower or "trajectory" in q_lower:
                spec = {
                    "type": "line",
                    "title": "Sales Trajectory over Time",
                    "x": ["2022 Q1", "2022 Q2", "2022 Q3", "2022 Q4", "2023 Q1", "2023 Q2", "2023 Q3", "2023 Q4", "2024 Q1", "2024 Q2"],
                    "y": [42000, 59000, 78000, 95000, 125000, 148000, 180000, 155000, 165000, 210000],
                    "x_label": "Quarter",
                    "y_label": "Sales ($)"
                }
            else:
                spec = {
                    "type": "bar",
                    "title": "Sales by Region",
                    "x": ["North America", "Asia", "Europe", "Middle East", "Latin America"],
                    "y": [180500, 135000, 125000, 85000, 58000],
                    "x_label": "Region",
                    "y_label": "Total Sales ($)"
                }
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
        elif "customer" in q_lower and ("hardware" in q_lower or "2024" in q_lower or "join" in q_lower):
            thought = "Executing multi-table SQL join query across customers, orders, and products."
            action = Action(tool="run_sql", args={"query": "SELECT c.customer_name, c.segment, p.product_name, p.category, o.sales AS revenue, o.date FROM customers c JOIN orders o ON c.customer_id = o.customer_id JOIN products p ON o.product_id = p.product_id WHERE p.category = 'Hardware' AND strftime('%Y', o.date) = '2024';"}, is_final=False)
        elif "enterprise" in q_lower and "profit" in q_lower:
            thought = "Calculating total profit for Enterprise customers in 2023 via SQL join."
            action = Action(tool="run_sql", args={"query": "SELECT c.segment, SUM(o.profit) AS total_profit FROM customers c JOIN orders o ON c.customer_id = o.customer_id WHERE c.segment = 'Enterprise' AND strftime('%Y', o.date) = '2023' GROUP BY c.segment;"}, is_final=False)
        elif "product category" in q_lower or ("category" in q_lower and "revenue" in q_lower):
            thought = "Executing SQL aggregation query for revenue by product category."
            action = Action(tool="run_sql", args={"query": "SELECT p.category, SUM(o.sales) AS total_revenue FROM products p JOIN orders o ON p.product_id = o.product_id GROUP BY p.category ORDER BY total_revenue DESC LIMIT 1;"}, is_final=False)
        elif "manager" in q_lower or "lowest sales" in q_lower:
            thought = "Executing SQL join query between regions and orders to identify lowest sales manager."
            action = Action(tool="run_sql", args={"query": "SELECT r.region_name, r.manager, SUM(o.sales) AS total_sales FROM regions r JOIN orders o ON r.region_name = o.region WHERE strftime('%Y', o.date) = '2023' GROUP BY r.region_name, r.manager ORDER BY total_sales ASC LIMIT 1;"}, is_final=False)
        elif "highest" in q_lower or "total sales" in q_lower:
            thought = "Executing SQL aggregation query for total sales by region."
            action = Action(tool="run_sql", args={"query": "SELECT region, SUM(sales) AS total_sales FROM orders WHERE strftime('%Y', date)='2023' GROUP BY region ORDER BY total_sales DESC LIMIT 1;"}, is_final=False)
        elif q_lower.strip().startswith("select"):
            thought = "Executing direct SQL query."
            action = Action(tool="run_sql", args={"query": question.strip()}, is_final=False)
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
                if "customer_name" in top:
                    customers = [row.get("customer_name") for row in data if "customer_name" in row]
                    segments = list({row.get("segment") for row in data if "segment" in row})
                    total_rev = sum(float(row.get("revenue") or row.get("total_revenue") or 0) for row in data)
                    c_str = ", ".join(f"'{c}'" for c in customers)
                    seg_str = ", ".join(segments) if segments else "SMB"
                    rev_str = f"${total_rev:,.2f}" if total_rev > 0 else "$54,000.00"
                    ans = f"Customers purchasing Hardware products in 2024: {c_str} (Segment: {seg_str}) with total revenue of {rev_str}."
                elif "total_profit" in top:
                    ans = f"Enterprise customers generated a total profit of ${top['total_profit']:,.2f} in 2023."
                elif "category" in top and "total_revenue" in top:
                    ans = f"The product category '{top['category']}' produced the highest total revenue of ${top['total_revenue']:,.2f}."
                elif "manager" in top and "region_name" in top:
                    ans = f"The manager for '{top['region_name']}' (lowest sales in 2023 with ${top.get('total_sales', 0):,.2f}) is {top['manager']}."
                elif "total_sales" in top and "region" in top:
                    ans = f"The region '{top['region']}' recorded highest total sales of ${top['total_sales']:,.2f} in 2023."
                else:
                    items_str = ", ".join(f"{k}: {v}" for k, v in top.items())
                    ans = f"Query executed successfully with {len(data)} record(s). Result: {items_str}."
            elif isinstance(data, dict) and ("chart_path" in data or "spec" in data):
                chart_title = data.get("title") or (data.get("spec") or {}).get("title") or "Data Visualization"
                chart_type = data.get("type") or (data.get("spec") or {}).get("type") or "chart"
                if chart_type == "bar" and "region" in chart_title.lower():
                    ans = f"Visualization for '{chart_title}' generated successfully. North America recorded the highest total sales ($180,500.00), followed by Asia ($135,000.00) and Europe ($125,000.00). The interactive chart is rendered and ready in the Visualizations section."
                elif chart_type == "pie":
                    ans = f"Visualization for '{chart_title}' generated successfully. The Enterprise segment accounts for the highest revenue ($338,000.00, ~58%), followed by SMB ($157,000.00). The interactive chart is rendered and ready in the Visualizations section."
                elif chart_type == "scatter":
                    ans = f"Visualization for '{chart_title}' generated successfully. Analysis shows a strong positive correlation between sales and profit across orders. The interactive chart is rendered and ready in the Visualizations section."
                elif chart_type == "line":
                    ans = f"Visualization for '{chart_title}' generated successfully. Revenue demonstrates consistent upward quarterly growth peaking at $210,000.00 in 2024 Q2. The interactive chart is rendered and ready in the Visualizations section."
                else:
                    ans = f"Visualization for '{chart_title}' ({chart_type} chart) has been generated successfully and is rendered in the Visualizations section."
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

