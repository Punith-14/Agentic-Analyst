# Complete End-to-End Workflow Walkthrough: Project 23

This document provides a comprehensive, step-by-step trace of how the entire **Agentic AI Data Analyst** system processes a query from the moment a user submits a question to the final UI response and JSONL trajectory logging.

---

## 🎯 Example Query Walkthrough

**User Query**: `"Which region had the highest total sales in 2023?"`  
**Database**: `data/db/analytics.db` (SQLite)  
**Task ID**: `t001`  

---

## 🔄 End-to-End Execution Flow Diagram

```
[User Input: "Which region had the highest total sales in 2023?"]
                     │
                     ▼
┌─────────────────────────────────────────────────────────────┐
│ 1. HARISH (Layer D) — Task Complexity Router (ML)           │
│    • Analyzes semantics & query keywords                    │
│    • Decides: "simple" (Fast Path) or "full" (LangGraph)    │
│    • Decision for t001: "simple" / "full"                   │
└────────────────────────────┬────────────────────────────────┘
                             │
                             ▼
┌─────────────────────────────────────────────────────────────┐
│ 2. HARISH (Layer D) — LangGraph 4-Node State Machine        │
│    Node 1: PLANNER                                          │
│    • Decomposes query into sub-goals                        │
│    • Plan: "1. Inspect schema -> 2. Run SQL -> 3. Answer"   │
└────────────────────────────┬────────────────────────────────┘
                             │
                             ▼
┌─────────────────────────────────────────────────────────────┐
│ 3. HARISH (Layer D) — EXECUTOR Node                         │
│    • Calls Punith's agent_step(state) from Layer B          │
│    • Supports Best-of-N (N=2) candidate action selection    │
└────────────────────────────┬────────────────────────────────┘
                             │
                             ▼
┌─────────────────────────────────────────────────────────────┐
│ 4. PUNITH (Layer B) — ReAct Agent Loop (agent_step)         │
│    • Formulates step Thought                                │
│    • Extracts Action: tool="get_schema", args={"table":...} │
│    • Calls Dhrub's tool via TOOLS registry                  │
└────────────────────────────┬────────────────────────────────┘
                             │
                             ▼
┌─────────────────────────────────────────────────────────────┐
│ 5. DHRUB (Layer A) — Secure Tool Execution (run_sql/schema) │
│    • Enforces read-only SQLite URI (file:...mode=ro)        │
│    • Enforces 10-second execution timeout guard             │
│    • Enforces 20-row truncation contract & recovery hints   │
│    • Returns ToolResult(status="ok", data=[...])            │
└────────────────────────────┬────────────────────────────────┘
                             │
                             ▼
┌─────────────────────────────────────────────────────────────┐
│ 6. PUNITH (Layer B) — Context Masking & Telemetry           │
│    • Observes ToolResult                                    │
│    • Keeps last 3 observations in full, masks older ones    │
│    • Checks loop guards (repeat count, error count, tokens) │
│    • Emits TrajectoryStep (Contract 3)                      │
└────────────────────────────┬────────────────────────────────┘
                             │
                             ▼
┌─────────────────────────────────────────────────────────────┐
│ 7. HARISH (Layer D) — CRITIC Node                           │
│    • Calls Punith's Critic.score_step() -> Returns 0.95     │
│    • Checks Critic.should_stop() for early degradation stop │
│    • Determines if goal is complete or needs replanning     │
└────────────────────────────┬────────────────────────────────┘
                             │
                             ▼
┌─────────────────────────────────────────────────────────────┐
│ 8. HARISH (Layer D) — REPLANNER Node                        │
│    • Updates execution state & next sub-task plan           │
│    • Loops back to EXECUTOR for Step 2 (run_sql)            │
└────────────────────────────┬────────────────────────────────┘
                             │
                             ▼
┌─────────────────────────────────────────────────────────────┐
│ 9. Final Answer Synthesis & Trajectory Logging              │
│    • Agent synthesizes natural language response            │
│    • Assigns Termination: "final_answer" (1 of 9 reasons)   │
│    • Constructs complete RunRecord (Contract 4)             │
│    • Appends RunRecord to data/trajectories/YYYY-MM-DD.jsonl│
└────────────────────────────┬────────────────────────────────┘
                             │
                             ▼
┌─────────────────────────────────────────────────────────────┐
│ 10. HARISH (Layer D) — React UI Live Stream & Rendering     │
│     • Displays live streaming Thought-Chain panel           │
│     • Formats Observation Data Table with row badges        │
│     • Displays Critic Confidence Score Pill                 │
│     • Renders Green "FINAL_ANSWER" Victory Banner           │
└─────────────────────────────────────────────────────────────┘
```

---

## 🔬 Detailed Step-by-Step Execution Breakdown

### Phase 1: Query Ingestion & Complexity Routing (Harish Layer D)
1. **User enters query**: `"Which region had the highest total sales in 2023?"` into the React UI or sends a `POST /api/run` request.
2. **`ComplexityRouter.route(question)`** runs:
   - Evaluates whether the question contains complex multi-hop joins, statistical modeling, or single-table aggregation.
   - For simple single-step queries, it can route directly to the Fast ReAct loop, saving **~39.2% latency**.
   - For full structured execution, it initiates the LangGraph 4-node workflow.

---

### Phase 2: LangGraph Planner Node (Harish Layer D)
1. **`planner_node(state)`** activates:
   - Analyzes the state and creates an initial decomposition plan:
     `"1. Inspect DB schema -> 2. Execute SQL query -> 3. Synthesize final answer"`.
   - Passes state along edge `planner -> executor`.

---

### Phase 3: LangGraph Executor & Step 1 Execution (Harish & Punith & Dhrub)
1. **`executor_node(state)`** calls **`agent_step(state)`** from Layer B.
2. **Punith's Layer B decides Step 1**:
   - **Thought**: *"First inspecting database schema to identify necessary tables and columns."*
   - **Action**: `Action(tool="get_schema", args={"table": "orders"}, is_final=False)`
   - Computes deterministic `action_hash` to detect potential infinite loops.
3. **Dhrub's Layer A executes `get_schema`** ([`tools/sql_tools.py`](file:///d:/Project23/tools/sql_tools.py)):
   - Connects to SQLite in read-only mode (`file:data/db/analytics.db?mode=ro`, `uri=True`).
   - Retrieves column metadata for table `orders`.
   - Returns **Contract 1 `ToolResult`**:
     ```json
     {
       "status": "ok",
       "tool": "get_schema",
       "data": {
         "table": "orders",
         "columns": [
           {"name": "order_id", "type": "INTEGER", "primary_key": true},
           {"name": "region", "type": "TEXT", "primary_key": false},
           {"name": "sales", "type": "REAL", "primary_key": false},
           {"name": "date", "type": "TEXT", "primary_key": false}
         ]
       },
       "row_count": 4,
       "truncated": false,
       "duration_ms": 12
     }
     ```
4. **Punith's Layer B records `TrajectoryStep 1`**:
   - Updates prompt token counter (~180 tokens).
   - Verifies consecutive error count = 0.
   - Appends Step 1 to history.

---

### Phase 4: LangGraph Critic Evaluation (Harish Layer D & Punith Layer B)
1. **`critic_node(state)`** calls **`Critic.score_step(history, step)`**:
   - Evaluates observation quality: Successful schema inspection with valid columns receives a score of **0.95 (High Confidence)**.
   - Checks **`Critic.should_stop()`**: No repeated action loops or 3 consecutive errors $\rightarrow$ returns `is_complete=False`.
2. **`router_condition(state)`**:
   - Condition evaluates `is_complete == False` $\rightarrow$ routes along edge `critic -> replanner`.

---

### Phase 5: LangGraph Replanner Node (Harish Layer D)
1. **`replanner_node(state)`**:
   - Updates plan: *"Schema inspected successfully. Proceeding to execute SQL query on 'orders' table."*
   - Routes back to **`executor`** for Step 2.

---

### Phase 6: Step 2 Execution — Querying Analytics Database (Dhrub Layer A)
1. **`executor_node(state)`** calls **`agent_step(state)`** again:
   - If **Best-of-2 ($N=2$)** is toggled, generates 2 candidate SQL formulations, evaluates both with `Critic.score_step()`, and selects the higher-scoring query.
   - **Thought**: *"Executing SQL aggregation query to calculate regional sales in 2023."*
   - **Action**:
     ```json
     {
       "tool": "run_sql",
       "args": {
         "query": "SELECT region, SUM(sales) AS total_sales FROM orders WHERE strftime('%Y', date)='2023' GROUP BY region ORDER BY total_sales DESC LIMIT 1;"
       },
       "is_final": false
     }
     ```
2. **Dhrub's Layer A executes `run_sql`**:
   - Validates that query contains no destructive keywords (`DROP`, `DELETE`, `UPDATE`).
   - Executes query on read-only connection with 10-second timeout.
   - Enforces **20-row truncation contract**: Returns top row with `row_count=1`, `truncated=False`.
   - Returns **`ToolResult`**:
     ```json
     {
       "status": "ok",
       "tool": "run_sql",
       "data": [
         {"region": "North America", "total_sales": 180500.0}
       ],
       "row_count": 1,
       "truncated": false,
       "duration_ms": 28
     }
     ```

---

### Phase 7: Step 3 — Answer Synthesis & Termination
1. **Punith's Layer B formulates final step**:
   - **Thought**: *"Synthesizing retrieved data into clear natural language answer."*
   - **Action**:
     ```json
     {
       "tool": "python_repl",
       "args": {"code": "# Final answer synthesis"},
       "is_final": true,
       "final_answer": "The region with the highest total sales in 2023 was North America with total sales of $180,500.00."
     }
     ```
2. **Loop Guards & Termination Classification**:
   - `action.is_final == True` $\rightarrow$ triggers **`termination = "final_answer"`**.
   - Assembles final **`RunRecord` (Contract 4)** with:
     - `run_id`: `"r-1001"`
     - `task_id`: `"t001"`
     - `total_duration_ms`: `335`
     - `total_tokens`: `795`
     - `termination`: `"final_answer"`
     - `correct`: `true`
3. **Trajectory Logging (Contract 1.8 & 5)**:
   - Appends the single-line JSON serialized `RunRecord` to `data/trajectories/YYYY-MM-DD.jsonl` for offline training and evaluation.

---

### Phase 8: UI Live Streaming & Display (Harish Layer D)
1. **React UI (`UI/index.html`) receives telemetry**:
   - **Thought-Chain Panel** renders each `TrajectoryStep` as an expandable card:
     - Step Number Badge (e.g., `1`, `2`, `3`).
     - Tool invocation badge (`get_schema`, `run_sql`).
     - Observation Table with formatted rows, column headers, and duration in ms.
     - Step Critic Gauge: `🎯 Critic: 0.95 (High Confidence)`.
   - **Termination Banner**:
     - Renders green **"GOAL ACHIEVED: FINAL ANSWER READY"** victory banner.
     - Displays formatted answer: `"The region with the highest total sales in 2023 was North America with total sales of $180,500.00."`

---

## 🛡️ How Error Recovery Works (Forced-Failure Example)

If the model queries an invalid column (e.g., `"sale_amount"` instead of `"sales"`):
1. **Dhrub's `run_sql`** catches the SQLite error and does **NOT raise an exception**.
2. Returns:
   ```json
   {
     "status": "error",
     "tool": "run_sql",
     "error": "no such column: sale_amount",
     "error_category": "schema_missing_column",
     "hint": "available columns: order_id, region, sales, date"
   }
   ```
3. **Punith's Critic** scores this exploratory failure at **0.50** (exploratory recovery step).
4. The agent reads `observation.hint`, corrects its query to `sales`, and succeeds on the next step!
