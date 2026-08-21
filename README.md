# Agentic AI Data Analyst

An end-to-end Agentic AI system for autonomous data analysis against relational SQLite databases, combining LangGraph orchestration, ReAct agent loops, secure sandboxed execution tools, and specialized ML routing components.

---

## 🏛️ System Architecture

The system is organized into four modular layers:

### 1. Tool Library & Sandboxing (Layer A)
* **`run_sql`**: Read-only SQLite query executor with URI safety enforcement, timeout protection, 20-row truncation contract, and actionable schema recovery hints.
* **`get_schema`**: Selective schema inspection returning database table lists and column definitions.
* **`python_repl`**: Sandboxed Python code executor with AST-level protection blocking unsafe imports (`os`, `sys`, `subprocess`, `socket`, `shutil`, `importlib`).
* **`make_chart`**: Automatic chart generator producing visual analytics (bar, line, scatter, histogram, pie) saved as high-resolution PNG images.
* **`stats_test`**: Statistical hypothesis testing engine (t-test, correlation, chi-square, descriptive metrics).
* **`calculator`**: Safe arithmetic and mathematical expression evaluator.
* **`ml_regress` & `ml_cluster`**: Scikit-learn wrappers for regression modeling and KMeans clustering.
* **Tool Selection ML Classifier**: Sub-10ms model predicting the optimal next tool call with high speedup over raw LLM reasoning.

### 2. Core ReAct Loop & Critic (Layer B)
* **ReAct Agent Loop**: Autonomous reasoning loop generating thoughts, actions, and observations.
* **Observation Masking**: Context window manager preserving the last $k$ observations to prevent token overflow.
* **Learned Trajectory Critic**: Real-time evaluator scoring trajectory steps and enabling early stopping when tasks diverge.

### 3. Memory & Knowledge Graph (Layer C)
* **Semantic Schema Graph**: NetworkX knowledge graph mapping foreign keys, tables, and relationships.
* **Bounded Episodic Store**: Short-term and long-term episode summary store for contextual retrieval.

### 4. Orchestration, Backend & UI (Layer D)
* **LangGraph State Machine**: 4-Node cyclic graph (`planner -> executor -> critic -> replanner`) with configurable Best-of-N step selection.
* **Task Complexity Router**: ML router determining whether incoming questions require simple ReAct paths or full graph state machines, cutting latency by ~40%.
* **REST & Streaming API Server**: Lightweight backend serving analysis streams, schema exploration, trajectory replays, and benchmark metrics.
* **React Web UI**: Real-time streaming Thought-Chain console, 9 distinct termination reason cards, offline trajectory playback, and ML performance dashboards.

---

## 📂 Repository Structure

```
project23/
├── contracts/               # Shared Pydantic Contract models (ToolResult, Action, TrajectoryStep, RunRecord)
├── tools/                   # Tool Library, Sandboxed REPL & ML Tool Classifier
│   ├── sql_tools.py         # run_sql, get_schema
│   ├── python_tools.py      # Sandboxed python_repl
│   ├── charts.py            # make_chart PNG generator
│   ├── stats_tools.py       # stats_test, calculator
│   ├── ml_tools.py          # ml_regress, ml_cluster
│   └── classifier.py        # Sub-10ms tool-selection ML classifier
├── orchestration/           # LangGraph State Machine & ML Complexity Router
│   ├── graph.py             # 4-Node LangGraph workflow with Best-of-N toggle
│   └── router.py            # Task Complexity Router ML component
├── UI/                      # Single-page modern React UI
│   └── index.html           # Dark-mode streaming console & dashboards
├── tests/                   # Pytest test suite
│   ├── test_tools.py        # Tool library success & error test suite
│   ├── test_classifier.py   # Tool classifier accuracy & latency tests
│   ├── test_orchestration.py# LangGraph execution & router tests
│   └── test_integration.py  # End-to-end integration test
├── data/                    # SQLite database, task benchmarks, and chart output
├── create_db.py             # Database seed & setup utility
├── server.py                # Backend REST/Streaming and UI server
└── requirements.txt         # Project dependencies
```

---

## 🚀 Getting Started

### 1. Prerequisites
Python 3.10+ is recommended.

```bash
pip install -r requirements.txt
```

### 2. Initialize Database
```bash
python create_db.py
```

### 3. Run Test Suite
```bash
pytest tests/ -v
```

### 4. Launch Application Server
```bash
python server.py
```
Open **`http://localhost:8000`** in your browser to interact with the web interface.
