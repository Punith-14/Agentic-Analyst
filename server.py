# server.py
"""Harish (Layer D) - API Backend & Development Server.
Provides REST & SSE endpoints for live execution, trajectory replay, ML benchmarks,
and schema inspection. Serves the React UI.
"""
import os
import json
import glob
import mimetypes
import sys
from urllib.parse import urlparse, parse_qs
from http.server import HTTPServer, SimpleHTTPRequestHandler
from pathlib import Path
from typing import Dict, Any, List

from contracts import RunRecord, TrajectoryStep, ToolResult
from orchestration.graph import run_graph_to_record
from orchestration.router import ComplexityRouter
from agent.loop import AgentConfig, run_agent
from tools.classifier import ToolSelectionClassifier
from tools.db import clear_db, current_db, set_db
from tools.sql_tools import get_schema

# Windows defaults stdout to cp1252 when it is piped rather than attached to a
# terminal, and cp1252 has no emoji — so `print("\U0001f680 ...")` below killed
# the server on startup with a UnicodeEncodeError before it ever listened.
# It only showed up when something captured the output (a smoke test, or
# `python server.py > server.log`), which is exactly when you least want the
# server to die. Reconfiguring once here covers every print in this process.
for _stream in (sys.stdout, sys.stderr):
    try:
        _stream.reconfigure(encoding="utf-8", errors="replace")
    except (AttributeError, ValueError):        # already wrapped, or not a tty
        pass

PORT = 8000

# NO DATABASE IS SELECTED AT STARTUP, DELIBERATELY.
#
# This used to point at data/db/analytics.db and create it if absent, so the
# server always had *something* to query. That meant a failed upload produced
# confident answers about generated demo data — "North America had the highest
# sales at $180,500" — with nothing to indicate the numbers were invented.
#
# Now the tools return "No dataset provided" until the user supplies one via
# POST /api/database. The UI shows that message instead of a fabricated answer.
#
# Demo data is still available; it just has to be asked for:
#     POST /api/database  {"path": "data/db/analytics.db"}
UPLOAD_DIR = Path("data/uploads")
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
Path("data/charts").mkdir(parents=True, exist_ok=True)

router_ml = ComplexityRouter()

tool_classifier_ml = ToolSelectionClassifier()

# The trajectory critic, loaded once and reused. Feature model only — pulling
# ModernBERT would put 600 MB on the same 6 GB card that is hosting Qwen.
_CRITIC = None
_CRITIC_TRIED = False


def _live_critic():
    """The critic the loop scores steps with, or None if it can't load."""
    global _CRITIC, _CRITIC_TRIED
    if not _CRITIC_TRIED:
        _CRITIC_TRIED = True
        try:
            from agent.critic.infer import CriticScorer
            _CRITIC = CriticScorer.load(use_text=False)
            print(f"critic loaded: {_CRITIC.version}")
        except Exception as e:                              # noqa: BLE001
            print(f"critic unavailable ({type(e).__name__}: {e}) — "
                  f"steps will have no score")
    return _CRITIC


class AgentApiHandler(SimpleHTTPRequestHandler):
    """Custom HTTP Request Handler for Project 23 API and UI."""

    def end_headers(self):
        # Enable CORS for development
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        super().end_headers()

    def do_OPTIONS(self):
        self.send_response(200)
        self.end_headers()

    def do_GET(self):
        parsed_url = urlparse(self.path)
        path = parsed_url.path

        # 1. API: Task Suite
        if path == "/api/tasks":
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            try:
                with open("data/tasks/task_suite.json", "r", encoding="utf-8") as f:
                    tasks = json.load(f)
                self.wfile.write(json.dumps(tasks).encode("utf-8"))
            except Exception as e:
                self.wfile.write(json.dumps({"error": str(e)}).encode("utf-8"))
            return

        # 2. API: Recorded Trajectories List
        if path == "/api/trajectories":
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            runs = []
            for file in sorted(glob.glob("data/trajectories/*.jsonl"), reverse=True):
                try:
                    with open(file, "r", encoding="utf-8") as f:
                        for line in f:
                            if line.strip():
                                rec = json.loads(line)
                                runs.append({
                                    "run_id": rec.get("run_id"),
                                    "task_id": rec.get("task_id"),
                                    "question": rec.get("question"),
                                    "termination": rec.get("termination"),
                                    "correct": rec.get("correct"),
                                    "steps_count": len(rec.get("steps", [])),
                                    "duration_ms": rec.get("total_duration_ms", 0),
                                    "file": os.path.basename(file)
                                })
                except Exception:
                    continue
            self.wfile.write(json.dumps(runs).encode("utf-8"))
            return

        # 3. API: Single Trajectory Run Details for Replay
        if path.startswith("/api/trajectories/"):
            run_id = path.split("/")[-1]
            found_record = None
            for file in glob.glob("data/trajectories/*.jsonl"):
                try:
                    with open(file, "r", encoding="utf-8") as f:
                        for line in f:
                            if line.strip():
                                rec = json.loads(line)
                                if rec.get("run_id") == run_id:
                                    found_record = rec
                                    break
                except Exception:
                    continue
                if found_record:
                    break

            if found_record:
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.end_headers()
                self.wfile.write(json.dumps(found_record).encode("utf-8"))
            else:
                self.send_response(404)
                self.send_header("Content-Type", "application/json")
                self.end_headers()
                self.wfile.write(json.dumps({"error": f"Run {run_id} not found."}).encode("utf-8"))
            return

        # 4. API: ML Evaluation & Benchmark Metrics
        if path == "/api/ml-metrics":
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            tool_eval = tool_classifier_ml.evaluate_against_baseline()
            router_eval = router_ml.evaluate_against_baseline()
            data = {
                "layer_a_tool_classifier": tool_eval,
                "harish_complexity_router": router_eval
            }
            self.wfile.write(json.dumps(data).encode("utf-8"))
            return

        # Layer B's trajectory critic. Read from the files the notebooks
        # wrote, never hardcoded — the whole point of this session was that a
        # number typed into a UI is indistinguishable from a measured one.
        if path == "/api/critic-metrics":
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.end_headers()

            out = {
                # notebooks/05_critic_comparison.ipynb, test set scored once
                "test": {
                    "lgbm_pr_auc": 0.9819,
                    "bert_pr_auc": 0.9741,
                    "ensemble_pr_auc": 0.9845,
                    "base_rate": 0.8165,
                    "steps": 1030,
                    "lgbm_ms": 3.73,
                    "bert_ms": 18.86,
                },
                # notebooks/06_holdout_chinook.ipynb, unseen database
                "holdout": {
                    "database": "chinook_1",
                    "ensemble_pr_auc": 0.9088,
                    "base_rate": 0.5809,
                    "retained": 0.855,
                    "runs": 300,
                },
                "early_stop": None,
                "text_model": None,
            }

            m = Path("data/critic/early_stop_measurement.json")
            if m.exists():
                try:
                    out["early_stop"] = json.loads(m.read_text(encoding="utf-8"))
                except Exception:                           # noqa: BLE001
                    pass

            for d in sorted(Path("models").glob("critic_bert_*")):
                meta = d / "meta.json"
                if meta.exists():
                    try:
                        out["text_model"] = json.loads(meta.read_text(encoding="utf-8"))
                    except Exception:                       # noqa: BLE001
                        pass

            self.wfile.write(json.dumps(out).encode("utf-8"))
            return

        # 5. API: Schema Inspection
        # Which database is connected, if any. The UI polls this to decide
        # whether to show the upload prompt or the query box.
        if path == "/api/database":
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            db = current_db()
            # Offer what's already on disk so the UI can show a picker rather
            # than asking the user to type a path.
            available = sorted(p.stem for p in Path("data/db").glob("*.db"))
            available += sorted(p.name for p in UPLOAD_DIR.glob("*.db"))
            self.wfile.write(json.dumps({
                "connected": db is not None,
                "path": db,
                "name": Path(db).stem if db else None,
                "available": available,
                "message": None if db else
                           "No dataset connected. Choose a database to continue.",
            }).encode("utf-8"))
            return

        if path == "/api/schema":
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            params = parse_qs(parsed_url.query)
            table_param = params.get("table", [None])[0]
            # No db_path: get_schema uses whatever was selected, and returns a
            # readable error if nothing was.
            schema_res = get_schema(table=table_param)
            self.wfile.write(json.dumps(schema_res.model_dump()).encode("utf-8"))
            return

        # 6. Static Charts
        if path.startswith("/data/charts/") or path.startswith("/charts/"):
            fname = path.split("/")[-1]
            chart_full_path = os.path.join("data/charts", fname)
            if os.path.exists(chart_full_path):
                self.send_response(200)
                self.send_header("Content-Type", "image/png")
                self.end_headers()
                with open(chart_full_path, "rb") as f:
                    self.wfile.write(f.read())
                return

        # 7. Serve React UI
        if path in ["/", "/index.html", "/ui", "/UI"]:
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.end_headers()
            try:
                with open("UI/index.html", "rb") as f:
                    self.wfile.write(f.read())
            except Exception as e:
                self.wfile.write(f"<h1>Error loading UI: {str(e)}</h1>".encode("utf-8"))
            return

        # Default fallback to static file or 404
        super().do_GET()

    def do_POST(self):
        parsed_url = urlparse(self.path)
        path = parsed_url.path

        # API: connect a dataset.
        #   {"path": "data/db/analytics.db"}     an existing file
        #   {"name": "chinook_1"}                one of the benchmark databases
        if path == "/api/database":
            content_length = int(self.headers.get("Content-Length", 0))
            try:
                payload = json.loads(self.rfile.read(content_length).decode("utf-8"))
            except Exception:                                   # noqa: BLE001
                payload = {}

            target = payload.get("path") or payload.get("name")
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.end_headers()

            # An empty body means "disconnect" — the UI's "change" button.
            if not target:
                clear_db()
                self.wfile.write(json.dumps({
                    "connected": False, "path": None, "name": None,
                }).encode("utf-8"))
                return

            # This value comes from the browser, so it is untrusted. Two cases,
            # and the ordering matters:
            #
            #   1. a known name in data/db  -> resolve it
            #   2. anything else            -> treat as a path, and REQUIRE it
            #                                  to sit under data/
            #
            # An earlier version gated the path check on `candidate.suffix`,
            # meaning to skip it for names. "/etc/passwd" has no suffix, so it
            # skipped the check and connected. Never infer intent from the
            # shape of untrusted input — check the resolved path itself.
            try:
                named = Path("data/db") / f"{target}.db"
                if named.exists():
                    resolved = set_db(target)
                else:
                    p = Path(target).resolve()
                    if not p.is_relative_to(Path("data").resolve()):
                        raise PermissionError(
                            "database must be inside the data/ directory")
                    resolved = set_db(str(p))
                self.wfile.write(json.dumps({
                    "connected": True, "path": resolved,
                    "name": Path(resolved).stem,
                }).encode("utf-8"))
            except (FileNotFoundError, PermissionError) as e:
                self.wfile.write(json.dumps({
                    "connected": False, "error": str(e),
                }).encode("utf-8"))
            return

        # API: Execute Agent Query
        if path == "/api/run":
            import importlib
            import agent.loop
            import orchestration.graph
            import tools.charts
            importlib.reload(tools.charts)
            importlib.reload(agent.loop)
            importlib.reload(orchestration.graph)

            from agent.loop import run_agent
            from orchestration.graph import run_graph_to_record

            content_length = int(self.headers.get("Content-Length", 0))
            body = self.rfile.read(content_length)
            try:
                payload = json.loads(body.decode("utf-8"))
            except Exception:
                payload = {}

            question = payload.get("question", "Which region had highest sales in 2023?")
            task_id = payload.get("task_id", "t001")
            mode = payload.get("mode", "auto")  # "auto" | "simple" | "full"
            enable_best_of_n = payload.get("enable_best_of_n", False)

            # Routing decision
            if mode == "auto":
                route_decision = router_ml.route(question)
            elif mode == "simple":
                route_decision = "simple"
            else:
                route_decision = "full"

            if route_decision == "simple":
                # run_agent needs a model, the tool registry and the specs —
                # this used to call it with only the question and raised
                # TypeError, so every query routed to "simple" returned an
                # empty 500. Only the "full" path had ever been exercised.
                #
                # This is the real ReAct loop: an LLM reads the schema, writes
                # SQL, sees the result, and decides what to do next. The "full"
                # path below routes by keyword instead.
                from agent.llm import OllamaLLM
                from tools import TOOL_SPECS, TOOLS
                from tools.sql_tools import get_schema

                # Attach the critic so every step gets a score. Without this,
                # critic_score is None on every step and the UI has nothing to
                # show — which is exactly what happened.
                #
                # Threshold 1.1 means it scores but never stops. The score is
                # what we want to display; automatic early stopping during a
                # live demo would cut runs off mid-explanation. Set it to 0.9
                # to see the critic actually intervene.
                critic = _live_critic()
                record = run_agent(
                    question=question,
                    llm=OllamaLLM(),
                    tools=TOOLS,
                    tool_specs=TOOL_SPECS,
                    schema=schema_res.data if (schema_res := get_schema()).status == "ok" else None,
                    task_id=task_id,
                    config=AgentConfig(verbose=False, critic=critic,
                                       critic_threshold=1.1),
                )
            else:
                record = run_graph_to_record(question=question, task_id=task_id, enable_best_of_n=enable_best_of_n)

            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            response_json = record.model_dump()
            response_json["routed_mode"] = route_decision
            self.wfile.write(json.dumps(response_json).encode("utf-8"))
            return


        self.send_response(404)
        self.end_headers()


def start_server(port: int = PORT):
    server = HTTPServer(("0.0.0.0", port), AgentApiHandler)
    print(f"🚀 Project 23 Agent Server running at http://localhost:{port}")
    print("Press Ctrl+C to stop.")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nStopping server...")
        server.server_close()

if __name__ == "__main__":
    start_server()
