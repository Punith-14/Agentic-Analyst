# server.py
"""Harish (Layer D) - API Backend & Development Server.
Provides REST & SSE endpoints for live execution, trajectory replay, ML benchmarks,
and schema inspection. Serves the React UI.
"""
import os
import json
import glob
import mimetypes
from urllib.parse import urlparse, parse_qs
from http.server import HTTPServer, SimpleHTTPRequestHandler
from pathlib import Path
from typing import Dict, Any, List

from contracts import RunRecord, TrajectoryStep, ToolResult
from orchestration.graph import run_graph_to_record
from orchestration.router import ComplexityRouter
from agent.loop import run_agent
from tools.classifier import ToolSelectionClassifier
from tools.sql_tools import get_schema
from create_db import create_analytics_database

PORT = 8000
DB_PATH = "data/db/analytics.db"

# Initialize DB & Chart Storage on server start
if not os.path.exists(DB_PATH):
    create_analytics_database(DB_PATH)
Path("data/charts").mkdir(parents=True, exist_ok=True)

router_ml = ComplexityRouter()

tool_classifier_ml = ToolSelectionClassifier()


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
                "dhrub_tool_classifier": tool_eval,
                "harish_complexity_router": router_eval
            }
            self.wfile.write(json.dumps(data).encode("utf-8"))
            return

        # 5. API: Schema Inspection
        if path == "/api/schema":
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            params = parse_qs(parsed_url.query)
            table_param = params.get("table", [None])[0]
            schema_res = get_schema(table=table_param, db_path=DB_PATH)
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
                record = run_agent(question=question, task_id=task_id)
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
