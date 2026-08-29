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
if router_ml.model is None and os.path.exists("data/trajectories"):
    try:
        router_ml.train_on_trajectories()
    except Exception:
        pass

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

        # 4. API: Live Database Schema & Table Introspection
        if path.startswith("/api/schema"):
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            params = parse_qs(parsed_url.query)
            target_db = params.get("db", ["analytics.db"])[0]
            
            db_file = os.path.join("data", "db", target_db)
            if not os.path.exists(db_file):
                db_file = os.path.join("data", "db", "analytics.db")
                if not os.path.exists(db_file):
                    db_file = "data.db"

            abs_db_path = os.path.abspath(db_file)
            
            # List all available DB files in data/db
            avail_dbs = [os.path.basename(f) for f in glob.glob("data/db/*.db")]
            if not avail_dbs:
                avail_dbs = ["analytics.db"]

            schema_data = {"database": os.path.basename(db_file), "db_path": db_file, "available_databases": avail_dbs, "tables": []}
            try:
                conn = sqlite3.connect(abs_db_path, timeout=5.0)
                cursor = conn.cursor()
                cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%' ORDER BY name;")
                table_names = [r[0] for r in cursor.fetchall()]
                
                def _safe_cell(v):
                    if v is None or isinstance(v, (int, float, str, bool)):
                        return v
                    if isinstance(v, bytes):
                        return v.decode("utf-8", errors="replace")
                    return str(v)

                for t in table_names:
                    # Clean identifier
                    safe_t = t.replace('"', '""')
                    # Columns
                    cursor.execute(f'PRAGMA table_info("{safe_t}");')
                    cols = []
                    for row in cursor.fetchall():
                        cols.append({
                            "cid": row[0],
                            "name": str(row[1]),
                            "type": str(row[2] or "TEXT"),
                            "notnull": bool(row[3]),
                            "default_value": str(row[4]) if row[4] is not None else None,
                            "pk": bool(row[5])
                        })
                    
                    # Foreign Keys
                    cursor.execute(f'PRAGMA foreign_key_list("{safe_t}");')
                    fks = []
                    for fk in cursor.fetchall():
                        fks.append({
                            "from": str(fk[3]),
                            "to_table": str(fk[2]),
                            "to_col": str(fk[4])
                        })
                    
                    # Row Count
                    try:
                        cursor.execute(f'SELECT COUNT(*) FROM "{safe_t}";')
                        row_count = int(cursor.fetchone()[0])
                    except Exception:
                        row_count = 0

                    # Sample 3 rows
                    sample_rows = []
                    try:
                        cursor.execute(f'SELECT * FROM "{safe_t}" LIMIT 3;')
                        col_names = [c["name"] for c in cols]
                        for r in cursor.fetchall():
                            sample_rows.append({col_names[idx]: _safe_cell(r[idx]) for idx in range(min(len(col_names), len(r)))})
                    except Exception:
                        sample_rows = []

                    schema_data["tables"].append({
                        "name": t,
                        "row_count": row_count,
                        "columns": cols,
                        "foreign_keys": fks,
                        "sample_rows": sample_rows
                    })
                conn.close()
            except Exception as e:
                schema_data["error"] = str(e)
            
            self.wfile.write(json.dumps(schema_data).encode("utf-8"))
            return



        # 5. API: Real-Time Complexity Route Prediction
        if path == "/api/route":

            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            params = parse_qs(parsed_url.query)
            q = params.get("q", [""])[0]
            decision, conf, lat = router_ml.route_with_confidence(q)
            res = {
                "decision": decision,
                "confidence": round(conf, 3),
                "latency_ms": round(lat, 2),
                "label": "Simple Query (Fast ReAct)" if decision == "simple" else "Complex Query (Full Graph)"
            }
            self.wfile.write(json.dumps(res).encode("utf-8"))
            return

        # 5. API: ML Evaluation & Benchmark Metrics
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

        # 5b. API: Trajectories for Replay Studio
        if path == "/api/trajectories":
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            trajectories = []
            traj_file = os.path.join("data", "trajectories", "2026-08-27-holdout.jsonl")
            if os.path.exists(traj_file):
                try:
                    with open(traj_file, "r", encoding="utf-8") as f:
                        for line in f:
                            line = line.strip()
                            if line:
                                try:
                                    trajectories.append(json.loads(line))
                                    if len(trajectories) >= 25:
                                        break
                                except Exception:
                                    continue
                except Exception:
                    pass
            self.wfile.write(json.dumps(trajectories).encode("utf-8"))
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

            try:
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
            except Exception as e:
                import traceback
                traceback.print_exc()
                self.send_response(500)
                self.send_header("Content-Type", "application/json")
                self.end_headers()
                err_payload = {"status": "error", "error": str(e), "traceback": traceback.format_exc()}
                self.wfile.write(json.dumps(err_payload).encode("utf-8"))
            return


        # API: Train Complexity Router
        if path == "/api/train-router":
            res = router_ml.train_on_trajectories()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(json.dumps(res).encode("utf-8"))
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
