"""End-to-end check: does the whole thing actually work?

    python scripts/smoke_test.py              # backend, ~30 seconds
    python scripts/smoke_test.py --with-agent # also runs a real query (needs Ollama)

Starts the server, exercises every endpoint, and prints a manual checklist for
the parts only a browser can verify.

THE FUNCTIONS HERE ARE check_*, NOT test_*, ON PURPOSE
This file matches pytest's default *_test.py pattern. When its functions
were named test_*, pytest collected them and ran check_unit_tests(), which
shells out to pytest, which collected this file again... until the machine
died. pytest.ini now restricts collection to tests/, and these names are
the second line of defence.

WHY THIS EXISTS SEPARATELY FROM pytest
The unit tests call functions. This calls the running server over HTTP, which
is what the frontend does — and that is where the interesting failures live.
The path-traversal hole in /api/database passed every unit test and was only
found by posting "/etc/passwd" at a live server.
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
BASE = "http://localhost:8000"

PASS, FAIL = 0, 0


def check(name: str, ok: bool, detail: str = "") -> bool:
    global PASS, FAIL
    if ok:
        PASS += 1
        print(f"  \033[92mPASS\033[0m  {name}")
    else:
        FAIL += 1
        print(f"  \033[91mFAIL\033[0m  {name}")
        if detail:
            print(f"        {detail}")
    return ok


def get(path: str) -> dict:
    with urllib.request.urlopen(BASE + path, timeout=30) as r:
        return json.loads(r.read())


def post(path: str, payload: dict, timeout: int = 30) -> dict:
    req = urllib.request.Request(
        BASE + path, data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"}, method="POST")
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read())


def section(title: str) -> None:
    print(f"\n\033[1m{title}\033[0m")
    print("─" * 66)


# ======================================================================


def check_unit_tests() -> None:
    section("1. Unit tests")
    # Streamed, not captured. 140 tests take 30-60s and a silent subprocess
    # looks identical to a hung one.
    print("  running pytest (30-60s, --skip-unit to skip) ...\n")
    r = subprocess.run([sys.executable, "-m", "pytest", "-q", "--tb=line"],
                       cwd=ROOT, text=True)
    print()
    check("pytest suite", r.returncode == 0,
          "run `pytest -q` on its own to see which tests failed")


def check_no_database_selected() -> None:
    section("2. No dataset connected — must refuse, never invent")

    d = get("/api/database")
    check("startup has no database connected", d["connected"] is False,
          f"connected={d['connected']} path={d.get('path')}")
    check("server lists databases the UI can offer",
          isinstance(d.get("available"), list) and len(d["available"]) > 0,
          f"available={d.get('available')}")

    s = get("/api/schema")
    check("get_schema refuses without a database", s["status"] == "error",
          f"status={s['status']}")
    check("the refusal is readable by a user",
          "upload" in (s.get("error", "") + (s.get("hint") or "")).lower(),
          f"error={s.get('error')}")
    check("no data is returned", s.get("data") is None)


def check_connect_and_schema() -> None:
    section("3. Connecting a dataset")

    d = post("/api/database", {"name": "chinook_1"})
    check("connect by name", d.get("connected") is True, json.dumps(d))

    s = get("/api/schema")
    check("schema returns ok", s["status"] == "ok")
    check("EVERY table is returned, not just names",
          isinstance(s["data"], dict) and "Album" in s["data"],
          f"data keys={list(s['data'])[:5] if isinstance(s['data'], dict) else s['data']}")
    check("tables include their columns",
          "columns" in s["data"].get("Album", {}),
          "this is the fix worth 75% -> 13% SQL errors")
    check("foreign keys are included",
          "foreign_keys" in s["data"].get("Album", {}))

    post("/api/database", {"name": "college_2"})
    s2 = get("/api/schema")
    check("switching database switches the schema",
          "student" in s2["data"] and "Album" not in s2["data"],
          f"tables={sorted(s2['data'])[:4]}")

    post("/api/database", {})
    d3 = get("/api/database")
    check("disconnect works (the UI 'change' button)",
          d3["connected"] is False)


def check_security() -> None:
    section("4. Security — untrusted paths from the browser")

    attacks = ["/etc/passwd", "../../../etc/passwd", "/etc/hosts",
               "data/../../../etc/passwd", "C:/Windows/System32/config/SAM",
               "~/.ssh/id_rsa"]
    for path in attacks:
        d = post("/api/database", {"path": path})
        check(f"refuses {path}", d.get("connected") is False,
              f"CONNECTED TO {path} — path traversal is open")

    d = post("/api/database", {"path": "data/db/chinook_1.db"})
    check("but a legitimate path still works", d.get("connected") is True)

    post("/api/database", {"name": "chinook_1"})


def check_python_timeout() -> None:
    section("5. python_repl timeout — the demo-killer")

    sys.path.insert(0, str(ROOT))
    from tools.python_tools import python_repl

    t0 = time.perf_counter()
    r = python_repl("while True: pass", timeout=3)
    elapsed = time.perf_counter() - t0

    check("an infinite loop is killed, not left hanging",
          r.status == "error" and r.error_category == "timeout",
          f"status={r.status} category={r.error_category}")
    check(f"returned in {elapsed:.1f}s rather than never", elapsed < 8,
          f"took {elapsed:.1f}s")
    check("normal code still runs",
          python_repl("result = sum(range(10))").data == "45")


def check_ui_served() -> None:
    section("6. Frontend is served, and shows live data")
    try:
        with urllib.request.urlopen(BASE + "/", timeout=15) as r:
            html = r.read().decode("utf-8", "replace")
        check("index.html is served", r.status == 200 and len(html) > 1000,
              f"{len(html)} bytes")

        for token, why in [
            ("/api/database", "asks which dataset is connected"),
            ("/api/schema", "Schema tab reads the live database"),
            ("/api/tasks", "task suite comes from the real question set"),
            ("/api/critic-metrics", "ML Engine shows the trajectory critic"),
            ("dbInfo.connected", "the Run button is gated on it"),
            ("pipelineStage", "the stage bar follows the run"),
            # No check for a live per-step critic score: it is computed and
            # stored, but deliberately not displayed. duration_ms is one of
            # the critic's features and a cold Ollama start makes step 0 take
            # ~8.6s — slower than 99.9% of training steps — so the critic
            # reads a normal cold start as a doomed run. Measured results are
            # on the ML Engine tab, where conditions match training.
        ]:
            check(f"UI calls {token}", token in html, why)

        # Four panels used to display invented content. Assert it stays gone.
        #
        # Comments are stripped first — the code explaining *why* these were
        # removed naturally mentions them by name, and matching that would
        # fail forever.
        code = re.sub(r"//[^\n]*", "", html)
        code = re.sub(r"/\*.*?\*/", "", code, flags=re.S)

        for token, what in [
            ("FALLBACK_TRAJECTORIES", "a canned successful run"),
            ("PRESET_TASKS", "hardcoded questions for the wrong database"),
            ("order_id", "a hardcoded schema for tables that may not exist"),
            ("180500", "fabricated chart data"),
        ]:
            check(f"no hardcoded {what}", token not in code,
                  f"{token!r} is back in index.html (outside a comment)")
    except urllib.error.URLError as e:
        check("index.html is served", False, str(e))


def check_critic_metrics() -> None:
    section("7. Critic metrics endpoint")
    try:
        d = get("/api/critic-metrics")
    except Exception as e:                                   # noqa: BLE001
        check("/api/critic-metrics responds", False, f"{type(e).__name__}: {e}")
        return

    check("test-set results present",
          d.get("test", {}).get("ensemble_pr_auc") is not None,
          json.dumps(d)[:160])
    check("holdout results present",
          d.get("holdout", {}).get("retained") is not None)
    check("early-stopping measurement read from disk",
          d.get("early_stop") is not None,
          "run scripts/measure_early_stop.py to generate it")
    if d.get("early_stop"):
        e = d["early_stop"]
        print(f"        PR-AUC {d['test']['ensemble_pr_auc']} · "
              f"{e['steps_saved_pct']:.0%} steps saved · "
              f"{e['answers_lost']} answers lost · "
              f"{d['holdout']['retained']:.0%} retained on {d['holdout']['database']}")


def check_real_agent() -> None:
    section("8. A real query end to end (needs Ollama)")
    post("/api/database", {"name": "chinook_1"})
    print("  running a question through the agent, up to 2 minutes ...")
    t0 = time.perf_counter()
    try:
        rec = post("/api/run", {"question": "How many albums are there?"},
                   timeout=180)
    except Exception as e:                                   # noqa: BLE001
        check("agent answered", False, f"{type(e).__name__}: {e}")
        return

    check(f"agent answered in {time.perf_counter()-t0:.0f}s",
          bool(rec.get("final_answer")), json.dumps(rec)[:200])
    check("the trajectory was recorded", len(rec.get("steps", [])) > 0)
    print(f"        answer: {rec.get('final_answer')}")
    print(f"        steps : {[s['action']['tool'] for s in rec.get('steps', []) if s.get('action')]}")


# ======================================================================


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--with-agent", action="store_true",
                    help="also run a real question (needs Ollama running)")
    ap.add_argument("--skip-unit", action="store_true")
    args = ap.parse_args()

    print("=" * 66)
    print("  SMOKE TEST — backend + frontend")
    print("=" * 66)

    if not args.skip_unit:
        check_unit_tests()

    print("\nstarting server ", end="", flush=True)
    srv = subprocess.Popen([sys.executable, "server.py"], cwd=ROOT,
                           stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                           text=True)
    try:
        for _ in range(30):                     # wait for it to accept requests
            time.sleep(1)
            print(".", end="", flush=True)
            try:
                get("/api/database")
                print(" ready")
                break
            except Exception:                   # noqa: BLE001
                if srv.poll() is not None:
                    print("\n\n  server exited during startup:\n")
                    print(srv.stdout.read())
                    sys.exit(1)
        else:
            print("\n  server never became ready — is port 8000 already in use?")
            print("  check with:  netstat -ano | findstr :8000")
            sys.exit(1)

        check_no_database_selected()
        check_connect_and_schema()
        check_security()
        check_python_timeout()
        check_ui_served()
        check_critic_metrics()
        if args.with_agent:
            check_real_agent()

    finally:
        srv.terminate()
        try:
            srv.wait(timeout=5)
        except subprocess.TimeoutExpired:
            srv.kill()

    print("\n" + "=" * 66)
    print(f"  {PASS} passed, {FAIL} failed")
    print("=" * 66)

    print("""
  CHECK IN A BROWSER — these need eyes, not asserts:

    python server.py     then open  http://localhost:8000

    [ ] An amber "No dataset connected" panel appears on load
    [ ] The Run button is greyed out and cannot be clicked
    [ ] Clicking "chinook_1" turns the panel green
    [ ] The Run button becomes clickable
    [ ] Asking "How many albums are there?" returns 347
    [ ] "change" returns you to the amber panel
    [ ] Refreshing the page keeps the connection (server-side state)
""")

    sys.exit(1 if FAIL else 0)


if __name__ == "__main__":
    main()
