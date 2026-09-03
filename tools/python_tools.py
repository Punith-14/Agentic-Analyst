# tools/python_tools.py
"""Dhrub (Layer A) - Secure Python REPL Executor.
Executes Python code in a restricted scope with safe built-ins and math/data functions.
Follows Contract 1 (ToolResult) and safety constraints.
"""
import io
import json
import subprocess
import sys
import time
import math
import traceback
import contextlib
from typing import Any, Dict, Optional
from contracts import ToolResult

TIMEOUT_SECONDS = 10


def python_repl(code: str, timeout: int = TIMEOUT_SECONDS, **kwargs) -> ToolResult:
    """Restricted Python executor with a real timeout.

    - Runs in a separate process, killed after `timeout` seconds
    - Blocks obvious dangerous imports
    - Captures printed output or a `result` variable
    - Never raises; returns ToolResult(status="error") instead

    NOT A SECURITY SANDBOX, despite what this docstring used to claim. The
    import block is string matching and is bypassable — in CPython
    `print.__self__` is the builtins module, so `print.__self__.__import__` is
    one hop from anything. It stops the model doing something silly by
    accident. It does not stop someone trying.

    If server.py ever exposes this to untrusted users, it needs a real sandbox
    (container, seccomp, or a restricted interpreter), not this.

    THE TIMEOUT IS THE PART THAT MATTERS. Previously the docstring promised
    ten seconds and the code measured elapsed time without ever enforcing it,
    so `while True:` from the model hung the process forever — taking down an
    overnight generation run, or a live demo, with no error and no output.
    """
    start_time = time.time()

    # 🔒 Security Guardrail: Block dangerous modules and filesystem access
    forbidden_modules = [
        "os", "sys", "subprocess", "shutil", "socket", "pathlib", 
        "builtins", "requests", "importlib", "urllib", "http", "eval", "exec"
    ]
    
    code_lower = code.lower()
    for mod in forbidden_modules:
        # Check import variations
        if f"import {mod}" in code_lower or f"from {mod}" in code_lower or f"__import__('{mod}')" in code_lower:
            duration_ms = int((time.time() - start_time) * 1000)
            return ToolResult(
                status="error",
                tool="python_repl",
                error=f"Security Violation: Importing '{mod}' module is restricted for safety.",
                error_full=f"Blocked attempt to access restricted module: {mod}",
                error_category="permission",
                duration_ms=duration_ms,
                hint="Use pure Python data processing or standard math/json operations only."
            )

    # Capture standard output (print statements)
    stdout_buffer = io.StringIO()

    # Restricted, safe globals
    safe_globals = {
        "__builtins__": {
            "print": print,
            "range": range,
            "len": len,
            "int": int,
            "float": float,
            "str": str,
            "list": list,
            "dict": dict,
            "set": set,
            "tuple": tuple,
            "sum": sum,
            "min": min,
            "max": max,
            "abs": abs,
            "round": round,
            "zip": zip,
            "enumerate": enumerate,
            "sorted": sorted,
            "isinstance": isinstance,
            "bool": bool,
            "any": any,
            "all": all,
            "map": map,
            "filter": filter,
            "True": True,
            "False": False,
            "None": None,
        },
        "math": math,
    }
    
    # A plain subprocess, NOT multiprocessing.
    #
    # multiprocessing with spawn (the only mode on Windows) re-imports the
    # parent's __main__ module in every child. Under pytest, __main__ is the
    # pytest runner — so the child re-ran pytest, which collected the tests,
    # which called this function, which spawned another child. That is a fork
    # bomb, and it took a machine down.
    #
    # subprocess re-imports nothing: it runs `python -c <runner>` and passes the
    # user's code in on stdin. Killable, no shared state, identical behaviour
    # on Windows and Linux.
    #
    # A thread would not work either — Python cannot interrupt a thread stuck
    # in `while True`, so a thread-based timeout returns while the thread runs
    # on forever, leaking a core per bad query.
    try:
        proc = subprocess.run(
            [sys.executable, "-I", "-c", _RUNNER],
            input=code, capture_output=True, text=True, timeout=timeout,
        )
    except subprocess.TimeoutExpired:
        return ToolResult(
            status="error",
            tool="python_repl",
            error=f"Execution timed out after {timeout}s.",
            error_full=f"process killed; code was:\n{code[:500]}",
            error_category="timeout",
            duration_ms=int((time.time() - start_time) * 1000),
            hint="Avoid unbounded loops. Work on the query results directly.",
        )

    duration_ms = int((time.time() - start_time) * 1000)

    try:
        res = json.loads(proc.stdout)
    except Exception:                                    # noqa: BLE001
        # The child died without printing a result — a segfault, or the OS
        # killed it. Report it rather than pretending it succeeded.
        return ToolResult(
            status="error", tool="python_repl",
            error="Execution failed without producing a result.",
            error_full=f"exit code {proc.returncode}\nstderr:\n{proc.stderr[:500]}",
            error_category="runtime", duration_ms=duration_ms,
            hint="Simplify the code and try again.",
        )

    if res["ok"]:
        return ToolResult(status="ok", tool="python_repl",
                          data=res["output"], duration_ms=duration_ms)

    return ToolResult(
        status="error", tool="python_repl",
        error=res["error"][:200], error_full=res["traceback"],
        error_category=res["category"], duration_ms=duration_ms,
        hint=res["hint"],
    )


# The program the child interpreter runs. Reads the user's code from stdin,
# executes it in the restricted namespace, prints one JSON object to stdout.
#
# A string rather than an imported function on purpose: `python -c` with -I
# (isolated) starts a clean interpreter that imports nothing from this project,
# so there is no path back into pytest, the server, or this module.
_RUNNER = r'''
import contextlib, io, json, math, sys, traceback

code = sys.stdin.read()

safe_globals = {
    "__builtins__": {
        "print": print, "range": range, "len": len, "int": int,
        "float": float, "str": str, "list": list, "dict": dict,
        "set": set, "tuple": tuple, "sum": sum, "min": min, "max": max,
        "abs": abs, "round": round, "zip": zip, "enumerate": enumerate,
        "sorted": sorted, "isinstance": isinstance, "bool": bool,
        "any": any, "all": all, "map": map, "filter": filter,
        "True": True, "False": False, "None": None,
    },
    "math": math,
}
safe_locals = {}
buf = io.StringIO()

try:
    with contextlib.redirect_stdout(buf):
        exec(code, safe_globals, safe_locals)
    out = buf.getvalue().strip()
    if not out:
        out = (str(safe_locals["result"]) if "result" in safe_locals
               else "Code executed successfully with no output.")
    res = {"ok": True, "output": out}
except SyntaxError as e:
    res = {"ok": False, "error": "SyntaxError: %s at line %s" % (e.msg, e.lineno),
           "traceback": traceback.format_exc(), "category": "syntax",
           "hint": "Check Python syntax and indentation."}
except TypeError as e:
    res = {"ok": False, "error": "TypeError: %s" % e,
           "traceback": traceback.format_exc(), "category": "type_error",
           "hint": "Verify variable data types."}
except BaseException as e:
    res = {"ok": False, "error": "Python Execution Error: %s" % e,
           "traceback": traceback.format_exc(), "category": "runtime",
           "hint": "Check variable definitions and function arguments."}

sys.__stdout__.write(json.dumps(res))
'''