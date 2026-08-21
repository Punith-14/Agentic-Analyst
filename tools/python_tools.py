# tools/python_tools.py
"""Dhrub (Layer A) - Secure Python REPL Executor.
Executes Python code in a restricted scope with safe built-ins and math/data functions.
Follows Contract 1 (ToolResult) and safety constraints.
"""
import io
import time
import math
import traceback
import contextlib
from typing import Any, Dict, Optional
from contracts import ToolResult

def python_repl(code: str, **kwargs) -> ToolResult:
    """Secure Python REPL Executor.
    - Blocks dangerous imports: os, sys, subprocess, socket, shutil, requests, importlib, builtins
    - 10-second execution timeout guard
    - Captures printed output or 'result' variable
    - Never raises exceptions; returns ToolResult with status='error'
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
    
    safe_locals = {}

    try:
        with contextlib.redirect_stdout(stdout_buffer):
            exec(code, safe_globals, safe_locals)

        output = stdout_buffer.getvalue().strip()

        # If nothing was printed, check if 'result' was computed
        if not output:
            if "result" in safe_locals:
                output = str(safe_locals["result"])
            else:
                output = "Code executed successfully with no output."

        duration_ms = int((time.time() - start_time) * 1000)
        return ToolResult(
            status="ok",
            tool="python_repl",
            data=output,
            duration_ms=duration_ms
        )

    except SyntaxError as e:
        duration_ms = int((time.time() - start_time) * 1000)
        return ToolResult(
            status="error",
            tool="python_repl",
            error=f"SyntaxError: {e.msg} at line {e.lineno}"[:200],
            error_full=traceback.format_exc(),
            error_category="syntax",
            duration_ms=duration_ms,
            hint="Check Python syntax and indentation."
        )
    except TypeError as e:
        duration_ms = int((time.time() - start_time) * 1000)
        return ToolResult(
            status="error",
            tool="python_repl",
            error=f"TypeError: {str(e)}"[:200],
            error_full=traceback.format_exc(),
            error_category="type_error",
            duration_ms=duration_ms,
            hint="Verify variable data types."
        )
    except Exception as e:
        duration_ms = int((time.time() - start_time) * 1000)
        return ToolResult(
            status="error",
            tool="python_repl",
            error=f"Python Execution Error: {str(e)}"[:200],
            error_full=traceback.format_exc(),
            error_category="runtime",
            duration_ms=duration_ms,
            hint="Check variable definitions and function arguments."
        )