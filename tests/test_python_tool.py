import pytest
import sys
from pathlib import Path

# Ensure project root is in sys.path
sys.path.append(str(Path(__file__).parent.parent))

from tools.python_tools import python_repl


def test_python_repl_basic_execution():
    """Test standard execution with print output."""
    code = "x = 10\ny = 20\nprint(x + y)"
    res = python_repl(code)
    
    assert res.status == "ok"
    assert res.data == "30"
    assert res["status"] == "ok"
    assert res["data"] == "30"


def test_python_repl_result_variable():
    """Test retrieving 'result' variable when nothing is printed."""
    code = "a = [1, 2, 3, 4]\nresult = sum(a)"
    res = python_repl(code)
    
    assert res.status == "ok"
    assert res.data == "10"


def test_python_repl_security_block():
    """Test that importing OS or subprocess is blocked by safety guardrails."""
    code = "import os\nprint(os.getcwd())"
    res = python_repl(code)
    
    assert res.status == "error"
    assert res.error_category == "permission"
    assert "Security Violation" in res.error