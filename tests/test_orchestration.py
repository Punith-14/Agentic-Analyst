# tests/test_orchestration.py
"""Tests for Harish's Orchestration Layer (Layer D).
Tests ComplexityRouter, 4-node LangGraph State Machine, Best-of-N selection,
and RunRecord emission.
"""
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

from contracts import RunRecord, TrajectoryStep, VALID_TERMINATIONS
from orchestration.router import ComplexityRouter
from orchestration.graph import build_graph, run_graph_to_record

def test_complexity_router_classification():
    router = ComplexityRouter()
    
    # Simple query should route to 'simple'
    simple_res = router.route("Which region had the highest total sales in 2023?")
    assert simple_res == "simple"

    # Complex query should route to 'full'
    complex_res = router.route("Compare total sales across regions and join customer segment with correlation regression")
    assert complex_res == "full"

def test_complexity_router_evaluation():
    router = ComplexityRouter()
    eval_res = router.evaluate_against_baseline()
    assert eval_res["router_accuracy"] >= 0.85
    assert eval_res["latency_saved_percent"] > 20.0
    assert "comparison_table" in eval_res

def test_langgraph_execution_to_run_record():
    """Test full LangGraph execution producing a Contract 4 RunRecord."""
    record = run_graph_to_record(
        question="Which region had the highest total sales in 2023?",
        task_id="t001",
        enable_best_of_n=False
    )
    
    assert isinstance(record, RunRecord)
    assert record.termination in VALID_TERMINATIONS
    assert len(record.steps) >= 2
    assert record.final_answer is not None
    assert all(isinstance(s, TrajectoryStep) for s in record.steps)

def test_langgraph_best_of_n_toggle():
    """Test Best-of-N (N=2) step selection execution."""
    record_bon = run_graph_to_record(
        question="Which region had the highest total sales in 2023?",
        task_id="t001",
        enable_best_of_n=True
    )
    assert isinstance(record_bon, RunRecord)
    assert len(record_bon.steps) > 0