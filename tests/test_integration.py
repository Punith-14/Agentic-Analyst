# tests/test_integration.py
"""Section 3.5: Weekly Integration Test.
Tests all four layers end-to-end against the frozen contracts.
"""
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

from contracts import RunRecord, TrajectoryStep, VALID_TERMINATIONS
from orchestration.graph import build_graph, run_graph_to_record

def test_end_to_end():
    """End-to-end integration test from Section 3.5 of the specification."""
    # Run the full orchestrated graph
    result = run_graph_to_record(
        question="Which region had the highest sales in 2023?",
        task_id="t001"
    )

    assert isinstance(result, RunRecord)
    assert result.termination in VALID_TERMINATIONS
    assert all(isinstance(s, TrajectoryStep) for s in result.steps)
    assert result.model_name and result.quantisation

    # Verify every observation obeys the 20-row truncation contract
    for s in result.steps:
        if s.observation and s.observation.data and isinstance(s.observation.data, list):
            assert len(s.observation.data) <= 20

    print("✅ Integration test passed: Graph produced valid RunRecord conforming to all 6 Contracts.")

if __name__ == "__main__":
    test_end_to_end()
