# tests/test_classifier.py
"""Tests for Dhrub's Tool-Selection ML Classifier."""
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

from tools.classifier import ToolSelectionClassifier

def test_tool_classifier_predictions():
    clf = ToolSelectionClassifier()
    
    # Test DB query prediction
    p1 = clf.predict("Which region had the highest total sales in 2023?")
    assert p1 == "run_sql"

    # Test schema inspection prediction
    p2 = clf.predict("Show database tables and column schema", step_index=1)
    assert p2 == "get_schema"

    # Test visualization prediction
    p3 = clf.predict("Plot a bar chart of 2023 sales")
    assert p3 == "make_chart"

    # Test stats prediction
    p4 = clf.predict("Check correlation and p-value between sales and profit")
    assert p4 == "stats_test"

    # Test ML regression prediction
    p5 = clf.predict("Train regression model to predict sales from quantity")
    assert p5 == "ml_regress"

def test_tool_classifier_evaluation():
    clf = ToolSelectionClassifier()
    eval_res = clf.evaluate_against_baseline()
    
    assert eval_res["classifier_accuracy"] >= 0.85
    assert eval_res["classifier_latency_ms"] < 10.0  # sub-10ms requirement
    assert eval_res["speedup_factor"] > 10.0
    assert "comparison_table" in eval_res
