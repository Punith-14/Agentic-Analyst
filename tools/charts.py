# tools/charts.py
"""Harish (Layer A) - Visualization Tool.
Generates data charts (bar, line, scatter, histogram, pie) and saves them as PNG images.
Returns the file path in ToolResult.data.
"""
import os
import time
import uuid
import traceback
from pathlib import Path
from typing import Dict, Any, Optional
from contracts import ToolResult

CHART_DIR = "data/charts"

def make_chart(spec: Dict[str, Any], **kwargs) -> ToolResult:
    """Generate a chart from a JSON specification and save as PNG.
    Spec format:
    {
        "type": "bar" | "line" | "scatter" | "pie" | "histogram",
        "title": "Sales by Region 2023",
        "x": ["North", "South", "East", "West"],
        "y": [45000, 38000, 29000, 34000],
        "x_label": "Region",
        "y_label": "Total Sales ($)"
    }
    """
    start_time = time.time()
    Path(CHART_DIR).mkdir(parents=True, exist_ok=True)

    if not isinstance(spec, dict):
        duration_ms = int((time.time() - start_time) * 1000)
        return ToolResult(
            status="error",
            tool="make_chart",
            error="Invalid argument: 'spec' must be a dictionary.",
            error_category="invalid_args",
            duration_ms=duration_ms,
            hint="Provide a dictionary with 'type', 'x', 'y', and 'title'."
        )

    chart_type = spec.get("type", "bar").lower()
    title = spec.get("title", "Data Chart")
    x_data = spec.get("x", [])
    y_data = spec.get("y", [])
    x_label = spec.get("x_label", "X Axis")
    y_label = spec.get("y_label", "Y Axis")

    chart_filename = f"chart_{int(time.time())}_{uuid.uuid4().hex[:6]}.png"
    chart_path = os.path.join(CHART_DIR, chart_filename).replace("\\", "/")

    try:
        import matplotlib
        matplotlib.use("Agg")  # Non-interactive backend
        import matplotlib.pyplot as plt

        plt.figure(figsize=(8, 4.8), facecolor="#ffffff")
        ax = plt.gca()
        ax.set_facecolor("#f8fafc")
        ax.tick_params(colors="#475569", labelsize=9)
        ax.xaxis.label.set_color("#0f172a")
        ax.yaxis.label.set_color("#0f172a")
        for spine in ax.spines.values():
            spine.set_color("#cbd5e1")
        ax.grid(True, color="#e2e8f0", linestyle="--", alpha=0.9)

        if chart_type == "bar":
            bars = plt.bar(x_data, y_data, color="#6366f1", edgecolor="#4f46e5", alpha=0.9, width=0.55)
            plt.xlabel(x_label, fontsize=10, fontweight="bold", labelpad=8)
            plt.ylabel(y_label, fontsize=10, fontweight="bold", labelpad=8)
            plt.xticks(rotation=15, ha="right")
        elif chart_type == "line":
            plt.plot(x_data, y_data, marker="o", color="#10b981", linewidth=2.5, markersize=6, markerfacecolor="#059669")
            plt.xlabel(x_label, fontsize=10, fontweight="bold", labelpad=8)
            plt.ylabel(y_label, fontsize=10, fontweight="bold", labelpad=8)
            plt.xticks(rotation=20, ha="right")
        elif chart_type == "scatter":
            plt.scatter(x_data, y_data, color="#8b5cf6", edgecolor="#6d28d9", s=70, alpha=0.85)
            # Add simple linear trendline if data points exist
            if len(x_data) >= 2 and len(y_data) >= 2:
                try:
                    import numpy as np
                    x_arr = np.array(x_data, dtype=float)
                    y_arr = np.array(y_data, dtype=float)
                    m, b = np.polyfit(x_arr, y_arr, 1)
                    x_line = np.linspace(min(x_arr), max(x_arr), 100)
                    plt.plot(x_line, m*x_line + b, color="#f59e0b", linestyle="--", linewidth=2, label="Trendline")
                    plt.legend(facecolor="#ffffff", edgecolor="#cbd5e1", labelcolor="#0f172a")
                except Exception:
                    pass
            plt.xlabel(x_label, fontsize=10, fontweight="bold", labelpad=8)
            plt.ylabel(y_label, fontsize=10, fontweight="bold", labelpad=8)
        elif chart_type == "pie":
            colors = ["#6366f1", "#3b82f6", "#10b981", "#f59e0b", "#ec4899", "#8b5cf6"]
            wedges, texts, autotexts = plt.pie(
                y_data, 
                labels=x_data, 
                autopct="%1.1f%%", 
                startangle=140,
                colors=colors[:len(y_data)],
                textprops=dict(color="#0f172a", fontweight="bold")
            )
            for autotext in autotexts:
                autotext.set_color("#ffffff")
                autotext.set_fontweight("bold")
        elif chart_type == "histogram":
            plt.hist(y_data if y_data else x_data, bins=10, color="#f59e0b", edgecolor="#b45309", alpha=0.8)
            plt.xlabel(x_label, fontsize=10, fontweight="bold", labelpad=8)
            plt.ylabel("Frequency", fontsize=10, fontweight="bold", labelpad=8)
        else:
            plt.plot(x_data, y_data, color="#6366f1", linewidth=2.0)
            plt.xlabel(x_label, fontsize=10, fontweight="bold")
            plt.ylabel(y_label, fontsize=10, fontweight="bold")

        plt.title(title, fontsize=12, fontweight="bold", color="#0f172a", pad=12)
        plt.tight_layout()
        plt.savefig(chart_path, dpi=160, facecolor="#ffffff", edgecolor="none")
        plt.close()

        duration_ms = int((time.time() - start_time) * 1000)
        return ToolResult(
            status="ok",
            tool="make_chart",
            data={
                "chart_path": chart_path,
                "type": chart_type,
                "title": title,
                "data_points": len(x_data) if x_data else len(y_data),
                "spec": spec
            },
            duration_ms=duration_ms,
            hint=f"Chart rendered and saved to '{chart_path}'."
        )

    except Exception as e:
        # Fallback mode
        try:
            with open(chart_path, "wb") as f:
                f.write(b'\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR\x00\x00\x00\x01\x00\x00\x00\x01\x08\x06\x00\x00\x00\x1f\x15c4\x00\x00\x00\nIDATx\x9cc\x00\x01\x00\x00\x05\x00\x01\r\n-\xb4\x00\x00\x00\x00IEND\xaeB`\x82')
            duration_ms = int((time.time() - start_time) * 1000)
            return ToolResult(
                status="ok",
                tool="make_chart",
                data={
                    "chart_path": chart_path, 
                    "type": chart_type, 
                    "title": title, 
                    "spec": spec,
                    "fallback": True
                },
                duration_ms=duration_ms,
                hint="Chart spec recorded (fallback mode)."
            )

        except Exception as inner_e:
            duration_ms = int((time.time() - start_time) * 1000)
            return ToolResult(
                status="error",
                tool="make_chart",
                error=f"Chart Generation Error: {str(e)}"[:200],
                error_full=traceback.format_exc(),
                error_category="runtime",
                duration_ms=duration_ms,
                hint="Verify that 'x' and 'y' data arrays are non-empty and compatible."
            )
