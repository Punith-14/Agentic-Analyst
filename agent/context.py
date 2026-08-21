# agent/context.py
"""Punith (Layer B) - Context Management and Observation Masking.
Retains the last 3 observations in full, masking older observations to save GPU context.
"""
from typing import List
from contracts import TrajectoryStep

def apply_context_masking(history: List[TrajectoryStep], keep_last_n: int = 3) -> List[dict]:
    """Mask observations older than keep_last_n steps to conserve tokens."""
    masked_history = []
    total_steps = len(history)

    for idx, step in enumerate(history):
        step_dict = step.model_dump()
        # If older than keep_last_n, mask the detailed data payload
        if idx < (total_steps - keep_last_n):
            if step_dict.get("observation") and step_dict["observation"].get("data"):
                step_dict["observation"]["data"] = "[DATA MASKED TO SAVE CONTEXT - Summary: rows processed]"
                step_dict["observation_truncated"] = True
        masked_history.append(step_dict)

    return masked_history
