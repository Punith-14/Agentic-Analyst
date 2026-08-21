# agent/critic/infer.py
"""Punith (Layer B) - Learned Trajectory Critic Inference Module.
Scores trajectory steps and recommends early stopping to cut unnecessary steps.
"""
from typing import List
from contracts import TrajectoryStep

class Critic:
    """Learned Trajectory Critic (Punith's ML Component).
    Scores step quality (0.0 - 1.0) and decides if the run should terminate early.
    """
    version: str = "v1.0-deberta-critic"

    def __init__(self, version: str = "v1.0-deberta-critic"):
        self.version = version

    def score_step(self, history: List[TrajectoryStep], step: TrajectoryStep) -> float:
        """Score a single trajectory step between 0.0 (detrimental) and 1.0 (optimal).
        Exploratory recovery steps are scored 0.5 as required by contract.
        """
        # If observation is successful and has meaningful data
        if step.observation:
            if step.observation.status == "ok":
                # High score if schema found or SQL returned rows
                if step.observation.data:
                    return 0.95
                return 0.80
            elif step.observation.status == "error":
                # Check if error has recovery hint (exploratory recovery step -> 0.5)
                if step.observation.hint:
                    return 0.50
                return 0.20

        # Action is final answer
        if step.action and step.action.is_final:
            return 1.00

        return 0.70

    def should_stop(self, history: List[TrajectoryStep]) -> bool:
        """Determine whether the trajectory should terminate early due to degradation."""
        if not history:
            return False

        # Guard 1: Three consecutive low-scoring steps
        if len(history) >= 3:
            recent_scores = [self.score_step(history[:i], history[i]) for i in range(len(history)-3, len(history))]
            if all(s < 0.3 for s in recent_scores):
                return True

        # Guard 2: Excessive repeated actions
        last_step = history[-1]
        if last_step.repeat_count >= 2:
            return True

        return False
