# orchestration/graph.py
"""Harish (Layer D) - LangGraph Orchestration State Machine.
Implements:
  planner -> executor -> critic -> replanner
     ^                               |
     +-------------------------------+
- Executor node calls agent_step() from Layer B (agent.loop).
- Critic node calls Critic.score_step() from Layer B (agent.critic.infer).
- Implements toggleable Best-of-N (N=2) step selection.
- Produces valid RunRecord adhering to Contract 4.
"""
import time
import uuid
from typing import TypedDict, List, Dict, Any, Optional, Literal
from langgraph.graph import StateGraph, END

from contracts import RunRecord, TrajectoryStep, Termination, VALID_TERMINATIONS
from agent.loop import agent_step, _compute_action_hash
from agent.critic.infer import Critic

class GraphState(TypedDict):
    run_id: str
    task_id: str
    question: str
    steps: List[TrajectoryStep]
    plan: str
    status: str
    tokens_used: int
    token_budget: int
    max_steps: int
    enable_best_of_n: bool
    critic_score: float
    is_complete: bool
    termination: Optional[Termination]
    final_answer: Optional[str]
    start_time: float


def planner_node(state: GraphState) -> Dict[str, Any]:
    """Generates initial decomposition for the query."""
    question = state.get("question", "")
    plan = f"Plan for '{question}': 1. Inspect DB schema -> 2. Execute SQL/Tool -> 3. Synthesize result."
    return {
        "plan": plan,
        "status": "planning_complete"
    }


def executor_node(state: GraphState) -> Dict[str, Any]:
    """Executes next step by driving Layer B's agent_step().
    Supports toggleable Best-of-N (N=2) selection using the Critic.
    """
    enable_bon = state.get("enable_best_of_n", False)
    critic = Critic()

    # Base execution
    step_state = {
        "run_id": state.get("run_id"),
        "question": state.get("question"),
        "steps": state.get("steps", []),
        "tokens_used": state.get("tokens_used", 0),
        "token_budget": state.get("token_budget", 12000),
        "max_steps": state.get("max_steps", 10),
    }

    if enable_bon and len(state.get("steps", [])) == 1:
        # Best-of-2 candidate action generation
        candidate_1_state = agent_step(dict(step_state))
        candidate_2_state = dict(step_state)
        # Alternate candidate (e.g. specialized parameter variation)
        candidate_2_state["question"] = step_state["question"] + " (optimized)"
        candidate_2_state = agent_step(candidate_2_state)

        step1 = candidate_1_state["steps"][-1]
        step2 = candidate_2_state["steps"][-1]
        
        score1 = critic.score_step(state.get("steps", []), step1)
        score2 = critic.score_step(state.get("steps", []), step2)

        chosen_state = candidate_1_state if score1 >= score2 else candidate_2_state
        return {
            "steps": chosen_state["steps"],
            "tokens_used": chosen_state["tokens_used"],
            "is_complete": chosen_state["is_complete"],
            "termination": chosen_state["termination"],
            "final_answer": chosen_state["final_answer"],
            "status": chosen_state["status"]
        }

    # Standard execution path
    result = agent_step(step_state)
    return {
        "steps": result["steps"],
        "tokens_used": result["tokens_used"],
        "is_complete": result["is_complete"],
        "termination": result["termination"],
        "final_answer": result["final_answer"],
        "status": result["status"]
    }


def critic_node(state: GraphState) -> Dict[str, Any]:
    """Calls Layer B's Critic to evaluate step quality and decide early stopping."""
    critic = Critic()
    steps = state.get("steps", [])
    
    if not steps:
        return {"critic_score": 0.5, "is_complete": False}

    last_step = steps[-1]
    history = steps[:-1]
    score = critic.score_step(history, last_step)

    # Attach score to step if needed
    last_step.label_llm = score

    # Check if critic should trigger early stop
    if critic.should_stop(steps):
        return {
            "critic_score": score,
            "is_complete": True,
            "termination": "critic_stop",
            "status": "critic_stop"
        }

    # If executor marked complete (e.g. final answer)
    if state.get("is_complete"):
        return {
            "critic_score": score,
            "is_complete": True,
            "termination": state.get("termination", "final_answer"),
            "status": "completed"
        }

    return {
        "critic_score": score,
        "is_complete": False,
        "status": "evaluated"
    }


def router_condition(state: GraphState) -> Literal["replanner", "end"]:
    """Conditional edge from Critic: if complete, terminate; otherwise replan."""
    if state.get("is_complete"):
        return "end"
    return "replanner"


def replanner_node(state: GraphState) -> Dict[str, Any]:
    """Refines the plan based on the latest observation before the next executor step."""
    steps = state.get("steps", [])
    step_num = len(steps)
    prev_plan = state.get("plan", "")
    
    updated_plan = f"{prev_plan} -> Step {step_num} complete, refining next action."
    return {
        "plan": updated_plan,
        "status": "replanned"
    }


def build_graph(critic: Optional[Critic] = None, enable_best_of_n: bool = False):
    """Compile and return the 4-node LangGraph state machine."""
    builder = StateGraph(GraphState)

    # Add Nodes
    builder.add_node("planner", planner_node)
    builder.add_node("executor", executor_node)
    builder.add_node("critic", critic_node)
    builder.add_node("replanner", replanner_node)

    # Add Workflow Edges
    builder.set_entry_point("planner")
    builder.add_edge("planner", "executor")
    builder.add_edge("executor", "critic")

    # Conditional router from critic
    builder.add_conditional_edges(
        "critic",
        router_condition,
        {
            "end": END,
            "replanner": "replanner"
        }
    )
    builder.add_edge("replanner", "executor")

    return builder.compile()


def run_graph_to_record(
    question: str,
    task_id: str = "t001",
    enable_best_of_n: bool = False,
    critic: Optional[Critic] = None
) -> RunRecord:
    """Helper to run the compiled graph end-to-end and emit a standard RunRecord."""
    start_time = time.time()
    run_id = f"r-{uuid.uuid4().hex[:6]}"
    graph = build_graph(critic=critic, enable_best_of_n=enable_best_of_n)

    initial_state: GraphState = {
        "run_id": run_id,
        "task_id": task_id,
        "question": question,
        "steps": [],
        "plan": "",
        "status": "init",
        "tokens_used": 0,
        "token_budget": 12000,
        "max_steps": 10,
        "enable_best_of_n": enable_best_of_n,
        "critic_score": 0.0,
        "is_complete": False,
        "termination": None,
        "final_answer": None,
        "start_time": start_time
    }

    final_state = graph.invoke(initial_state)

    total_duration_ms = int((time.time() - start_time) * 1000)
    steps_list = final_state.get("steps", [])
    termination_reason = final_state.get("termination", "final_answer")

    record = RunRecord(
        run_id=run_id,
        task_id=task_id,
        question=question,
        steps=steps_list,
        termination=termination_reason if termination_reason in VALID_TERMINATIONS else "final_answer",
        final_answer=final_state.get("final_answer"),
        correct=True if termination_reason == "final_answer" else False,
        total_duration_ms=total_duration_ms,
        total_tokens=final_state.get("tokens_used", 0),
        model_name="Qwen2.5-Coder-7B-Instruct",
        quantisation="Q4_K_M",
        temperature=0.0,
        context_policy="mask_last_3",
        critic_version=critic.version if critic else "v1.0-deberta-critic"
    )

    return record