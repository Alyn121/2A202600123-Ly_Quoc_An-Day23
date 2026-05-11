"""Routing functions for conditional edges."""

from __future__ import annotations

from .state import AgentState, Route


def route_after_classify(state: AgentState) -> str:
    """Map classified route to the next graph node.

    Uses route value from classify_node to determine next step.
    Falls back safely to "answer" for unknown routes.
    """
    route = state.get("route", Route.SIMPLE.value)
    mapping = {
        Route.SIMPLE.value: "answer",
        Route.TOOL.value: "tool",
        Route.MISSING_INFO.value: "clarify",
        Route.RISKY.value: "risky_action",
        Route.ERROR.value: "retry",
    }
    # Safe fallback to answer node for unknown routes
    next_node = mapping.get(route, "answer")
    return next_node


def route_after_retry(state: AgentState) -> str:
    """Decide whether to retry tool or escalate to dead-letter.

    Implements bounded retry: if attempt >= max_attempts, route to dead_letter.
    Otherwise, return to tool for another attempt.
    """
    attempt = int(state.get("attempt", 0))
    max_attempts = int(state.get("max_attempts", 3))
    
    if attempt >= max_attempts:
        # Max retries exhausted, escalate to dead letter
        return "dead_letter"
    
    # Continue retrying
    return "tool"


def route_after_evaluate(state: AgentState) -> str:
    """Decide whether tool result is satisfactory or needs retry.

    This is the critical 'done?' gate that enables retry loops.
    If evaluation_result is "needs_retry", route back to retry node.
    Otherwise, proceed to answer node.
    """
    evaluation = state.get("evaluation_result", "success")
    
    if evaluation == "needs_retry":
        # Tool result not satisfactory, route to retry
        return "retry"
    
    # Tool result satisfactory, proceed to answer
    return "answer"


def route_after_approval(state: AgentState) -> str:
    """Route based on approval decision for risky actions.

    If approved, proceed to tool evaluation.
    If rejected, route to clarify (safe fallback) rather than executing risky action.
    """
    approval = state.get("approval") or {}
    approved = approval.get("approved", False)
    
    if approved:
        # Approval granted, proceed to tool
        return "tool"
    else:
        # Approval denied, ask for clarification instead of executing risky action
        return "clarify"