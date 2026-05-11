"""Graph construction.

This module is intentionally import-safe. It imports LangGraph only inside the builder so unit tests
that check schema/metrics can run even if students are still debugging graph wiring.
"""

from __future__ import annotations

from typing import Any

from .nodes import (
    answer_node,
    approval_node,
    ask_clarification_node,
    classify_node,
    dead_letter_node,
    evaluate_node,
    finalize_node,
    intake_node,
    retry_or_fallback_node,
    risky_action_node,
    tool_node,
)
from .routing import route_after_approval, route_after_classify, route_after_evaluate, route_after_retry
from .state import AgentState


def build_graph(checkpointer: Any | None = None):
    """Build and compile the complete LangGraph workflow.

    Graph structure (paths to finalize):
    - START → intake → classify → (conditional route)
      - simple → answer → finalize → END
      - tool → tool → evaluate → (conditional)
        - success → answer → finalize → END
        - needs_retry → retry → (conditional)
          - max_attempts not reached → tool (retry loop)
          - max_attempts reached → dead_letter → finalize → END
      - missing_info → clarify → finalize → END
      - risky → risky_action → approval → (conditional)
        - approved → tool → evaluate → ... (same as tool path)
        - denied → clarify → finalize → END
      - error → retry (same as tool retry loop)

    All paths eventually reach finalize → END.
    Retry loop is bounded by max_attempts.
    Risky actions require HITL approval.
    """
    try:
        from langgraph.graph import END, START, StateGraph
    except Exception as exc:  # pragma: no cover - helpful install error
        raise RuntimeError("LangGraph is required. Run: pip install -e '.[dev]' or pip install langgraph") from exc

    graph = StateGraph(AgentState)
    
    # Add all nodes
    graph.add_node("intake", intake_node)
    graph.add_node("classify", classify_node)
    graph.add_node("answer", answer_node)
    graph.add_node("tool", tool_node)
    graph.add_node("evaluate", evaluate_node)
    graph.add_node("clarify", ask_clarification_node)
    graph.add_node("risky_action", risky_action_node)
    graph.add_node("approval", approval_node)
    graph.add_node("retry", retry_or_fallback_node)
    graph.add_node("dead_letter", dead_letter_node)
    graph.add_node("finalize", finalize_node)

    # Main flow
    graph.add_edge(START, "intake")
    graph.add_edge("intake", "classify")
    
    # Classify routes to 5 different paths
    graph.add_conditional_edges("classify", route_after_classify)
    
    # Tool → evaluate → retry loop or answer
    graph.add_edge("tool", "evaluate")
    graph.add_conditional_edges("evaluate", route_after_evaluate)
    
    # Clarify → finalize (safe path for missing info or denied approval)
    graph.add_edge("clarify", "finalize")
    
    # Risky → approval → tool/clarify
    graph.add_edge("risky_action", "approval")
    graph.add_conditional_edges("approval", route_after_approval)
    
    # Retry bounded by max_attempts
    graph.add_conditional_edges("retry", route_after_retry)
    
    # Terminal nodes
    graph.add_edge("answer", "finalize")
    graph.add_edge("dead_letter", "finalize")
    graph.add_edge("finalize", END)

    return graph.compile(checkpointer=checkpointer)