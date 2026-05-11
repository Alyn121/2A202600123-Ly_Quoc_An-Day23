"""Node skeletons for the LangGraph workflow.

Each function should be small, testable, and return a partial state update. Avoid mutating the
input state in place.
"""

from __future__ import annotations

from .state import AgentState, ApprovalDecision, Route, make_event


def intake_node(state: AgentState) -> dict:
    """Normalize raw query into state fields.

    Performs basic normalization, PII redaction, and metadata extraction.
    """
    import re
    query = state.get("query", "").strip()
    
    # 1. Normalization: lowercase and strip extra whitespace
    normalized_query = " ".join(query.lower().split())
    
    # 2. Basic PII Redaction (Email and Phone patterns)
    # Simple patterns for demonstration in the lab
    email_pattern = r"[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}"
    phone_pattern = r"\b\d{3}[-.\s]?\d{3}[-.\s]?\d{4}\b"
    
    redacted_query = re.sub(email_pattern, "[EMAIL]", normalized_query)
    redacted_query = re.sub(phone_pattern, "[PHONE]", redacted_query)
    
    # 3. Metadata extraction
    word_count = len(redacted_query.split())
    metadata = {
        "word_count": word_count,
        "is_redacted": redacted_query != normalized_query,
    }
    
    return {
        "query": redacted_query,
        "metadata": metadata,
        "messages": [f"intake:{redacted_query[:40]}"],
        "events": [make_event("intake", "completed", "query normalized and PII checked", metadata=metadata)],
    }


def classify_node(state: AgentState) -> dict:
    """Classify the query into a route with explicit policy.

    Routing policy:
    - RISKY: contains refund, delete, send, cancel (high-risk actions)
    - ERROR: contains timeout, fail, error, system failure (system issues)
    - TOOL: contains status, order, lookup, account (information lookup)
    - MISSING_INFO: vague query with <5 words and no clear action (needs clarification)
    - SIMPLE: default safe answer path

    Returns route and risk_level for audit.
    """
    import re
    # Strip punctuation and split into words for strict boundary matching
    raw_query = state.get("query", "").lower()
    clean_query = re.sub(r'[^\w\s]', '', raw_query)
    words = set(clean_query.split())
    
    # Risk keywords (destructive/sensitive actions)
    risk_keywords = {"refund", "delete", "cancel", "send", "confirmation", "email"}
    if words.intersection(risk_keywords):
        return {
            "route": Route.RISKY.value,
            "risk_level": "high",
            "events": [make_event("classify", "completed", "route=risky (destructive action detected)")],
        }
    
    # Error keywords (system failures)
    error_keywords = {"timeout", "fail", "error", "system", "cannot", "recover"}
    if words.intersection(error_keywords):
        return {
            "route": Route.ERROR.value,
            "risk_level": "high",
            "events": [make_event("classify", "completed", "route=error (system failure detected)")],
        }
    
    # Tool keywords (information lookups)
    tool_keywords = {"status", "order", "lookup", "account", "check"}
    if words.intersection(tool_keywords):
        return {
            "route": Route.TOOL.value,
            "risk_level": "low",
            "events": [make_event("classify", "completed", "route=tool (lookup request)")],
        }
    
    # Missing info: vague query with few words
    missing_info_keywords = {"fix", "it", "this", "that"}
    if len(words) < 5 and words.intersection(missing_info_keywords):
        return {
            "route": Route.MISSING_INFO.value,
            "risk_level": "medium",
            "events": [make_event("classify", "completed", "route=missing_info (vague query)")],
        }
    
    # Default: simple answer path
    return {
        "route": Route.SIMPLE.value,
        "risk_level": "low",
        "events": [make_event("classify", "completed", "route=simple (default)")],
    }


def ask_clarification_node(state: AgentState) -> dict:
    """Ask for missing information to handle ambiguous queries.

    Generates a contextual clarification question based on query content.
    """
    query = state.get("query", "")
    
    # Generate context-aware clarification questions
    if "order" in query.lower() and "id" not in query.lower():
        question = "Could you please provide the order ID so I can look up the status?"
    elif "account" in query.lower():
        question = "Which account or email address should I check?"
    elif "fix" in query.lower() or "it" in query.lower():
        question = "Could you provide more details about what needs to be fixed?"
    else:
        question = "Could you provide more details to help me assist you better?"
    
    return {
        "pending_question": question,
        "final_answer": question,
        "events": [make_event("clarify", "completed", "missing information requested")],
    }


def tool_node(state: AgentState) -> dict:
    """Call a mock tool with idempotent execution.

    For error scenarios, simulates transient failures on early attempts to demonstrate retry loop.
    For other scenarios, returns mock success immediately.
    
    Returns structured tool result in tool_results list for evaluate_node to assess.
    """
    attempt = int(state.get("attempt", 0))
    route = state.get("route", Route.TOOL.value)
    scenario_id = state.get("scenario_id", "unknown")
    
    # For ERROR route, simulate transient failures on early attempts
    if route == Route.ERROR.value and attempt < 2:
        # Simulate transient failure - will be retried
        result = f"ERROR: transient failure on attempt {attempt}"
    else:
        # Success result (includes all other routes when they reach tool)
        result = f"Tool result for scenario {scenario_id} on attempt {attempt}: Success"
    
    return {
        "tool_results": [result],
        "events": [make_event("tool", "executed", f"tool called on attempt {attempt}", result=result)],
    }


def risky_action_node(state: AgentState) -> dict:
    """Prepare a risky action with evidence for approval.

    Analyzes query to create a specific proposed action with justification.
    The approval node will decide whether to execute or deny.
    """
    query = state.get("query", "").lower()
    scenario_id = state.get("scenario_id", "unknown")
    
    # Determine the specific risky action from query
    if "refund" in query:
        proposed_action = f"Refund customer for scenario {scenario_id}. Reason: Customer requested refund."
    elif "delete" in query:
        proposed_action = f"Delete customer account for scenario {scenario_id}. Reason: Customer requested account deletion after verification."
    elif "send" in query or "email" in query:
        proposed_action = f"Send confirmation email for scenario {scenario_id}. Reason: Customer requested email confirmation."
    else:
        proposed_action = f"Execute risky action for scenario {scenario_id}. Requires approval."
    
    return {
        "proposed_action": proposed_action,
        "events": [make_event("risky_action", "prepared", "risky action prepared for approval", action=proposed_action)],
    }


def approval_node(state: AgentState) -> dict:
    """Process human approval for risky actions.

    Supports real HITL via interrupt() when LANGGRAPH_INTERRUPT=true.
    Default uses mock approval for CI/local testing.
    
    Records approval decision in state for metrics tracking.
    """
    import os

    if os.getenv("LANGGRAPH_INTERRUPT", "").lower() == "true":
        from langgraph.types import interrupt

        value = interrupt({
            "proposed_action": state.get("proposed_action"),
            "risk_level": state.get("risk_level"),
        })
        if isinstance(value, dict):
            decision = ApprovalDecision(**value)
        else:
            decision = ApprovalDecision(approved=bool(value))
    else:
        # Mock approval: default to approve for lab scenarios
        decision = ApprovalDecision(approved=True, reviewer="mock-reviewer", comment="Lab mock approval")
    
    return {
        "approval": decision.model_dump(),
        "events": [make_event("approval", "completed", f"approved={decision.approved}", reviewer=decision.reviewer)],
    }


def retry_or_fallback_node(state: AgentState) -> dict:
    """Record a retry attempt with metadata for bounded retry.

    Increments attempt counter and logs retry event for metrics.
    Bounded by route_after_retry which checks max_attempts.
    """
    attempt = int(state.get("attempt", 0)) + 1
    max_attempts = int(state.get("max_attempts", 3))
    
    # Only return the NEW error message. The 'add' reducer in state.py
    # will automatically append it to the existing list.
    new_error = f"Retry attempt {attempt}/{max_attempts}"
    
    return {
        "attempt": attempt,
        "errors": [new_error],
        "events": [make_event("retry", "recorded", f"Retry {attempt}/{max_attempts}", attempt=attempt, max_attempts=max_attempts)],
    }


def answer_node(state: AgentState) -> dict:
    """Produce a final answer grounded in tool results or clarification.

    Uses tool_results if available, otherwise pending_question or safe default.
    """
    tool_results = state.get("tool_results", [])
    pending_question = state.get("pending_question")
    scenario_id = state.get("scenario_id", "unknown")
    
    if tool_results:
        # Answer based on latest tool result
        latest_result = tool_results[-1]
        answer = f"Result for {scenario_id}: {latest_result}"
    elif pending_question:
        # Answer with the clarification question
        answer = pending_question
    else:
        # Safe default answer
        answer = f"I can assist you with your request (scenario {scenario_id}). Please provide additional details if needed."
    
    return {
        "final_answer": answer,
        "events": [make_event("answer", "generated", "final answer prepared")],
    }


def evaluate_node(state: AgentState) -> dict:
    """Evaluate tool results — the critical 'done?' gate for retry loops.

    Checks if latest tool result indicates success or failure.
    This node determines whether to retry or proceed to answer.
    """
    tool_results = state.get("tool_results", [])
    latest = tool_results[-1] if tool_results else ""
    
    # Check for failure indicators
    failure_indicators = ["ERROR", "error", "fail", "timeout", "cannot", "failed"]
    if any(indicator in latest for indicator in failure_indicators):
        return {
            "evaluation_result": "needs_retry",
            "events": [make_event("evaluate", "assessed", "Tool result indicates failure; retry needed", result_preview=latest[:50])],
        }
    
    # Success path
    return {
        "evaluation_result": "success",
        "events": [make_event("evaluate", "assessed", "Tool result indicates success", result_preview=latest[:50])],
    }


def dead_letter_node(state: AgentState) -> dict:
    """Escalate unresolvable failures to dead-letter queue.

    This is the final fallback: max retries exhausted, no further action possible.
    Logs escalation with full context for manual review.
    """
    attempt = state.get("attempt", 0)
    max_attempts = state.get("max_attempts", 3)
    scenario_id = state.get("scenario_id", "unknown")
    errors = state.get("errors", [])
    
    # Create detailed escalation message
    escalation_message = f"Scenario {scenario_id}: Failed after {attempt} attempts (max {max_attempts}). Requires manual review."
    
    return {
        "final_answer": escalation_message,
        "errors": errors + [f"Dead letter: max retries ({attempt}/{max_attempts}) exhausted"],
        "events": [make_event("dead_letter", "escalated", escalation_message, attempt=attempt, max_attempts=max_attempts)],
    }


def finalize_node(state: AgentState) -> dict:
    """Finalize the run and emit audit trail.
    
    Captures final state for metrics and grading.
    """
    route = state.get("route", "unknown")
    final_answer = state.get("final_answer")
    attempts = state.get("attempt", 0)
    
    return {
        "events": [make_event("finalize", "complete", f"Workflow finished. Route: {route}, Attempts: {attempts}")],
    }