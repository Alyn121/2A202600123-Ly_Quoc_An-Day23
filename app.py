import os
import streamlit as st
import uuid
import sqlite3

# MUST be set before importing graph so that interrupt logic in approval_node is activated
os.environ["LANGGRAPH_INTERRUPT"] = "true"

from langgraph_agent_lab.graph import build_graph
from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.types import Command
from langgraph_agent_lab.state import ApprovalDecision

# Set up page layout
st.set_page_config(page_title="AI Support Agent", page_icon="🤖", layout="centered")

# --- Initialize Application State and Persistence ---
def init_graph():
    conn = sqlite3.connect("demo_checkpoints.db", check_same_thread=False)
    checkpointer = SqliteSaver(conn)
    app = build_graph(checkpointer=checkpointer)
    return app

st.title("🤖 LangGraph Support Agent Demo")
st.markdown("A production-style ticket agent with HITL (Human-in-the-loop) for risky actions.")

# Ensure we have a thread_id
if "thread_id" not in st.session_state:
    st.session_state.thread_id = str(uuid.uuid4())

# Initialize the graph
if "app" not in st.session_state:
    st.session_state.app = init_graph()

app = st.session_state.app
config = {"configurable": {"thread_id": st.session_state.thread_id}}

# --- Retrieve Graph State ---
current_state = app.get_state(config)
state_values = current_state.values if current_state else {}

# Keep track of local chat history manually for Streamlit UI
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
    st.session_state.chat_history.append({"role": "assistant", "content": "Hello! I am your AI support agent. How can I help you today?"})

# Display Chat
for msg in st.session_state.chat_history:
    st.chat_message(msg["role"]).write(msg["content"])

# --- Check for Interrupts (Pending Approval) ---
is_interrupted = False
pending_action = None

if current_state and current_state.tasks:
    # Look for tasks that are blocked waiting for an interrupt to be resumed
    for task in current_state.tasks:
        if task.interrupts:
            is_interrupted = True
            # In our nodes.py, interrupt was called with a dict containing proposed_action and risk_level
            interrupt_val = task.interrupts[0].value
            if isinstance(interrupt_val, dict):
                 pending_action = interrupt_val.get("proposed_action", "Unknown Action")
            else:
                 pending_action = str(interrupt_val)
            break

if is_interrupted:
    st.warning("⚠️ **Human Approval Required**")
    st.info(f"**Proposed Action:** {pending_action}")
    
    col1, col2 = st.columns(2)
    with col1:
        if st.button("✅ Approve Action", use_container_width=True, type="primary"):
            # Resume the graph with an approved decision
            decision = {"approved": True, "reviewer": "streamlit_user", "comment": "Approved via UI"}
            with st.spinner("Processing approved action..."):
                for chunk in app.stream(Command(resume=decision), config, stream_mode="updates"):
                    pass # Just consume the stream
            
            # Fetch final answer and add to chat
            final_state = app.get_state(config).values
            final_answer = final_state.get("final_answer", "Action executed.")
            st.session_state.chat_history.append({"role": "assistant", "content": final_answer})
            st.rerun()

    with col2:
        if st.button("❌ Deny Action", use_container_width=True):
            # Resume the graph with a denied decision
            decision = {"approved": False, "reviewer": "streamlit_user", "comment": "Denied via UI"}
            with st.spinner("Processing rejection..."):
                for chunk in app.stream(Command(resume=decision), config, stream_mode="updates"):
                    pass # Just consume the stream
            
            # Fetch final answer and add to chat
            final_state = app.get_state(config).values
            final_answer = final_state.get("final_answer", "Action denied.")
            st.session_state.chat_history.append({"role": "assistant", "content": final_answer})
            st.rerun()

# --- Chat Input ---
# Disable input if we are waiting for human approval
user_query = st.chat_input("Type your support request here...", disabled=is_interrupted)

if user_query:
    # Add user message to UI
    st.session_state.chat_history.append({"role": "user", "content": user_query})
    st.chat_message("user").write(user_query)
    
    with st.spinner("Agent is thinking..."):
        # Invoke the graph with the user query
        # Since it's a completely new turn, we might want to just append or reset.
        # But based on the state machine, "intake" expects "query" to be updated.
        input_data = {"query": user_query, "scenario_id": f"UI-{uuid.uuid4().hex[:6]}"}
        
        # We use stream to execute. We ignore chunks in UI to just print the final result or wait for interrupt.
        for chunk in app.stream(input_data, config, stream_mode="updates"):
             pass
        
    st.rerun()

# --- Sidebar for Debugging ---
with st.sidebar:
    st.header("Graph Status")
    st.write(f"**Thread ID:** `{st.session_state.thread_id}`")
    
    if st.button("Clear Session"):
        st.session_state.clear()
        st.rerun()
        
    st.subheader("Internal State (Debug)")
    if current_state and current_state.values:
        st.json({
            "route": current_state.values.get("route"),
            "attempt": current_state.values.get("attempt"),
            "errors": current_state.values.get("errors", []),
            "events_count": len(current_state.values.get("events", [])),
            "tool_results": current_state.values.get("tool_results", []),
        })
