"""Page 3: LangGraph Agent Workflow & MCP Tool Execution — BankShield AI.

Renders the live LangGraph StateGraph, agent reasoning profiles, Model Context Protocol (MCP)
tool call telemetry, conditional validation routing, self-correction loops, and shared typed state.
"""

from pathlib import Path
import sys

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import json
import pandas as pd
import streamlit as st

from orchestration.langgraph_orchestrator import LangGraphInvestigationOrchestrator
from services.data_service import DataService

st.set_page_config(page_title="LangGraph & MCP Workflow — BankShield AI", page_icon="🔄", layout="wide")

# Session state setup
if "data_service" not in st.session_state:
    st.session_state.data_service = DataService()
if "orchestrator" not in st.session_state:
    st.session_state.orchestrator = LangGraphInvestigationOrchestrator(data_service=st.session_state.data_service)

orchestrator = st.session_state.orchestrator

st.title("🔄 LangGraph State Machine & MCP Tool Execution")
st.caption("Inspect live LangGraph nodes, state transitions, Model Context Protocol (MCP) tools, and Critic routing")

# Fetch latest investigation from session or run default demo if none
latest_inv = st.session_state.get("latest_investigation")

if not latest_inv:
    st.info("💡 Running initial live LangGraph investigation on benchmark transaction `TXN-10198`...")
    latest_inv = orchestrator.investigate_transaction("TXN-10198")
    st.session_state["latest_investigation"] = latest_inv

tx_id = latest_inv.get("transaction_id", "N/A")
total_ms = latest_inv.get("total_duration_ms", 0)
exec_trace = latest_inv.get("execution_trace", [])
node_history = latest_inv.get("node_history", [])
plan = latest_inv.get("plan", {})
evidence = latest_inv.get("evidence", {})
risk_res = latest_inv.get("risk_result", {})
validation = latest_inv.get("final_report", {}).get("validation_result", {})
mcp_calls = latest_inv.get("mcp_tool_calls", [])
retry_count = latest_inv.get("retry_count", 0)
corr_applied = latest_inv.get("correction_applied", False)

# Top Bar
col_top1, col_top2, col_top3 = st.columns([2, 1, 1])
with col_top1:
    st.markdown(f"**Target Transaction:** ` {tx_id} ` | **LangGraph Status:** `{latest_inv.get('status', 'completed').upper()}`")
with col_top2:
    st.markdown(f"**Critic Recalculation:** `{'MATCH ✅' if validation.get('score_matches') else 'DISCREPANCY ❌'}`")
with col_top3:
    st.markdown(f"**Self-Correction Retries:** `{retry_count}/1` (Applied: `{corr_applied}`)")

st.markdown("---")

# Tabbed Layout: Graph View, Agent Reasoning, MCP Telemetry, State Inspector
tab_graph, tab_agents, tab_mcp, tab_state = st.tabs([
    "🕸️ LangGraph Architecture & Nodes",
    "🧠 Agent Reasoning & Responsibilities",
    "🔌 Model Context Protocol (MCP) Tools",
    "📦 Shared InvestigationState Inspector",
])

# ==============================================================================
# TAB 1: LANGGRAPH ARCHITECTURE & STATE MACHINE
# ==============================================================================
with tab_graph:
    st.subheader("1. LangGraph StateGraph Architecture & Conditional Routing")
    st.write(
        """
        The workflow is modeled as a compiled **LangGraph `StateGraph`** with explicit nodes, typed transitions,
        and conditional branching on the Critic Agent's mathematical validation outcome:
        """
    )

    col_g1, col_g2 = st.columns([3, 2])

    with col_g1:
        st.markdown(
            """
            ```mermaid
            flowchart TD
                START([START]) --> P[planner_node<br/><i>Planner Agent</i>]
                P --> R[research_node<br/><i>Research Agent via MCP</i>]
                R --> D[risk_assessment_node<br/><i>Risk Rules Engine</i>]
                D --> C[critic_node<br/><i>Critic 9-Gate Audit</i>]
                
                C -->|VALIDATED| Rep[report_generation_node<br/><i>Compile 16-Field Dossier</i>]
                C -->|VALIDATION_FAILED & retry < 1| Fix[correction_node<br/><i>Recalibrate Score</i>]
                Fix --> C
                
                Rep --> END_NODE([END])
                
                style START fill:#E0E7FF,stroke:#4F46E5
                style P fill:#F1F5F9,stroke:#64748B
                style R fill:#FEF3C7,stroke:#D97706
                style D fill:#EDE9FE,stroke:#7C3AED
                style C fill:#FEE2E2,stroke:#DC2626
                style Fix fill:#FFEDD5,stroke:#EA580C
                style Rep fill:#DCFCE7,stroke:#16A34A
                style END_NODE fill:#E0E7FF,stroke:#4F46E5
            ```
            """
        )

    with col_g2:
        st.markdown("#### Live State Transitions for this Case:")
        for idx, node in enumerate(node_history, start=1):
            if node == "correction_node":
                badge = "🔄 **Node (Self-Correction Loop)**"
                color = "#EA580C"
            elif node == "critic_node":
                badge = "🛡️ **Node (Audit Gate)**"
                color = "#DC2626"
            elif node == "research_node":
                badge = "🔬 **Node (MCP Tools)**"
                color = "#D97706"
            else:
                badge = "⚙️ **Node**"
                color = "#1E3A8A"

            st.markdown(
                f"""
                <div style="background-color: #F8FAFC; border-left: 4px solid {color}; padding: 6px 12px; margin-bottom: 6px; border-radius: 4px; font-size: 0.9rem;">
                    Step {idx}: {badge} — <code>{node}</code>
                </div>
                """,
                unsafe_allow_html=True,
            )

        st.caption("Execution completed safely within bounded retry limits.")

# ==============================================================================
# TAB 2: AGENT REASONING & RESPONSIBILITIES
# ==============================================================================
with tab_agents:
    st.subheader("2. Specialized Agent Reasoning Profiles")
    st.caption("Clearly distinguishes autonomous agent reasoning from graph node routing and MCP tool calls")

    c_a1, c_a2 = st.columns(2)

    with c_a1:
        st.markdown(
            """
            <div style="background-color: #F8FAFC; border: 1px solid #CBD5E1; padding: 14px; border-radius: 6px; margin-bottom: 12px;">
                <strong>🧭 1. Investigation Planner Agent:</strong><br>
                • <strong>Role:</strong> Formulate an explicit 6-step dynamic roadmap.<br>
                • <strong>Reasoning Goal:</strong> Validates input transaction ID, verifies objective, and defines required tool calls.<br>
                • <strong>Output Contract:</strong> Populates <code>state['plan']</code> with ordered execution tasks.
            </div>
            """,
            unsafe_allow_html=True,
        )

        st.markdown(
            """
            <div style="background-color: #F8FAFC; border: 1px solid #CBD5E1; padding: 14px; border-radius: 6px; margin-bottom: 12px;">
                <strong>🔬 2. Transaction Research Agent:</strong><br>
                • <strong>Role:</strong> Autonomous Tool Operator.<br>
                • <strong>Reasoning Goal:</strong> Connects to MCP Server via <code>BankShieldMCPClient</code>; queries past history, computes z-scores, velocity, and unseen merchant flags.<br>
                • <strong>Output Contract:</strong> Populates <code>state['evidence']</code> and logs tool telemetry.
            </div>
            """,
            unsafe_allow_html=True,
        )

    with c_a2:
        st.markdown(
            """
            <div style="background-color: #F8FAFC; border: 1px solid #CBD5E1; padding: 14px; border-radius: 6px; margin-bottom: 12px;">
                <strong>⚖️ 3. Risk Assessment Agent:</strong><br>
                • <strong>Role:</strong> Deterministic Decision Supporter.<br>
                • <strong>Reasoning Goal:</strong> Evaluates 5 transparent rules capped at 100 points. Associates every point with empirical evidence.<br>
                • <strong>Output Contract:</strong> Populates <code>state['risk_result']</code> and maps to Low/Med/High/Critical priority tiers.
            </div>
            """,
            unsafe_allow_html=True,
        )

        st.markdown(
            """
            <div style="background-color: #F8FAFC; border: 1px solid #CBD5E1; padding: 14px; border-radius: 6px; margin-bottom: 12px;">
                <strong>🛡️ 4. Critic & Validation Agent:</strong><br>
                • <strong>Role:</strong> Independent Four-Eyes Quality Auditor.<br>
                • <strong>Reasoning Goal:</strong> Independently recomputes all mathematical operations; enforces 9 quality gates; triggers conditional correction on failure.<br>
                • <strong>Output Contract:</strong> Populates <code>state['validation_status']</code> ('VALIDATED' or 'VALIDATION_FAILED').
            </div>
            """,
            unsafe_allow_html=True,
        )

# ==============================================================================
# TAB 3: MODEL CONTEXT PROTOCOL (MCP) TOOLS & TELEMETRY
# ==============================================================================
with tab_mcp:
    st.subheader("3. Model Context Protocol (MCP) Standardized Tool Invocations")
    st.write(
        """
        The Research Agent and Critic Agent do not use ad-hoc hardcoded function calls.
        Instead, they communicate with the **local MCP Server (`mcp_server.server`)** via the **`BankShieldMCPClient`**,
        adhering strictly to the Model Context Protocol specification:
        """
    )

    # Display MCP tools discovered from server
    mcp_client = orchestrator.mcp_client
    registered_tools = mcp_client.list_tools()

    with st.expander("🔌 View Registered MCP Server Tool Schemas (Model Context Protocol)", expanded=False):
        for tool in registered_tools:
            st.markdown(f"**Tool:** ` {tool.get('name')} `")
            st.caption(tool.get("description", ""))
            st.code(json.dumps(tool.get("input_schema", {}), indent=2), language="json")

    st.markdown("#### Live MCP Tool Call Log for Current Investigation:")
    tool_telemetry = mcp_client.get_telemetry_history()

    if tool_telemetry:
        tel_rows = []
        for call in tool_telemetry[-8:]:
            tel_rows.append({
                "Timestamp": call.get("timestamp", "-")[:19],
                "MCP Tool Name": call.get("tool_name"),
                "Protocol": call.get("protocol", "MCP/2.0"),
                "Status": call.get("status"),
                "Duration (ms)": f"{call.get('duration_ms', 0):.2f}",
                "Arguments": json.dumps(call.get("arguments", {})),
            })
        st.dataframe(pd.DataFrame(tel_rows), use_container_width=True, hide_index=True)
    else:
        st.info("No MCP calls recorded yet.")

# ==============================================================================
# TAB 4: SHARED TYPED INVESTIGATION STATE
# ==============================================================================
with tab_state:
    st.subheader("4. Shared Typed InvestigationState Viewer")
    st.write("Inspect the exact TypedDict state dictionary passed between LangGraph nodes:")

    # Clean state view for display
    state_display = {
        "investigation_id": latest_inv.get("investigation_id"),
        "transaction_id": latest_inv.get("transaction_id"),
        "node_history": latest_inv.get("node_history"),
        "validation_status": latest_inv.get("validation_status"),
        "retry_count": latest_inv.get("retry_count"),
        "correction_applied": latest_inv.get("correction_applied"),
        "risk_result": latest_inv.get("risk_result"),
        "plan": latest_inv.get("plan"),
        "mcp_tool_calls_count": len(latest_inv.get("mcp_tool_calls", [])),
        "total_duration_ms": latest_inv.get("total_duration_ms"),
    }
    st.json(state_display)

st.markdown("---")

# Section 5: Orchestration Execution Trace Table
st.subheader("5. Sequential Orchestration Trace (All Graph Events)")
if exec_trace:
    trace_rows = []
    for item in exec_trace:
        trace_rows.append({
            "Timestamp": item.get("timestamp", "-")[:19],
            "LangGraph Node": item.get("node", "-"),
            "Agent / Actor": item.get("agent_name", "-"),
            "Action Performed": item.get("action", "-"),
            "Status": item.get("status", "-"),
            "Duration (ms)": f"{item.get('duration_ms', 0):.2f}",
            "Summary Output": item.get("output_summary", "-"),
        })
    st.dataframe(pd.DataFrame(trace_rows), use_container_width=True, hide_index=True)
