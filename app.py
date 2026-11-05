"""BankShield AI — Streamlit Application Entrypoint.

An Agentic AI System for Banking Transaction Risk Investigation and Decision Support.
"""

from pathlib import Path
import sys

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import streamlit as st

st.set_page_config(
    page_title="BankShield AI — Transaction Risk Investigation",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom enterprise styling
st.markdown(
    """
    <style>
    .main-header {
        font-size: 2.2rem;
        font-weight: 700;
        color: #1E3A8A;
        margin-bottom: 0.2rem;
    }
    .sub-header {
        font-size: 1.1rem;
        color: #475569;
        margin-bottom: 1.5rem;
    }
    .metric-card {
        background-color: #F8FAFC;
        border: 1px solid #E2E8F0;
        border-radius: 8px;
        padding: 16px;
        margin-bottom: 12px;
    }
    .agent-badge {
        display: inline-block;
        padding: 4px 10px;
        border-radius: 12px;
        font-size: 0.85rem;
        font-weight: 600;
        margin-right: 6px;
    }
    .synthetic-banner {
        background-color: #FEF3C7;
        color: #92400E;
        padding: 10px 16px;
        border-radius: 6px;
        border-left: 4px solid #F59E0B;
        margin-bottom: 1.5rem;
        font-size: 0.9rem;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# Sidebar navigation & status
with st.sidebar:
    st.image("https://img.icons8.com/fluency/96/shield.png", width=64)
    st.title("BankShield AI")
    st.caption("Agentic Decision Support System v1.0.0")

    st.markdown("---")
    st.markdown("**Navigation Pages:**")
    st.markdown("1. 📊 **Executive Overview**")
    st.markdown("2. 🔍 **Investigate Transaction**")
    st.markdown("3. 🔄 **Agent Workflow**")
    st.markdown("4. 📋 **Investigation Reports**")
    st.markdown("5. 📖 **Methodology & Limitations**")

    st.markdown("---")
    st.info(
        "💡 **Quick Demo Flow (5-Min):**\n\n"
        "1. Open **Executive Overview**\n"
        "2. Run **Investigate Transaction** on preset demo transaction `TXN-10198`\n"
        "3. Inspect **Agent Workflow** multi-agent trace\n"
        "4. Review **Investigation Reports** & download dossier\n"
        "5. Conclude on **Methodology & Limitations**"
    )

    st.markdown("---")
    st.caption("Autonomous Agentic Decision Support System | Fully Offline & Deterministic")

# Main Page Content
st.markdown('<div class="main-header">🛡️ BankShield AI</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="sub-header">An Agentic AI System for Banking Transaction Risk Investigation and Decision Support</div>',
    unsafe_allow_html=True,
)

st.markdown(
    """
    <div class="synthetic-banner">
        ⚠️ <strong>NOTICE:</strong> This application operates on <strong>synthetic demonstration data</strong> for analyst decision-support.
        High-risk indicators flag anomalies for human inspection, <em>not</em> automated adverse actions.
        No real customer credentials or PII are used.
    </div>
    """,
    unsafe_allow_html=True,
)

col1, col2 = st.columns([3, 2])

with col1:
    st.markdown("### Welcome to BankShield AI")
    st.write(
        """
        In modern retail banking, compliance analysts face overwhelming volumes of suspicious transaction alerts.
        Manual investigation requires tedious record retrieval, cross-referencing chronological account behavior,
        computing statistical baselines, and documenting audit findings.

        **BankShield AI** introduces an autonomous multi-agent pipeline that transforms this manual bottleneck:
        """
    )

    st.markdown(
        """
        - 🧭 **Planner Agent Node**: Decomposes the investigation into an ordered, dynamic 6-step roadmap in LangGraph state.
        - 🔬 **Research Agent Node (MCP Client)**: Invokes standardized Model Context Protocol (MCP) tools to query synthetic records and compute baselines.
        - ⚖️ **Risk Assessment Agent Node**: Evaluates transparent, weighted rules capped at 100 points with evidence attribution.
        - 🛡️ **Critic & Validation Agent Node**: Conducts an independent 9-gate audit and recalculates scores via MCP validation tools.
        - 🔄 **Conditional Correction Node**: Evaluates validation gates; triggers bounded self-correction loops when discrepancies occur.
        """
    )

    st.markdown("---")
    st.markdown("### Getting Started")
    st.write("Select a page from the sidebar navigation to begin the demonstration:")

    c_btn1, c_btn2 = st.columns(2)
    with c_btn1:
        if st.button("📊 Launch Executive Overview", use_container_width=True, type="primary"):
            st.switch_page("pages/1_Executive_Overview.py")
    with c_btn2:
        if st.button("🔍 Launch Transaction Investigation", use_container_width=True):
            st.switch_page("pages/2_Investigate_Transaction.py")

with col2:
    st.markdown("### LangGraph & MCP Architecture")
    st.markdown(
        """
        ```mermaid
        flowchart TD
            User([Compliance Analyst]) <--> UI[Streamlit Interface]
            UI --> Orch[LangGraph Orchestrator]
            subgraph StateGraph [LangGraph State Machine]
                direction TB
                P[planner_node] --> R[research_node]
                R --> D[risk_assessment_node]
                D --> C[critic_node]
                C -->|VALIDATED| Rep[report_generation_node]
                C -->|FAILED & retry < 1| Fix[correction_node]
                Fix --> C
                Rep --> EndNode([END])
            end
            Orch --> StateGraph
            subgraph MCPLayer [Model Context Protocol Server]
                MCPClient[MCP Client] <--> MCPServer[Local MCP Server]
                MCPServer <--> T1[(get_transaction)]
                MCPServer <--> T2[(get_account_history)]
                MCPServer <--> T3[(calculate_deviation)]
                MCPServer <--> T4[(identify_indicators)]
                MCPServer <--> T5[(validate_investigation)]
            end
            R <--> MCPLayer
            C <--> MCPLayer
        ```
        """
    )

    st.markdown(
        """
        <div class="metric-card">
            <strong>Key LangGraph & MCP Guarantees:</strong><br>
            • Typed Shared State (InvestigationState)<br>
            • Standardized Tool Invocation via Model Context Protocol (MCP)<br>
            • Conditional Routing with Bounded Retries (Max 1 Loop)<br>
            • 100% Deterministic Core (Zero Mandatory LLMs, No Paid APIs)
        </div>
        """,
        unsafe_allow_html=True,
    )
