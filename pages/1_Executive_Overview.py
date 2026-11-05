"""Page 1: Executive Overview Dashboard — BankShield AI.

Displays high-level portfolio metrics, transaction volume breakdowns, risk priority
distributions, validation audit rates, and recent investigation activity.
"""

from pathlib import Path
import sys

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from orchestration.langgraph_orchestrator import LangGraphInvestigationOrchestrator
from services.data_service import DataService

st.set_page_config(page_title="Executive Overview — BankShield AI", page_icon="📊", layout="wide")

# Initialize shared session services
if "data_service" not in st.session_state:
    st.session_state.data_service = DataService()

if "orchestrator" not in st.session_state:
    st.session_state.orchestrator = LangGraphInvestigationOrchestrator(data_service=st.session_state.data_service)

data_service = st.session_state.data_service
orchestrator = st.session_state.orchestrator

# Pre-populate sample investigations if session history is empty so executive overview has real data to display
if len(orchestrator.get_investigation_history()) == 0:
    # Seed with 4 illustrative investigations (Critical, High, Medium, Low)
    sample_seed_ids = ["TXN-10198", "TXN-10471", "TXN-10015", "TXN-10078"]
    for sid in sample_seed_ids:
        try:
            orchestrator.investigate_transaction(sid)
        except Exception:
            pass

# UI Header
st.title("📊 Executive Overview & Portfolio Risk Monitoring")
st.caption("Real-time operational metrics across synthetic banking transactions and agentic investigations")

st.markdown(
    """
    <div style="background-color: #FEF3C7; color: #92400E; padding: 10px 16px; border-radius: 6px; border-left: 4px solid #F59E0B; margin-bottom: 1.2rem; font-size: 0.9rem;">
        ⚠️ <strong>SYNTHETIC DATA DISCLAIMER:</strong> All portfolio figures and transaction records are synthetically generated for demonstration.
        High risk scores reflect rule-based anomaly detection for human decision support, not definitive fraud.
    </div>
    """,
    unsafe_allow_html=True,
)

# Load raw transaction dataset
tx_df, acc_df = data_service.load_data()
history = orchestrator.get_investigation_history()

# Compute live KPIs from actual data
total_txns = len(tx_df)
total_investigated = len(history)

# Count priority tiers from completed investigations
priority_counts = {"Critical": 0, "High": 0, "Medium": 0, "Low": 0}
validated_count = 0
needs_review_count = 0

for inv in history:
    rep = inv.get("final_report", {})
    p = rep.get("priority", "Low").capitalize()
    if p in priority_counts:
        priority_counts[p] += 1
    else:
        priority_counts[p] = 1

    status = rep.get("status", inv.get("validation_status", "NEEDS_REVIEW"))
    if status == "VALIDATED":
        validated_count += 1
    else:
        needs_review_count += 1

high_critical_cases = priority_counts["Critical"] + priority_counts["High"]
val_pass_rate = round((validated_count / total_investigated * 100), 1) if total_investigated > 0 else 100.0

# 6 Executive Metric Cards
c1, c2, c3, c4, c5, c6 = st.columns(6)

with c1:
    st.metric(label="Total Transactions", value=f"{total_txns:,}")
with c2:
    st.metric(label="Investigated Cases", value=f"{total_investigated}", delta=f"{total_investigated} in session")
with c3:
    st.metric(label="High / Critical", value=f"{high_critical_cases}", delta="Require review", delta_color="inverse")
with c4:
    st.metric(label="Needs Review / Warning", value=f"{needs_review_count}", delta="Audit flag", delta_color="inverse")
with c5:
    st.metric(label="Validation Pass Rate", value=f"{val_pass_rate}%", delta="Critic integrity")
with c6:
    st.metric(label="Active Accounts", value=f"{len(acc_df)}")

st.markdown("---")

# Visual Charts Row
chart_col1, chart_col2 = st.columns(2)

with chart_col1:
    st.subheader("Risk Priority Distribution (Investigated Cases)")
    pri_df = pd.DataFrame(
        [{"Priority Tier": k, "Count": v} for k, v in priority_counts.items()]
    )
    fig_pri = px.pie(
        pri_df,
        names="Priority Tier",
        values="Count",
        color="Priority Tier",
        color_discrete_map={
            "Critical": "#EF4444",
            "High": "#F97316",
            "Medium": "#FBBF24",
            "Low": "#10B981",
        },
        hole=0.45,
    )
    fig_pri.update_layout(margin=dict(t=20, b=20, l=20, r=20), height=320)
    st.plotly_chart(fig_pri, use_container_width=True)

with chart_col2:
    st.subheader("Transaction Amount Volume by Channel")
    if "channel" in tx_df.columns and "amount" in tx_df.columns:
        channel_vol = tx_df.groupby("channel")["amount"].agg(["sum", "count"]).reset_index()
        fig_channel = px.bar(
            channel_vol,
            x="channel",
            y="sum",
            text="count",
            labels={"sum": "Total Volume (INR)", "channel": "Channel", "count": "Txn Count"},
            color="channel",
            color_discrete_sequence=px.colors.qualitative.Prism,
        )
        fig_channel.update_traces(texttemplate="%{text} txns", textposition="outside")
        fig_channel.update_layout(margin=dict(t=20, b=20, l=20, r=20), height=320, showlegend=False)
        st.plotly_chart(fig_channel, use_container_width=True)
    else:
        st.info("Transaction channel data not available for visualization.")

st.markdown("---")

# Recent Investigation Activity Table
st.subheader("Recent Multi-Agent Investigation Cases")
if history:
    table_rows = []
    for inv in reversed(history[-10:]):
        rep = inv.get("final_report", {})
        tx = rep.get("transaction_details", {})
        val = rep.get("validation_result", {})
        table_rows.append({
            "Investigation ID": rep.get("investigation_id", "-")[:8] + "...",
            "Transaction ID": rep.get("transaction_id", "-"),
            "Account ID": rep.get("account_id", "-"),
            "Amount (INR)": f"{float(tx.get('amount', 0)):,.2f}" if tx.get("amount") else "-",
            "Merchant": tx.get("merchant", "-"),
            "Risk Score": f"{rep.get('risk_score', 0)}/100",
            "Priority Tier": rep.get("priority", "Low"),
            "Critic Status": rep.get("status", "VALIDATED"),
            "Execution Time": f"{inv.get('total_duration_ms', 0):.2f} ms",
        })

    hist_df = pd.DataFrame(table_rows)

    def highlight_priority(val):
        if val == "Critical":
            return "background-color: #FEE2E2; color: #991B1B; font-weight: bold;"
        elif val == "High":
            return "background-color: #FFEDD5; color: #9A3412; font-weight: bold;"
        elif val == "Medium":
            return "background-color: #FEF3C7; color: #92400E;"
        elif val == "Low":
            return "background-color: #D1FAE5; color: #065F46;"
        return ""

    styled_df = hist_df.style.map(highlight_priority, subset=["Priority Tier"])
    st.dataframe(styled_df, use_container_width=True, hide_index=True)
else:
    st.info("No investigations completed in this session yet. Navigate to 'Investigate Transaction' to run an investigation.")
