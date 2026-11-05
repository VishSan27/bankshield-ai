"""Page 4: Investigation Reports — BankShield AI.

Provides investigation audit history, filtering by transaction ID and priority tiers,
full evidence dossier inspection, and report export in JSON, CSV, and text formats.
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
from tools.report_tools import format_report_text

st.set_page_config(page_title="Investigation Reports — BankShield AI", page_icon="📋", layout="wide")

# Session state setup
if "data_service" not in st.session_state:
    st.session_state.data_service = DataService()
if "orchestrator" not in st.session_state:
    st.session_state.orchestrator = LangGraphInvestigationOrchestrator(data_service=st.session_state.data_service)

orchestrator = st.session_state.orchestrator

st.title("📋 Investigation Dossiers & Audit Reports")
st.caption("Access persistent session dossiers, review human-analyst recommendations, and export records")

history = orchestrator.get_investigation_history()

if not history:
    st.warning("No investigations recorded in this session yet. Seed an investigation from 'Investigate Transaction'.")
    st.stop()

# Filter Controls
col_f1, col_f2, col_f3 = st.columns(3)

with col_f1:
    search_tx = st.text_input("Filter by Transaction ID:", placeholder="e.g. TXN-10198")
with col_f2:
    priority_filter = st.multiselect(
        "Filter by Priority Tier:",
        options=["Critical", "High", "Medium", "Low"],
        default=["Critical", "High", "Medium", "Low"],
    )
with col_f3:
    status_filter = st.multiselect(
        "Filter by Critic Status:",
        options=["VALIDATED", "NEEDS_REVIEW", "VALIDATION_FAILED"],
        default=["VALIDATED", "NEEDS_REVIEW", "VALIDATION_FAILED"],
    )

# Filter history
filtered_history = []
for inv in history:
    rep = inv.get("final_report", {})
    t_id = rep.get("transaction_id", "")
    pri = rep.get("priority", "Low").capitalize()
    stat = rep.get("status", "VALIDATED")

    if search_tx and search_tx.upper() not in t_id.upper():
        continue
    if pri not in priority_filter:
        continue
    if stat not in status_filter:
        continue
    filtered_history.append(inv)

st.write(f"Showing **{len(filtered_history)}** of **{len(history)}** recorded investigations.")

# History Table
table_data = []
for inv in filtered_history:
    rep = inv.get("final_report", {})
    tx = rep.get("transaction_details", {})
    table_data.append({
        "Investigation ID": rep.get("investigation_id", "-")[:8] + "...",
        "Transaction ID": rep.get("transaction_id", "-"),
        "Account ID": rep.get("account_id", "-"),
        "Amount (INR)": f"{float(tx.get('amount', 0)):,.2f}" if tx.get("amount") else "-",
        "Risk Score": f"{rep.get('risk_score', 0)}/100",
        "Priority Tier": rep.get("priority", "Low"),
        "Critic Status": rep.get("status", "VALIDATED"),
        "Timestamp": rep.get("timestamp", "-")[:19],
    })

if table_data:
    st.dataframe(pd.DataFrame(table_data), use_container_width=True, hide_index=True)

st.markdown("---")

# Select specific report to inspect in detail
st.subheader("Inspect Full 16-Field Investigation Dossier")

report_options = {
    f"{inv.get('final_report', {}).get('transaction_id')} — Priority: {inv.get('final_report', {}).get('priority')} (Score: {inv.get('final_report', {}).get('risk_score')})": inv
    for inv in filtered_history
}

if report_options:
    selected_label = st.selectbox("Select Case to Inspect:", options=list(report_options.keys()))
    chosen_inv = report_options[selected_label]
    chosen_rep = chosen_inv.get("final_report", {})

    col_view1, col_view2 = st.columns([3, 2])

    with col_view1:
        st.markdown(f"### Case Summary: `{chosen_rep.get('transaction_id')}`")
        st.write(f"**Investigation ID:** `{chosen_rep.get('investigation_id')}`")
        st.write(f"**Timestamp:** `{chosen_rep.get('timestamp')}`")
        st.write(f"**Assigned Priority:** `{chosen_rep.get('priority')}` (Score: {chosen_rep.get('risk_score')}/100)")
        st.write(f"**Recommended Analyst Action:** {chosen_rep.get('recommendation')}")
        st.write(f"**Executive Narrative:** {chosen_rep.get('summary')}")

        st.markdown("#### Evidence Summary & Risk Indicators:")
        ev = chosen_rep.get("evidence", {})
        st.write(f"- Summary: {ev.get('evidence_summary', 'None')}")
        for r in chosen_rep.get("triggered_rules", []):
            st.markdown(f"- ⚠️ **{r.get('rule_name')}** (+{r.get('score_contribution')} pts): {r.get('explanation')}")

    with col_view2:
        st.markdown("### Export & Download")

        # JSON Export
        json_str = json.dumps(chosen_rep, indent=2)
        st.download_button(
            label="💾 Download Dossier (JSON)",
            data=json_str,
            file_name=f"investigation_{chosen_rep.get('transaction_id')}.json",
            mime="application/json",
            use_container_width=True,
        )

        # Formatted Plain Text Export
        text_rep = format_report_text(chosen_rep)
        st.download_button(
            label="📄 Download Formatted Report (TXT)",
            data=text_rep,
            file_name=f"investigation_{chosen_rep.get('transaction_id')}.txt",
            mime="text/plain",
            use_container_width=True,
        )

        # Batch CSV Export
        all_reps = [inv.get("final_report", {}) for inv in filtered_history]
        csv_df = pd.DataFrame([
            {
                "investigation_id": r.get("investigation_id"),
                "transaction_id": r.get("transaction_id"),
                "account_id": r.get("account_id"),
                "risk_score": r.get("risk_score"),
                "priority": r.get("priority"),
                "status": r.get("status"),
                "recommendation": r.get("recommendation"),
                "timestamp": r.get("timestamp"),
            }
            for r in all_reps
        ])
        st.download_button(
            label="📊 Download Filtered Session Cases (CSV)",
            data=csv_df.to_csv(index=False),
            file_name="bankshield_investigations.csv",
            mime="text/csv",
            use_container_width=True,
        )

    # Formatted Plain Text Dossier Preview
    with st.expander("📄 View Formatted Official Plain-Text Audit Dossier"):
        st.code(text_rep, language="text")
