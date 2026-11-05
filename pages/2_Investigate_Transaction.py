"""Page 2: Investigate Transaction — BankShield AI.

Interactive workbench allowing compliance analysts to select or input transactions,
trigger single or batch agent investigations, view real-time evidence retrieval,
inspect deterministic rule contributions, review critic audit checks, and generate decision briefings.
"""

from pathlib import Path
import sys

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from orchestration.langgraph_orchestrator import LangGraphInvestigationOrchestrator
from services.data_service import DataService
from services.llm_summary_service import generate_executive_narrative

st.set_page_config(page_title="Investigate Transaction — BankShield AI", page_icon="🔍", layout="wide")

# Session state setup
if "data_service" not in st.session_state:
    st.session_state.data_service = DataService()
if "orchestrator" not in st.session_state:
    st.session_state.orchestrator = LangGraphInvestigationOrchestrator(data_service=st.session_state.data_service)

data_service = st.session_state.data_service
orchestrator = st.session_state.orchestrator

st.title("🔍 Banking Transaction Risk Investigation Workbench")
st.caption("Execute autonomous multi-agent risk analysis with evidence attribution and critic validation")

# Preset benchmark transactions for quick 5-minute presentation
PRESET_DEMO_CASES = {
    "TXN-10198 (High Risk — Extreme Amount & Unseen Merchant)": "TXN-10198",
    "TXN-10471 (Critical Risk — Compound High Amount + 5 Failed Attempts)": "TXN-10471",
    "TXN-10015 (Medium Risk — Repeated Failed Authorizations)": "TXN-10015",
    "TXN-10016 (Medium Risk — Off-Hours Nocturnal Activity)": "TXN-10016",
    "TXN-10078 (Low Risk — Routine Daytime Commercial Activity)": "TXN-10078",
    "TXN-99999 (Error Scenario — Non-Existent Record)": "TXN-99999",
}

tab_single, tab_batch = st.tabs(["🎯 Single Transaction Investigation", "📦 Batch Investigation Pipeline"])

# ==============================================================================
# TAB 1: SINGLE INVESTIGATION
# ==============================================================================
with tab_single:
    col_input1, col_input2 = st.columns([2, 1])

    with col_input1:
        st.subheader("1. Select or Search Transaction")
        preset_choice = st.selectbox(
            "Recommended Demonstration Scenarios:",
            options=list(PRESET_DEMO_CASES.keys()),
            index=0,
            help="Select one of the pre-configured scenarios designed to illustrate distinct risk behaviors.",
        )
        selected_preset_id = PRESET_DEMO_CASES[preset_choice]

        all_tx_ids = data_service.get_all_transaction_ids()
        custom_id_input = st.text_input(
            "Or enter any Transaction ID:",
            value=selected_preset_id,
            help="Type an existing transaction ID (e.g. TXN-10001) or test error handling.",
        )

    with col_input2:
        st.subheader("Investigation Scope")
        investigation_objective = st.text_input(
            "Investigation Objective:",
            value="Investigate transaction risk indicators and baseline deviations",
        )
        enable_llm_briefing = st.checkbox(
            "Generate Natural-Language Briefing (Optional Local LLM / Deterministic Fallback)",
            value=True,
            help="Uses local Ollama if running, otherwise transparently synthesizes deterministic briefing.",
        )

    run_btn = st.button("🚀 Run Multi-Agent Investigation", type="primary", use_container_width=True)

    if run_btn:
        target_id = custom_id_input.strip()
        if not target_id:
            st.error("Please enter or select a valid Transaction ID.")
        else:
            with st.spinner(f"Orchestrating agents for transaction {target_id}..."):
                # Execute orchestrated workflow
                investigation_result = orchestrator.investigate_transaction(
                    transaction_id=target_id,
                    objective=investigation_objective,
                )
                st.session_state["latest_investigation"] = investigation_result

    # Display results if available in session
    if "latest_investigation" in st.session_state:
        res = st.session_state["latest_investigation"]
        report = res.get("final_report", {})
        tx = report.get("transaction_details", {})
        risk_res = res.get("risk_result", {})
        validation = report.get("validation_result", {})
        plan = res.get("plan", {})
        evidence = res.get("evidence", {})

        st.markdown("---")
        st.subheader("2. Investigation Results & Dossier")

        # Handle non-existent error case
        if res.get("status") == "failed" or not tx:
            st.error(f"❌ Investigation could not be completed for ID: {res.get('transaction_id')}")
            st.warning(f"Error Detail: {report.get('error', 'Transaction not found in database.')}")
            st.info(
                "Critic & Orchestrator successfully handled the invalid input without application crash."
            )
        else:
            # High-level outcome banners
            score = report.get("risk_score", 0)
            priority = report.get("priority", "Low")
            val_status = report.get("status", "VALIDATED")

            # Color theme
            if priority == "Critical":
                color_bg = "#FEE2E2"
                color_border = "#EF4444"
                text_color = "#991B1B"
            elif priority == "High":
                color_bg = "#FFEDD5"
                color_border = "#F97316"
                text_color = "#9A3412"
            elif priority == "Medium":
                color_bg = "#FEF3C7"
                color_border = "#FBBF24"
                text_color = "#92400E"
            else:
                color_bg = "#D1FAE5"
                color_border = "#10B981"
                text_color = "#065F46"

            st.markdown(
                f"""
                <div style="background-color: {color_bg}; border-left: 6px solid {color_border}; padding: 16px; border-radius: 6px; margin-bottom: 20px;">
                    <div style="font-size: 1.3rem; font-weight: 700; color: {text_color};">
                        Priority Level: {priority.upper()} (Risk Score: {score}/100) | Critic Audit: {val_status}
                    </div>
                    <div style="font-size: 0.95rem; color: #374151; margin-top: 6px;">
                        <strong>Analyst Recommendation:</strong> {report.get('recommendation', 'Review case.')}
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

            # 5 Summary KPI columns with LangGraph and MCP indicators
            m1, m2, m3, m4, m5 = st.columns(5)
            with m1:
                st.metric("Amount", f"INR {float(tx.get('amount', 0)):,.2f}")
            with m2:
                st.metric("Account ID", tx.get("account_id", "N/A"))
            with m3:
                st.metric("LangGraph Nodes", f"{len(res.get('node_history', []))} Nodes", delta="State Machine")
            with m4:
                st.metric("MCP Tool Calls", f"{len(res.get('mcp_tool_calls', []))} Tools", delta="Model Context Protocol")
            with m5:
                retry_c = res.get("retry_count", 0)
                st.metric("Critic Retries", f"{retry_c}/1", delta="Bounded Loop" if retry_c > 0 else "First-Pass Pass")

            # LangGraph State Route Badge
            node_path_str = " ➔ ".join([f"`{n}`" for n in res.get("node_history", [])])
            st.info(f"🕸️ **LangGraph Active Execution Path:** {node_path_str}")

            st.markdown("### Risk Indicator Contributions")
            triggered_rules = report.get("triggered_rules", [])

            col_rules, col_chart = st.columns([3, 2])

            with col_rules:
                if triggered_rules:
                    for rule in triggered_rules:
                        r_name = rule.get("rule_name", rule.get("name", "Rule"))
                        r_pts = rule.get("score_contribution", rule.get("weight", 0))
                        r_exp = rule.get("explanation", "")
                        r_ref = rule.get("evidence_ref", "")

                        st.markdown(
                            f"""
                            <div style="background-color: #F8FAFC; border: 1px solid #E2E8F0; padding: 10px 14px; border-radius: 6px; margin-bottom: 8px;">
                                <div style="display: flex; justify-content: space-between;">
                                    <span style="font-weight: 600; color: #1E293B;">⚠️ {r_name.replace('_', ' ').title()}</span>
                                    <span style="font-weight: 700; color: #DC2626;">+{r_pts} Points</span>
                                </div>
                                <div style="font-size: 0.85rem; color: #475569; margin-top: 4px;">{r_exp}</div>
                                <div style="font-size: 0.8rem; color: #64748B; margin-top: 2px;"><em>Supporting Evidence: {r_ref}</em></div>
                            </div>
                            """,
                            unsafe_allow_html=True,
                        )
                else:
                    st.success("✅ No anomalous risk indicators were triggered for this transaction.")

                untriggered = report.get("untriggered_rules", [])
                if untriggered:
                    st.caption(f"Evaluated normal indicators: {', '.join([u.get('rule_name', u.get('name', '')) for u in untriggered])}")

            with col_chart:
                # Score gauge chart
                fig_gauge = go.Figure(
                    go.Indicator(
                        mode="gauge+number",
                        value=score,
                        domain={"x": [0, 1], "y": [0, 1]},
                        title={"text": "Computed Risk Score"},
                        gauge={
                            "axis": {"range": [0, 100]},
                            "bar": {"color": color_border},
                            "steps": [
                                {"range": [0, 30], "color": "#D1FAE5"},
                                {"range": [30, 60], "color": "#FEF3C7"},
                                {"range": [60, 80], "color": "#FFEDD5"},
                                {"range": [80, 100], "color": "#FEE2E2"},
                            ],
                            "threshold": {
                                "line": {"color": "black", "width": 4},
                                "thickness": 0.75,
                                "value": score,
                            },
                        },
                    )
                )
                fig_gauge.update_layout(height=260, margin=dict(t=30, b=10, l=20, r=20))
                st.plotly_chart(fig_gauge, use_container_width=True)

            # Executive Decision Briefing (Optional LLM or deterministic fallback)
            st.markdown("### Executive Decision Support Briefing")
            briefing = generate_executive_narrative(report, use_llm_if_available=True)
            st.markdown(
                f"""
                <div style="background-color: #EFF6FF; border: 1px solid #BFDBFE; padding: 14px; border-radius: 6px; font-size: 0.95rem; color: #1E3A8A;">
                    <div style="font-size: 0.8rem; font-weight: 600; color: #3B82F6; margin-bottom: 6px;">
                        OPERATIONAL MODE: {briefing.get('mode')}
                    </div>
                    {briefing.get('narrative').replace('\n', '<br>')}
                </div>
                """,
                unsafe_allow_html=True,
            )

            # Technical Details & Critic Audit Accordion
            with st.expander("🔬 View Technical Evidence, Account Baseline & Critic 9-Gate Audit", expanded=False):
                col_tech1, col_tech2 = st.columns(2)

                with col_tech1:
                    st.markdown("**Critic Agent Audit Checkpoints:**")
                    checks = validation.get("checks_performed", [])
                    for chk in checks:
                        chk_name = chk.get("name", "").replace("_", " ").title()
                        passed = chk.get("passed", False)
                        icon = "✅" if passed else "❌"
                        st.markdown(f"{icon} **{chk_name}**: {chk.get('details', '')}")

                    st.markdown(f"**Score Integrity Check:** Reported={score} | Recalculated={validation.get('recalculated_score')} (Match: {validation.get('score_matches')})")

                with col_tech2:
                    st.markdown("**Historical Account Baseline:**")
                    profile = evidence.get("account_profile", {})
                    st.write(f"- Historical Average Amount: INR {profile.get('historical_average_amount', 0):,.2f}")
                    st.write(f"- Daily Velocity Baseline: {profile.get('historical_transaction_frequency', 0)} txns/day")
                    st.write(f"- Usual Active Hours: {profile.get('usual_transaction_hours', 'N/A')}")
                    st.write(f"- Account Age: {profile.get('account_age_days', 0)} days")

                    dev = evidence.get("deviation_analysis", {})
                    st.write(f"- Amount Z-Score: {dev.get('z_score', 'N/A')}")
                    st.write(f"- Amount Deviation Ratio: {dev.get('deviation_ratio', 'N/A')}x baseline")

# ==============================================================================
# TAB 2: BATCH INVESTIGATION
# ==============================================================================
with tab_batch:
    st.subheader("Batch Transaction Pipeline")
    st.write("Execute high-throughput multi-agent investigation across multiple transactions simultaneously.")

    all_ids = data_service.get_all_transaction_ids()
    batch_selection = st.multiselect(
        "Select Transactions for Batch Investigation:",
        options=all_ids,
        default=all_ids[:5],
        help="Choose a subset of transactions to process through the pipeline.",
    )

    if st.button("⚡ Run Batch Investigation", type="primary"):
        if not batch_selection:
            st.warning("Please select at least one transaction.")
        else:
            with st.spinner(f"Processing batch of {len(batch_selection)} transactions..."):
                batch_res = orchestrator.investigate_batch(batch_selection)
                st.session_state["latest_batch"] = batch_res

    if "latest_batch" in st.session_state:
        b_res = st.session_state["latest_batch"]
        summary = b_res.get("batch_summary", {})
        results_list = b_res.get("results", [])

        st.markdown("---")
        st.subheader("Batch Investigation Summary")

        bc1, bc2, bc3, bc4 = st.columns(4)
        with bc1:
            st.metric("Total Processed", summary.get("total_investigated", 0))
        with bc2:
            st.metric("Critical Cases", summary.get("critical_cases", 0))
        with bc3:
            st.metric("High Cases", summary.get("high_cases", 0))
        with bc4:
            st.metric("Average Risk Score", f"{summary.get('average_risk_score', 0):.1f}/100")

        # Table of batch results
        batch_rows = []
        for r in results_list:
            rep = r.get("final_report", {})
            tx_info = rep.get("transaction_details", {})
            batch_rows.append({
                "Transaction ID": r.get("transaction_id"),
                "Account ID": tx_info.get("account_id", "-"),
                "Amount (INR)": f"{float(tx_info.get('amount', 0)):,.2f}" if tx_info.get("amount") else "-",
                "Risk Score": rep.get("risk_score", 0),
                "Priority": rep.get("priority", "Low"),
                "Critic Status": rep.get("status", "VALIDATED"),
                "Recommendation": rep.get("recommendation", "-"),
            })

        st.dataframe(pd.DataFrame(batch_rows), use_container_width=True, hide_index=True)
