"""Generate High-Resolution Presentation Diagrams for BankShield AI.

Generates 4 publication-quality PNG diagrams for insertion into PowerPoint:
1. docs/diagrams/slide3_process_comparison.png (Slide 3: Manual vs. Agentic Workflow)
2. docs/diagrams/slide4_system_architecture.png (Slide 4: LangGraph State Machine & MCP Architecture)
3. docs/diagrams/slide5_investigation_workbench.png (Slide 5: Live UI Workbench & Risk Gauge)
4. docs/diagrams/slide6_critic_quality_gates.png (Slide 6: Critic Agent 9-Gate Quality Audit)
"""

from pathlib import Path
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, Circle, Wedge
import numpy as np

# Ensure diagrams directory exists
OUT_DIR = Path(__file__).resolve().parent.parent / "docs" / "diagrams"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# Shared Color Palette matching presentation deck
COLOR_PRIMARY = "#1E3A8A"      # Deep Navy
COLOR_SECONDARY = "#0F172A"    # Dark Slate
COLOR_ACCENT = "#D97706"       # Amber / Gold
COLOR_SUCCESS = "#059669"      # Emerald Green
COLOR_DANGER = "#DC2626"       # Crimson Red
COLOR_MUTED = "#64748B"        # Slate Gray
COLOR_BG_LIGHT = "#F8FAFC"     # Card Light Gray
COLOR_BORDER = "#CBD5E1"       # Light Slate Border


# ==============================================================================
# DIAGRAM 1: PROCESS COMPARISON (SLIDE 3)
# ==============================================================================
def generate_diagram_slide3():
    fig, ax = plt.subplots(figsize=(14, 7.5), dpi=300)
    fig.patch.set_facecolor("#FFFFFF")
    ax.set_facecolor("#FFFFFF")
    ax.set_xlim(0, 14)
    ax.set_ylim(0, 8)
    ax.axis("off")

    # Header Banner
    ax.text(7.0, 7.5, "BANKING RISK INVESTIGATION WORKFLOW COMPARISON", 
            ha="center", va="center", fontsize=18, fontweight="bold", color=COLOR_PRIMARY)
    ax.text(7.0, 7.1, "Conventional Manual Banking Bottlenecks vs. BankShield AI Autonomous Multi-Agent Pipeline", 
            ha="center", va="center", fontsize=11, color=COLOR_MUTED)

    # ------------------ LEFT COLUMN: TRADITIONAL MANUAL ------------------
    box_left = FancyBboxPatch((0.5, 0.4), 6.0, 6.3, boxstyle="round,pad=0.2",
                              facecolor="#FEF2F2", edgecolor="#FCA5A5", linewidth=2)
    ax.add_patch(box_left)

    ax.text(3.5, 6.3, "TRADITIONAL MANUAL WORKFLOW", ha="center", va="center",
            fontsize=13, fontweight="bold", color=COLOR_DANGER)
    ax.text(3.5, 6.0, "High Operational Latency (10 - 15 Minutes per Alert)", ha="center", va="center",
            fontsize=9.5, fontweight="semibold", color="#991B1B")

    steps_manual = [
        ("1. Alert Ingestion", "Monitoring engine flags transaction anomaly alert"),
        ("2. Fragmented DB Querying", "Manual SQL lookup across core banking, card logs & history"),
        ("3. Mental Math Estimation", "Analyst mentally estimates spending deviation & normal hours"),
        ("4. Subjective Risk Weighting", "Inconsistent heuristics applied across analyst shifts"),
        ("5. Manual Narrative Typing", "Repetitive typing of 16-field dossier leads to fatigue & errors"),
    ]

    y_pos = 5.2
    for title, desc in steps_manual:
        step_box = FancyBboxPatch((1.0, y_pos - 0.35), 5.0, 0.65, boxstyle="round,pad=0.1",
                                  facecolor="#FFFFFF", edgecolor="#F87171", linewidth=1.2)
        ax.add_patch(step_box)
        ax.text(1.2, y_pos + 0.08, title, fontsize=10, fontweight="bold", color=COLOR_SECONDARY)
        ax.text(1.2, y_pos - 0.15, desc, fontsize=8, color="#475569")
        
        if y_pos > 1.6:
            ax.annotate("", xy=(3.5, y_pos - 0.45), xytext=(3.5, y_pos - 0.35),
                        arrowprops=dict(arrowstyle="->", color=COLOR_DANGER, lw=1.5))
        y_pos -= 0.85

    # Bottom summary box manual
    summary_box_left = FancyBboxPatch((1.0, 0.6), 5.0, 0.6, boxstyle="round,pad=0.1",
                                      facecolor="#FEE2E2", edgecolor="#EF4444", linewidth=1.5)
    ax.add_patch(summary_box_left)
    ax.text(3.5, 0.9, "[!] Bottleneck: 10-15 Min Latency | Subjective | Zero Cross-Audit",
            ha="center", va="center", fontsize=9, fontweight="bold", color="#7F1D1D")

    # ------------------ RIGHT COLUMN: BANKSHIELD AI ------------------
    box_right = FancyBboxPatch((7.5, 0.4), 6.0, 6.3, boxstyle="round,pad=0.2",
                               facecolor="#F0FDF4", edgecolor="#86EFAC", linewidth=2)
    ax.add_patch(box_right)

    ax.text(10.5, 6.3, "BANKSHIELD AI AGENTIC PIPELINE", ha="center", va="center",
            fontsize=13, fontweight="bold", color=COLOR_SUCCESS)
    ax.text(10.5, 6.0, "Sub-Second Computation (~27 ms) with Independent Critic Audit", ha="center", va="center",
            fontsize=9.5, fontweight="semibold", color="#166534")

    steps_agentic = [
        ("1. Planner Node (LangGraph)", "Decomposes investigation into dynamic 6-step roadmap in state"),
        ("2. Research Node (MCP Client)", "Calls 4 MCP tools; computes exact z-scores, velocity & history"),
        ("3. Risk Assessment Node", "Evaluates 5 deterministic rules capped at 100 with evidence citations"),
        ("4. Critic Node & 9-Gate Audit", "Independently recalculates math; triggers self-correction loop if needed"),
        ("5. Report Generation Node", "Compiles standardized 16-field dossier with JSON / CSV / TXT export"),
    ]

    y_pos = 5.2
    for title, desc in steps_agentic:
        step_box = FancyBboxPatch((8.0, y_pos - 0.35), 5.0, 0.65, boxstyle="round,pad=0.1",
                                  facecolor="#FFFFFF", edgecolor="#4ADE80", linewidth=1.2)
        ax.add_patch(step_box)
        ax.text(8.2, y_pos + 0.08, title, fontsize=10, fontweight="bold", color=COLOR_PRIMARY)
        ax.text(8.2, y_pos - 0.15, desc, fontsize=8, color="#475569")
        
        if y_pos > 1.6:
            ax.annotate("", xy=(10.5, y_pos - 0.45), xytext=(10.5, y_pos - 0.35),
                        arrowprops=dict(arrowstyle="->", color=COLOR_SUCCESS, lw=1.5))
        y_pos -= 0.85

    # Bottom summary box agentic
    summary_box_right = FancyBboxPatch((8.0, 0.6), 5.0, 0.6, boxstyle="round,pad=0.1",
                                       facecolor="#DCFCE7", edgecolor="#22C55E", linewidth=1.5)
    ax.add_patch(summary_box_right)
    ax.text(10.5, 0.9, "[OK] Speedup: ~27 ms Latency | 100% Explainable | Critic Verified",
            ha="center", va="center", fontsize=9, fontweight="bold", color="#14532D")

    # Center Comparison Divider
    ax.plot([7.0, 7.0], [0.6, 6.7], linestyle="--", color="#94A3B8", linewidth=1.5)
    vs_circle = Circle((7.0, 3.6), 0.35, facecolor="#F8FAFC", edgecolor="#64748B", linewidth=1.5)
    ax.add_patch(vs_circle)
    ax.text(7.0, 3.6, "VS", ha="center", va="center", fontsize=10, fontweight="bold", color=COLOR_SECONDARY)

    plt.tight_layout()
    p = OUT_DIR / "slide3_process_comparison.png"
    plt.savefig(p, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Generated: {p}")


# ==============================================================================
# DIAGRAM 2: SYSTEM ARCHITECTURE (SLIDE 4) - COMPLETE COMMUNICATION & RESOURCE EDITION
# ==============================================================================
def generate_diagram_slide4():
    fig, ax = plt.subplots(figsize=(16, 9.4), dpi=300)
    fig.patch.set_facecolor("#FFFFFF")
    ax.set_facecolor("#FFFFFF")
    ax.set_xlim(0, 16)
    ax.set_ylim(0, 9.4)
    ax.axis("off")

    # ------------------ TOP MAIN HEADER ------------------
    ax.text(8.0, 9.08, "BANKSHIELD AI — SYSTEM ARCHITECTURE & COMMUNICATION LAYERS", 
            ha="center", va="center", fontsize=18, fontweight="bold", color=COLOR_PRIMARY)
    ax.text(8.0, 8.75, "Inter-Agent State Bus  |  MCP Tool Communication Protocol  |  Resource Allocation & Execution Governance", 
            ha="center", va="center", fontsize=10.5, color=COLOR_MUTED)

    # ==========================================================================
    # LAYER 1: USER & PRESENTATION LAYER (TOP ROW)
    # ==========================================================================
    layer1_box = FancyBboxPatch((0.6, 7.65), 14.8, 0.92, boxstyle="round,pad=0.1",
                                facecolor="#EFF6FF", edgecolor="#3B82F6", linewidth=1.8)
    ax.add_patch(layer1_box)

    # Layer Tag Pill
    l1_tag = FancyBboxPatch((0.9, 8.18), 3.2, 0.3, boxstyle="round,pad=0.03",
                            facecolor="#1D4ED8", edgecolor="#1D4ED8", linewidth=1.0)
    ax.add_patch(l1_tag)
    ax.text(2.5, 8.33, "LAYER 1: PRESENTATION LAYER", ha="center", va="center", fontsize=8.2, fontweight="bold", color="#FFFFFF")

    # User Box
    u_box = FancyBboxPatch((4.5, 7.78), 3.1, 0.62, boxstyle="round,pad=0.05",
                           facecolor="#FFFFFF", edgecolor="#93C5FD", linewidth=1.2)
    ax.add_patch(u_box)
    ax.text(6.05, 8.15, "Compliance Analyst", ha="center", va="center", fontsize=9.2, fontweight="bold", color=COLOR_PRIMARY)
    ax.text(6.05, 7.92, "Submits Transaction ID (e.g. TXN-10198)", ha="center", va="center", fontsize=7.2, color="#475569")

    # Double Arrow User <-> UI
    ax.annotate("", xy=(8.0, 8.09), xytext=(7.6, 8.09), arrowprops=dict(arrowstyle="<->", color="#2563EB", lw=2.0))

    # Streamlit UI Box
    ui_box = FancyBboxPatch((8.1, 7.78), 7.0, 0.62, boxstyle="round,pad=0.05",
                            facecolor="#FFFFFF", edgecolor="#93C5FD", linewidth=1.2)
    ax.add_patch(ui_box)
    ax.text(11.6, 8.15, "Streamlit Enterprise Dashboard (Multi-Page Web UI)", ha="center", va="center", fontsize=9.2, fontweight="bold", color=COLOR_PRIMARY)
    ax.text(11.6, 7.92, "Interactive Workbench  |  Plotly Risk Gauge  |  Live State Inspector  |  JSON/CSV Export", ha="center", va="center", fontsize=7.2, color="#475569")

    # Connecting Arrow Layer 1 -> Layer 2
    ax.annotate("", xy=(8.0, 7.22), xytext=(8.0, 7.65), arrowprops=dict(arrowstyle="->", color=COLOR_PRIMARY, lw=2.2))
    conn_pill1 = FancyBboxPatch((6.0, 7.35), 4.0, 0.26, boxstyle="round,pad=0.03",
                                facecolor="#FFFFFF", edgecolor=COLOR_PRIMARY, linewidth=1.0)
    ax.add_patch(conn_pill1)
    ax.text(8.0, 7.48, "Dispatches Case to LangGraph Orchestrator", ha="center", va="center", fontsize=7.2, fontweight="bold", color=COLOR_PRIMARY)

    # ==========================================================================
    # LAYER 2: LANGGRAPH MULTI-AGENT STATE MACHINE & COMMUNICATION BUS (MIDDLE)
    # ==========================================================================
    layer2_box = FancyBboxPatch((0.6, 3.35), 14.8, 3.85, boxstyle="round,pad=0.12",
                                facecolor="#F8FAFC", edgecolor=COLOR_PRIMARY, linewidth=2.2)
    ax.add_patch(layer2_box)

    # Layer Tag Pill
    l2_tag = FancyBboxPatch((0.9, 6.82), 7.2, 0.3, boxstyle="round,pad=0.03",
                            facecolor=COLOR_PRIMARY, edgecolor=COLOR_PRIMARY, linewidth=1.0)
    ax.add_patch(l2_tag)
    ax.text(4.5, 6.97, "LAYER 2: MULTI-AGENT STATE MACHINE (LangGraph StateGraph & State Bus)", ha="center", va="center", fontsize=8.2, fontweight="bold", color="#FFFFFF")

    ax.text(15.1, 6.97, "Inter-Agent Protocol: Typed State Transitions (No Free-Text Drift)", ha="right", va="center", fontsize=7.5, fontweight="bold", color=COLOR_MUTED)

    # --- 5 AGENT CARDS (TOP ROW OF LAYER 2) ---
    agent_cards = [
        # (step, node_name, agent_lbl, desc, x, w, bg, bdr, badge_col)
        ("STEP 1", "planner_node", "Planner Agent", "Decomposes goal into\n6-step execution plan", 0.9, 2.5, "#DBEAFE", "#3B82F6", "#1D4ED8"),
        ("STEP 2", "research_node", "Research Agent", "Calls 4 MCP tools;\nqueries account history", 3.8, 2.6, "#E0E7FF", "#6366F1", "#4338CA"),
        ("STEP 3", "risk_node", "Risk Assessment Agent", "Scores 5 weighted rules\n(0-100) with evidence", 6.8, 2.6, "#FEF3C7", "#F59E0B", "#B45309"),
        ("STEP 4", "critic_node", "Critic Agent (Audit)", "Independent 9-gate audit;\nrecalculates score via MCP", 9.8, 2.6, "#F3E8FF", "#A855F7", "#7E22CE"),
        ("STEP 5", "report_node", "Report Generator", "Synthesizes standardized\n16-field regulatory dossier", 12.8, 2.3, "#DCFCE7", "#22C55E", "#15803D"),
    ]

    for step_num, node_name, agent_lbl, desc, x_pos, w_card, bg_col, bdr_col, badge_col in agent_cards:
        ac = FancyBboxPatch((x_pos, 5.3), w_card, 1.4, boxstyle="round,pad=0.06",
                            facecolor=bg_col, edgecolor=bdr_col, linewidth=1.5)
        ax.add_patch(ac)

        s_pill = FancyBboxPatch((x_pos + 0.12, 6.32), 1.1, 0.24, boxstyle="round,pad=0.03",
                                facecolor=badge_col, edgecolor=badge_col, linewidth=1.0)
        ax.add_patch(s_pill)
        ax.text(x_pos + 0.67, 6.44, step_num, ha="center", va="center", fontsize=7.0, fontweight="bold", color="#FFFFFF")

        ax.text(x_pos + w_card - 0.12, 6.44, node_name, ha="right", va="center", fontsize=6.8, color="#64748B", style="italic")
        ax.text(x_pos + 0.12, 6.0, agent_lbl, fontsize=8.8, fontweight="bold", color=COLOR_SECONDARY)
        ax.text(x_pos + 0.12, 5.55, desc, fontsize=7.2, color="#334155")

    # Direct Forward Pipeline Flow Arrows
    ax.annotate("", xy=(3.75, 5.95), xytext=(3.45, 5.95), arrowprops=dict(arrowstyle="->", color="#3B82F6", lw=2.0))
    ax.annotate("", xy=(6.75, 5.95), xytext=(6.45, 5.95), arrowprops=dict(arrowstyle="->", color="#6366F1", lw=2.0))
    ax.annotate("", xy=(9.75, 5.95), xytext=(9.45, 5.95), arrowprops=dict(arrowstyle="->", color="#F59E0B", lw=2.0))
    ax.annotate("", xy=(12.75, 5.95), xytext=(12.45, 5.95), arrowprops=dict(arrowstyle="->", color=COLOR_SUCCESS, lw=2.4))
    ax.text(12.6, 6.1, "OK", ha="center", va="center", fontsize=6.5, fontweight="bold", color=COLOR_SUCCESS)

    # --- THE INTER-AGENT COMMUNICATION BUS: SHARED TYPED STATE (MIDDLE OF LAYER 2) ---
    bus_box = FancyBboxPatch((0.9, 4.3), 14.2, 0.68, boxstyle="round,pad=0.06",
                             facecolor="#F0FDFA", edgecolor="#0D9488", linewidth=1.8)
    ax.add_patch(bus_box)

    bus_tag = FancyBboxPatch((1.1, 4.65), 5.6, 0.24, boxstyle="round,pad=0.03",
                             facecolor="#0D9488", edgecolor="#0D9488", linewidth=1.0)
    ax.add_patch(bus_tag)
    ax.text(3.9, 4.77, "INTER-AGENT STATE BUS: InvestigationState (TypedDict)", ha="center", va="center", fontsize=7.5, fontweight="bold", color="#FFFFFF")

    ax.text(14.8, 4.77, "Blackboard Architecture: Structured Memory Buffer | Zero Unparsed Chat", ha="right", va="center", fontsize=7.2, fontweight="bold", color="#0F766E")

    # Bus Fields and Read/Write Annotations
    bus_rw = [
        (2.15, "Writes: plan['steps']", "1. Planner"),
        (5.1, "Reads: plan | Writes: evidence, indicators", "2. Research"),
        (8.1, "Reads: evidence | Writes: risk_score, rules", "3. Risk Engine"),
        (11.1, "Reads: evidence & score | Writes: audit status", "4. Critic"),
        (13.95, "Reads: full state -> compiles dossier", "5. Report"),
    ]
    for x_rw, rw_text, ag_name in bus_rw:
        ax.text(x_rw, 4.45, rw_text, ha="center", va="center", fontsize=6.7, fontweight="bold", color="#115E59")
        # Bidirectional arrows connecting agent card to state bus
        ax.annotate("", xy=(x_rw, 4.98), xytext=(x_rw, 5.3),
                    arrowprops=dict(arrowstyle="<->", color="#0D9488", lw=1.6))

    # --- BOTTOM OF LAYER 2: RESOURCE ALLOCATION & SELF-CORRECTION ---
    # Left Card: Resource Allocation & Governance Box
    gov_box = FancyBboxPatch((0.9, 3.45), 7.6, 0.72, boxstyle="round,pad=0.05",
                             facecolor="#F1F5F9", edgecolor="#475569", linewidth=1.4)
    ax.add_patch(gov_box)
    ax.text(1.1, 3.98, "RESOURCE ALLOCATION & EXECUTION GOVERNANCE:", fontsize=8.0, fontweight="bold", color="#0F172A")
    ax.text(1.1, 3.75, "- Bounded Execution Budget: max_retries = 1 strictly prevents runaway execution loops and CPU exhaustion.", fontsize=7.0, color="#334155")
    ax.text(1.1, 3.55, "- State Memory Bounds: Strict TypedDict schema ensures fixed memory footprint | Tracks duration_ms & node_history.", fontsize=7.0, color="#334155")

    # Right Card: Conditional Self-Correction Loop
    corr_box = FancyBboxPatch((8.8, 3.45), 6.3, 0.72, boxstyle="round,pad=0.05",
                              facecolor="#FEE2E2", edgecolor="#EF4444", linewidth=1.4)
    ax.add_patch(corr_box)
    ax.text(9.0, 3.98, "CONDITIONAL SELF-CORRECTION (correction_node):", fontsize=8.0, fontweight="bold", color="#991B1B")
    ax.text(9.0, 3.75, "Triggered if Critic flags discrepancy. Recalibrates score to verified value. Bounded to 1 retry.", fontsize=7.0, color="#7F1D1D")
    ax.text(9.0, 3.55, "Safe Exit: If validation fails after 1 retry, forces finalization with audit warnings attached.", fontsize=7.0, color="#7F1D1D")

    # Downward arrow Critic -> Correction
    ax.annotate("", xy=(11.5, 4.17), xytext=(11.5, 4.3), arrowprops=dict(arrowstyle="->", color=COLOR_DANGER, lw=1.8))
    # Return loop arrow Correction -> Critic
    ax.annotate("", xy=(10.5, 4.3), xytext=(10.5, 4.17), arrowprops=dict(arrowstyle="->", color="#B91C1C", lw=1.8))

    # ==========================================================================
    # LAYER 3: MODEL CONTEXT PROTOCOL (MCP 2.0) & DATA LAYER (BOTTOM CONTAINER)
    # ==========================================================================
    layer3_box = FancyBboxPatch((0.6, 0.35), 14.8, 2.7, boxstyle="round,pad=0.12",
                                facecolor="#FAF5FF", edgecolor="#7E22CE", linewidth=2.0)
    ax.add_patch(layer3_box)

    # Layer Tag Pill
    l3_tag = FancyBboxPatch((0.9, 2.68), 7.8, 0.28, boxstyle="round,pad=0.03",
                            facecolor="#7E22CE", edgecolor="#7E22CE", linewidth=1.0)
    ax.add_patch(l3_tag)
    ax.text(4.8, 2.82, "LAYER 3: MODEL CONTEXT PROTOCOL (MCP 2.0) TOOL COMMUNICATION & DATA ACCESS", ha="center", va="center", fontsize=8.0, fontweight="bold", color="#FFFFFF")

    ax.text(15.1, 2.82, "Agent-to-Tool Protocol: Standardized JSON-RPC Request-Response Schemas",
            ha="right", va="center", fontsize=7.5, fontweight="bold", color="#6B21A8")

    # --- 5 STANDARDIZED MCP TOOLS ---
    mcp_tools = [
        ("T1: get_transaction", "Retrieves transaction amount,\nchannel & merchant details", 0.9, 2.6),
        ("T2: get_account_history", "Queries chronological records\n& 30-day baseline profile", 3.8, 2.6),
        ("T3: calculate_deviation", "Computes sample mean, std dev,\nand amount z-score", 6.8, 2.6),
        ("T4: identify_indicators", "Detects velocity bursts, unseen\nmerchant, nocturnal window", 9.8, 2.6),
        ("T5: validate_investigation", "Performs independent Critic\nscore recalculation & audit", 12.8, 2.3),
    ]

    for t_name, t_desc, tx_pos, tw_card in mcp_tools:
        tb = FancyBboxPatch((tx_pos, 1.35), tw_card, 1.1, boxstyle="round,pad=0.06",
                            facecolor="#FFFFFF", edgecolor="#C084FC", linewidth=1.2)
        ax.add_patch(tb)
        ax.text(tx_pos + 0.15, 2.2, t_name, fontsize=8.5, fontweight="bold", color="#581C87")
        ax.text(tx_pos + 0.15, 1.7, t_desc, fontsize=7.2, color="#4A044E")

    # --- BOTTOM DATA STORE BAR ---
    db_bar = FancyBboxPatch((0.9, 0.52), 14.2, 0.65, boxstyle="round,pad=0.06",
                            facecolor="#F1F5F9", edgecolor="#64748B", linewidth=1.4)
    ax.add_patch(db_bar)
    ax.text(1.2, 0.92, "Synthetic Banking Data Store: transactions.csv (500 records)  |  account_profiles.csv (100 accounts)",
            fontsize=8.5, fontweight="bold", color=COLOR_PRIMARY)
    ax.text(1.2, 0.68, "Verified In-Memory Access: Tools query verified synthetic files strictly via typed MCP JSON-RPC schemas | 100% Deterministic & Offline",
            fontsize=7.2, color="#334155")

    # Connect MCP Tools to Data Store
    ax.annotate("", xy=(8.0, 1.17), xytext=(8.0, 1.35), arrowprops=dict(arrowstyle="<->", color="#7E22CE", lw=1.5))

    # ==========================================================================
    # INTER-LAYER PROTOCOL CONNECTORS (RESEARCH -> MCP & CRITIC -> TOOL 5)
    # ==========================================================================
    # Connector 1: Research Agent (Step 2) -> MCP Client (Tools 1-4)
    ax.annotate("", xy=(5.1, 2.96), xytext=(5.1, 4.3),
                arrowprops=dict(arrowstyle="<->", color="#6366F1", lw=2.2, linestyle="dashed"))
    mcp_call_pill = FancyBboxPatch((3.7, 3.1), 2.8, 0.26, boxstyle="round,pad=0.03",
                                   facecolor="#EEF2FF", edgecolor="#6366F1", linewidth=1.0)
    ax.add_patch(mcp_call_pill)
    ax.text(5.1, 3.23, "MCP Tool Calls (Tools 1-4)", ha="center", va="center", fontsize=7.0, fontweight="bold", color="#4338CA")

    # Connector 2: Critic Agent (Step 4) -> Tool 5 (Validation)
    ax.annotate("", xy=(12.0, 2.96), xytext=(12.0, 4.3),
                arrowprops=dict(arrowstyle="<->", color="#7E22CE", lw=2.2, linestyle="dashed"))
    mcp_audit_pill = FancyBboxPatch((10.4, 3.1), 3.2, 0.26, boxstyle="round,pad=0.03",
                                    facecolor="#FAF5FF", edgecolor="#7E22CE", linewidth=1.0)
    ax.add_patch(mcp_audit_pill)
    ax.text(12.0, 3.23, "MCP Audit Verification (Tool 5)", ha="center", va="center", fontsize=7.0, fontweight="bold", color="#6B21A8")

    plt.tight_layout()
    p = OUT_DIR / "slide4_system_architecture.png"
    plt.savefig(p, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Generated clean communication & resource architecture diagram: {p}")


# ==============================================================================
# DIAGRAM 3: UI WORKBENCH & RISK GAUGE MOCKUP (SLIDE 5)
# ==============================================================================
def generate_diagram_slide5():
    fig, ax = plt.subplots(figsize=(14, 7.5), dpi=300)
    fig.patch.set_facecolor("#FFFFFF")
    ax.set_facecolor("#FFFFFF")
    ax.set_xlim(0, 14)
    ax.set_ylim(0, 7.8)
    ax.axis("off")

    # Header
    ax.text(7.0, 7.4, "STREAMLIT INTERACTIVE WORKBENCH DEMONSTRATION", 
            ha="center", va="center", fontsize=17, fontweight="bold", color=COLOR_PRIMARY)
    ax.text(7.0, 7.0, "Live Multi-Agent Risk Investigation for High-Risk Benchmark Case: TXN-10198", 
            ha="center", va="center", fontsize=10.5, color=COLOR_MUTED)

    # Top Case Status Bar
    status_bar = FancyBboxPatch((0.6, 6.0), 12.8, 0.7, boxstyle="round,pad=0.1",
                                facecolor="#F8FAFC", edgecolor=COLOR_PRIMARY, linewidth=1.5)
    ax.add_patch(status_bar)
    ax.text(1.0, 6.35, "Target Transaction: TXN-10198", fontsize=9.5, fontweight="bold", color=COLOR_PRIMARY)
    ax.text(5.0, 6.35, "Latency: 38.2 ms", fontsize=9.5, fontweight="bold", color=COLOR_SECONDARY)
    ax.text(7.8, 6.35, "Critic Audit: 100% MATCH (Score: 60) [PASS]", fontsize=9.5, fontweight="bold", color=COLOR_SUCCESS)
    ax.text(11.8, 6.35, "Retries: 0 / 1", fontsize=9.5, fontweight="bold", color="#475569")

    # Left Card: Gauge Meter
    gauge_card = FancyBboxPatch((0.6, 1.4), 5.5, 4.3, boxstyle="round,pad=0.15",
                                facecolor="#FFFFFF", edgecolor="#CBD5E1", linewidth=1.5)
    ax.add_patch(gauge_card)
    ax.text(3.35, 5.3, "DETERMINISTIC RISK SCORE GAUGE", ha="center", va="center", fontsize=11, fontweight="bold", color=COLOR_PRIMARY)

    # Draw Semi-Circular Gauge Meter
    center = (3.35, 3.4)
    r = 1.4
    # Wedge sectors
    ax.add_patch(Wedge(center, r, 0, 45, facecolor="#EF4444", alpha=0.9, width=0.4))    # Critical
    ax.add_patch(Wedge(center, r, 45, 90, facecolor="#F97316", alpha=0.9, width=0.4))   # High
    ax.add_patch(Wedge(center, r, 90, 135, facecolor="#FACC15", alpha=0.9, width=0.4))  # Medium
    ax.add_patch(Wedge(center, r, 135, 180, facecolor="#22C55E", alpha=0.9, width=0.4)) # Low

    # Needle for Score 60 (maps between 90 and 45 deg, approx 72 deg)
    angle_rad = np.radians(72)
    needle_x = center[0] + (r - 0.1) * np.cos(angle_rad)
    needle_y = center[1] + (r - 0.1) * np.sin(angle_rad)
    ax.plot([center[0], needle_x], [center[1], needle_y], color="#0F172A", lw=3.0)
    ax.plot(center[0], center[1], "o", color="#0F172A", markersize=8)

    # Score Callout Text
    ax.text(3.35, 2.7, "60 / 100", ha="center", va="center", fontsize=22, fontweight="bold", color="#EA580C")
    ax.text(3.35, 2.3, "PRIORITY TIER: HIGH", ha="center", va="center", fontsize=11, fontweight="bold", color="#C2410C")
    ax.text(3.35, 1.8, "Tiers: Low (0-29) | Medium (30-59) | High (60-79) | Critical (80-100)",
            ha="center", va="center", fontsize=7.5, color="#64748B")

    # Right Card: Evidence Attribution
    evidence_card = FancyBboxPatch((6.5, 1.4), 6.9, 4.3, boxstyle="round,pad=0.15",
                                   facecolor="#FFFFFF", edgecolor="#CBD5E1", linewidth=1.5)
    ax.add_patch(evidence_card)
    ax.text(6.8, 5.3, "TRIGGERED EVIDENCE & POINT BREAKDOWN", fontsize=11, fontweight="bold", color=COLOR_PRIMARY)

    ev_items = [
        ("[!] +30 Points", "Extreme Amount Deviation", "INR 48,000 exceeds 5x historical mean (INR 3,200) | z = 24.19"),
        ("[!] +20 Points", "Unseen Merchant Identifier", "First-ever recorded transaction at 'ElectroWorld Global'"),
        ("[!] +10 Points", "Nocturnal Execution Window", "Transaction executed at 03:15 AM (Flagged window: 02:00 - 05:00)"),
        ("[-] +0 Points", "Daily Velocity Burst", "Frequency within normal baseline (1 txn in 24h vs avg 1.4)"),
        ("[-] +0 Points", "Authorization Failures", "Zero consecutive authorization failures prior to transaction"),
    ]

    y_ev = 4.7
    for pts, title, detail in ev_items:
        color_pts = COLOR_PRIMARY if "+0" in pts else "#9A3412"
        ax.text(6.8, y_ev, pts, fontsize=9, fontweight="bold", color=color_pts)
        ax.text(8.2, y_ev, title, fontsize=9, fontweight="bold", color=COLOR_SECONDARY)
        ax.text(6.8, y_ev - 0.22, detail, fontsize=7.5, color="#475569")
        y_ev -= 0.65

    # Bottom Decision Support Bar
    rec_box = FancyBboxPatch((0.6, 0.4), 12.8, 0.75, boxstyle="round,pad=0.1",
                             facecolor="#FFFBEB", edgecolor="#F59E0B", linewidth=1.5)
    ax.add_patch(rec_box)
    ax.text(0.9, 0.9, "RECOMMENDED HUMAN ANALYST ACTION:", fontsize=8.5, fontweight="bold", color="#92400E")
    ax.text(0.9, 0.62, "Initiate prompt cardholder telephone verification. Request two-factor confirmation for ElectroWorld Global before settlement clearance.",
            fontsize=8, color="#78350F")

    plt.tight_layout()
    p = OUT_DIR / "slide5_investigation_workbench.png"
    plt.savefig(p, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Generated: {p}")


# ==============================================================================
# DIAGRAM 4: CRITIC 9-GATE QUALITY AUDIT (SLIDE 6)
# ==============================================================================
def generate_diagram_slide6():
    fig, ax = plt.subplots(figsize=(14, 7.5), dpi=300)
    fig.patch.set_facecolor("#FFFFFF")
    ax.set_facecolor("#FFFFFF")
    ax.set_xlim(0, 14)
    ax.set_ylim(0, 7.8)
    ax.axis("off")

    # Header
    ax.text(7.0, 7.4, "CRITIC AGENT: 9-GATE QUALITY AUDIT ARCHITECTURE", 
            ha="center", va="center", fontsize=17, fontweight="bold", color=COLOR_PRIMARY)
    ax.text(7.0, 7.0, "Enforcing Banking 'Four-Eyes' Compliance: Independent Mathematical Re-Audit & Self-Correction", 
            ha="center", va="center", fontsize=10.5, color=COLOR_MUTED)

    # 9 Gate Grid (3 rows x 3 cols)
    gates = [
        ("Gate 1: Transaction ID Valid", "Validates ID string integrity, structure, and non-empty record identifier.", "PASSED [OK]"),
        ("Gate 2: Supporting Records", "Verifies supporting historical account records and profile baseline exist.", "PASSED [OK]"),
        ("Gate 3: Math Recalculation", "Independently re-computes score from raw indicators via MCP validation tool.", "MATCHED (100%)"),
        ("Gate 4: Score Range Valid", "Enforces mathematical ceiling [0, 100] with score capping logic.", "PASSED [OK]"),
        ("Gate 5: Priority Mapping", "Verifies deterministic thresholds (Low, Medium, High, Critical) match score.", "PASSED [OK]"),
        ("Gate 6: Rule Consistency", "Audits that every triggered rule is backed by verified empirical evidence.", "PASSED [OK]"),
        ("Gate 7: Missing Data Warnings", "Differentiates true missing attributes from zero-risk attributes.", "PASSED [OK]"),
        ("Gate 8: Prohibited Words", "Blocks non-neutral terms ('guilty', 'fraudulent', 'thief') to avoid legal bias.", "PASSED [OK]"),
        ("Gate 9: 16-Field Schema", "Ensures all mandatory regulatory audit fields are fully populated.", "PASSED [OK]"),
    ]

    col_coords = [0.6, 4.8, 9.0]
    row_coords = [5.3, 3.7, 2.1]

    idx = 0
    for r_y in row_coords:
        for c_x in col_coords:
            g_title, g_desc, g_status = gates[idx]
            g_box = FancyBboxPatch((c_x, r_y - 1.1), 4.0, 1.25, boxstyle="round,pad=0.1",
                                   facecolor="#F8FAFC", edgecolor="#CBD5E1", linewidth=1.2)
            ax.add_patch(g_box)
            ax.text(c_x + 0.2, r_y - 0.15, g_title, fontsize=9, fontweight="bold", color=COLOR_PRIMARY)
            ax.text(c_x + 0.2, r_y - 0.45, g_desc, fontsize=7.5, color="#475569", wrap=True)
            
            badge_box = FancyBboxPatch((c_x + 2.3, r_y - 0.95), 1.6, 0.3, boxstyle="round,pad=0.04",
                                       facecolor="#DCFCE7", edgecolor="#22C55E", linewidth=1.0)
            ax.add_patch(badge_box)
            ax.text(c_x + 3.1, r_y - 0.8, g_status, ha="center", va="center", fontsize=7.2, fontweight="bold", color="#15803D")
            idx += 1

    # Bottom Callout: Self-Correction Loop
    audit_footer = FancyBboxPatch((0.6, 0.4), 12.4, 0.95, boxstyle="round,pad=0.1",
                                  facecolor="#FEF3C7", edgecolor="#F59E0B", linewidth=1.8)
    ax.add_patch(audit_footer)
    ax.text(0.9, 1.05, "CONDITIONAL SELF-CORRECTION & BOUNDED RETRIES:", fontsize=9, fontweight="bold", color="#92400E")
    ax.text(0.9, 0.78, "- If Critic detects a calculation discrepancy or ungrounded conclusion, LangGraph conditionally branches to correction_node.", fontsize=8, color="#78350F")
    ax.text(0.9, 0.55, "- Recalibrates score to independently audited value and enforces max_retries = 1, guaranteeing safe termination without infinite loops.", fontsize=8, color="#78350F")

    plt.tight_layout()
    p = OUT_DIR / "slide6_critic_quality_gates.png"
    plt.savefig(p, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Generated: {p}")


if __name__ == "__main__":
    print("Generating 4 presentation diagrams...")
    generate_diagram_slide3()
    generate_diagram_slide4()
    generate_diagram_slide5()
    generate_diagram_slide6()
    print("All 4 presentation diagrams successfully generated in docs/diagrams/!")
