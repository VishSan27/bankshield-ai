"""Typed Shared State for LangGraph Orchestration in BankShield AI.

Defines the complete state schema passed across LangGraph nodes, tracking
input goals, task roadmaps, raw records, MCP tool calls, empirical indicators,
risk scores, validation audits, retry counters, and granular execution telemetry.
"""

from typing import Any, Dict, List, Optional
from typing_extensions import TypedDict


class InvestigationState(TypedDict, total=False):
    """Typed shared investigation state used across all LangGraph nodes."""

    # 1. Target & Session Identifiers
    investigation_id: str
    transaction_id: str
    objective: str
    created_at: str

    # 2. Planning State (PlannerAgent)
    plan: Dict[str, Any]
    current_step_index: int

    # 3. Research & Data State (ResearchAgent via MCP Tools)
    transaction_data: Optional[Dict[str, Any]]
    account_profile: Optional[Dict[str, Any]]
    account_history: List[Dict[str, Any]]
    deviation_analysis: Dict[str, Any]
    indicators: Dict[str, Any]
    evidence: Dict[str, Any]
    missing_data_warnings: List[str]
    mcp_tool_calls: List[Dict[str, Any]]

    # 4. Risk Assessment State (RiskAssessmentAgent)
    risk_result: Dict[str, Any]
    risk_score: int
    raw_score: int
    priority: str
    triggered_rules: List[Dict[str, Any]]
    untriggered_rules: List[Dict[str, Any]]
    missing_data_rules: List[Dict[str, Any]]
    recommendation: str

    # 5. Critic & Validation State (CriticAgent & Recalculation Node)
    validation_result: Dict[str, Any]
    validation_status: str  # 'VALIDATED', 'NEEDS_REVIEW', 'VALIDATION_FAILED'
    score_matches: bool
    recalculated_score: int
    recalculated_priority: str
    validation_checks: List[Dict[str, Any]]

    # 6. Retry & Self-Correction Governance
    retry_count: int
    max_retries: int
    correction_applied: bool
    correction_history: List[Dict[str, Any]]

    # 7. Final Output & Reporting
    final_report: Dict[str, Any]
    status: str  # 'completed', 'failed', 'partial'
    errors: List[str]
    warnings: List[str]

    # 8. Execution Trace & Graph Telemetry
    node_history: List[str]
    execution_trace: List[Dict[str, Any]]
    start_time: float
    total_duration_ms: float
