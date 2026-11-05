"""LangGraph Multi-Agent Investigation Orchestrator for BankShield AI.

Coordinates the full multi-agent investigation lifecycle using LangGraph StateGraph:
- Shared typed state (InvestigationState)
- Graph nodes for PlannerAgent, ResearchAgent (via MCP Client), RiskAssessmentAgent, CriticAgent,
  CorrectionNode, and ReportGenerationNode
- Explicit edges and conditional routing based on Critic validation outcomes
- Bounded retry limits and safe termination
- Real node execution tracking, state transitions, and MCP tool telemetry
- 100% deterministic operation without mandatory LLMs
"""

from datetime import datetime, timezone
import logging
from pathlib import Path
import sys
import time
from typing import Any, Dict, List, Literal, Optional
import uuid

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from langgraph.graph import END, START, StateGraph

from agents.critic_agent import CriticAgent
from agents.planner_agent import PlannerAgent
from agents.risk_assessment_agent import RiskAssessmentAgent
from orchestration.investigation_state import InvestigationState
from services.data_service import DataService
from services.mcp_client import BankShieldMCPClient
from tools.report_tools import generate_investigation_report
from tools.risk_rules import create_default_engine

logger = logging.getLogger("LangGraphOrchestrator")


def build_investigation_graph(
    data_service: Optional[DataService] = None,
    mcp_client: Optional[BankShieldMCPClient] = None,
    max_retries: int = 1,
) -> Any:
    """Construct and compile the LangGraph StateGraph workflow for BankShield AI.

    Args:
        data_service: Optional DataService instance.
        mcp_client: Optional BankShieldMCPClient for standardized tool invocation.
        max_retries: Maximum permitted self-correction/recalculation cycles.

    Returns:
        Compiled LangGraph workflow executable.
    """
    ds = data_service or DataService()
    client = mcp_client or BankShieldMCPClient()
    planner_agent = PlannerAgent()
    risk_agent = RiskAssessmentAgent()
    critic_agent = CriticAgent()

    # --------------------------------------------------------------------------
    # Graph Node 1: Planner Agent
    # --------------------------------------------------------------------------
    def planner_node(state: InvestigationState) -> Dict[str, Any]:
        """Formulate ordered investigation execution plan."""
        start_ms = time.perf_counter()
        tx_id = state.get("transaction_id", "")
        obj = state.get("objective", "Investigate transaction risk indicators")

        logger.info("[LangGraph] Node 'planner' invoked for tx: %s", tx_id)
        plan = planner_agent.plan_investigation(tx_id, objective=obj)

        elapsed = round((time.perf_counter() - start_ms) * 1000, 2)
        trace_entry = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "node": "planner_node",
            "agent_name": "PlannerAgent",
            "action": "plan_investigation",
            "status": plan.get("status", "completed"),
            "duration_ms": elapsed,
            "output_summary": f"Generated {len(plan.get('steps', []))} ordered execution steps.",
        }

        node_hist = list(state.get("node_history", [])) + ["planner_node"]
        trace = list(state.get("execution_trace", [])) + [trace_entry]

        # Handle early validation failure for non-existent ID
        errors = list(state.get("errors", []))
        if plan.get("status") == "error":
            errors.append(plan.get("error", "Invalid transaction identifier"))

        return {
            "plan": plan,
            "node_history": node_hist,
            "execution_trace": trace,
            "errors": errors,
        }

    # --------------------------------------------------------------------------
    # Graph Node 2: Research Agent (via Model Context Protocol)
    # --------------------------------------------------------------------------
    def research_node(state: InvestigationState) -> Dict[str, Any]:
        """Invoke MCP tools for evidence retrieval, deviation analysis, and indicators."""
        start_ms = time.perf_counter()
        tx_id = state.get("transaction_id", "")

        logger.info("[LangGraph] Node 'research' invoking MCP tools for tx: %s", tx_id)
        mcp_calls = []

        # MCP Tool Call 1: get_transaction
        tx_resp = client.get_transaction(tx_id)
        tx_data = tx_resp.get("data")
        mcp_calls.append({
            "tool": "get_transaction",
            "status": tx_resp.get("status"),
            "details": f"Amount: {tx_data.get('amount') if tx_data else 'None'}",
        })

        if not tx_data:
            elapsed = round((time.perf_counter() - start_ms) * 1000, 2)
            trace_entry = {
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "node": "research_node",
                "agent_name": "ResearchAgent",
                "action": "mcp_tool_execution",
                "status": "error",
                "duration_ms": elapsed,
                "output_summary": f"Transaction '{tx_id}' not found via MCP server.",
            }
            return {
                "transaction_data": None,
                "account_profile": None,
                "account_history": [],
                "indicators": {},
                "deviation_analysis": {},
                "evidence": {"evidence_summary": "Transaction not found."},
                "missing_data_warnings": [f"Transaction ID {tx_id} not found in database"],
                "mcp_tool_calls": mcp_calls,
                "node_history": list(state.get("node_history", [])) + ["research_node"],
                "execution_trace": list(state.get("execution_trace", [])) + [trace_entry],
                "errors": list(state.get("errors", [])) + [f"Transaction {tx_id} not found"],
            }

        acc_id = tx_data.get("account_id", "")

        # MCP Tool Call 2: get_account_history
        acc_resp = client.get_account_history(acc_id)
        acc_history = acc_resp.get("history", [])
        acc_profile = acc_resp.get("profile", {})
        mcp_calls.append({
            "tool": "get_account_history",
            "status": acc_resp.get("status"),
            "details": f"Records: {len(acc_history)}, Profile: {bool(acc_profile)}",
        })

        # MCP Tool Call 3: calculate_transaction_deviation
        dev_resp = client.calculate_transaction_deviation(tx_id)
        deviation = dev_resp.get("deviation", {})
        mcp_calls.append({
            "tool": "calculate_transaction_deviation",
            "status": dev_resp.get("status"),
            "details": f"z-score: {deviation.get('z_score', 'N/A')}",
        })

        # MCP Tool Call 4: identify_transaction_indicators
        ind_resp = client.identify_transaction_indicators(tx_id)
        indicators = ind_resp.get("indicators", {})
        mcp_calls.append({
            "tool": "identify_transaction_indicators",
            "status": ind_resp.get("status"),
            "details": f"Indicators evaluated: {len(indicators)}",
        })

        # Assemble evidence bundle
        evidence_bundle = {
            "transaction_details": tx_data,
            "account_profile": acc_profile,
            "account_history": acc_history,
            "deviation_analysis": deviation,
            "indicators": indicators,
            "evidence_summary": f"Evaluated transaction {tx_id} against {len(acc_history)} historical records.",
            "tool_calls": client.get_telemetry_history()[-4:],
        }

        elapsed = round((time.perf_counter() - start_ms) * 1000, 2)
        trace_entry = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "node": "research_node",
            "agent_name": "ResearchAgent",
            "action": "mcp_tool_execution",
            "status": "success",
            "duration_ms": elapsed,
            "output_summary": f"Executed 4 MCP tools; computed z-score {deviation.get('z_score', 0)}.",
        }

        return {
            "transaction_data": tx_data,
            "account_profile": acc_profile,
            "account_history": acc_history,
            "deviation_analysis": deviation,
            "indicators": indicators,
            "evidence": evidence_bundle,
            "missing_data_warnings": [],
            "mcp_tool_calls": mcp_calls,
            "node_history": list(state.get("node_history", [])) + ["research_node"],
            "execution_trace": list(state.get("execution_trace", [])) + [trace_entry],
        }

    # --------------------------------------------------------------------------
    # Graph Node 3: Risk Assessment Agent
    # --------------------------------------------------------------------------
    def risk_assessment_node(state: InvestigationState) -> Dict[str, Any]:
        """Apply deterministic risk scoring rules capped at 100 points."""
        start_ms = time.perf_counter()
        evidence = state.get("evidence", {})

        logger.info("[LangGraph] Node 'risk_assessment' evaluating deterministic rules")
        risk_result = risk_agent.assess_risk(evidence)

        score = risk_result.get("risk_score", 0)
        raw_score = risk_result.get("raw_score", score)
        priority = risk_result.get("priority", "Low")
        trig = risk_result.get("triggered_rules", [])
        untrig = risk_result.get("untriggered_rules", [])
        missing = risk_result.get("missing_data_rules", [])
        rec = risk_result.get("recommended_action", "Review transaction.")

        elapsed = round((time.perf_counter() - start_ms) * 1000, 2)
        trace_entry = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "node": "risk_assessment_node",
            "agent_name": "RiskAssessmentAgent",
            "action": "assess_risk",
            "status": "success",
            "duration_ms": elapsed,
            "output_summary": f"Computed score {score}/100 ({priority} priority, {len(trig)} rules triggered).",
        }

        return {
            "risk_result": risk_result,
            "risk_score": score,
            "raw_score": raw_score,
            "priority": priority,
            "triggered_rules": trig,
            "untriggered_rules": untrig,
            "missing_data_rules": missing,
            "recommendation": rec,
            "node_history": list(state.get("node_history", [])) + ["risk_assessment_node"],
            "execution_trace": list(state.get("execution_trace", [])) + [trace_entry],
        }

    # --------------------------------------------------------------------------
    # Graph Node 4: Critic & Validation Agent
    # --------------------------------------------------------------------------
    def critic_node(state: InvestigationState) -> Dict[str, Any]:
        """Independently audit numerical calculations and quality gates via Critic & MCP."""
        start_ms = time.perf_counter()
        tx_data = state.get("transaction_data") or {}
        evidence = state.get("evidence") or {}
        risk_res = state.get("risk_result") or {}
        plan = state.get("plan") or {}
        tx_id = state.get("transaction_id", "")

        logger.info("[LangGraph] Node 'critic' executing 9-gate quality audit for tx: %s", tx_id)

        # Agent-level 9-gate audit
        val_res = critic_agent.validate_investigation(
            transaction_data=tx_data,
            evidence=evidence,
            risk_result=risk_res,
            plan=plan,
        )

        # Also invoke MCP independent validation tool for cross-protocol audit
        trig_names = [r.get("rule_name", "") for r in state.get("triggered_rules", [])]
        mcp_val = client.validate_investigation(
            transaction_id=tx_id,
            reported_score=state.get("risk_score", 0),
            priority=state.get("priority", "Low"),
            triggered_rule_names=trig_names,
        )

        score_matches = val_res.get("score_matches", True) and mcp_val.get("score_matches", True)
        recalculated_score = val_res.get("recalculated_score", state.get("risk_score", 0))

        val_status = val_res.get("status", "VALIDATED")
        if not score_matches:
            val_status = "VALIDATION_FAILED"

        elapsed = round((time.perf_counter() - start_ms) * 1000, 2)
        trace_entry = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "node": "critic_node",
            "agent_name": "CriticAgent",
            "action": "validate_investigation",
            "status": val_status,
            "duration_ms": elapsed,
            "output_summary": f"Audit Status: {val_status} (Score match: {score_matches}, Recalculated: {recalculated_score}).",
        }

        return {
            "validation_result": val_res,
            "validation_status": val_status,
            "score_matches": score_matches,
            "recalculated_score": recalculated_score,
            "recalculated_priority": val_res.get("recalculated_priority", state.get("priority")),
            "validation_checks": val_res.get("checks_performed", []),
            "node_history": list(state.get("node_history", [])) + ["critic_node"],
            "execution_trace": list(state.get("execution_trace", [])) + [trace_entry],
        }

    # --------------------------------------------------------------------------
    # Graph Node 5: Correction / Recalculation Node
    # --------------------------------------------------------------------------
    def correction_node(state: InvestigationState) -> Dict[str, Any]:
        """Perform bounded recalibration / self-correction when Critic flags discrepancies."""
        start_ms = time.perf_counter()
        current_retries = state.get("retry_count", 0) + 1
        recalc_score = state.get("recalculated_score", state.get("risk_score", 0))

        logger.warning(
            "[LangGraph] Node 'correction' triggered! Discrepancy detected (Attempt %d/%d). Adjusting score to %d",
            current_retries,
            state.get("max_retries", max_retries),
            recalc_score,
        )

        correction_event = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "retry_cycle": current_retries,
            "original_score": state.get("risk_score"),
            "corrected_score": recalc_score,
            "reason": "Critic Agent identified mathematical or priority discrepancy; applied verified recalculation.",
        }

        # Apply correction to risk result
        corrected_risk_res = dict(state.get("risk_result", {}))
        corrected_risk_res["risk_score"] = recalc_score
        corrected_risk_res["total_score"] = recalc_score
        engine = create_default_engine()
        corrected_priority = engine.get_priority(recalc_score)
        corrected_risk_res["priority"] = corrected_priority

        elapsed = round((time.perf_counter() - start_ms) * 1000, 2)
        trace_entry = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "node": "correction_node",
            "agent_name": "CorrectionEngine",
            "action": "recalculate_and_correct",
            "status": "correction_applied",
            "duration_ms": elapsed,
            "output_summary": f"Correction cycle {current_retries}: Score calibrated to {recalc_score} ({corrected_priority}).",
        }

        corr_hist = list(state.get("correction_history", [])) + [correction_event]

        return {
            "retry_count": current_retries,
            "correction_applied": True,
            "correction_history": corr_hist,
            "risk_result": corrected_risk_res,
            "risk_score": recalc_score,
            "priority": corrected_priority,
            "node_history": list(state.get("node_history", [])) + ["correction_node"],
            "execution_trace": list(state.get("execution_trace", [])) + [trace_entry],
        }

    # --------------------------------------------------------------------------
    # Graph Node 6: Report Generation Node
    # --------------------------------------------------------------------------
    def report_generation_node(state: InvestigationState) -> Dict[str, Any]:
        """Compile final 16-field standardized investigation dossier."""
        start_ms = time.perf_counter()
        inv_id = state.get("investigation_id", str(uuid.uuid4()))
        tx_data = state.get("transaction_data") or {"transaction_id": state.get("transaction_id")}
        evidence = state.get("evidence") or {}
        risk_res = state.get("risk_result") or {}
        val_res = state.get("validation_result") or {}
        plan = state.get("plan") or {}

        logger.info("[LangGraph] Node 'report_generation' compiling 16-field report")

        report = generate_investigation_report(
            investigation_id=inv_id,
            transaction=tx_data,
            evidence=evidence,
            risk_result=risk_res,
            validation_result=val_res,
            plan=plan,
        )

        overall_status = "completed" if report.get("status") == "VALIDATED" else "needs_review"
        if not state.get("transaction_data") and state.get("errors"):
            overall_status = "failed"
            report["error"] = state.get("errors", ["Investigation error"])[0]

        elapsed = round((time.perf_counter() - start_ms) * 1000, 2)
        trace_entry = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "node": "report_generation_node",
            "agent_name": "ReportGenerator",
            "action": "generate_report",
            "status": overall_status,
            "duration_ms": elapsed,
            "output_summary": f"Dossier finalized. Status: {report.get('status', 'VALIDATED')}.",
        }

        # Calculate total workflow execution latency
        start_time = state.get("start_time", time.perf_counter())
        total_latency = round((time.perf_counter() - start_time) * 1000, 2)

        return {
            "final_report": report,
            "status": overall_status,
            "total_duration_ms": total_latency,
            "node_history": list(state.get("node_history", [])) + ["report_generation_node"],
            "execution_trace": list(state.get("execution_trace", [])) + [trace_entry],
        }

    # --------------------------------------------------------------------------
    # Conditional Routing Logic: Critic Validation Gate
    # --------------------------------------------------------------------------
    def route_critic_decision(state: InvestigationState) -> Literal["correction_node", "report_generation_node"]:
        """Route conditionally based on validation outcome and retry limits.

        Rules:
        - If validation passed (VALIDATED): route to report generation.
        - If validation failed (VALIDATION_FAILED) and retries < max_retries: route to correction.
        - If retries >= max_retries: terminate safely by routing to report generation (flagged).
        """
        val_status = state.get("validation_status", "VALIDATED")
        retries = state.get("retry_count", 0)
        limit = state.get("max_retries", max_retries)

        if val_status == "VALIDATION_FAILED" and retries < limit:
            logger.warning("[LangGraph Routing] Routing to correction_node (Retry %d < Limit %d)", retries, limit)
            return "correction_node"

        logger.info("[LangGraph Routing] Routing to report_generation_node (Status: %s)", val_status)
        return "report_generation_node"

    # --------------------------------------------------------------------------
    # StateGraph Construction & Edge Definition
    # --------------------------------------------------------------------------
    workflow = StateGraph(InvestigationState)

    # Add all 6 nodes
    workflow.add_node("planner_node", planner_node)
    workflow.add_node("research_node", research_node)
    workflow.add_node("risk_assessment_node", risk_assessment_node)
    workflow.add_node("critic_node", critic_node)
    workflow.add_node("correction_node", correction_node)
    workflow.add_node("report_generation_node", report_generation_node)

    # Linear forward flow
    workflow.add_edge(START, "planner_node")
    workflow.add_edge("planner_node", "research_node")
    workflow.add_edge("research_node", "risk_assessment_node")
    workflow.add_edge("risk_assessment_node", "critic_node")

    # Conditional branching from Critic node
    workflow.add_conditional_edges(
        "critic_node",
        route_critic_decision,
        {
            "correction_node": "correction_node",
            "report_generation_node": "report_generation_node",
        },
    )

    # Correction loop back to risk assessment / critic audit
    workflow.add_edge("correction_node", "critic_node")

    # Final termination edge
    workflow.add_edge("report_generation_node", END)

    # Compile the graph
    app = workflow.compile()
    return app


class LangGraphInvestigationOrchestrator:
    """Master Multi-Agent Investigation Orchestrator powered by LangGraph and MCP."""

    def __init__(
        self,
        data_service: Optional[DataService] = None,
        mcp_client: Optional[BankShieldMCPClient] = None,
        max_retries: int = 1,
    ) -> None:
        """Initialize orchestrator with data service, MCP client, and compiled LangGraph.

        Args:
            data_service: DataService instance.
            mcp_client: BankShieldMCPClient instance.
            max_retries: Bounded retry limit for self-correction.
        """
        self.data_service = data_service or DataService()
        self.mcp_client = mcp_client or BankShieldMCPClient()
        self.max_retries = max_retries

        # Compile the LangGraph StateGraph
        self.graph = build_investigation_graph(
            data_service=self.data_service,
            mcp_client=self.mcp_client,
            max_retries=self.max_retries,
        )

        self.session_history: List[Dict[str, Any]] = []

    def investigate_transaction(
        self,
        transaction_id: str,
        objective: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Execute end-to-end multi-agent investigation via LangGraph and MCP.

        Args:
            transaction_id: Transaction identifier to investigate.
            objective: Optional investigation objective.

        Returns:
            Dictionary matching standard investigation output schema with LangGraph state.
        """
        inv_id = f"INV-{uuid.uuid4().hex[:8].upper()}"
        obj_text = objective or "Investigate banking transaction risk indicators and baseline deviations"
        start_wall_time = time.perf_counter()

        # Initial state initialization
        initial_state: InvestigationState = {
            "investigation_id": inv_id,
            "transaction_id": transaction_id.strip() if transaction_id else "",
            "objective": obj_text,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "retry_count": 0,
            "max_retries": self.max_retries,
            "correction_applied": False,
            "correction_history": [],
            "node_history": [],
            "execution_trace": [],
            "errors": [],
            "warnings": [],
            "mcp_tool_calls": [],
            "start_time": start_wall_time,
        }

        try:
            # Execute the LangGraph StateGraph!
            final_state = self.graph.invoke(initial_state)

            final_report = final_state.get("final_report", {})
            execution_trace = final_state.get("execution_trace", [])
            total_duration_ms = final_state.get("total_duration_ms", 0.0)
            validation_status = final_state.get("validation_status", "VALIDATED")
            status = final_state.get("status", "completed")

            result_bundle = {
                "investigation_id": inv_id,
                "transaction_id": transaction_id,
                "status": status,
                "plan": final_state.get("plan", {}),
                "evidence": final_state.get("evidence", {}),
                "risk_result": final_state.get("risk_result", {}),
                "validation_status": validation_status,
                "validation_result": final_state.get("validation_result", {}),
                "final_report": final_report,
                "execution_trace": execution_trace,
                "total_duration_ms": total_duration_ms,
                "node_history": final_state.get("node_history", []),
                "mcp_tool_calls": final_state.get("mcp_tool_calls", []),
                "retry_count": final_state.get("retry_count", 0),
                "correction_applied": final_state.get("correction_applied", False),
            }

            self.session_history.append(result_bundle)
            return result_bundle

        except Exception as exc:
            logger.error("LangGraph investigation failed with exception: %s", exc, exc_info=True)
            elapsed_ms = round((time.perf_counter() - start_wall_time) * 1000, 2)
            fallback_bundle = {
                "investigation_id": inv_id,
                "transaction_id": transaction_id,
                "status": "failed",
                "error": str(exc),
                "final_report": {
                    "investigation_id": inv_id,
                    "transaction_id": transaction_id,
                    "status": "VALIDATION_FAILED",
                    "error": str(exc),
                    "priority": "Low",
                    "risk_score": 0,
                    "recommendation": "Investigation failed due to unexpected exception.",
                },
                "execution_trace": [{
                    "node": "graph_error",
                    "agent_name": "Orchestrator",
                    "status": "error",
                    "duration_ms": elapsed_ms,
                    "output_summary": f"Unhandled graph exception: {exc}",
                }],
                "total_duration_ms": elapsed_ms,
                "node_history": ["graph_error"],
                "mcp_tool_calls": [],
                "retry_count": 0,
                "correction_applied": False,
            }
            self.session_history.append(fallback_bundle)
            return fallback_bundle

    def investigate_batch(
        self,
        transaction_ids: List[str],
        objective: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Execute batch investigations across multiple transaction IDs.

        Args:
            transaction_ids: List of transaction IDs to process.
            objective: Optional shared objective.

        Returns:
            Dictionary containing individual results and aggregated batch summary.
        """
        results: List[Dict[str, Any]] = []
        scores: List[int] = []
        priority_breakdown: Dict[str, int] = {"Critical": 0, "High": 0, "Medium": 0, "Low": 0}

        for tid in transaction_ids:
            res = self.investigate_transaction(tid, objective=objective)
            results.append(res)
            score = res.get("final_report", {}).get("risk_score", 0)
            p = res.get("final_report", {}).get("priority", "Low").capitalize()
            scores.append(score)
            priority_breakdown[p] = priority_breakdown.get(p, 0) + 1

        avg_score = round(sum(scores) / len(scores), 1) if scores else 0.0

        batch_summary = {
            "total_investigated": len(results),
            "priority_breakdown": priority_breakdown,
            "average_risk_score": avg_score,
            "critical_cases": priority_breakdown.get("Critical", 0),
            "high_cases": priority_breakdown.get("High", 0),
            "medium_cases": priority_breakdown.get("Medium", 0),
            "low_cases": priority_breakdown.get("Low", 0),
        }

        return {
            "results": results,
            "batch_summary": batch_summary,
        }

    def get_investigation_history(self) -> List[Dict[str, Any]]:
        """Return history of all investigations executed in session."""
        return list(self.session_history)

    def get_execution_trace(self) -> List[Dict[str, Any]]:
        """Return execution trace from most recent investigation."""
        if self.session_history:
            return self.session_history[-1].get("execution_trace", [])
        return []

    def get_mermaid_diagram(self) -> str:
        """Export the compiled LangGraph workflow as a Mermaid diagram string."""
        try:
            return self.graph.get_graph().draw_mermaid()
        except Exception:
            return (
                "graph TD;\n"
                "  __start__ --> planner_node;\n"
                "  planner_node --> research_node;\n"
                "  research_node --> risk_assessment_node;\n"
                "  risk_assessment_node --> critic_node;\n"
                "  critic_node -->|VALIDATED| report_generation_node;\n"
                "  critic_node -->|FAILED| correction_node;\n"
                "  correction_node --> critic_node;\n"
                "  report_generation_node --> __end__;\n"
            )
