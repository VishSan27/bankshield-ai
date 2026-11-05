"""Investigation Orchestrator for BankShield AI.

Coordinates the multi-agent investigation workflow:
1. PlannerAgent structures the execution steps.
2. ResearchAgent retrieves transactional evidence and computes behavioral indicators.
3. RiskAssessmentAgent evaluates risk scoring rules and priority tier.
4. CriticAgent performs an independent audit and score recalculation.
   (Automatically executes one retry if critical validation errors arise)
5. Synthesizes a comprehensive final report and maintains session history and telemetry.
"""

from datetime import datetime, timezone
import logging
import random
import time
from typing import Any, Dict, List, Optional
import uuid

from pydantic import BaseModel, Field

from agents.critic_agent import CriticAgent
from agents.planner_agent import PlannerAgent
from agents.research_agent import ResearchAgent
from agents.risk_assessment_agent import RiskAssessmentAgent

# Ensure deterministic execution
random.seed(42)

logger = logging.getLogger(__name__)

# Try importing DataService
try:
    from services.data_service import DataService
except ImportError:
    DataService = None

# Try importing report tools
try:
    from tools.report_tools import generate_investigation_report
except ImportError:
    generate_investigation_report = None


class ExecutionTraceEntry(BaseModel):
    """Telemetry record for a single stage in the orchestration workflow."""

    agent_name: str = Field(..., description="Agent or component that executed the action")
    action: str = Field(..., description="Operation or step identifier executed")
    status: str = Field(..., description="Execution status: completed, failed, retried")
    duration_ms: float = Field(..., description="Execution duration in milliseconds")
    output_summary: str = Field(..., description="Brief summary of produced artifact")


class InvestigationOrchestrator:
    """Master orchestrator coordinating all analytical agents and workflow lifecycle."""

    def __init__(self, data_service: Optional[Any] = None) -> None:
        """Initialize the orchestrator and all member agents.

        Args:
            data_service: DataService instance. If None, instantiates a default DataService.
        """
        if data_service is not None:
            self.data_service = data_service
        elif DataService is not None:
            try:
                self.data_service = DataService()
            except Exception as exc:
                logger.warning("Could not instantiate DataService in Orchestrator: %s", exc)
                self.data_service = None
        else:
            self.data_service = None

        # Instantiate analytical agents
        self.planner = PlannerAgent()
        self.researcher = ResearchAgent(data_service=self.data_service)
        self.risk_assessor = RiskAssessmentAgent()
        self.critic = CriticAgent()

        # Session memory
        self.history: List[Dict[str, Any]] = []
        self.global_trace: List[Dict[str, Any]] = []

        logger.info(
            "InvestigationOrchestrator initialized with data_service=%s",
            type(self.data_service).__name__ if self.data_service else "None",
        )

    def _record_trace(
        self,
        trace_list: List[Dict[str, Any]],
        agent_name: str,
        action: str,
        status: str,
        duration_ms: float,
        output_summary: str,
    ) -> None:
        """Record an execution trace entry into local and session telemetry."""
        entry = {
            "agent_name": agent_name,
            "action": action,
            "status": status,
            "duration_ms": duration_ms,
            "output_summary": output_summary,
        }
        trace_list.append(entry)
        self.global_trace.append(entry)

    def _build_default_report(
        self,
        investigation_id: str,
        transaction: Dict[str, Any],
        evidence: Dict[str, Any],
        risk_result: Dict[str, Any],
        validation_result: Dict[str, Any],
        plan: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Fallback report generator if report_tools is not yet present."""
        return {
            "investigation_id": investigation_id,
            "transaction_id": transaction.get("transaction_id", "UNKNOWN"),
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "transaction_summary": {
                "transaction_id": transaction.get("transaction_id"),
                "account_id": transaction.get("account_id"),
                "amount": transaction.get("amount"),
                "timestamp": transaction.get("timestamp"),
                "merchant": transaction.get("merchant"),
                "channel": transaction.get("channel"),
            },
            "risk_assessment": {
                "risk_score": risk_result.get("risk_score", 0),
                "priority": risk_result.get("priority", "Low"),
                "triggered_rules": risk_result.get("triggered_rules", []),
                "recommended_action": risk_result.get("recommended_action", ""),
                "assessment_summary": risk_result.get("assessment_summary", ""),
            },
            "evidence_metrics": {
                "deviation_analysis": evidence.get("deviation_analysis", {}),
                "active_indicators": evidence.get("indicators", {}),
                "missing_warnings": evidence.get("missing_data_warnings", []),
            },
            "audit_validation": {
                "status": validation_result.get("status", "UNKNOWN"),
                "validation_score": validation_result.get("validation_score", 0.0),
                "score_matches": validation_result.get("score_matches", True),
                "recalculated_score": validation_result.get("recalculated_score", 0),
                "errors": validation_result.get("errors", []),
                "warnings": validation_result.get("warnings", []),
            },
            "investigation_plan": {
                "objective": plan.get("objective"),
                "steps_completed": sum(
                    1 for s in plan.get("steps", []) if s.get("status") == "completed"
                ),
                "total_steps": len(plan.get("steps", [])),
            },
        }

    def investigate_transaction(
        self,
        transaction_id: str,
        objective: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Execute a full multi-agent investigation lifecycle for a single transaction.

        Flow:
            1. PlannerAgent structures execution plan.
            2. Steps are executed in order, with status updated dynamically.
            3. ResearchAgent retrieves context and behavioral indicators.
            4. RiskAssessmentAgent computes deterministic risk score and actions.
            5. CriticAgent performs quality audit and independent recalculation.
               - Retries risk evaluation if critical validation discrepancies occur.
            6. Report is synthesized via report_tools.
            7. Results and telemetry are persisted to session history.

        Args:
            transaction_id: Target transaction ID to evaluate.
            objective: Optional investigative focus. Defaults to 'Investigate transaction risk'.

        Returns:
            Dict containing full dossier, agent outputs, execution trace, final report,
            validation status, and timing metrics.
        """
        overall_start = time.perf_counter()
        investigation_objective = objective or "Investigate transaction risk"
        logger.info(
            "Starting coordinated investigation for transaction_id='%s'",
            transaction_id,
        )

        execution_trace: List[Dict[str, Any]] = []

        # -------------------------------------------------------------
        # Step 1: Planning
        # -------------------------------------------------------------
        t0 = time.perf_counter()
        plan = self.planner.plan_investigation(transaction_id, investigation_objective)
        d_plan = round((time.perf_counter() - t0) * 1000, 2)

        if plan.get("status") == "error":
            self._record_trace(
                execution_trace,
                "PlannerAgent",
                "plan_investigation",
                "failed",
                d_plan,
                f"Planning aborted: {plan.get('error')}",
            )
            total_duration = round((time.perf_counter() - overall_start) * 1000, 2)
            error_result = {
                "investigation_id": plan.get("investigation_id"),
                "transaction_id": transaction_id,
                "objective": investigation_objective,
                "plan": plan,
                "evidence": {},
                "risk_result": {},
                "validation_result": {},
                "validation_status": "VALIDATION_FAILED",
                "final_report": {"error": plan.get("error")},
                "execution_trace": execution_trace,
                "total_duration_ms": total_duration,
                "status": "failed",
            }
            self.history.append(error_result)
            return error_result

        self._record_trace(
            execution_trace,
            "PlannerAgent",
            "plan_investigation",
            "completed",
            d_plan,
            f"Created plan with {len(plan.get('steps', []))} steps",
        )

        # Execution State Containers
        evidence: Dict[str, Any] = {}
        risk_result: Dict[str, Any] = {}
        validation_result: Dict[str, Any] = {}
        final_report: Dict[str, Any] = {}
        plan_steps = plan.get("steps", [])

        # Execute Plan Steps in Order
        for step in plan_steps:
            step_name = step.get("name")
            step["status"] = "in_progress"
            s_start = time.perf_counter()

            try:
                # ---------------------------------------------------------
                # Step: validate_input
                # ---------------------------------------------------------
                if step_name == "validate_input":
                    if isinstance(transaction_id, str) and transaction_id.strip():
                        step["status"] = "completed"
                        summary = f"Validated transaction identifier '{transaction_id.strip()}'"
                    else:
                        step["status"] = "failed"
                        summary = "Invalid transaction ID format"

                    s_duration = round((time.perf_counter() - s_start) * 1000, 2)
                    self._record_trace(
                        execution_trace,
                        step.get("assigned_agent", "PlannerAgent"),
                        step_name,
                        step["status"],
                        s_duration,
                        summary,
                    )
                    if step["status"] == "failed":
                        break

                # ---------------------------------------------------------
                # Step: retrieve_data
                # ---------------------------------------------------------
                elif step_name == "retrieve_data":
                    # ResearchAgent retrieves data and establishes baseline
                    evidence = self.researcher.investigate_transaction(transaction_id)
                    tx_found = evidence.get("transaction_details") is not None
                    step["status"] = "completed" if tx_found else "failed"

                    s_duration = round((time.perf_counter() - s_start) * 1000, 2)
                    summary = (
                        f"Retrieved transaction record (found={tx_found}) and "
                        f"{len(evidence.get('account_history', []))} historical events"
                    )
                    self._record_trace(
                        execution_trace,
                        step.get("assigned_agent", "ResearchAgent"),
                        step_name,
                        step["status"],
                        s_duration,
                        summary,
                    )
                    if step["status"] == "failed":
                        break

                # ---------------------------------------------------------
                # Step: calculate_indicators
                # ---------------------------------------------------------
                elif step_name == "calculate_indicators":
                    indicators = evidence.get("indicators", {})
                    deviation = evidence.get("deviation_analysis", {})
                    active_count = sum(1 for v in indicators.values() if v)
                    step["status"] = "completed"

                    s_duration = round((time.perf_counter() - s_start) * 1000, 2)
                    summary = (
                        f"Computed {active_count} active indicators; "
                        f"deviation z-score: {deviation.get('z_score', 'N/A')}"
                    )
                    self._record_trace(
                        execution_trace,
                        step.get("assigned_agent", "ResearchAgent"),
                        step_name,
                        step["status"],
                        s_duration,
                        summary,
                    )

                # ---------------------------------------------------------
                # Step: assess_risk
                # ---------------------------------------------------------
                elif step_name == "assess_risk":
                    risk_result = self.risk_assessor.assess_risk(evidence)
                    step["status"] = "completed" if risk_result.get("status") == "success" else "failed"

                    s_duration = round((time.perf_counter() - s_start) * 1000, 2)
                    summary = (
                        f"Risk score: {risk_result.get('risk_score')}/100, "
                        f"Priority: {risk_result.get('priority')}, "
                        f"{len(risk_result.get('triggered_rules', []))} rules triggered"
                    )
                    self._record_trace(
                        execution_trace,
                        step.get("assigned_agent", "RiskAssessmentAgent"),
                        step_name,
                        step["status"],
                        s_duration,
                        summary,
                    )

                # ---------------------------------------------------------
                # Step: validate_results (Critic validation with 1 retry)
                # ---------------------------------------------------------
                elif step_name == "validate_results":
                    tx_data = evidence.get("transaction_details", {})
                    validation_result = self.critic.validate_investigation(
                        transaction_data=tx_data,
                        evidence=evidence,
                        risk_result=risk_result,
                        plan=plan,
                    )

                    # RETRY POLICY: If critical validation errors occur, attempt ONE recalculation/retry
                    if not validation_result.get("is_valid", True) and validation_result.get("errors"):
                        r_start = time.perf_counter()
                        logger.warning(
                            "Validation failed with %d critical errors. Initiating ONE retry...",
                            len(validation_result.get("errors", [])),
                        )

                        # Retry Action: Reconcile risk score if independent recalculation differed
                        if not validation_result.get("score_matches", True):
                            recalc_score = validation_result.get("recalculated_score", 0)
                            risk_result["risk_score"] = recalc_score
                            risk_result["priority"] = self.critic._determine_expected_priority(recalc_score)
                            risk_result["assessment_summary"] = (
                                f"Risk assessment recalculated during validation retry. "
                                f"Updated risk score: {recalc_score}/100, Priority: {risk_result['priority']}."
                            )

                        # Re-run validation audit
                        retry_validation = self.critic.validate_investigation(
                            transaction_data=tx_data,
                            evidence=evidence,
                            risk_result=risk_result,
                            plan=plan,
                        )
                        r_duration = round((time.perf_counter() - r_start) * 1000, 2)

                        self._record_trace(
                            execution_trace,
                            "CriticAgent",
                            "retry_validation",
                            "completed" if retry_validation.get("is_valid") else "failed",
                            r_duration,
                            f"Retry audit status: {retry_validation.get('status')}",
                        )
                        validation_result = retry_validation

                    step["status"] = "completed" if validation_result.get("is_valid") else "failed"
                    s_duration = round((time.perf_counter() - s_start) * 1000, 2)
                    summary = (
                        f"Audit status: {validation_result.get('status')} "
                        f"({validation_result.get('validation_score')}% pass rate, "
                        f"{len(validation_result.get('errors', []))} errors)"
                    )
                    self._record_trace(
                        execution_trace,
                        step.get("assigned_agent", "CriticAgent"),
                        step_name,
                        step["status"],
                        s_duration,
                        summary,
                    )

                # ---------------------------------------------------------
                # Step: generate_report
                # ---------------------------------------------------------
                elif step_name == "generate_report":
                    tx_data = evidence.get("transaction_details", {})
                    inv_id = plan.get("investigation_id", str(uuid.uuid4()))

                    # Dynamic resolution if report_tools was imported after orchestrator instantiation
                    report_fn = generate_investigation_report
                    if report_fn is None:
                        try:
                            from tools.report_tools import generate_investigation_report as dynamic_report_fn
                            report_fn = dynamic_report_fn
                        except ImportError:
                            report_fn = None

                    if report_fn is not None:
                        try:
                            final_report = report_fn(
                                inv_id,
                                tx_data,
                                evidence,
                                risk_result,
                                validation_result,
                                plan,
                            )
                        except Exception as r_err:
                            logger.warning(
                                "generate_investigation_report failed: %s. Using default generator.",
                                r_err,
                            )
                            final_report = self._build_default_report(
                                inv_id, tx_data, evidence, risk_result, validation_result, plan
                            )
                    else:
                        final_report = self._build_default_report(
                            inv_id, tx_data, evidence, risk_result, validation_result, plan
                        )

                    step["status"] = "completed"
                    s_duration = round((time.perf_counter() - s_start) * 1000, 2)
                    summary = f"Synthesized final report for investigation {inv_id}"
                    self._record_trace(
                        execution_trace,
                        step.get("assigned_agent", "InvestigationOrchestrator"),
                        step_name,
                        step["status"],
                        s_duration,
                        summary,
                    )

            except Exception as step_exc:
                step["status"] = "failed"
                s_duration = round((time.perf_counter() - s_start) * 1000, 2)
                logger.error("Exception during step %s: %s", step_name, step_exc, exc_info=True)
                self._record_trace(
                    execution_trace,
                    step.get("assigned_agent", "Orchestrator"),
                    step_name,
                    "failed",
                    s_duration,
                    f"Unhandled error: {str(step_exc)}",
                )
                break

        # Finalize overall plan status
        failed_steps = [s for s in plan_steps if s.get("status") == "failed"]
        plan["status"] = "completed" if not failed_steps else "failed"

        total_duration = round((time.perf_counter() - overall_start) * 1000, 2)
        overall_status = "completed" if plan["status"] == "completed" else "failed"

        investigation_result = {
            "investigation_id": plan.get("investigation_id"),
            "transaction_id": transaction_id,
            "objective": investigation_objective,
            "plan": plan,
            "evidence": evidence,
            "risk_result": risk_result,
            "validation_result": validation_result,
            "validation_status": validation_result.get("status", "UNKNOWN"),
            "final_report": final_report,
            "execution_trace": execution_trace,
            "total_duration_ms": total_duration,
            "status": overall_status,
        }

        # Store in session memory
        self.history.append(investigation_result)

        logger.info(
            "Investigation finished for %s in %.2fms with status=%s, validation=%s",
            transaction_id,
            total_duration,
            overall_status,
            validation_result.get("status"),
        )
        return investigation_result

    def investigate_batch(
        self,
        transaction_ids: List[str],
        objective: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Execute coordinated multi-agent investigations across a batch of transactions.

        Args:
            transaction_ids: List of transaction identifiers.
            objective: High-level objective for the batch run.

        Returns:
            Dict containing individual investigation results, consolidated batch summary,
            and batch execution telemetry.
        """
        batch_start = time.perf_counter()
        batch_objective = objective or "Investigate batch transactions risk"
        logger.info(
            "Starting batch investigation for %d transactions",
            len(transaction_ids) if isinstance(transaction_ids, list) else 0,
        )

        batch_trace: List[Dict[str, Any]] = []

        # Step 1: Batch Planning
        batch_plan = self.planner.plan_batch_investigation(
            transaction_ids, objective=batch_objective
        )
        self._record_trace(
            batch_trace,
            "PlannerAgent",
            "plan_batch_investigation",
            "completed" if batch_plan.get("status") == "planned" else "failed",
            0.0,
            f"Created batch plan for {len(transaction_ids)} transactions",
        )

        results: List[Dict[str, Any]] = []
        for tx_id in transaction_ids:
            res = self.investigate_transaction(tx_id, objective=f"Batch evaluation of {tx_id}")
            results.append(res)

        # Step 2: Batch Summary Aggregation
        total_count = len(results)
        successful_count = sum(1 for r in results if r.get("status") == "completed")
        failed_count = total_count - successful_count

        priority_breakdown = {"Critical": 0, "High": 0, "Medium": 0, "Low": 0}
        risk_scores: List[int] = []
        critical_cases: List[str] = []

        for r in results:
            risk_res = r.get("risk_result", {})
            priority = str(risk_res.get("priority", "Low")).capitalize()
            if priority in priority_breakdown:
                priority_breakdown[priority] += 1
            else:
                priority_breakdown[priority] = 1

            score = risk_res.get("risk_score")
            if score is not None:
                risk_scores.append(int(score))

            if priority in ("High", "Critical"):
                critical_cases.append(r.get("transaction_id", ""))

        avg_score = round(sum(risk_scores) / len(risk_scores), 1) if risk_scores else 0.0
        total_batch_duration = round((time.perf_counter() - batch_start) * 1000, 2)

        batch_summary = {
            "batch_id": batch_plan.get("batch_id", str(uuid.uuid4())),
            "total_investigated": total_count,
            "successful_count": successful_count,
            "failed_count": failed_count,
            "priority_breakdown": priority_breakdown,
            "average_risk_score": avg_score,
            "critical_cases_count": len(critical_cases),
            "critical_cases": critical_cases,
            "total_duration_ms": total_batch_duration,
        }

        self._record_trace(
            batch_trace,
            "InvestigationOrchestrator",
            "aggregate_batch_results",
            "completed",
            0.0,
            f"Aggregated {total_count} transactions: avg score {avg_score}, {len(critical_cases)} high/critical cases",
        )

        return {
            "batch_id": batch_summary["batch_id"],
            "results": results,
            "batch_summary": batch_summary,
            "execution_trace": batch_trace,
        }

    def get_investigation_history(self) -> List[Dict[str, Any]]:
        """Return a copy of all completed investigations in this session."""
        return list(self.history)

    def get_execution_trace(self) -> List[Dict[str, Any]]:
        """Return a copy of all execution telemetry entries recorded across investigations."""
        return list(self.global_trace)
