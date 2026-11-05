"""Planner Agent for BankShield AI.

Responsible for decomposing fraud investigation objectives into ordered,
deterministic execution plans for single transactions and batches.
"""

from datetime import datetime, timezone
import logging
import random
from typing import Any, Dict, List, Optional
import uuid

from pydantic import BaseModel, Field

# Ensure reproducibility
random.seed(42)

logger = logging.getLogger(__name__)


class PlanStep(BaseModel):
    """Represents a discrete step in an investigation workflow."""

    step_number: int = Field(
        ..., description="Ordered 1-based index of the execution step"
    )
    name: str = Field(..., description="Unique machine-readable step identifier")
    description: str = Field(..., description="Summary of the analytical step")
    assigned_agent: str = Field(
        ..., description="Agent designated to execute this step"
    )
    required_tools: List[str] = Field(
        default_factory=list, description="List of tools required"
    )
    expected_output: str = Field(
        ..., description="Expected artifact or data produced"
    )
    status: str = Field(
        default="pending",
        description="Current step status: pending, in_progress, completed, failed",
    )


class InvestigationPlan(BaseModel):
    """Structured execution plan for a single transaction investigation."""

    investigation_id: str = Field(
        default_factory=lambda: str(uuid.uuid4()),
        description="Unique UUID for this investigation",
    )
    transaction_id: str = Field(
        ..., description="Target transaction identifier under review"
    )
    objective: str = Field(
        default="Investigate transaction risk",
        description="Primary investigation objective",
    )
    created_at: str = Field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat(),
        description="ISO 8601 UTC creation timestamp",
    )
    steps: List[PlanStep] = Field(
        default_factory=list, description="Ordered execution steps"
    )
    status: str = Field(
        default="planned",
        description="Overall plan status: planned, in_progress, completed, error",
    )
    error: Optional[str] = Field(
        default=None, description="Error message if planning failed"
    )


class BatchInvestigationPlan(BaseModel):
    """Structured execution plan for a batch of transactions."""

    investigation_id: str = Field(
        default_factory=lambda: str(uuid.uuid4()),
        description="Unique UUID for this batch investigation",
    )
    batch_id: str = Field(
        default_factory=lambda: str(uuid.uuid4()),
        description="Unique batch identifier",
    )
    transaction_ids: List[str] = Field(
        ..., description="List of transaction identifiers"
    )
    objective: str = Field(
        default="Investigate batch transaction risk",
        description="Batch investigation objective",
    )
    created_at: str = Field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat(),
        description="ISO 8601 UTC creation timestamp",
    )
    transaction_plans: List[Dict[str, Any]] = Field(
        default_factory=list,
        description="Individual investigation plans for each transaction",
    )
    summary_step: Dict[str, Any] = Field(
        ..., description="Final aggregation step for the batch"
    )
    status: str = Field(
        default="planned",
        description="Overall batch plan status: planned, in_progress, completed, error",
    )
    error: Optional[str] = Field(
        default=None, description="Error message if planning failed"
    )


class PlannerAgent:
    """Agent responsible for structuring and sequencing transaction investigations."""

    STANDARD_STEPS = [
        {
            "step_number": 1,
            "name": "validate_input",
            "description": "Validate transaction input format and existence",
            "assigned_agent": "PlannerAgent",
            "required_tools": ["input_validator"],
            "expected_output": "Validation status and confirmed transaction parameters",
        },
        {
            "step_number": 2,
            "name": "retrieve_data",
            "description": "Retrieve transaction details, account profile, and historical transaction records",
            "assigned_agent": "ResearchAgent",
            "required_tools": [
                "get_transaction",
                "get_account_history",
                "get_recent_transactions",
            ],
            "expected_output": "Transaction details, account profile, and historical transaction records",
        },
        {
            "step_number": 3,
            "name": "calculate_indicators",
            "description": "Calculate behavioral deviation metrics and identify anomaly risk indicators",
            "assigned_agent": "ResearchAgent",
            "required_tools": [
                "calculate_transaction_deviation",
                "identify_transaction_indicators",
            ],
            "expected_output": "Statistical deviation metrics and computed risk indicators",
        },
        {
            "step_number": 4,
            "name": "assess_risk",
            "description": "Apply risk assessment rules, compute risk score, and determine priority tier",
            "assigned_agent": "RiskAssessmentAgent",
            "required_tools": ["evaluate_risk_rules"],
            "expected_output": "Risk score (0-100), priority tier, triggered rules, and recommended action",
        },
        {
            "step_number": 5,
            "name": "validate_results",
            "description": "Audit calculations, verify evidence consistency, and validate priority classification",
            "assigned_agent": "CriticAgent",
            "required_tools": ["critic_validator"],
            "expected_output": "Independent validation report, recalculation score, and check results",
        },
        {
            "step_number": 6,
            "name": "generate_report",
            "description": "Synthesize research evidence, risk evaluation, and audit results into an investigation report",
            "assigned_agent": "InvestigationOrchestrator",
            "required_tools": ["generate_investigation_report"],
            "expected_output": "Comprehensive investigation dossier and action recommendation",
        },
    ]

    def __init__(self) -> None:
        """Initialize PlannerAgent with structured logger."""
        logger.info("PlannerAgent initialized successfully.")

    def plan_investigation(
        self,
        transaction_id: str,
        objective: str = "Investigate transaction risk",
    ) -> Dict[str, Any]:
        """Create an ordered, structured investigation plan for a single transaction.

        Args:
            transaction_id: Target transaction identifier. Must be a non-empty string.
            objective: High-level goal of the investigation.

        Returns:
            Dict containing the investigation plan or an error plan if validation fails.
        """
        logger.info(
            "Planning investigation for transaction_id='%s' with objective='%s'",
            transaction_id,
            objective,
        )

        # Input validation
        if not isinstance(transaction_id, str) or not transaction_id.strip():
            error_msg = "Invalid transaction_id: must be a non-empty string."
            logger.warning("Planning failed: %s", error_msg)
            error_plan = InvestigationPlan(
                transaction_id=str(transaction_id) if transaction_id is not None else "",
                objective=objective,
                steps=[],
                status="error",
                error=error_msg,
            )
            return error_plan.model_dump()

        sanitized_tx_id = transaction_id.strip()
        steps = [
            PlanStep(
                step_number=step["step_number"],
                name=step["name"],
                description=step["description"],
                assigned_agent=step["assigned_agent"],
                required_tools=list(step["required_tools"]),
                expected_output=step["expected_output"],
                status="pending",
            )
            for step in self.STANDARD_STEPS
        ]

        plan = InvestigationPlan(
            transaction_id=sanitized_tx_id,
            objective=objective,
            steps=steps,
            status="planned",
            error=None,
        )

        logger.info(
            "Created plan %s with %d steps for transaction %s",
            plan.investigation_id,
            len(plan.steps),
            sanitized_tx_id,
        )
        return plan.model_dump()

    def plan_batch_investigation(
        self,
        transaction_ids: List[str],
        objective: str = "Investigate batch transactions risk",
    ) -> Dict[str, Any]:
        """Create a coordinated plan for a batch of transactions including summary aggregation.

        Args:
            transaction_ids: List of target transaction identifiers.
            objective: High-level goal of the batch investigation.

        Returns:
            Dict containing individual transaction plans and a summary aggregation step.
        """
        logger.info(
            "Planning batch investigation for %d transactions with objective='%s'",
            len(transaction_ids) if isinstance(transaction_ids, list) else 0,
            objective,
        )

        if not isinstance(transaction_ids, list) or len(transaction_ids) == 0:
            error_msg = "Invalid transaction_ids: must be a non-empty list of transaction identifiers."
            logger.warning("Batch planning failed: %s", error_msg)
            return {
                "investigation_id": str(uuid.uuid4()),
                "batch_id": str(uuid.uuid4()),
                "transaction_ids": transaction_ids if isinstance(transaction_ids, list) else [],
                "objective": objective,
                "created_at": datetime.now(timezone.utc).isoformat(),
                "transaction_plans": [],
                "summary_step": {},
                "status": "error",
                "error": error_msg,
            }

        transaction_plans: List[Dict[str, Any]] = []
        for tx_id in transaction_ids:
            tx_plan = self.plan_investigation(
                tx_id,
                objective=f"Evaluate risk indicators for transaction {tx_id} within batch",
            )
            transaction_plans.append(tx_plan)

        summary_step = {
            "step_number": len(transaction_plans) + 1,
            "name": "aggregate_batch_results",
            "description": "Aggregate individual investigation findings and compile batch risk summary",
            "assigned_agent": "InvestigationOrchestrator",
            "required_tools": ["generate_batch_summary"],
            "expected_output": "Consolidated batch investigation metrics and high-priority case list",
            "status": "pending",
        }

        batch_plan = BatchInvestigationPlan(
            transaction_ids=transaction_ids,
            objective=objective,
            transaction_plans=transaction_plans,
            summary_step=summary_step,
            status="planned",
            error=None,
        )

        logger.info(
            "Created batch plan %s containing %d transactions",
            batch_plan.batch_id,
            len(transaction_plans),
        )
        return batch_plan.model_dump()
