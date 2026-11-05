"""Evaluation Service for BankShield AI.

Provides automated evaluation suites and benchmark test scenarios
to validate multi-agent orchestration, risk rule accuracy, critic audits,
priority classification, and graceful error handling.
"""

from datetime import datetime, timezone
import logging
import random
import time
from typing import Any, Dict, List, Optional

# Ensure reproducibility
random.seed(42)

logger = logging.getLogger(__name__)

PRIORITY_RANK: Dict[str, int] = {
    "Low": 1,
    "Medium": 2,
    "High": 3,
    "Critical": 4,
}

# Standard benchmark test scenarios mapped to synthetic transaction dataset
TEST_SCENARIOS: List[Dict[str, Any]] = [
    {
        "scenario_name": "High Risk - Extreme Amount & Unrecognized Merchant",
        "description": "Transaction amount significantly exceeds historical baseline at an unrecognized merchant during nocturnal hours",
        "transaction_id": "TXN-10198",
        "expected_min_priority": "High",
        "expected_indicators": ["high_amount", "new_merchant", "unusual_time"],
        "is_error_scenario": False,
    },
    {
        "scenario_name": "High Risk - Multiple Compound Anomalies",
        "description": "High amount transaction preceded by 5 consecutive failed authorization attempts and nocturnal activity",
        "transaction_id": "TXN-10471",
        "expected_min_priority": "High",
        "expected_indicators": ["high_amount", "failed_attempts", "new_merchant"],
        "is_error_scenario": False,
    },
    {
        "scenario_name": "Medium Risk - Multiple Failed Authorization Attempts",
        "description": "Transaction preceded by 4 failed authorization attempts",
        "transaction_id": "TXN-10015",
        "expected_min_priority": "Medium",
        "expected_indicators": ["failed_attempts"],
        "is_error_scenario": False,
    },
    {
        "scenario_name": "Medium Risk - Nocturnal Window Activity",
        "description": "Transaction executed during off-hours nocturnal window (03:00) with unrecognized merchant",
        "transaction_id": "TXN-10016",
        "expected_min_priority": "Medium",
        "expected_indicators": ["unusual_time"],
        "is_error_scenario": False,
    },
    {
        "scenario_name": "Low Risk - Routine Daytime Transaction",
        "description": "Normal daytime transaction within historical average at recognized merchant",
        "transaction_id": "TXN-10078",
        "expected_min_priority": "Low",
        "expected_indicators": [],
        "is_error_scenario": False,
    },
    {
        "scenario_name": "Error Handling - Non-Existent Transaction Identifier",
        "description": "Attempting to investigate non-existent ID 'TXN-99999' should be handled gracefully without crashing",
        "transaction_id": "TXN-99999",
        "expected_min_priority": "Low",
        "expected_indicators": [],
        "is_error_scenario": True,
    },
]


def run_evaluation(
    orchestrator: Any,
    scenarios: Optional[List[Dict[str, Any]]] = None,
) -> Dict[str, Any]:
    """Execute end-to-end evaluation suite against an InvestigationOrchestrator.

    Args:
        orchestrator: Instance of InvestigationOrchestrator or equivalent runner.
        scenarios: Optional list of test scenario specifications. Defaults to TEST_SCENARIOS.

    Returns:
        Structured evaluation metrics containing total_scenarios, passed, failed,
        results per scenario, validation_pass_rate, average_execution_time,
        priority_distribution, and error_handling_results.
    """
    if orchestrator is None:
        raise ValueError("An active orchestrator instance is required to run evaluation.")

    eval_scenarios = scenarios or TEST_SCENARIOS
    total_scenarios = len(eval_scenarios)
    passed_count = 0
    failed_count = 0
    results: List[Dict[str, Any]] = []
    durations_ms: List[float] = []
    priority_distribution: Dict[str, int] = {
        "Critical": 0,
        "High": 0,
        "Medium": 0,
        "Low": 0,
    }
    error_handling_results: Dict[str, Any] = {}

    logger.info("Beginning test evaluation across %d scenarios", total_scenarios)

    for scenario in eval_scenarios:
        sc_name = scenario.get("scenario_name", "Unnamed Scenario")
        tx_id = scenario.get("transaction_id", "")
        expected_min_priority = scenario.get("expected_min_priority", "Low").capitalize()
        expected_indicators = scenario.get("expected_indicators", [])
        is_error_sc = bool(scenario.get("is_error_scenario", False))

        start_time = time.perf_counter()
        scenario_passed = True
        failure_reasons: List[str] = []
        actual_priority = "Low"
        actual_indicators: List[str] = []
        actual_score = 0
        investigation_output: Dict[str, Any] = {}

        try:
            # Invoke orchestrator
            if hasattr(orchestrator, "investigate_transaction"):
                investigation_output = orchestrator.investigate_transaction(tx_id)
            elif hasattr(orchestrator, "investigate"):
                investigation_output = orchestrator.investigate(tx_id)
            elif callable(orchestrator):
                investigation_output = orchestrator(tx_id)
            else:
                raise AttributeError("Orchestrator does not have an investigate method")

            elapsed_ms = round((time.perf_counter() - start_time) * 1000, 2)
            durations_ms.append(elapsed_ms)

            # Evaluate outcome based on scenario type
            if is_error_sc:
                # Error scenario: verify graceful handling without crash
                status = investigation_output.get("status", "")
                plan_err = investigation_output.get("plan", {}).get("error")
                rep_err = investigation_output.get("final_report", {}).get("error")
                has_error_flag = bool(status == "failed" or plan_err or rep_err)

                if has_error_flag:
                    scenario_passed = True
                    error_handling_results = {
                        "scenario_name": sc_name,
                        "transaction_id": tx_id,
                        "handled_gracefully": True,
                        "status": status,
                        "duration_ms": elapsed_ms,
                        "error_message": plan_err or rep_err or "Handled as expected",
                    }
                else:
                    scenario_passed = False
                    failure_reasons.append("Expected error handling status 'failed' or error message, but received success.")
                    error_handling_results = {
                        "scenario_name": sc_name,
                        "transaction_id": tx_id,
                        "handled_gracefully": False,
                        "status": status,
                        "duration_ms": elapsed_ms,
                        "error_message": "Failed to flag non-existent transaction",
                    }

            else:
                # Standard investigation scenario
                final_rep = investigation_output.get("final_report", {})
                risk_res = investigation_output.get("risk_result", {})

                actual_score = int(final_rep.get("risk_score", risk_res.get("risk_score", 0)))
                actual_priority = str(
                    final_rep.get("priority", risk_res.get("priority", "Low"))
                ).capitalize()

                if actual_priority in priority_distribution:
                    priority_distribution[actual_priority] += 1
                else:
                    priority_distribution[actual_priority] = 1

                # 1. Verify Priority Threshold
                actual_rank = PRIORITY_RANK.get(actual_priority, 1)
                expected_rank = PRIORITY_RANK.get(expected_min_priority, 1)
                if actual_rank < expected_rank:
                    scenario_passed = False
                    failure_reasons.append(
                        f"Priority tier '{actual_priority}' lower than expected minimum '{expected_min_priority}'."
                    )

                # 2. Verify Expected Indicators
                triggered_rules = final_rep.get("triggered_rules", risk_res.get("triggered_rules", []))
                actual_indicators = [
                    r.get("rule_name", r.get("name", ""))
                    for r in triggered_rules
                    if isinstance(r, dict)
                ]

                # Check if each expected indicator was triggered
                for exp_ind in expected_indicators:
                    if exp_ind not in actual_indicators:
                        scenario_passed = False
                        failure_reasons.append(
                            f"Expected risk indicator '{exp_ind}' was not triggered in investigation (actual: {actual_indicators})."
                        )

        except Exception as exc:
            elapsed_ms = round((time.perf_counter() - start_time) * 1000, 2)
            durations_ms.append(elapsed_ms)
            scenario_passed = False
            failure_reasons.append(f"Unhandled exception during investigation: {exc}")
            logger.error("Error evaluating scenario '%s': %s", sc_name, exc, exc_info=True)

        if scenario_passed:
            passed_count += 1
        else:
            failed_count += 1

        results.append(
            {
                "scenario_name": sc_name,
                "transaction_id": tx_id,
                "passed": scenario_passed,
                "duration_ms": elapsed_ms,
                "expected_min_priority": expected_min_priority,
                "actual_priority": actual_priority,
                "actual_score": actual_score,
                "expected_indicators": expected_indicators,
                "actual_indicators": actual_indicators,
                "failure_reasons": failure_reasons,
            }
        )

    pass_rate = round((passed_count / total_scenarios) * 100, 2) if total_scenarios > 0 else 0.0
    avg_duration = round(sum(durations_ms) / len(durations_ms), 2) if durations_ms else 0.0

    evaluation_report = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "total_scenarios": total_scenarios,
        "passed": passed_count,
        "failed": failed_count,
        "validation_pass_rate": pass_rate,
        "average_execution_time": avg_duration,
        "priority_distribution": priority_distribution,
        "error_handling_results": error_handling_results,
        "results": results,
    }

    logger.info(
        "Evaluation completed: %d/%d passed (%.1f%%), avg time: %.2fms",
        passed_count,
        total_scenarios,
        pass_rate,
        avg_duration,
    )

    return evaluation_report


class EvaluationService:
    """Service wrapper for running test evaluations against orchestrators."""

    def __init__(self, orchestrator: Optional[Any] = None) -> None:
        """Initialize EvaluationService with optional orchestrator.

        Args:
            orchestrator: Optional InvestigationOrchestrator instance.
        """
        self.orchestrator = orchestrator

    def run_evaluation(
        self,
        orchestrator: Optional[Any] = None,
        scenarios: Optional[List[Dict[str, Any]]] = None,
    ) -> Dict[str, Any]:
        """Run evaluation suite using either the instance orchestrator or passed orchestrator."""
        target_orch = orchestrator or self.orchestrator
        if target_orch is None:
            raise ValueError("No orchestrator provided for evaluation.")
        return run_evaluation(target_orch, scenarios=scenarios)
