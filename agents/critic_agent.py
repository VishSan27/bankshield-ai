"""Critic and Validation Agent for BankShield AI.

Conducts rigorous, independent post-investigation audits of research evidence,
mathematical calculations, rule consistency, and compliance terminology.
Independently recalculates risk scores to guarantee auditability and integrity.
"""

from datetime import datetime, timezone
import logging
import math
import random
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field

# Ensure reproducibility
random.seed(42)

logger = logging.getLogger(__name__)

# Try importing configuration thresholds
try:
    from config.settings import (
        PRIORITY_THRESHOLDS,
        RISK_WEIGHTS,
        PriorityLevel,
        PriorityThresholds,
        settings,
    )
except ImportError:
    RISK_WEIGHTS = {
        "high_amount": 30,
        "unusual_frequency": 25,
        "new_merchant": 20,
        "failed_attempts": 15,
        "unusual_time": 10,
    }
    PRIORITY_THRESHOLDS = [
        (80, "Critical"),
        (60, "High"),
        (30, "Medium"),
        (0, "Low"),
    ]
    settings = None

# Try importing risk engine
try:
    from tools.risk_rules import RiskEngine, create_default_engine
except ImportError:
    RiskEngine = None
    create_default_engine = None


class ValidationCheck(BaseModel):
    """Audit verification check outcome."""

    name: str = Field(..., description="Machine-readable check identifier")
    passed: bool = Field(..., description="Whether the check passed")
    details: str = Field(..., description="Detailed explanation of audit findings")
    severity: str = Field(
        ..., description="Severity level: CRITICAL, WARNING, or INFO"
    )


class ValidationResult(BaseModel):
    """Consolidated critic validation outcome."""

    is_valid: bool = Field(
        ..., description="True if investigation passed all critical audits"
    )
    checks_performed: List[Dict[str, Any]] = Field(
        default_factory=list, description="List of all audit checks evaluated"
    )
    errors: List[str] = Field(
        default_factory=list, description="Actionable critical audit failures"
    )
    warnings: List[str] = Field(
        default_factory=list, description="Non-critical warnings or discrepancies"
    )
    validation_score: float = Field(
        ..., ge=0.0, le=100.0, description="Percentage of passed audit checks"
    )
    recalculated_score: int = Field(
        ..., ge=0, le=100, description="Independently recalculated risk score"
    )
    score_matches: bool = Field(
        ..., description="True if recalculated score matches reported score"
    )
    status: str = Field(
        ...,
        description="Audit status: VALIDATED, NEEDS_REVIEW, or VALIDATION_FAILED",
    )


class CriticAgent:
    """Agent responsible for auditing and validating investigative integrity."""

    # Prohibited words that constitute unsupported conclusions or premature fraud determinations
    PROHIBITED_TERMS = [
        "fraudulent transaction",
        "proven fraud",
        "confirmed thief",
        "guilty user",
        "stolen card",
        "illegal act",
    ]

    def __init__(self, risk_engine: Optional[Any] = None) -> None:
        """Initialize CriticAgent with an independent risk engine for recalculation.

        Args:
            risk_engine: Optional RiskEngine instance for independent recalculation.
        """
        if risk_engine is not None:
            self.risk_engine = risk_engine
        elif create_default_engine is not None:
            try:
                self.risk_engine = create_default_engine()
            except Exception as exc:
                logger.warning("Could not initialize default RiskEngine in CriticAgent: %s", exc)
                self.risk_engine = None
        else:
            self.risk_engine = None

        logger.info(
            "CriticAgent initialized with independent risk_engine=%s",
            type(self.risk_engine).__name__ if self.risk_engine else "Built-in",
        )

    def _independent_recalculate_risk(self, indicators: Dict[str, Any]) -> int:
        """Independently recalculate the risk score using official rules and weights."""
        if self.risk_engine is not None and hasattr(self.risk_engine, "evaluate"):
            try:
                res = self.risk_engine.evaluate(indicators)
                if isinstance(res, dict):
                    return int(round(float(res.get("total_score", res.get("risk_score", 0)))))
                return int(round(float(getattr(res, "total_score", getattr(res, "risk_score", 0)))))
            except Exception as exc:
                logger.warning("Error running engine evaluate in CriticAgent: %s", exc)

        # Built-in independent calculation using weights
        score = 0
        weights = dict(RISK_WEIGHTS)
        for key, weight in weights.items():
            val = indicators.get(key)
            is_detected = (bool(val.get("detected", False))) if isinstance(val, dict) else bool(val)
            if is_detected:
                score += weight
        return min(100, max(0, score))

    def _determine_expected_priority(self, score: int) -> str:
        """Map score to expected priority using standard thresholds."""
        for threshold, tier in PRIORITY_THRESHOLDS:
            if score >= threshold:
                return tier
        return "Low"

    def validate_investigation(
        self,
        transaction_data: Dict[str, Any],
        evidence: Dict[str, Any],
        risk_result: Dict[str, Any],
        plan: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Validate the full dossier of an investigation against 9 strict quality gates.

        Args:
            transaction_data: Raw or retrieved transaction attributes.
            evidence: Evidence bundle produced by ResearchAgent.
            risk_result: Risk evaluation output produced by RiskAssessmentAgent.
            plan: Investigation plan produced by PlannerAgent.

        Returns:
            Dict containing validation outcome, audit checklist, errors, warnings,
            recalculated score, score match flag, and status.
        """
        logger.info("Executing comprehensive investigation audit")
        checks: List[Dict[str, Any]] = []
        errors: List[str] = []
        warnings: List[str] = []

        # Safe attribute extraction
        tx_data = transaction_data if isinstance(transaction_data, dict) else {}
        ev_data = evidence if isinstance(evidence, dict) else {}
        rk_data = risk_result if isinstance(risk_result, dict) else {}
        pl_data = plan if isinstance(plan, dict) else {}

        tx_details = ev_data.get("transaction_details") or tx_data
        tx_id = (
            tx_details.get("transaction_id")
            or pl_data.get("transaction_id")
            or ""
        )
        indicators = ev_data.get("indicators") or {}
        reported_score = int(round(float(rk_data.get("risk_score", rk_data.get("total_score", 0)))))
        reported_priority = str(rk_data.get("priority", "Low")).capitalize()

        # Independently recalculate score
        recalculated_score = self._independent_recalculate_risk(indicators)
        score_matches = (recalculated_score == reported_score)

        # -------------------------------------------------------------
        # Check 1: transaction_id_valid
        # -------------------------------------------------------------
        c1_passed = bool(isinstance(tx_id, str) and tx_id.strip())
        if c1_passed:
            c1_details = f"Transaction identifier '{tx_id}' is valid and non-empty."
            c1_sev = "INFO"
        else:
            c1_details = "Transaction identifier is missing, blank, or improperly formatted."
            c1_sev = "CRITICAL"
            errors.append(c1_details)
        checks.append({
            "name": "transaction_id_valid",
            "passed": c1_passed,
            "details": c1_details,
            "severity": c1_sev,
        })

        # -------------------------------------------------------------
        # Check 2: supporting_records_exist
        # -------------------------------------------------------------
        has_tx = bool(tx_details and isinstance(tx_details, dict) and tx_details.get("amount") is not None)
        has_history_key = "account_history" in ev_data
        history_len = len(ev_data.get("account_history", []))

        if not has_tx:
            c2_passed = False
            c2_sev = "CRITICAL"
            c2_details = "Core transaction record is missing from evidence bundle."
            errors.append(c2_details)
        elif not has_history_key:
            c2_passed = False
            c2_sev = "WARNING"
            c2_details = "Evidence bundle lacks historical transaction collection attribute."
            warnings.append(c2_details)
        elif history_len == 0:
            c2_passed = True
            c2_sev = "WARNING"
            c2_details = f"Transaction record present (${tx_details.get('amount')}), but historical records are empty."
            warnings.append(c2_details)
        else:
            c2_passed = True
            c2_sev = "INFO"
            c2_details = f"Supporting transaction records exist with {history_len} historical events."
        checks.append({
            "name": "supporting_records_exist",
            "passed": c2_passed,
            "details": c2_details,
            "severity": c2_sev,
        })

        # -------------------------------------------------------------
        # Check 3: numerical_calculations
        # -------------------------------------------------------------
        dev_data = ev_data.get("deviation_analysis", {})
        dev_math_ok = True
        dev_notes: List[str] = []

        if history_len > 0 and isinstance(dev_data, dict):
            reported_mean = dev_data.get("historical_mean")
            amounts = [
                float(h["amount"])
                for h in ev_data.get("account_history", [])
                if isinstance(h, dict) and "amount" in h
            ]
            if amounts and reported_mean is not None:
                calc_mean = sum(amounts) / len(amounts)
                if abs(calc_mean - float(reported_mean)) > 1.0:
                    dev_math_ok = False
                    dev_notes.append(
                        f"Mean amount discrepancy: reported {reported_mean} vs computed {calc_mean:.2f}"
                    )

        if not score_matches:
            c3_passed = False
            c3_sev = "CRITICAL"
            c3_details = (
                f"Numerical calculation discrepancy: reported risk score ({reported_score}) "
                f"does not match independently audited score ({recalculated_score})."
            )
            errors.append(c3_details)
        elif not dev_math_ok:
            c3_passed = False
            c3_sev = "WARNING"
            c3_details = f"Deviation metrics discrepancy: {'; '.join(dev_notes)}"
            warnings.append(c3_details)
        else:
            c3_passed = True
            c3_sev = "INFO"
            c3_details = (
                f"Numerical calculations verified. Reported risk score ({reported_score}) "
                f"perfectly matches independent recalculation ({recalculated_score})."
            )
        checks.append({
            "name": "numerical_calculations",
            "passed": c3_passed,
            "details": c3_details,
            "severity": c3_sev,
        })

        # -------------------------------------------------------------
        # Check 4: score_range_valid
        # -------------------------------------------------------------
        c4_passed = (0 <= reported_score <= 100)
        if c4_passed:
            c4_details = f"Risk score ({reported_score}) is within the allowable range [0, 100]."
            c4_sev = "INFO"
        else:
            c4_details = f"Risk score ({reported_score}) is outside the allowable range [0, 100]."
            c4_sev = "CRITICAL"
            errors.append(c4_details)
        checks.append({
            "name": "score_range_valid",
            "passed": c4_passed,
            "details": c4_details,
            "severity": c4_sev,
        })

        # -------------------------------------------------------------
        # Check 5: priority_mapping_correct
        # -------------------------------------------------------------
        expected_priority = self._determine_expected_priority(reported_score)
        c5_passed = (reported_priority == expected_priority)
        if c5_passed:
            c5_details = (
                f"Priority '{reported_priority}' correctly corresponds to risk score "
                f"{reported_score} per configuration thresholds."
            )
            c5_sev = "INFO"
        else:
            c5_details = (
                f"Priority mapping error: risk score {reported_score} should map to "
                f"'{expected_priority}', but was designated as '{reported_priority}'."
            )
            c5_sev = "CRITICAL"
            errors.append(c5_details)
        checks.append({
            "name": "priority_mapping_correct",
            "passed": c5_passed,
            "details": c5_details,
            "severity": c5_sev,
        })

        # -------------------------------------------------------------
        # Check 6: explanation_consistency
        # -------------------------------------------------------------
        triggered_rules = rk_data.get("triggered_rules", [])
        inconsistent_rules: List[str] = []

        for rule in triggered_rules:
            rule_id = rule.get("rule_id", rule.get("name", "")) if isinstance(rule, dict) else str(rule)
            # Find matching indicator
            indicator_val = indicators.get(rule_id)
            if indicator_val is False:
                inconsistent_rules.append(
                    f"Rule '{rule_id}' was triggered, but indicator in evidence is False"
                )

        if inconsistent_rules:
            c6_passed = False
            c6_sev = "CRITICAL"
            c6_details = f"Explanation inconsistency detected: {'; '.join(inconsistent_rules)}."
            errors.append(c6_details)
        else:
            c6_passed = True
            c6_sev = "INFO"
            c6_details = (
                f"All {len(triggered_rules)} triggered rules have consistent, supporting indicators in evidence."
            )
        checks.append({
            "name": "explanation_consistency",
            "passed": c6_passed,
            "details": c6_details,
            "severity": c6_sev,
        })

        # -------------------------------------------------------------
        # Check 7: missing_data_warnings
        # -------------------------------------------------------------
        unflagged_gaps: List[str] = []
        evidence_warnings = ev_data.get("missing_data_warnings", [])

        if ev_data.get("account_profile") is None and not any("profile" in str(w).lower() for w in evidence_warnings):
            unflagged_gaps.append("Missing account profile not noted in missing_data_warnings")
        if history_len == 0 and not any("history" in str(w).lower() for w in evidence_warnings):
            unflagged_gaps.append("Empty account history not noted in missing_data_warnings")

        if unflagged_gaps:
            c7_passed = False
            c7_sev = "WARNING"
            c7_details = f"Data gaps present without warnings: {'; '.join(unflagged_gaps)}."
            warnings.append(c7_details)
        else:
            c7_passed = True
            c7_sev = "INFO"
            c7_details = (
                f"Missing data logging verified ({len(evidence_warnings)} warnings appropriately recorded)."
            )
        checks.append({
            "name": "missing_data_warnings",
            "passed": c7_passed,
            "details": c7_details,
            "severity": c7_sev,
        })

        # -------------------------------------------------------------
        # Check 8: unsupported_conclusions
        # -------------------------------------------------------------
        summary_text = str(rk_data.get("assessment_summary", "")).lower()
        action_text = str(rk_data.get("recommended_action", "")).lower()
        found_prohibited: List[str] = []

        for term in self.PROHIBITED_TERMS:
            if term in summary_text or term in action_text:
                found_prohibited.append(term)

        # Check if high/critical priority exists with zero triggered rules
        unsupported_severity = (reported_priority in ("High", "Critical") and len(triggered_rules) == 0)

        if found_prohibited:
            c8_passed = False
            c8_sev = "CRITICAL"
            c8_details = f"Prohibited non-neutral terminology detected: {', '.join(found_prohibited)}."
            errors.append(c8_details)
        elif unsupported_severity:
            c8_passed = False
            c8_sev = "CRITICAL"
            c8_details = f"Priority '{reported_priority}' assigned without any supporting triggered rules."
            errors.append(c8_details)
        else:
            c8_passed = True
            c8_sev = "INFO"
            c8_details = "Investigative conclusions are fully substantiated and adhere to neutral language standards."
        checks.append({
            "name": "unsupported_conclusions",
            "passed": c8_passed,
            "details": c8_details,
            "severity": c8_sev,
        })

        # -------------------------------------------------------------
        # Check 9: required_fields_present
        # -------------------------------------------------------------
        missing_fields: List[str] = []
        for field in ["risk_score", "priority", "triggered_rules", "recommended_action", "assessment_summary"]:
            if field not in rk_data:
                missing_fields.append(f"risk_result.{field}")
        for field in ["investigation_id", "steps", "status"]:
            if field not in pl_data:
                missing_fields.append(f"plan.{field}")
        for field in ["transaction_details", "indicators"]:
            if field not in ev_data:
                missing_fields.append(f"evidence.{field}")

        if missing_fields:
            c9_passed = False
            c9_sev = "CRITICAL"
            c9_details = f"Required investigative fields missing: {', '.join(missing_fields)}."
            errors.append(c9_details)
        else:
            c9_passed = True
            c9_sev = "INFO"
            c9_details = "All mandatory fields across plan, evidence, and risk assessment are populated."
        checks.append({
            "name": "required_fields_present",
            "passed": c9_passed,
            "details": c9_details,
            "severity": c9_sev,
        })

        # -------------------------------------------------------------
        # Synthesis & Scoring
        # -------------------------------------------------------------
        passed_count = sum(1 for c in checks if c["passed"])
        validation_score = round((passed_count / len(checks)) * 100.0, 1)

        if len(errors) > 0:
            status = "VALIDATION_FAILED"
            is_valid = False
        elif len(warnings) > 0:
            status = "NEEDS_REVIEW"
            is_valid = True
        else:
            status = "VALIDATED"
            is_valid = True

        output = ValidationResult(
            is_valid=is_valid,
            checks_performed=checks,
            errors=errors,
            warnings=warnings,
            validation_score=validation_score,
            recalculated_score=recalculated_score,
            score_matches=score_matches,
            status=status,
        )

        logger.info(
            "Audit completed with status=%s, validation_score=%.1f%%, %d errors, %d warnings",
            status,
            validation_score,
            len(errors),
            len(warnings),
        )
        return output.model_dump()
