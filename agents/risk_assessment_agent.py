"""Risk Assessment Agent for BankShield AI.

Evaluates transactional evidence against deterministic risk scoring rules,
determines priority tiers, and suggests compliance-aligned operational actions.
Adheres strictly to neutral investigative terminology (never labels transactions
as fraudulent; uses 'requires investigation' language).
"""

import logging
import random
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field

# Ensure reproducibility
random.seed(42)

logger = logging.getLogger(__name__)

# Try importing from config
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


class FallbackRiskEngine:
    """Internal fallback engine implementing the standard BankShield AI risk evaluation."""

    def __init__(self) -> None:
        """Initialize fallback rules with default weights."""
        self.weights = dict(RISK_WEIGHTS)
        self.rules_metadata = {
            "high_amount": {
                "name": "High Amount Deviation",
                "description": "Transaction amount deviates substantially from historical baseline average",
            },
            "unusual_frequency": {
                "name": "Rapid Velocity Anomaly",
                "description": "Burst in transaction velocity exceeds typical daily frequency",
            },
            "new_merchant": {
                "name": "Unrecognized Merchant",
                "description": "Transaction initiated at a merchant not observed in prior account history",
            },
            "failed_attempts": {
                "name": "Elevated Authentication Failures",
                "description": "Multiple consecutive failed authorization attempts preceded this transaction",
            },
            "unusual_time": {
                "name": "Off-Hours Activity",
                "description": "Transaction occurred outside typical diurnal operational window",
            },
        }

    def evaluate(self, indicators: Dict[str, Any]) -> Dict[str, Any]:
        """Evaluate indicators against rule weights and assign priority tier."""
        total_score = 0
        triggered_rules: List[Dict[str, Any]] = []
        untriggered_rules: List[Dict[str, Any]] = []
        missing_data_rules: List[Dict[str, Any]] = []

        for rule_key, weight in self.weights.items():
            meta = self.rules_metadata.get(
                rule_key, {"name": rule_key.replace("_", " ").title(), "description": rule_key}
            )
            rule_entry = {
                "rule_id": rule_key,
                "name": meta["name"],
                "weight": weight,
                "description": meta["description"],
            }

            if rule_key not in indicators or indicators[rule_key] is None:
                missing_data_rules.append(rule_entry)
            else:
                val = indicators[rule_key]
                is_detected = (bool(val.get("detected", False))) if isinstance(val, dict) else bool(val)
                if is_detected:
                    total_score += weight
                    triggered_rules.append(rule_entry)
                else:
                    untriggered_rules.append(rule_entry)

        # Map score to priority using settings or threshold intervals
        priority = "Low"
        for threshold, tier in PRIORITY_THRESHOLDS:
            if total_score >= threshold:
                priority = tier
                break

        return {
            "total_score": min(total_score, 100),
            "priority": priority,
            "triggered_rules": triggered_rules,
            "untriggered_rules": untriggered_rules,
            "missing_data_rules": missing_data_rules,
        }


class RiskAssessmentOutput(BaseModel):
    """Pydantic model representing structured risk assessment findings."""

    risk_score: int = Field(
        ..., ge=0, le=100, description="Normalized risk score between 0 and 100"
    )
    priority: str = Field(
        ..., description="Investigation priority: Low, Medium, High, or Critical"
    )
    triggered_rules: List[Dict[str, Any]] = Field(
        default_factory=list, description="Rules triggered with weights and explanations"
    )
    untriggered_rules: List[Dict[str, Any]] = Field(
        default_factory=list, description="Rules evaluated but not triggered"
    )
    missing_data_rules: List[Dict[str, Any]] = Field(
        default_factory=list, description="Rules that could not be evaluated due to missing data"
    )
    recommended_action: str = Field(
        ..., description="Standardized operational action recommendation"
    )
    assessment_summary: str = Field(
        ..., description="Investigative synthesis avoiding subjective labels"
    )
    status: str = Field(
        default="success", description="Assessment status: success or error"
    )


class RiskAssessmentAgent:
    """Agent responsible for deterministic risk scoring and action recommendations."""

    RECOMMENDED_ACTIONS = {
        "Critical": "Immediate manual review required. Escalate to senior analyst.",
        "High": "Priority review recommended. Flag for analyst attention.",
        "Medium": "Standard review. Monitor for additional suspicious activity.",
        "Low": "No immediate action required. Continue routine monitoring.",
    }

    def __init__(self, risk_engine: Optional[Any] = None) -> None:
        """Initialize RiskAssessmentAgent with an underlying RiskEngine.

        Args:
            risk_engine: Optional custom RiskEngine instance. If None, uses default engine.
        """
        if risk_engine is not None:
            self.risk_engine = risk_engine
        elif create_default_engine is not None:
            try:
                self.risk_engine = create_default_engine()
            except Exception as exc:
                logger.warning("Could not create default RiskEngine via factory: %s", exc)
                self.risk_engine = FallbackRiskEngine()
        else:
            self.risk_engine = FallbackRiskEngine()

        logger.info(
            "RiskAssessmentAgent initialized with risk_engine=%s",
            type(self.risk_engine).__name__,
        )

    def assess_risk(self, evidence: Dict[str, Any]) -> Dict[str, Any]:
        """Assess risk indicators extracted from evidence and compute risk score.

        Args:
            evidence: Evidence bundle containing indicators, deviation analysis, and context.

        Returns:
            Dict containing risk_score, priority tier, triggered_rules, untriggered_rules,
            missing_data_rules, recommended_action, assessment_summary, and status.
        """
        logger.info("Conducting risk assessment on provided evidence bundle")

        if not isinstance(evidence, dict):
            error_msg = "Invalid evidence provided: expected a dictionary."
            logger.error(error_msg)
            return {
                "risk_score": 0,
                "priority": "Low",
                "triggered_rules": [],
                "untriggered_rules": [],
                "missing_data_rules": [],
                "recommended_action": self.RECOMMENDED_ACTIONS["Low"],
                "assessment_summary": f"Assessment halted: {error_msg}",
                "status": "error",
            }

        # Extract indicators from evidence
        indicators: Dict[str, Any] = {}
        if "indicators" in evidence and isinstance(evidence["indicators"], dict):
            indicators = evidence["indicators"]
        else:
            # Check if indicators are flattened at root
            indicators = {
                k: v
                for k, v in evidence.items()
                if k in ("high_amount", "unusual_frequency", "new_merchant", "failed_attempts", "unusual_time")
            }

        try:
            # Evaluate using engine
            engine_output: Any = self.risk_engine.evaluate(indicators)

            # Normalize output structure from either Pydantic, dataclass, or dict
            if isinstance(engine_output, dict):
                raw_score = engine_output.get("total_score", engine_output.get("risk_score", 0))
                priority = str(engine_output.get("priority", "Low"))
                triggered = engine_output.get("triggered_rules", [])
                missing = engine_output.get("missing_data_rules", [])
                untriggered = engine_output.get("untriggered_rules", [])
            else:
                raw_score = getattr(engine_output, "total_score", getattr(engine_output, "risk_score", 0))
                priority = str(getattr(engine_output, "priority", "Low"))
                triggered = getattr(engine_output, "triggered_rules", [])
                missing = getattr(engine_output, "missing_data_rules", [])
                untriggered = getattr(engine_output, "untriggered_rules", [])

            # Standardize priority capitalization
            priority_normalized = priority.capitalize()
            if priority_normalized not in self.RECOMMENDED_ACTIONS:
                # Fallback to closest match or Low
                priority_normalized = "Low"

            risk_score = int(round(float(raw_score)))
            # Enforce 0-100 bounding
            risk_score = max(0, min(100, risk_score))

            # Derive untriggered rules if not populated by external engine
            if not untriggered and hasattr(self.risk_engine, "rules"):
                triggered_ids = {
                    r.get("rule_id", r.get("name", "")) if isinstance(r, dict) else str(r)
                    for r in triggered
                }
                missing_ids = {
                    r.get("rule_id", r.get("name", "")) if isinstance(r, dict) else str(r)
                    for r in missing
                }
                for r in getattr(self.risk_engine, "rules", []):
                    rid = getattr(r, "id", getattr(r, "name", str(r)))
                    if rid not in triggered_ids and rid not in missing_ids:
                        untriggered.append({"rule_id": rid, "name": getattr(r, "name", rid)})

            # Select standardized action recommendation
            recommended_action = self.RECOMMENDED_ACTIONS.get(
                priority_normalized, self.RECOMMENDED_ACTIONS["Low"]
            )

            # Build objective, compliance-friendly investigative summary
            # strictly avoiding subjective or premature fraud conclusions
            if triggered:
                rule_names = [
                    r.get("name", r.get("rule_id", str(r))) if isinstance(r, dict) else str(r)
                    for r in triggered
                ]
                rules_str = ", ".join(rule_names)
                assessment_summary = (
                    f"Risk assessment computed an aggregate risk score of {risk_score}/100, "
                    f"categorized as '{priority_normalized}' priority. {len(triggered)} risk indicator(s) "
                    f"were triggered: {rules_str}. The transaction displays anomalous behavior that requires "
                    f"investigation and supervisory review before clearance."
                )
            else:
                assessment_summary = (
                    f"Risk assessment computed an aggregate risk score of {risk_score}/100, "
                    f"categorized as '{priority_normalized}' priority. No anomalous behavioral indicators "
                    f"were triggered. The transaction conforms to expected parameters; routine monitoring maintained."
                )

            result = RiskAssessmentOutput(
                risk_score=risk_score,
                priority=priority_normalized,
                triggered_rules=triggered if isinstance(triggered, list) else [],
                untriggered_rules=untriggered if isinstance(untriggered, list) else [],
                missing_data_rules=missing if isinstance(missing, list) else [],
                recommended_action=recommended_action,
                assessment_summary=assessment_summary,
                status="success",
            )

            logger.info(
                "Risk assessment concluded: score=%d, priority=%s, triggered_count=%d",
                risk_score,
                priority_normalized,
                len(triggered),
            )
            return result.model_dump()

        except Exception as exc:
            logger.error("Error during risk assessment evaluation: %s", exc, exc_info=True)
            return {
                "risk_score": 0,
                "priority": "Low",
                "triggered_rules": [],
                "untriggered_rules": [],
                "missing_data_rules": [],
                "recommended_action": self.RECOMMENDED_ACTIONS["Low"],
                "assessment_summary": f"Assessment encountered an error during evaluation: {str(exc)}",
                "status": "error",
            }
