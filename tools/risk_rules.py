"""Deterministic Risk Scoring Engine for BankShield AI.

Provides rule-based evaluation of fraud risk indicators, calculating
transparent, explainable risk scores with evidence attribution,
score capping, priority tier mapping, and missing data discrimination.
"""

from dataclasses import dataclass, field
import logging
import random
from typing import Any, Callable, Dict, List, Optional, Tuple

from pydantic import BaseModel, Field

# Ensure reproducibility
random.seed(42)

logger = logging.getLogger(__name__)

# Import configuration settings with robust fallbacks
try:
    from config.settings import PRIORITY_THRESHOLDS, RISK_WEIGHTS
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


class RiskResult(BaseModel):
    """Pydantic model representing deterministic risk evaluation findings."""

    total_score: int = Field(
        ...,
        ge=0,
        le=100,
        description="Final normalized risk score capped at 100",
    )
    raw_score: int = Field(
        ...,
        description="Cumulative score before applying 100 capping",
    )
    priority: str = Field(
        ...,
        description="Priority classification tier: Critical, High, Medium, or Low",
    )
    triggered_rules: List[Dict[str, Any]] = Field(
        default_factory=list,
        description="Rules triggered with score contribution, explanation, and evidence",
    )
    untriggered_rules: List[Dict[str, Any]] = Field(
        default_factory=list,
        description="Rules evaluated but not triggered",
    )
    missing_data_rules: List[Dict[str, Any]] = Field(
        default_factory=list,
        description="Rules that could not be evaluated due to missing evidence data",
    )

    @property
    def risk_score(self) -> int:
        """Alias for total_score for cross-module compatibility."""
        return self.total_score

    def to_dict(self) -> Dict[str, Any]:
        """Convert result model to dictionary."""
        return self.model_dump()

    def __getitem__(self, item: str) -> Any:
        """Enable dictionary-style bracket access."""
        return getattr(self, item)


@dataclass
class RiskRule:
    """Definition of a deterministic risk scoring rule.

    Attributes:
        name: Unique rule identifier (matches indicator name, e.g. 'high_amount').
        weight: Score points added when this rule triggers.
        condition_fn: Optional callable (indicator_data, all_indicators) -> (is_triggered, is_missing, evidence).
        threshold_description: Human-readable description of the condition threshold.
        explanation_template: Format template for generating explanation of why rule triggered.
    """

    name: str
    weight: int
    condition_fn: Optional[Callable[..., Any]] = None
    threshold_description: str = ""
    explanation_template: str = ""


class RiskEngine:
    """Configurable deterministic risk scoring engine for transaction indicators."""

    def __init__(
        self,
        weights: Optional[Dict[str, int]] = None,
        thresholds: Optional[List[Tuple[int, str]]] = None,
    ) -> None:
        """Initialize RiskEngine with configurable weights and priority thresholds.

        Args:
            weights: Optional dictionary of rule names to integer point weights.
            thresholds: Optional list of (min_score, priority_tier) tuples sorted descending.
        """
        self.weights: Dict[str, int] = dict(weights or RISK_WEIGHTS)
        self.thresholds: List[Tuple[int, str]] = list(
            thresholds or PRIORITY_THRESHOLDS
        )
        # Ensure thresholds are sorted in descending order of score cutoff
        self.thresholds.sort(key=lambda t: t[0], reverse=True)
        self.rules: Dict[str, RiskRule] = {}

    def register_rule(self, rule: RiskRule) -> None:
        """Register a new risk scoring rule in the engine.

        If custom weights were provided during engine initialization, the rule's weight
        is updated to match the configured weight for its name.

        Args:
            rule: RiskRule instance to register.
        """
        if rule.name in self.weights:
            rule.weight = self.weights[rule.name]
        self.rules[rule.name] = rule
        logger.debug("Registered risk rule '%s' with weight %d", rule.name, rule.weight)

    def get_priority(self, score: int) -> str:
        """Map a numeric risk score (0-100) to its priority classification tier.

        Args:
            score: Integer risk score.

        Returns:
            Priority level string ('Critical', 'High', 'Medium', or 'Low').
        """
        for min_score, tier in self.thresholds:
            if score >= min_score:
                return tier
        return "Low"

    def evaluate(self, indicators: Dict[str, Any]) -> RiskResult:
        """Evaluate indicators against all registered rules and compute risk score.

        Strictly distinguishes between 'not triggered' and 'data missing'.
        Caps total score at 100 while tracking uncapped raw score.
        Produces human-readable explanations referencing actual evidence values.

        Args:
            indicators: Dictionary containing evaluated indicators.

        Returns:
            RiskResult with total_score, raw_score, priority, and rule breakdowns.
        """
        if not isinstance(indicators, dict):
            logger.warning("Invalid indicators object passed to evaluate: %s", type(indicators))
            indicators = {}

        # Handle nested indicators bundle if passed directly
        if "indicators" in indicators and isinstance(indicators["indicators"], dict):
            eval_map = indicators["indicators"]
        else:
            eval_map = indicators

        raw_score: int = 0
        triggered_rules: List[Dict[str, Any]] = []
        untriggered_rules: List[Dict[str, Any]] = []
        missing_data_rules: List[Dict[str, Any]] = []

        for rule_name, rule in self.rules.items():
            rule_entry_base = {
                "rule_name": rule.name,
                "rule_id": rule.name,
                "name": rule.name,
                "weight": rule.weight,
                "threshold_description": rule.threshold_description,
                "description": rule.threshold_description,
            }

            # 1. Check if indicator data is present in input
            if rule_name not in eval_map or eval_map[rule_name] is None:
                missing_data_rules.append(
                    {
                        **rule_entry_base,
                        "reason": f"Indicator '{rule_name}' missing from evidence bundle",
                    }
                )
                continue

            indicator_data = eval_map[rule_name]
            is_triggered: bool = False
            is_missing: bool = False
            evidence_text: str = ""

            # 2. Evaluate condition
            if rule.condition_fn is not None:
                try:
                    fn_res = rule.condition_fn(indicator_data, eval_map)
                    if isinstance(fn_res, tuple) and len(fn_res) == 3:
                        is_triggered, is_missing, evidence_text = fn_res
                    elif isinstance(fn_res, tuple) and len(fn_res) == 2:
                        is_triggered, is_missing = fn_res
                        evidence_text = str(getattr(indicator_data, "get", lambda k, d=None: d)("evidence", ""))
                    else:
                        is_triggered = bool(fn_res)
                except Exception as exc:
                    logger.error("Error evaluating condition_fn for rule '%s': %s", rule_name, exc)
                    is_missing = True
                    evidence_text = f"Evaluation error: {exc}"
            else:
                # Default evaluation logic inspecting dictionary or boolean
                if isinstance(indicator_data, dict):
                    is_missing = bool(indicator_data.get("missing_data", False))
                    is_triggered = bool(indicator_data.get("detected", False))
                    evidence_text = str(indicator_data.get("evidence", ""))
                elif isinstance(indicator_data, bool):
                    is_missing = False
                    is_triggered = indicator_data
                    evidence_text = f"Flag set to {indicator_data}"
                else:
                    is_missing = False
                    is_triggered = bool(indicator_data)
                    evidence_text = str(indicator_data)

            # 3. Categorize outcome
            if is_missing:
                missing_data_rules.append(
                    {
                        **rule_entry_base,
                        "reason": evidence_text or "Required evidence metrics unavailable",
                    }
                )
            elif is_triggered:
                raw_score += rule.weight

                # Build explanation
                if rule.explanation_template and evidence_text:
                    try:
                        explanation = rule.explanation_template.format(
                            rule_name=rule.name,
                            weight=rule.weight,
                            evidence=evidence_text,
                            description=rule.threshold_description,
                        )
                    except Exception:
                        explanation = f"{rule.threshold_description}: {evidence_text}"
                elif evidence_text:
                    explanation = f"{rule.threshold_description} (+{rule.weight} pts): {evidence_text}"
                else:
                    explanation = f"{rule.threshold_description} (+{rule.weight} pts)"

                triggered_rules.append(
                    {
                        **rule_entry_base,
                        "score_contribution": rule.weight,
                        "explanation": explanation,
                        "evidence_ref": evidence_text,
                    }
                )
            else:
                untriggered_rules.append(
                    {
                        **rule_entry_base,
                        "reason": "Condition threshold not met",
                    }
                )

        # Cap final total score at 100
        total_score = min(max(raw_score, 0), 100)
        priority = self.get_priority(total_score)

        return RiskResult(
            total_score=total_score,
            raw_score=raw_score,
            priority=priority,
            triggered_rules=triggered_rules,
            untriggered_rules=untriggered_rules,
            missing_data_rules=missing_data_rules,
        )


def create_default_engine(
    weights: Optional[Dict[str, int]] = None,
    thresholds: Optional[List[Tuple[int, str]]] = None,
) -> RiskEngine:
    """Create and configure a RiskEngine with the 5 standard BankShield AI rules.

    The 5 standard rules and default weights are:
    1. high_amount (30 pts)
    2. unusual_frequency (25 pts)
    3. new_merchant (20 pts)
    4. failed_attempts (15 pts)
    5. unusual_time (10 pts)

    Args:
        weights: Optional custom rule weight mapping.
        thresholds: Optional custom priority score tiers.

    Returns:
        Fully configured RiskEngine instance.
    """
    engine = RiskEngine(weights=weights, thresholds=thresholds)

    # 1. High Amount Rule (Weight: 30)
    engine.register_rule(
        RiskRule(
            name="high_amount",
            weight=engine.weights.get("high_amount", 30),
            threshold_description="Transaction amount significantly exceeds historical average (>= 3.0 std dev)",
            explanation_template="High Amount Anomaly (+{weight} pts): {evidence}",
        )
    )

    # 2. Unusual Frequency Rule (Weight: 25)
    engine.register_rule(
        RiskRule(
            name="unusual_frequency",
            weight=engine.weights.get("unusual_frequency", 25),
            threshold_description="Transaction frequency in recent 24h exceeds baseline velocity (>= 3.0x multiplier)",
            explanation_template="Velocity Burst (+{weight} pts): {evidence}",
        )
    )

    # 3. New Merchant Rule (Weight: 20)
    engine.register_rule(
        RiskRule(
            name="new_merchant",
            weight=engine.weights.get("new_merchant", 20),
            threshold_description="Transaction executed with previously unseen merchant",
            explanation_template="Unrecognized Merchant (+{weight} pts): {evidence}",
        )
    )

    # 4. Failed Attempts Rule (Weight: 15)
    engine.register_rule(
        RiskRule(
            name="failed_attempts",
            weight=engine.weights.get("failed_attempts", 15),
            threshold_description="Multiple consecutive failed authorization attempts preceded transaction (>= 3)",
            explanation_template="Authentication Failures (+{weight} pts): {evidence}",
        )
    )

    # 5. Unusual Time Rule (Weight: 10)
    engine.register_rule(
        RiskRule(
            name="unusual_time",
            weight=engine.weights.get("unusual_time", 10),
            threshold_description="Transaction initiated outside typical diurnal hours (00:00 - 05:00 nocturnal window)",
            explanation_template="Off-Hours Activity (+{weight} pts): {evidence}",
        )
    )

    return engine
