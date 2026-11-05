"""Transaction Research Agent for BankShield AI.

Gathers transactional evidence, account profiles, historical trends,
computes statistical deviations, and extracts key risk indicators.
"""

from datetime import datetime, timezone
import logging
import math
import random
import time
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field

# Ensure deterministic execution
random.seed(42)

logger = logging.getLogger(__name__)

# Try importing configuration thresholds
try:
    from config.settings import (
        AMOUNT_DEVIATION_THRESHOLD,
        FAILED_ATTEMPTS_THRESHOLD,
        FREQUENCY_MULTIPLIER_THRESHOLD,
        UNUSUAL_HOURS_END,
        UNUSUAL_HOURS_START,
        settings,
    )
except ImportError:
    AMOUNT_DEVIATION_THRESHOLD = 3.0
    FREQUENCY_MULTIPLIER_THRESHOLD = 3.0
    FAILED_ATTEMPTS_THRESHOLD = 3
    UNUSUAL_HOURS_START = 0
    UNUSUAL_HOURS_END = 5
    settings = None

# Try importing tools and services
try:
    from tools import transaction_tools
except ImportError:
    transaction_tools = None

try:
    from services.data_service import DataService
except ImportError:
    DataService = None


class ToolCallRecord(BaseModel):
    """Execution telemetry record for an agent tool call."""

    name: str = Field(..., description="Tool or function identifier called")
    input: Dict[str, Any] = Field(
        default_factory=dict, description="Input parameters passed"
    )
    output: Any = Field(
        None, description="Output or summarized result from tool call"
    )
    duration_ms: float = Field(..., description="Execution duration in milliseconds")
    status: str = Field(
        default="success", description="Status of tool execution: success, error"
    )


class ResearchEvidenceBundle(BaseModel):
    """Consolidated evidence package produced by the Research Agent."""

    transaction_details: Optional[Dict[str, Any]] = Field(
        default=None, description="Target transaction attributes"
    )
    account_profile: Optional[Dict[str, Any]] = Field(
        default=None, description="Associated account profile metadata"
    )
    account_history: List[Dict[str, Any]] = Field(
        default_factory=list, description="Historical transactions for account"
    )
    recent_transactions: List[Dict[str, Any]] = Field(
        default_factory=list, description="Transactions within recent 24-hour window"
    )
    deviation_analysis: Dict[str, Any] = Field(
        default_factory=dict, description="Numerical deviation metrics"
    )
    indicators: Dict[str, Any] = Field(
        default_factory=dict, description="Evaluated risk indicators"
    )
    evidence_summary: Dict[str, Any] = Field(
        default_factory=dict, description="Synthesized evidence overview"
    )
    tool_calls: List[Dict[str, Any]] = Field(
        default_factory=list, description="Audit trace of tools executed"
    )
    missing_data_warnings: List[str] = Field(
        default_factory=list, description="Data gaps or missing records flagged"
    )
    status: str = Field(
        default="success", description="Overall research status: success, partial, error"
    )


class ResearchAgent:
    """Agent responsible for conducting data retrieval and behavioral anomaly research."""

    def __init__(self, data_service: Optional[Any] = None) -> None:
        """Initialize ResearchAgent with DataService.

        Args:
            data_service: DataService instance for data access. If None, instantiates default.
        """
        if data_service is not None:
            self.data_service = data_service
        elif DataService is not None:
            try:
                self.data_service = DataService()
            except Exception as exc:
                logger.warning("Could not instantiate DataService: %s", exc)
                self.data_service = None
        else:
            self.data_service = None

        logger.info(
            "ResearchAgent initialized with data_service=%s",
            type(self.data_service).__name__ if self.data_service else "None",
        )

    def _execute_tool_with_telemetry(
        self,
        tool_name: str,
        input_args: Dict[str, Any],
        func: Any,
        tool_calls: List[Dict[str, Any]],
        *args: Any,
        **kwargs: Any,
    ) -> Any:
        """Execute a tool function while recording structured telemetry."""
        start_time = time.perf_counter()
        try:
            result = func(*args, **kwargs)
            duration_ms = round((time.perf_counter() - start_time) * 1000, 2)

            # Summarize output for trace log
            output_summary: Any
            if isinstance(result, list):
                output_summary = f"List with {len(result)} items"
            elif isinstance(result, dict):
                output_summary = {
                    k: (f"{len(v)} items" if isinstance(v, list) else v)
                    for k, v in list(result.items())[:5]
                }
            else:
                output_summary = str(result)[:100]

            tool_calls.append(
                {
                    "name": tool_name,
                    "input": input_args,
                    "output": output_summary,
                    "duration_ms": duration_ms,
                    "status": "success",
                }
            )
            return result
        except Exception as exc:
            duration_ms = round((time.perf_counter() - start_time) * 1000, 2)
            logger.error("Error executing tool %s: %s", tool_name, exc)
            tool_calls.append(
                {
                    "name": tool_name,
                    "input": input_args,
                    "output": str(exc),
                    "duration_ms": duration_ms,
                    "status": "error",
                }
            )
            return None

    def _fallback_calculate_deviation(
        self,
        transaction: Dict[str, Any],
        history: List[Dict[str, Any]],
    ) -> Dict[str, Any]:
        """Calculate amount deviation using standard statistical methods."""
        tx_amount = float(transaction.get("amount", 0.0))
        if not history:
            return {
                "historical_count": 0,
                "historical_mean": 0.0,
                "historical_std": 0.0,
                "target_amount": tx_amount,
                "z_score": 0.0,
                "deviation_ratio": 1.0,
                "is_deviating": False,
                "summary": "No historical transactions found for baseline deviation analysis.",
            }

        amounts = [float(h.get("amount", 0.0)) for h in history if "amount" in h]
        if not amounts:
            return {
                "historical_count": 0,
                "historical_mean": 0.0,
                "historical_std": 0.0,
                "target_amount": tx_amount,
                "z_score": 0.0,
                "deviation_ratio": 1.0,
                "is_deviating": False,
                "summary": "Historical transactions contain no amount values.",
            }

        mean = sum(amounts) / len(amounts)
        variance = sum((x - mean) ** 2 for x in amounts) / len(amounts)
        std = math.sqrt(variance)

        z_score = round((tx_amount - mean) / (std if std > 1e-4 else 1.0), 2)
        ratio = round(tx_amount / (mean if mean > 1e-4 else 1.0), 2)
        is_deviating = bool(z_score >= AMOUNT_DEVIATION_THRESHOLD or ratio >= 5.0)

        summary = (
            f"Amount ${tx_amount:,.2f} is {ratio:.1f}x the historical mean "
            f"(${mean:,.2f}) with z-score {z_score:.2f}."
        )

        return {
            "historical_count": len(amounts),
            "historical_mean": round(mean, 2),
            "historical_std": round(std, 2),
            "target_amount": tx_amount,
            "z_score": z_score,
            "deviation_ratio": ratio,
            "is_deviating": is_deviating,
            "summary": summary,
        }

    def _fallback_identify_indicators(
        self,
        transaction: Dict[str, Any],
        history: List[Dict[str, Any]],
        account_profile: Optional[Dict[str, Any]],
    ) -> Dict[str, Any]:
        """Identify standard behavioral risk indicators from transactional context."""
        tx_amount = float(transaction.get("amount", 0.0))
        failed_attempts = int(transaction.get("failed_attempts", 0))

        # Check hour
        ts_str = str(transaction.get("timestamp", ""))
        hour = -1
        try:
            # Handle ISO format or standard space separator
            dt = datetime.fromisoformat(ts_str.replace(" ", "T"))
            hour = dt.hour
        except Exception:
            hour = -1

        unusual_time = bool(
            UNUSUAL_HOURS_START <= hour <= UNUSUAL_HOURS_END
        ) if hour >= 0 else False

        # Check new merchant
        merchant = transaction.get("merchant")
        hist_merchants = {h.get("merchant") for h in history if h.get("merchant")}
        new_merchant = bool(
            transaction.get("is_new_merchant", False)
            or (merchant is not None and merchant not in hist_merchants)
        )

        # Check amount deviation
        deviation = self._fallback_calculate_deviation(transaction, history)
        high_amount = deviation.get("is_deviating", False)

        # Check frequency
        hist_count = len(history)
        avg_daily_freq = float(
            account_profile.get("historical_transaction_frequency", 1.0)
        ) if account_profile else 1.0
        # If velocity multiplier is exceeded
        unusual_frequency = bool(hist_count > 0 and (hist_count / max(avg_daily_freq, 0.1)) >= FREQUENCY_MULTIPLIER_THRESHOLD)

        # Check failed attempts
        failed_attempts_flag = bool(failed_attempts >= FAILED_ATTEMPTS_THRESHOLD)

        return {
            "high_amount": high_amount,
            "unusual_frequency": unusual_frequency,
            "new_merchant": new_merchant,
            "failed_attempts": failed_attempts_flag,
            "unusual_time": unusual_time,
        }

    def investigate_transaction(self, transaction_id: str) -> Dict[str, Any]:
        """Conduct comprehensive research and compile evidence for a transaction.

        Args:
            transaction_id: Transaction identifier to investigate.

        Returns:
            Dict containing detailed evidence bundle, indicators, tool call traces,
            missing data warnings, and research status.
        """
        logger.info("Conducting transaction research for transaction_id='%s'", transaction_id)
        tool_calls: List[Dict[str, Any]] = []
        missing_data_warnings: List[str] = []

        if not self.data_service:
            missing_data_warnings.append("DataService is unavailable. Research cannot query backend.")
            bundle = ResearchEvidenceBundle(
                transaction_details=None,
                account_profile=None,
                account_history=[],
                recent_transactions=[],
                deviation_analysis={},
                indicators={},
                evidence_summary={"error": "DataService unavailable"},
                tool_calls=[],
                missing_data_warnings=missing_data_warnings,
                status="error",
            )
            return bundle.model_dump()

        # Step 1: Retrieve Target Transaction
        tx_raw: Any = None
        if transaction_tools and hasattr(transaction_tools, "get_transaction"):
            tx_raw = self._execute_tool_with_telemetry(
                "get_transaction",
                {"transaction_id": transaction_id},
                transaction_tools.get_transaction,
                tool_calls,
                self.data_service,
                transaction_id,
            )
            if isinstance(tx_raw, dict) and "data" in tx_raw:
                transaction_details = tx_raw.get("data")
            elif isinstance(tx_raw, dict):
                transaction_details = tx_raw
            else:
                transaction_details = None
        else:
            transaction_details = self._execute_tool_with_telemetry(
                "get_transaction",
                {"transaction_id": transaction_id},
                self.data_service.get_transaction,
                tool_calls,
                transaction_id,
            )

        if not transaction_details:
            msg = f"Transaction ID '{transaction_id}' was not found in records."
            missing_data_warnings.append(msg)
            logger.warning(msg)
            bundle = ResearchEvidenceBundle(
                transaction_details=None,
                account_profile=None,
                account_history=[],
                recent_transactions=[],
                deviation_analysis={},
                indicators={},
                evidence_summary={"error": msg},
                tool_calls=tool_calls,
                missing_data_warnings=missing_data_warnings,
                status="error",
            )
            return bundle.model_dump()

        account_id = transaction_details.get("account_id")
        account_profile: Optional[Dict[str, Any]] = None
        account_history: List[Dict[str, Any]] = []
        recent_transactions: List[Dict[str, Any]] = []

        # Step 2: Retrieve Account Context if account_id is present
        if not account_id:
            missing_data_warnings.append("Transaction record lacks an account_id field.")
        else:
            # 2a. Account Profile
            account_profile = self._execute_tool_with_telemetry(
                "get_account_profile",
                {"account_id": account_id},
                self.data_service.get_account_profile,
                tool_calls,
                account_id,
            )
            if not account_profile:
                missing_data_warnings.append(
                    f"Account profile for account '{account_id}' was not found."
                )

            # 2b. Account History
            if transaction_tools and hasattr(transaction_tools, "get_account_history"):
                hist_raw = self._execute_tool_with_telemetry(
                    "get_account_history",
                    {"account_id": account_id},
                    transaction_tools.get_account_history,
                    tool_calls,
                    self.data_service,
                    account_id,
                )
                if isinstance(hist_raw, dict) and "history" in hist_raw:
                    account_history = hist_raw.get("history", [])
                elif isinstance(hist_raw, list):
                    account_history = hist_raw
                else:
                    account_history = []
            else:
                hist_raw = self._execute_tool_with_telemetry(
                    "get_account_history",
                    {"account_id": account_id},
                    self.data_service.get_account_history,
                    tool_calls,
                    account_id,
                )
                account_history = hist_raw if isinstance(hist_raw, list) else []

            if not account_history:
                missing_data_warnings.append(
                    f"No historical transactions found for account '{account_id}'."
                )

            # 2c. Recent Transactions (24 hours)
            if transaction_tools and hasattr(transaction_tools, "get_recent_transactions"):
                rec_raw = self._execute_tool_with_telemetry(
                    "get_recent_transactions",
                    {"account_id": account_id, "hours": 24},
                    transaction_tools.get_recent_transactions,
                    tool_calls,
                    self.data_service,
                    account_id,
                    24,
                )
                if isinstance(rec_raw, dict) and "recent_transactions" in rec_raw:
                    recent_transactions = rec_raw.get("recent_transactions", [])
                elif isinstance(rec_raw, dict) and "data" in rec_raw:
                    recent_transactions = rec_raw.get("data", [])
                elif isinstance(rec_raw, list):
                    recent_transactions = rec_raw
                else:
                    recent_transactions = []
            else:
                rec_raw = self._execute_tool_with_telemetry(
                    "get_recent_transactions",
                    {"account_id": account_id, "hours": 24},
                    self.data_service.get_recent_transactions,
                    tool_calls,
                    account_id,
                    hours=24,
                )
                recent_transactions = rec_raw if isinstance(rec_raw, list) else []

        # Step 3: Compute Behavioral Deviations
        deviation_analysis: Dict[str, Any]
        if transaction_tools and hasattr(transaction_tools, "calculate_transaction_deviation"):
            deviation_analysis = self._execute_tool_with_telemetry(
                "calculate_transaction_deviation",
                {"transaction_id": transaction_id, "history_len": len(account_history)},
                transaction_tools.calculate_transaction_deviation,
                tool_calls,
                transaction_details,
                account_history,
            ) or {}
        else:
            deviation_analysis = self._fallback_calculate_deviation(
                transaction_details, account_history
            )

        # Step 4: Identify Risk Indicators
        indicators: Dict[str, Any]
        if transaction_tools and hasattr(transaction_tools, "identify_transaction_indicators"):
            raw_ind = self._execute_tool_with_telemetry(
                "identify_transaction_indicators",
                {"transaction_id": transaction_id},
                transaction_tools.identify_transaction_indicators,
                tool_calls,
                transaction_details,
                account_history,
                account_profile,
            ) or {}
            # If tool returns indicators nested under 'indicators' key
            if isinstance(raw_ind, dict) and "indicators" in raw_ind and isinstance(raw_ind["indicators"], dict):
                ind_map = raw_ind["indicators"]
            elif isinstance(raw_ind, dict):
                ind_map = raw_ind
            else:
                ind_map = {}

            indicators = {}
            for k, v in ind_map.items():
                if isinstance(v, dict):
                    indicators[k] = bool(v.get("detected", False))
                else:
                    indicators[k] = bool(v)
        else:
            indicators = self._fallback_identify_indicators(
                transaction_details, account_history, account_profile
            )

        # Step 5: Synthesize Evidence Summary
        active_indicators = [
            k for k, v in indicators.items() if bool(v)
        ] if isinstance(indicators, dict) else []

        evidence_summary = {
            "transaction_id": transaction_id,
            "account_id": account_id,
            "amount": transaction_details.get("amount"),
            "timestamp": transaction_details.get("timestamp"),
            "merchant": transaction_details.get("merchant"),
            "channel": transaction_details.get("channel"),
            "failed_attempts": transaction_details.get("failed_attempts", 0),
            "historical_records_retrieved": len(account_history),
            "recent_24h_records_retrieved": len(recent_transactions),
            "active_indicators_count": len(active_indicators),
            "active_indicators": active_indicators,
            "deviation_z_score": deviation_analysis.get("z_score"),
            "deviation_ratio": deviation_analysis.get("deviation_ratio"),
            "has_profile": account_profile is not None,
            "missing_data_count": len(missing_data_warnings),
        }

        # Determine overall status
        status = "partial" if missing_data_warnings else "success"

        bundle = ResearchEvidenceBundle(
            transaction_details=transaction_details,
            account_profile=account_profile,
            account_history=account_history,
            recent_transactions=recent_transactions,
            deviation_analysis=deviation_analysis,
            indicators=indicators,
            evidence_summary=evidence_summary,
            tool_calls=tool_calls,
            missing_data_warnings=missing_data_warnings,
            status=status,
        )

        logger.info(
            "Research completed for %s with status=%s, %d indicators flagged, %d warnings",
            transaction_id,
            status,
            len(active_indicators),
            len(missing_data_warnings),
        )
        return bundle.model_dump()
