"""Transaction Research Tools for BankShield AI.

Provides standalone deterministic analytical tools used by the Research Agent
to inspect transactions, retrieve account history, compute deviations, and
extract behavioral risk indicators.
"""

from datetime import datetime, timedelta, timezone
import logging
import math
import random
from typing import Any, Dict, List, Optional

# Ensure reproducibility
random.seed(42)

logger = logging.getLogger(__name__)

# Import configuration thresholds with robust fallbacks
try:
    from config.settings import (
        AMOUNT_DEVIATION_THRESHOLD,
        FAILED_ATTEMPTS_THRESHOLD,
        FREQUENCY_MULTIPLIER_THRESHOLD,
        UNUSUAL_HOURS_END,
        UNUSUAL_HOURS_START,
    )
except ImportError:
    AMOUNT_DEVIATION_THRESHOLD = 3.0
    FREQUENCY_MULTIPLIER_THRESHOLD = 3.0
    FAILED_ATTEMPTS_THRESHOLD = 3
    UNUSUAL_HOURS_START = 0
    UNUSUAL_HOURS_END = 5


class IndicatorDict(dict):
    """Custom dictionary for risk indicators that evaluates truthiness via 'detected'."""

    def __bool__(self) -> bool:
        return bool(self.get("detected", False))


def get_transaction(data_service: Any, transaction_id: str) -> Dict[str, Any]:
    """Retrieve details for a single transaction by ID.

    Args:
        data_service: Instance of DataService providing dataset access.
        transaction_id: Unique transaction identifier string.

    Returns:
        Structured dict with 'status': 'success'|'error' and transaction data.
    """
    if data_service is None:
        return {
            "status": "error",
            "error": "DataService instance is required",
            "data": None,
            "transaction": None,
        }

    if not isinstance(transaction_id, str) or not transaction_id.strip():
        return {
            "status": "error",
            "error": "Invalid transaction_id: must be a non-empty string",
            "data": None,
            "transaction": None,
        }

    try:
        txn = data_service.get_transaction(transaction_id.strip())
        if txn is None:
            return {
                "status": "error",
                "error": f"Transaction '{transaction_id}' not found",
                "data": None,
                "transaction": None,
            }
        return {
            "status": "success",
            "data": txn,
            "transaction": txn,
            "transaction_id": transaction_id,
        }
    except Exception as exc:
        logger.error("Error retrieving transaction %s: %s", transaction_id, exc)
        return {
            "status": "error",
            "error": str(exc),
            "data": None,
            "transaction": None,
        }


def get_account_history(data_service: Any, account_id: str) -> Dict[str, Any]:
    """Retrieve historical transaction list and summary metrics for an account.

    Args:
        data_service: Instance of DataService.
        account_id: Unique account identifier string.

    Returns:
        Structured dict with 'status': 'success'|'error', history list, and stats.
    """
    if data_service is None:
        return {
            "status": "error",
            "error": "DataService instance is required",
            "history": [],
            "data": [],
            "stats": {},
        }

    if not isinstance(account_id, str) or not account_id.strip():
        return {
            "status": "error",
            "error": "Invalid account_id: must be a non-empty string",
            "history": [],
            "data": [],
            "stats": {},
        }

    try:
        history = data_service.get_account_history(account_id.strip())
        amounts = [
            float(t["amount"])
            for t in history
            if "amount" in t and t["amount"] is not None
        ]

        if amounts:
            mean_amt = sum(amounts) / len(amounts)
            var_amt = (
                sum((x - mean_amt) ** 2 for x in amounts) / (len(amounts) - 1)
                if len(amounts) > 1
                else 0.0
            )
            std_amt = math.sqrt(var_amt)
            min_amt = min(amounts)
            max_amt = max(amounts)
        else:
            mean_amt = 0.0
            std_amt = 0.0
            min_amt = 0.0
            max_amt = 0.0

        merchants = sorted(
            list(
                {
                    str(t["merchant"])
                    for t in history
                    if "merchant" in t and t["merchant"] is not None
                }
            )
        )
        locations = sorted(
            list(
                {
                    str(t["location"])
                    for t in history
                    if "location" in t and t["location"] is not None
                }
            )
        )

        stats = {
            "total_transactions": len(history),
            "total_amount": round(sum(amounts), 2),
            "mean_amount": round(mean_amt, 2),
            "std_amount": round(std_amt, 2),
            "min_amount": round(min_amt, 2),
            "max_amount": round(max_amt, 2),
            "unique_merchants_count": len(merchants),
            "unique_locations_count": len(locations),
            "known_merchants": merchants,
            "known_locations": locations,
        }

        return {
            "status": "success",
            "account_id": account_id,
            "history": history,
            "data": history,
            "stats": stats,
        }
    except Exception as exc:
        logger.error("Error retrieving history for account %s: %s", account_id, exc)
        return {
            "status": "error",
            "error": str(exc),
            "history": [],
            "data": [],
            "stats": {},
        }


def get_recent_transactions(
    data_service: Any,
    account_id: str,
    hours: int = 24,
) -> Dict[str, Any]:
    """Retrieve transactions within a recent time window for an account.

    Args:
        data_service: Instance of DataService.
        account_id: Unique account identifier string.
        hours: Lookback window in hours (default: 24).

    Returns:
        Structured dict with 'status': 'success'|'error', recent txns, and count.
    """
    if data_service is None:
        return {
            "status": "error",
            "error": "DataService instance is required",
            "recent_transactions": [],
            "data": [],
            "count": 0,
        }

    if not isinstance(account_id, str) or not account_id.strip():
        return {
            "status": "error",
            "error": "Invalid account_id: must be a non-empty string",
            "recent_transactions": [],
            "data": [],
            "count": 0,
        }

    try:
        recent = data_service.get_recent_transactions(account_id.strip(), hours=hours)
        return {
            "status": "success",
            "account_id": account_id,
            "hours": hours,
            "recent_transactions": recent,
            "data": recent,
            "count": len(recent),
        }
    except Exception as exc:
        logger.error(
            "Error retrieving recent transactions for account %s: %s",
            account_id,
            exc,
        )
        return {
            "status": "error",
            "error": str(exc),
            "recent_transactions": [],
            "data": [],
            "count": 0,
        }


def calculate_transaction_deviation(
    transaction: Dict[str, Any],
    history: List[Dict[str, Any]],
    threshold: float = AMOUNT_DEVIATION_THRESHOLD,
) -> Dict[str, Any]:
    """Calculate statistical amount deviation (z-score and ratio) against account history.

    Args:
        transaction: Target transaction dictionary.
        history: Historical transactions list for baseline calculation.
        threshold: Z-score cutoff threshold for flagging anomaly (default: 3.0).

    Returns:
        Structured dict with mean_amount, std_amount, transaction_amount,
        deviation_score (z-score), is_unusual (bool), and statistical summary.
    """
    if not isinstance(transaction, dict):
        return {
            "status": "error",
            "error": "Target transaction must be a dictionary",
            "mean_amount": 0.0,
            "std_amount": 0.0,
            "transaction_amount": 0.0,
            "deviation_score": 0.0,
            "z_score": 0.0,
            "deviation_ratio": 1.0,
            "is_unusual": False,
            "is_deviating": False,
            "threshold": threshold,
            "historical_count": 0,
            "summary": "Invalid transaction dictionary provided.",
        }

    try:
        tx_amount = float(transaction.get("amount", 0.0))
    except (ValueError, TypeError):
        tx_amount = 0.0

    target_id = transaction.get("transaction_id")
    # Exclude target transaction itself from historical baseline if present
    filtered_history = [
        h
        for h in history
        if not (
            target_id
            and isinstance(h, dict)
            and h.get("transaction_id") == target_id
        )
    ]
    eval_history = filtered_history if filtered_history else history

    amounts = [
        float(h["amount"])
        for h in eval_history
        if isinstance(h, dict) and "amount" in h and h["amount"] is not None
    ]


    if not amounts:
        summary = (
            f"No historical transactions found for baseline deviation analysis. "
            f"Target amount is ${tx_amount:,.2f}."
        )
        return {
            "status": "success",
            "mean_amount": 0.0,
            "std_amount": 0.0,
            "transaction_amount": round(tx_amount, 2),
            "deviation_score": 0.0,
            "z_score": 0.0,
            "deviation_ratio": 1.0,
            "is_unusual": False,
            "is_deviating": False,
            "threshold": threshold,
            "historical_count": 0,
            "summary": summary,
        }

    mean_amt = sum(amounts) / len(amounts)
    if len(amounts) > 1:
        var_amt = sum((x - mean_amt) ** 2 for x in amounts) / (len(amounts) - 1)
        std_amt = math.sqrt(var_amt)
    else:
        std_amt = 0.0

    # Calculate z-score
    if std_amt > 1e-4:
        z_score = round((tx_amount - mean_amt) / std_amt, 2)
    else:
        z_score = 0.0 if abs(tx_amount - mean_amt) < 1e-4 else 999.0

    # Calculate ratio relative to baseline
    denom = mean_amt if mean_amt > 1e-4 else 1.0
    ratio = round(tx_amount / denom, 2)

    # Flag as unusual if z-score >= threshold or extreme multiplier >= 5.0
    is_unusual = bool(z_score >= threshold or ratio >= 5.0)

    summary = (
        f"Transaction amount ${tx_amount:,.2f} is {ratio:.1f}x historical mean "
        f"(${mean_amt:,.2f}) with z-score {z_score:.2f} (std: ${std_amt:,.2f})."
    )

    return {
        "status": "success",
        "mean_amount": round(mean_amt, 2),
        "std_amount": round(std_amt, 2),
        "transaction_amount": round(tx_amount, 2),
        "deviation_score": z_score,
        "z_score": z_score,
        "deviation_ratio": ratio,
        "is_unusual": is_unusual,
        "is_deviating": is_unusual,
        "threshold": threshold,
        "historical_count": len(amounts),
        "summary": summary,
    }


def identify_transaction_indicators(
    transaction: Dict[str, Any],
    history: List[Dict[str, Any]],
    account_profile: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Identify and evaluate behavioral risk indicators against transaction context.

    Evaluates 6 standard indicators:
    - high_amount: Unusual transaction size relative to historical distribution.
    - unusual_frequency: Rapid velocity burst exceeding typical daily transaction volume.
    - new_merchant: Transaction initiated with a merchant never seen in history.
    - new_location: Transaction initiated in a location never seen in history.
    - failed_attempts: Multiple failed authorization attempts preceding transaction.
    - unusual_time: Transaction occurring during nocturnal off-hours.

    Only flags indicators where data exists; marks missing data explicitly.

    Args:
        transaction: Target transaction dictionary.
        history: Historical transactions list for account.
        account_profile: Associated account profile metadata (optional).

    Returns:
        Structured dict containing evaluated indicators with details and evidence.
    """
    if not isinstance(transaction, dict):
        return {
            "status": "error",
            "error": "Target transaction must be a dictionary",
            "indicators": {},
        }

    account_profile = account_profile or {}
    indicators: Dict[str, IndicatorDict] = {}

    # ---------------------------------------------------------
    # 1. High Amount Indicator
    # ---------------------------------------------------------
    raw_amount = transaction.get("amount")
    if raw_amount is None:
        indicators["high_amount"] = IndicatorDict(
            {
                "name": "high_amount",
                "detected": False,
                "value": None,
                "threshold": AMOUNT_DEVIATION_THRESHOLD,
                "description": "Transaction amount significantly higher than historical average",
                "evidence": "Amount value is missing from transaction record",
                "missing_data": True,
            }
        )
    else:
        try:
            tx_amount = float(raw_amount)
        except (ValueError, TypeError):
            tx_amount = 0.0

        deviation = calculate_transaction_deviation(transaction, history)
        z_score = deviation.get("z_score", 0.0)
        ratio = deviation.get("deviation_ratio", 1.0)
        mean_amt = deviation.get("mean_amount", 0.0)
        std_amt = deviation.get("std_amount", 0.0)

        # Check account profile fallback if history is small
        if len(history) < 2 and "historical_average_amount" in account_profile:
            try:
                prof_avg = float(account_profile["historical_average_amount"])
                if prof_avg > 0:
                    ratio = round(tx_amount / prof_avg, 2)
                    mean_amt = prof_avg
            except (ValueError, TypeError):
                pass

        is_high = bool(z_score >= AMOUNT_DEVIATION_THRESHOLD or ratio >= 5.0)
        indicators["high_amount"] = IndicatorDict(
            {
                "name": "high_amount",
                "detected": is_high,
                "value": z_score if z_score != 0.0 else ratio,
                "threshold": AMOUNT_DEVIATION_THRESHOLD,
                "description": "Transaction amount significantly higher than historical baseline",
                "evidence": (
                    f"Amount ${tx_amount:,.2f} is {ratio:.1f}x baseline mean (${mean_amt:,.2f}) "
                    f"with z-score {z_score:.2f} (std: ${std_amt:,.2f})"
                ),
                "missing_data": False,
            }
        )

    # ---------------------------------------------------------
    # 2. Unusual Frequency Indicator
    # ---------------------------------------------------------
    baseline_freq: Optional[float] = None
    if "historical_transaction_frequency" in account_profile and account_profile["historical_transaction_frequency"] is not None:
        try:
            baseline_freq = float(account_profile["historical_transaction_frequency"])
        except (ValueError, TypeError):
            baseline_freq = None

    # Count recent transactions in last 24h
    recent_count: int = 0
    ts_str = str(transaction.get("timestamp", "")).strip()
    txn_dt: Optional[datetime] = None
    if ts_str:
        try:
            txn_dt = datetime.fromisoformat(ts_str.replace(" ", "T"))
        except Exception:
            txn_dt = None

    if txn_dt and history:
        cutoff = txn_dt - timedelta(hours=24)
        for h in history:
            h_ts = str(h.get("timestamp", "")).strip()
            if not h_ts:
                continue
            try:
                h_dt = datetime.fromisoformat(h_ts.replace(" ", "T"))
                if cutoff <= h_dt <= txn_dt:
                    recent_count += 1
            except Exception:
                continue
    elif history:
        # Approximate: total history if very short
        recent_count = len(history)

    if baseline_freq is None and not history:
        indicators["unusual_frequency"] = IndicatorDict(
            {
                "name": "unusual_frequency",
                "detected": False,
                "value": None,
                "threshold": FREQUENCY_MULTIPLIER_THRESHOLD,
                "description": "Unusual burst in transaction velocity",
                "evidence": "Insufficient historical velocity data to compute frequency multiplier",
                "missing_data": True,
            }
        )
    else:
        effective_baseline = max(baseline_freq or 1.0, 0.1)
        velocity_multiplier = round(recent_count / effective_baseline, 2)
        is_frequent = bool(
            recent_count > 0 and velocity_multiplier >= FREQUENCY_MULTIPLIER_THRESHOLD
        )

        indicators["unusual_frequency"] = IndicatorDict(
            {
                "name": "unusual_frequency",
                "detected": is_frequent,
                "value": velocity_multiplier,
                "threshold": FREQUENCY_MULTIPLIER_THRESHOLD,
                "description": "Unusual burst in transaction velocity",
                "evidence": (
                    f"Observed {recent_count} transactions in 24h window vs baseline "
                    f"{effective_baseline:.2f}/day (multiplier: {velocity_multiplier:.2f}x, threshold: {FREQUENCY_MULTIPLIER_THRESHOLD}x)"
                ),
                "missing_data": False,
            }
        )

    # ---------------------------------------------------------
    # 3. New Merchant Indicator
    # ---------------------------------------------------------
    merchant = transaction.get("merchant")
    if merchant is None or not str(merchant).strip():
        indicators["new_merchant"] = IndicatorDict(
            {
                "name": "new_merchant",
                "detected": False,
                "value": None,
                "threshold": "Previously observed merchant",
                "description": "Transaction initiated at a merchant not seen in prior account history",
                "evidence": "Merchant data missing from transaction record",
                "missing_data": True,
            }
        )
    else:
        merchant_clean = str(merchant).strip()
        hist_merchants = {
            str(h.get("merchant", "")).strip()
            for h in history
            if h.get("merchant")
        }
        is_explicitly_new = bool(transaction.get("is_new_merchant", False))
        is_new = bool(is_explicitly_new or (hist_merchants and merchant_clean not in hist_merchants))

        indicators["new_merchant"] = IndicatorDict(
            {
                "name": "new_merchant",
                "detected": is_new,
                "value": merchant_clean,
                "threshold": "Previously observed merchant",
                "description": "Transaction initiated at a merchant not seen in prior account history",
                "evidence": (
                    f"Merchant '{merchant_clean}' has not been seen in account history "
                    f"({len(hist_merchants)} known merchants)"
                    if is_new
                    else f"Merchant '{merchant_clean}' matches known merchant history"
                ),
                "missing_data": False,
            }
        )

    # ---------------------------------------------------------
    # 4. New Location Indicator
    # ---------------------------------------------------------
    location = transaction.get("location")
    if location is None or not str(location).strip():
        indicators["new_location"] = IndicatorDict(
            {
                "name": "new_location",
                "detected": False,
                "value": None,
                "threshold": "Previously observed location",
                "description": "Transaction location not seen in prior account history",
                "evidence": "Location data missing from transaction record",
                "missing_data": True,
            }
        )
    else:
        location_clean = str(location).strip()
        hist_locations = {
            str(h.get("location", "")).strip()
            for h in history
            if h.get("location")
        }
        is_explicitly_new = bool(transaction.get("is_new_location", False))
        is_new = bool(is_explicitly_new or (hist_locations and location_clean not in hist_locations))

        indicators["new_location"] = IndicatorDict(
            {
                "name": "new_location",
                "detected": is_new,
                "value": location_clean,
                "threshold": "Previously observed location",
                "description": "Transaction location not seen in prior account history",
                "evidence": (
                    f"Location '{location_clean}' has not been seen in account history "
                    f"({len(hist_locations)} known locations)"
                    if is_new
                    else f"Location '{location_clean}' matches known account activity locations"
                ),
                "missing_data": False,
            }
        )

    # ---------------------------------------------------------
    # 5. Failed Attempts Indicator
    # ---------------------------------------------------------
    raw_failed = transaction.get("failed_attempts")
    if raw_failed is None:
        indicators["failed_attempts"] = IndicatorDict(
            {
                "name": "failed_attempts",
                "detected": False,
                "value": None,
                "threshold": FAILED_ATTEMPTS_THRESHOLD,
                "description": "Multiple failed authorization attempts prior to transaction",
                "evidence": "Failed authorization attempts data missing from transaction record",
                "missing_data": True,
            }
        )
    else:
        try:
            failed_count = int(raw_failed)
        except (ValueError, TypeError):
            failed_count = 0

        is_failed_flagged = bool(failed_count >= FAILED_ATTEMPTS_THRESHOLD)
        indicators["failed_attempts"] = IndicatorDict(
            {
                "name": "failed_attempts",
                "detected": is_failed_flagged,
                "value": failed_count,
                "threshold": FAILED_ATTEMPTS_THRESHOLD,
                "description": "Multiple failed authorization attempts prior to transaction",
                "evidence": (
                    f"{failed_count} consecutive failed attempts detected prior to transaction "
                    f"(threshold: {FAILED_ATTEMPTS_THRESHOLD})"
                ),
                "missing_data": False,
            }
        )

    # ---------------------------------------------------------
    # 6. Unusual Time Indicator
    # ---------------------------------------------------------
    if not ts_str or txn_dt is None:
        indicators["unusual_time"] = IndicatorDict(
            {
                "name": "unusual_time",
                "detected": False,
                "value": None,
                "threshold": f"{UNUSUAL_HOURS_START:02d}:00 - {UNUSUAL_HOURS_END:02d}:00",
                "description": "Transaction occurring outside typical diurnal operating hours",
                "evidence": "Timestamp data missing or unparseable in transaction record",
                "missing_data": True,
            }
        )
    else:
        hour = txn_dt.hour
        is_nocturnal = bool(UNUSUAL_HOURS_START <= hour <= UNUSUAL_HOURS_END)
        time_str = f"{hour:02d}:{txn_dt.minute:02d}"

        indicators["unusual_time"] = IndicatorDict(
            {
                "name": "unusual_time",
                "detected": is_nocturnal,
                "value": time_str,
                "threshold": f"{UNUSUAL_HOURS_START:02d}:00 - {UNUSUAL_HOURS_END:02d}:00",
                "description": "Transaction occurring outside typical diurnal operating hours",
                "evidence": (
                    f"Transaction initiated at {time_str}, within nocturnal window "
                    f"({UNUSUAL_HOURS_START:02d}:00 - {UNUSUAL_HOURS_END:02d}:00)"
                    if is_nocturnal
                    else f"Transaction initiated at {time_str}, outside unusual off-hours window"
                ),
                "missing_data": False,
            }
        )

    return {
        "status": "success",
        "transaction_id": transaction.get("transaction_id"),
        "indicators": indicators,
        **indicators,
    }
