"""Investigation Report Generation and Export Utilities for BankShield AI.

Provides functions to synthesize multi-agent findings into comprehensive
investigation reports containing all required dossier fields, format reports
into human-readable text, and export to JSON or CSV formats.
"""

import csv
from datetime import datetime, timezone
import json
import logging
from pathlib import Path
import random
from typing import Any, Dict, List, Optional
import uuid

# Ensure reproducibility
random.seed(42)

logger = logging.getLogger(__name__)


def generate_investigation_report(
    investigation_id: Optional[str],
    transaction: Dict[str, Any],
    evidence: Dict[str, Any],
    risk_result: Any,
    validation_result: Any,
    plan: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Synthesize evidence, risk scoring, and validation into a comprehensive report.

    Contains all 16 required investigation dossier fields:
    1. investigation_id (UUID)
    2. transaction_id
    3. account_id
    4. timestamp (ISO 8601 UTC)
    5. status (VALIDATED, NEEDS_REVIEW, or VALIDATION_FAILED)
    6. priority (Critical, High, Medium, or Low)
    7. risk_score (0-100)
    8. raw_score
    9. recommendation
    10. summary
    11. transaction_details
    12. evidence
    13. triggered_rules
    14. untriggered_rules
    15. missing_data_rules
    16. validation_result

    Args:
        investigation_id: Investigation UUID or None to auto-generate.
        transaction: Target transaction dictionary.
        evidence: Evidence bundle produced by ResearchAgent.
        risk_result: Risk evaluation result (dict or RiskResult model).
        validation_result: Audit result produced by CriticAgent (dict or model).
        plan: Optional investigation plan dictionary or model.

    Returns:
        Dossier dictionary containing all 16 required specification fields.
    """
    inv_id = str(investigation_id).strip() if investigation_id else str(uuid.uuid4())
    tx_dict = dict(transaction) if isinstance(transaction, dict) else {}
    ev_dict = dict(evidence) if isinstance(evidence, dict) else {}
    plan_dict = dict(plan) if isinstance(plan, dict) else (plan.model_dump() if hasattr(plan, "model_dump") else {})

    # Extract target identifiers
    txn_id = tx_dict.get("transaction_id", "UNKNOWN")
    acc_id = tx_dict.get("account_id", ev_dict.get("account_id", "UNKNOWN"))
    ts_now = datetime.now(timezone.utc).isoformat()

    # Normalize risk result
    if hasattr(risk_result, "model_dump"):
        risk_dict = risk_result.model_dump()
    elif isinstance(risk_result, dict):
        risk_dict = dict(risk_result)
    else:
        risk_dict = {}

    risk_score = int(risk_dict.get("risk_score", risk_dict.get("total_score", 0)))
    raw_score = int(risk_dict.get("raw_score", risk_score))
    priority = str(risk_dict.get("priority", "Low")).capitalize()
    triggered_rules = list(risk_dict.get("triggered_rules", []))
    untriggered_rules = list(risk_dict.get("untriggered_rules", []))
    missing_data_rules = list(risk_dict.get("missing_data_rules", []))

    # Normalize validation result
    if hasattr(validation_result, "model_dump"):
        val_dict = validation_result.model_dump()
    elif isinstance(validation_result, dict):
        val_dict = dict(validation_result)
    else:
        val_dict = {}

    val_status = val_dict.get("status")
    is_valid = val_dict.get("is_valid", True)
    val_errors = val_dict.get("errors", [])
    val_warnings = val_dict.get("warnings", [])

    # ---------------------------------------------------------
    # Derive Report Status (VALIDATED, NEEDS_REVIEW, VALIDATION_FAILED)
    # ---------------------------------------------------------
    if val_status in ("VALIDATION_FAILED", "FAILED") or (is_valid is False and val_errors):
        status = "VALIDATION_FAILED"
    elif (
        val_status in ("NEEDS_REVIEW", "WARNING", "PARTIAL")
        or val_warnings
        or missing_data_rules
        or not val_dict
    ):
        status = "NEEDS_REVIEW"
    elif val_status in ("VALIDATED", "PASSED", "SUCCESS") or is_valid is True:
        status = "VALIDATED"
    else:
        status = "NEEDS_REVIEW"

    # ---------------------------------------------------------
    # Derive Operational Recommendation
    # ---------------------------------------------------------
    # ---------------------------------------------------------
    # Derive Operational Recommendation (Human Decision Support)
    # ---------------------------------------------------------
    if status == "VALIDATION_FAILED":
        recommendation = (
            "MANUAL_AUDIT_REQUIRED: Audit identified discrepancies during validation; "
            "recommend senior compliance analyst review data integrity."
        )
    elif priority == "Critical":
        recommendation = (
            "IMMEDIATE_ANALYST_REVIEW: Critical priority threshold reached (score >= 80). "
            "Recommend immediate manual inspection of merchant verification and customer contact."
        )
    elif priority == "High":
        recommendation = (
            "PRIORITY_ANALYST_REVIEW: High priority threshold reached (score 60-79). "
            "Recommend expedited analyst inspection of amount deviation and recent transaction velocity."
        )
    elif priority == "Medium":
        recommendation = (
            "STANDARD_REVIEW: Medium priority threshold reached (score 30-59). "
            "Recommend standard analyst review and monitoring for recurring anomalous indicators."
        )
    else:
        recommendation = (
            "ROUTINE_MONITORING: Low priority threshold (score < 30). "
            "No immediate investigation required; proceed with standard transaction monitoring."
        )

    # ---------------------------------------------------------
    # Synthesize Executive Narrative Summary
    # ---------------------------------------------------------
    triggered_count = len(triggered_rules)
    rule_names = [r.get("rule_name", r.get("name", "Unknown")) for r in triggered_rules]
    rule_str = ", ".join(rule_names) if rule_names else "None"

    summary = (
        f"Investigation {inv_id} for transaction {txn_id} (Account: {acc_id}). "
        f"Computed risk score of {risk_score}/100 categorized as {priority} priority. "
        f"{triggered_count} rules triggered: [{rule_str}]. "
        f"Critic validation status: {status}. Recommended action: {recommendation}."
    )

    # Extract planner fields
    objective = plan_dict.get("objective", "Investigate banking transaction risk indicators")
    planner_tasks = plan_dict.get("steps", [])

    # Compile the full 16-field dossier with specification aliases
    report: Dict[str, Any] = {
        "investigation_id": inv_id,
        "transaction_id": txn_id,
        "account_id": acc_id,
        "timestamp": ts_now,
        "status": status,
        "priority": priority,
        "risk_score": risk_score,
        "raw_score": raw_score,
        "recommendation": recommendation,
        "summary": summary,
        "transaction_details": tx_dict,
        "evidence": ev_dict,
        "triggered_rules": triggered_rules,
        "untriggered_rules": untriggered_rules,
        "missing_data_rules": missing_data_rules,
        "validation_result": val_dict,
        "plan": plan_dict,
        # Standard 16-field compliance dossier specification:
        "investigation_objective": objective,
        "planner_task_list": planner_tasks,
        "evidence_collected": ev_dict,
        "risk_indicators": ev_dict.get("indicators", ev_dict.get("calculated_indicators", {})),
        "individual_rule_contributions": triggered_rules,
        "total_risk_score": risk_score,
        "assigned_priority": priority,
        "recommended_next_action": recommendation,
        "critic_validation_results": val_dict,
        "missing_evidence_and_limitations": missing_data_rules,
        "final_investigation_status": status,
    }

    return report


def format_report_text(report: Dict[str, Any]) -> str:
    """Format an investigation report dictionary into clean, structured plain text.

    Args:
        report: Investigation report dictionary.

    Returns:
        Formatted multi-line plain text report string.
    """
    if not isinstance(report, dict):
        return "Error: Invalid report dictionary provided."

    inv_id = report.get("investigation_id", "N/A")
    txn_id = report.get("transaction_id", "N/A")
    acc_id = report.get("account_id", "N/A")
    timestamp = report.get("timestamp", "N/A")
    status = report.get("status", "N/A")
    priority = report.get("priority", "N/A")
    risk_score = report.get("risk_score", 0)
    raw_score = report.get("raw_score", risk_score)
    recommendation = report.get("recommendation", "N/A")
    summary = report.get("summary", "N/A")

    tx_details = report.get("transaction_details", {})
    amount = tx_details.get("amount", "N/A")
    merchant = tx_details.get("merchant", "N/A")
    location = tx_details.get("location", "N/A")
    channel = tx_details.get("channel", "N/A")
    tx_time = tx_details.get("timestamp", "N/A")
    failed_attempts = tx_details.get("failed_attempts", 0)

    triggered_rules = report.get("triggered_rules", [])
    untriggered_rules = report.get("untriggered_rules", [])
    missing_data_rules = report.get("missing_data_rules", [])
    val_result = report.get("validation_result", {})

    lines: List[str] = [
        "=" * 78,
        "BANKSHIELD AI - TRANSACTION INVESTIGATION REPORT",
        "=" * 78,
        f"Investigation ID : {inv_id}",
        f"Report Timestamp : {timestamp}",
        f"Audit Status     : {status}",
        f"Priority Tier    : {priority}",
        f"Risk Score       : {risk_score}/100 (Raw Score: {raw_score})",
        "-" * 78,
        "1. TARGET TRANSACTION DETAILS",
        "-" * 78,
        f"  Transaction ID   : {txn_id}",
        f"  Account ID       : {acc_id}",
        f"  Amount           : ${float(amount):,.2f}" if isinstance(amount, (int, float)) else f"  Amount           : {amount}",
        f"  Timestamp        : {tx_time}",
        f"  Merchant         : {merchant}",
        f"  Location         : {location}",
        f"  Channel          : {channel}",
        f"  Failed Attempts  : {failed_attempts}",
        "-" * 78,
        "2. EXECUTIVE SUMMARY",
        "-" * 78,
        f"  {summary}",
        "-" * 78,
        "3. RISK RULES EVALUATION",
        "-" * 78,
        f"  Triggered Rules ({len(triggered_rules)}):",
    ]

    if triggered_rules:
        for r in triggered_rules:
            name = r.get("rule_name", r.get("name", "Unknown"))
            weight = r.get("score_contribution", r.get("weight", 0))
            expl = r.get("explanation", "")
            ev = r.get("evidence_ref", "")
            lines.append(f"    - [{name}] (+{weight} pts): {expl}")
            if ev and ev not in expl:
                lines.append(f"      Evidence: {ev}")
    else:
        lines.append("    (None)")

    lines.append(f"\n  Untriggered Rules ({len(untriggered_rules)}):")
    if untriggered_rules:
        for r in untriggered_rules:
            name = r.get("rule_name", r.get("name", "Unknown"))
            reason = r.get("reason", "Threshold not met")
            lines.append(f"    - [{name}]: {reason}")
    else:
        lines.append("    (None)")

    if missing_data_rules:
        lines.append(f"\n  Missing Data Rules ({len(missing_data_rules)}):")
        for r in missing_data_rules:
            name = r.get("rule_name", r.get("name", "Unknown"))
            reason = r.get("reason", "Data missing")
            lines.append(f"    - [{name}]: {reason}")

    lines.extend([
        "-" * 78,
        "4. CRITIC AUDIT & VALIDATION",
        "-" * 78,
        f"  Status             : {val_result.get('status', 'N/A')}",
        f"  Audit Pass Rate    : {val_result.get('validation_score', 'N/A')}%",
        f"  Score Verification : Reported {risk_score} vs Recalculated {val_result.get('recalculated_score', 'N/A')} (Match: {val_result.get('score_matches', 'N/A')})",
    ])

    val_errors = val_result.get("errors", [])
    if val_errors:
        lines.append("  Errors:")
        for err in val_errors:
            lines.append(f"    - {err}")

    val_warnings = val_result.get("warnings", [])
    if val_warnings:
        lines.append("  Warnings:")
        for warn in val_warnings:
            lines.append(f"    - {warn}")

    lines.extend([
        "-" * 78,
        "5. ACTION RECOMMENDATION",
        "-" * 78,
        f"  {recommendation}",
        "=" * 78,
    ])

    return "\n".join(lines)


def export_report_json(report: Dict[str, Any], filepath: str | Path) -> bool:
    """Export an investigation report to a formatted JSON file.

    Args:
        report: Report dictionary to persist.
        filepath: Destination filesystem path.

    Returns:
        True if export succeeded, False otherwise.
    """
    try:
        dest_path = Path(filepath)
        dest_path.parent.mkdir(parents=True, exist_ok=True)

        with open(dest_path, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2, default=str)

        logger.info("Successfully exported report JSON to '%s'", dest_path)
        return True
    except Exception as exc:
        logger.error("Failed to export report JSON to '%s': %s", filepath, exc)
        return False


def export_report_csv(reports: List[Dict[str, Any]], filepath: str | Path) -> bool:
    """Export a collection of investigation reports to a flattened CSV file.

    Args:
        reports: List of investigation report dictionaries.
        filepath: Destination filesystem path.

    Returns:
        True if export succeeded, False otherwise.
    """
    if not isinstance(reports, list):
        logger.error("Expected list of reports for CSV export, got %s", type(reports))
        return False

    try:
        dest_path = Path(filepath)
        dest_path.parent.mkdir(parents=True, exist_ok=True)

        fieldnames = [
            "investigation_id",
            "transaction_id",
            "account_id",
            "timestamp",
            "status",
            "priority",
            "risk_score",
            "raw_score",
            "recommendation",
            "summary",
            "triggered_rule_count",
            "triggered_rules",
            "missing_data_count",
            "validation_status",
            "amount",
            "merchant",
        ]

        rows: List[Dict[str, Any]] = []
        for r in reports:
            if not isinstance(r, dict):
                continue

            tx = r.get("transaction_details", {})
            trig = r.get("triggered_rules", [])
            trig_names = ";".join(
                [x.get("rule_name", x.get("name", "")) for x in trig if isinstance(x, dict)]
            )
            val = r.get("validation_result", {})

            row = {
                "investigation_id": r.get("investigation_id", ""),
                "transaction_id": r.get("transaction_id", ""),
                "account_id": r.get("account_id", ""),
                "timestamp": r.get("timestamp", ""),
                "status": r.get("status", ""),
                "priority": r.get("priority", ""),
                "risk_score": r.get("risk_score", 0),
                "raw_score": r.get("raw_score", 0),
                "recommendation": r.get("recommendation", ""),
                "summary": r.get("summary", ""),
                "triggered_rule_count": len(trig),
                "triggered_rules": trig_names,
                "missing_data_count": len(r.get("missing_data_rules", [])),
                "validation_status": val.get("status", r.get("status", "")),
                "amount": tx.get("amount", ""),
                "merchant": tx.get("merchant", ""),
            }
            rows.append(row)

        with open(dest_path, "w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)

        logger.info(
            "Successfully exported %d reports to CSV at '%s'",
            len(rows),
            dest_path,
        )
        return True
    except Exception as exc:
        logger.error("Failed to export reports CSV to '%s': %s", filepath, exc)
        return False
