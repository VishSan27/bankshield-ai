"""Tests for report generation and formatting tools in BankShield AI."""

import json
import os
import tempfile
import pytest

from tools.report_tools import (
    export_report_csv,
    export_report_json,
    format_report_text,
    generate_investigation_report,
)


@pytest.fixture
def sample_report_data():
    """Create sample inputs for report generation."""
    investigation_id = "INV-TEST-001"
    transaction = {
        "transaction_id": "TXN-10001",
        "account_id": "ACC-1001",
        "timestamp": "2026-10-10T14:30:00",
        "amount": 25000.0,
        "merchant": "Luxury Retailer",
        "location": "Mumbai",
        "channel": "online",
        "status": "completed",
    }
    evidence = {
        "evidence_summary": "Transaction amount 5x historical average",
        "relevant_transaction_ids": ["TXN-10001"],
        "calculated_indicators": {"high_amount": True},
        "supporting_values": {"deviation_score": 4.5},
        "missing_data_warnings": [],
    }
    risk_result = {
        "total_score": 65,
        "risk_score": 65,
        "priority": "High",
        "triggered_rules": [
            {
                "rule_name": "high_amount",
                "score_contribution": 30,
                "explanation": "Amount exceeds 5x average",
                "evidence_ref": "25000.0 INR",
            }
        ],
        "recommended_action": "Priority review recommended. Flag for analyst attention.",
    }
    validation_result = {
        "is_valid": True,
        "status": "VALIDATED",
        "checks_performed": [{"name": "numerical_calculations", "passed": True}],
        "errors": [],
        "warnings": [],
    }
    plan = {
        "steps": [
            {"step_number": 1, "name": "validate_input", "status": "completed"},
            {"step_number": 2, "name": "retrieve_data", "status": "completed"},
        ]
    }
    return {
        "investigation_id": investigation_id,
        "transaction": transaction,
        "evidence": evidence,
        "risk_result": risk_result,
        "validation_result": validation_result,
        "plan": plan,
    }


def test_generate_investigation_report(sample_report_data):
    """Test generating a structured report with all 16 specification fields."""
    report = generate_investigation_report(
        investigation_id=sample_report_data["investigation_id"],
        transaction=sample_report_data["transaction"],
        evidence=sample_report_data["evidence"],
        risk_result=sample_report_data["risk_result"],
        validation_result=sample_report_data["validation_result"],
        plan=sample_report_data["plan"],
    )

    required_fields = [
        "investigation_id",
        "timestamp",
        "transaction_id",
        "account_id",
        "transaction_details",
        "investigation_objective",
        "planner_task_list",
        "evidence_collected",
        "risk_indicators",
        "individual_rule_contributions",
        "total_risk_score",
        "assigned_priority",
        "recommended_next_action",
        "critic_validation_results",
        "missing_evidence_and_limitations",
        "final_investigation_status",
    ]

    for field in required_fields:
        assert field in report, f"Missing required specification field: {field}"

    assert report["final_investigation_status"] == "VALIDATED"
    assert report["total_risk_score"] == 65
    assert report["assigned_priority"] == "High"


def test_format_report_text(sample_report_data):
    """Test converting structured report into formatted human-readable text."""
    report = generate_investigation_report(
        investigation_id=sample_report_data["investigation_id"],
        transaction=sample_report_data["transaction"],
        evidence=sample_report_data["evidence"],
        risk_result=sample_report_data["risk_result"],
        validation_result=sample_report_data["validation_result"],
        plan=sample_report_data["plan"],
    )
    formatted = format_report_text(report)
    assert isinstance(formatted, str)
    assert "BANKSHIELD AI" in formatted
    assert "TXN-10001" in formatted
    assert "VALIDATED" in formatted


def test_export_report_json(sample_report_data):
    """Test exporting report to JSON file."""
    report = generate_investigation_report(
        investigation_id=sample_report_data["investigation_id"],
        transaction=sample_report_data["transaction"],
        evidence=sample_report_data["evidence"],
        risk_result=sample_report_data["risk_result"],
        validation_result=sample_report_data["validation_result"],
        plan=sample_report_data["plan"],
    )
    with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as tf:
        temp_path = tf.name

    try:
        success = export_report_json(report, temp_path)
        assert success is True
        with open(temp_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        assert data["investigation_id"] == "INV-TEST-001"
    finally:
        if os.path.exists(temp_path):
            os.remove(temp_path)


def test_export_report_csv(sample_report_data):
    """Test exporting reports list to CSV file."""
    report = generate_investigation_report(
        investigation_id=sample_report_data["investigation_id"],
        transaction=sample_report_data["transaction"],
        evidence=sample_report_data["evidence"],
        risk_result=sample_report_data["risk_result"],
        validation_result=sample_report_data["validation_result"],
        plan=sample_report_data["plan"],
    )
    with tempfile.NamedTemporaryFile(suffix=".csv", delete=False) as tf:
        temp_path = tf.name

    try:
        success = export_report_csv([report], temp_path)
        assert success is True
        assert os.path.getsize(temp_path) > 0
    finally:
        if os.path.exists(temp_path):
            os.remove(temp_path)
