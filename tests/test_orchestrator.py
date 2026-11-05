"""Tests for InvestigationOrchestrator in BankShield AI."""

import pytest
from orchestration.orchestrator import InvestigationOrchestrator
from services.data_service import DataService


@pytest.fixture
def orchestrator():
    """Create orchestrator with real DataService."""
    data_service = DataService()
    return InvestigationOrchestrator(data_service=data_service)


def test_orchestrator_single_investigation(orchestrator):
    """Test full multi-agent investigation lifecycle for a single transaction."""
    tx_ids = orchestrator.data_service.get_all_transaction_ids()
    tx_id = tx_ids[0]

    result = orchestrator.investigate_transaction(tx_id)

    assert result["status"] in ["completed", "success"]
    assert result["transaction_id"] == tx_id
    assert "final_report" in result
    assert "execution_trace" in result
    assert "validation_status" in result
    assert "total_duration_ms" in result

    # Check execution trace order
    trace = result["execution_trace"]
    assert len(trace) >= 4

    agent_names = [t["agent_name"] for t in trace]
    assert "PlannerAgent" in agent_names
    assert "ResearchAgent" in agent_names
    assert "RiskAssessmentAgent" in agent_names
    assert "CriticAgent" in agent_names


def test_orchestrator_batch_investigation(orchestrator):
    """Test batch investigation processing multiple transactions."""
    tx_ids = orchestrator.data_service.get_all_transaction_ids()[:3]

    batch_result = orchestrator.investigate_batch(tx_ids)

    assert "results" in batch_result
    assert "batch_summary" in batch_result
    assert len(batch_result["results"]) == 3

    summary = batch_result["batch_summary"]
    assert summary["total_investigated"] == 3
    assert "priority_breakdown" in summary
    assert "average_risk_score" in summary


def test_orchestrator_invalid_transaction_handling(orchestrator):
    """Test orchestrator gracefully handles non-existent transaction IDs."""
    result = orchestrator.investigate_transaction("TXN-NONEXISTENT-999")

    # Should not raise exception, but return structured failure / report
    assert result["status"] in ["failed", "partial", "completed"]
    assert result.get("transaction_id") == "TXN-NONEXISTENT-999"


def test_orchestrator_session_history(orchestrator):
    """Test session history accumulates investigation records."""
    tx_ids = orchestrator.data_service.get_all_transaction_ids()[:2]
    orchestrator.investigate_transaction(tx_ids[0])
    orchestrator.investigate_transaction(tx_ids[1])

    history = orchestrator.get_investigation_history()
    assert len(history) >= 2
