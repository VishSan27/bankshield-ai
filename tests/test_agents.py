"""Tests for the four specialized agents in BankShield AI."""

import pytest
from agents.critic_agent import CriticAgent
from agents.planner_agent import PlannerAgent
from agents.research_agent import ResearchAgent
from agents.risk_assessment_agent import RiskAssessmentAgent
from services.data_service import DataService


@pytest.fixture
def data_service():
    """Create DataService instance."""
    return DataService()


def test_planner_agent_standard_plan():
    """Test PlannerAgent generates an ordered 6-step plan."""
    planner = PlannerAgent()
    plan = planner.plan_investigation("TXN-10001", objective="Investigate transaction risk")

    assert plan["status"] == "planned"
    assert plan["transaction_id"] == "TXN-10001"
    assert "steps" in plan
    assert len(plan["steps"]) == 6

    step_names = [s["name"] for s in plan["steps"]]
    expected_names = [
        "validate_input",
        "retrieve_data",
        "calculate_indicators",
        "assess_risk",
        "validate_results",
        "generate_report",
    ]
    assert step_names == expected_names


def test_planner_agent_invalid_input():
    """Test PlannerAgent handles invalid / empty transaction IDs."""
    planner = PlannerAgent()
    plan = planner.plan_investigation("", objective="Test empty")
    assert plan["status"] in ["error", "failed"]
    assert "error" in plan


def test_research_agent_valid_transaction(data_service):
    """Test ResearchAgent collects evidence and computes indicators."""
    agent = ResearchAgent(data_service=data_service)
    tx_ids = data_service.get_all_transaction_ids()
    evidence = agent.investigate_transaction(tx_ids[0])

    assert evidence["status"] == "success"
    assert "transaction_details" in evidence
    assert "account_profile" in evidence
    assert "indicators" in evidence
    assert "deviation_analysis" in evidence
    assert len(evidence.get("tool_calls", [])) >= 3


def test_research_agent_nonexistent_transaction(data_service):
    """Test ResearchAgent handles nonexistent transaction IDs."""
    agent = ResearchAgent(data_service=data_service)
    evidence = agent.investigate_transaction("TXN-NONEXISTENT-999")
    assert evidence["status"] == "error"
    assert len(evidence.get("missing_data_warnings", [])) > 0


def test_risk_assessment_agent(data_service):
    """Test RiskAssessmentAgent evaluates collected evidence."""
    research_agent = ResearchAgent(data_service=data_service)
    risk_agent = RiskAssessmentAgent()

    tx_ids = data_service.get_all_transaction_ids()
    evidence = research_agent.investigate_transaction(tx_ids[0])
    assessment = risk_agent.assess_risk(evidence)

    assert assessment["status"] == "success"
    assert 0 <= assessment["risk_score"] <= 100
    assert assessment["priority"] in ["Low", "Medium", "High", "Critical"]
    assert "recommended_action" in assessment
    assert "fraudulent" not in assessment["recommended_action"].lower()


def test_critic_agent_validation(data_service):
    """Test CriticAgent validates investigation results and performs audits."""
    planner = PlannerAgent()
    research = ResearchAgent(data_service=data_service)
    risk_agent = RiskAssessmentAgent()
    critic = CriticAgent()

    tx_id = data_service.get_all_transaction_ids()[0]
    plan = planner.plan_investigation(tx_id)
    evidence = research.investigate_transaction(tx_id)
    risk_res = risk_agent.assess_risk(evidence)

    validation = critic.validate_investigation(
        transaction_data=evidence["transaction_details"],
        evidence=evidence,
        risk_result=risk_res,
        plan=plan,
    )

    assert "is_valid" in validation
    assert "status" in validation
    assert validation["status"] in ["VALIDATED", "NEEDS_REVIEW", "VALIDATION_FAILED"]
    assert "checks_performed" in validation
    assert len(validation["checks_performed"]) >= 8
    assert validation["score_matches"] is True


def test_critic_agent_tamper_detection(data_service):
    """Test that CriticAgent flags tampered or fabricated risk scores."""
    planner = PlannerAgent()
    research = ResearchAgent(data_service=data_service)
    risk_agent = RiskAssessmentAgent()
    critic = CriticAgent()

    tx_id = data_service.get_all_transaction_ids()[0]
    plan = planner.plan_investigation(tx_id)
    evidence = research.investigate_transaction(tx_id)
    risk_res = risk_agent.assess_risk(evidence)

    # Tamper with the reported score
    tampered_risk_res = dict(risk_res)
    tampered_risk_res["risk_score"] = 99  # fabricated score

    validation = critic.validate_investigation(
        transaction_data=evidence["transaction_details"],
        evidence=evidence,
        risk_result=tampered_risk_res,
        plan=plan,
    )

    # Must detect discrepancy
    assert validation["score_matches"] is False
    assert validation["status"] == "VALIDATION_FAILED"
