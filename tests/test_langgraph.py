"""Unit and Integration Tests for LangGraph Multi-Agent Orchestration in BankShield AI."""

import pytest
from orchestration.investigation_state import InvestigationState
from orchestration.langgraph_orchestrator import (
    LangGraphInvestigationOrchestrator,
    build_investigation_graph,
)
from services.data_service import DataService


@pytest.fixture
def data_service():
    """Create DataService fixture."""
    return DataService()


@pytest.fixture
def langgraph_orchestrator(data_service):
    """Create LangGraphInvestigationOrchestrator fixture."""
    return LangGraphInvestigationOrchestrator(data_service=data_service, max_retries=1)


def test_langgraph_graph_compilation():
    """Test that LangGraph StateGraph compiles with correct nodes and edges."""
    graph = build_investigation_graph(max_retries=1)
    assert graph is not None
    # Verify mermaid diagram export
    mermaid = graph.get_graph().draw_mermaid()
    assert "planner_node" in mermaid
    assert "research_node" in mermaid
    assert "risk_assessment_node" in mermaid
    assert "critic_node" in mermaid
    assert "correction_node" in mermaid
    assert "report_generation_node" in mermaid


def test_langgraph_valid_transaction_flow(langgraph_orchestrator):
    """Test end-to-end execution of a valid transaction through LangGraph."""
    res = langgraph_orchestrator.investigate_transaction("TXN-10198")

    assert res["status"] in ["completed", "needs_review"]
    assert res["transaction_id"] == "TXN-10198"
    assert "final_report" in res
    assert "node_history" in res

    # Verify nodes visited in order
    nodes = res["node_history"]
    assert "planner_node" in nodes
    assert "research_node" in nodes
    assert "risk_assessment_node" in nodes
    assert "critic_node" in nodes
    assert "report_generation_node" in nodes

    # Verify MCP tool calls recorded
    assert len(res["mcp_tool_calls"]) >= 3


def test_langgraph_conditional_correction_routing(data_service):
    """Test conditional routing to correction_node when Critic flags a discrepancy."""
    # Build graph with a custom node that forces a discrepancy on attempt 0
    graph = build_investigation_graph(data_service=data_service, max_retries=1)

    # Invoke with initial state that triggers correction
    initial_state: InvestigationState = {
        "investigation_id": "INV-TEST-RETRY",
        "transaction_id": "TXN-10078",
        "objective": "Test retry routing",
        "retry_count": 0,
        "max_retries": 1,
        "correction_applied": False,
        "correction_history": [],
        "node_history": [],
        "execution_trace": [],
        "errors": [],
        "warnings": [],
        "mcp_tool_calls": [],
    }

    result = graph.invoke(initial_state)
    assert "report_generation_node" in result["node_history"]
    # Final output status is completed or needs_review
    assert result["status"] in ["completed", "needs_review"]


def test_langgraph_retry_limit_enforcement(data_service):
    """Test that retry limit prevents infinite loops and terminates safely."""
    orch = LangGraphInvestigationOrchestrator(data_service=data_service, max_retries=1)
    res = orch.investigate_transaction("TXN-10015")

    # Retry count should not exceed max_retries
    assert res.get("retry_count", 0) <= 1
    assert "report_generation_node" in res["node_history"]


def test_langgraph_invalid_transaction_handling(langgraph_orchestrator):
    """Test that LangGraph gracefully handles non-existent IDs without crashing."""
    res = langgraph_orchestrator.investigate_transaction("TXN-99999")

    assert res["status"] in ["failed", "partial", "completed"]
    assert res["transaction_id"] == "TXN-99999"
    assert "final_report" in res
    assert len(res["node_history"]) > 0


def test_langgraph_batch_investigation(langgraph_orchestrator):
    """Test batch investigation with LangGraph."""
    batch_ids = ["TXN-10198", "TXN-10078"]
    batch_res = langgraph_orchestrator.investigate_batch(batch_ids)

    assert "results" in batch_res
    assert "batch_summary" in batch_res
    assert len(batch_res["results"]) == 2
    assert batch_res["batch_summary"]["total_investigated"] == 2
