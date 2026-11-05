"""Unit and Protocol Tests for Model Context Protocol (MCP) Server and Client in BankShield AI."""

import pytest
from mcp_server.server import create_mcp_server, get_mcp_server
from services.data_service import DataService
from services.mcp_client import BankShieldMCPClient


@pytest.fixture
def mcp_client():
    """Create BankShieldMCPClient fixture."""
    return BankShieldMCPClient()


def test_mcp_server_tool_registration():
    """Test that MCPServer registers all 5 required investigation tools."""
    server = get_mcp_server()
    assert server is not None
    # Verify tool names
    expected_tools = {
        "get_transaction",
        "get_account_history",
        "calculate_transaction_deviation",
        "identify_transaction_indicators",
        "validate_investigation",
    }
    client = BankShieldMCPClient(server)
    tools = client.list_tools()
    registered_names = {t["name"] for t in tools}
    assert expected_tools.issubset(registered_names)


def test_mcp_get_transaction_valid(mcp_client):
    """Test invoking 'get_transaction' tool through MCP client."""
    res = mcp_client.get_transaction("TXN-10198")
    assert res.get("status") == "success"
    assert res.get("transaction_id") == "TXN-10198"
    assert "data" in res
    assert res["data"]["amount"] > 0


def test_mcp_get_transaction_invalid(mcp_client):
    """Test invoking 'get_transaction' with non-existent ID via MCP client."""
    res = mcp_client.get_transaction("TXN-NONEXISTENT")
    assert res.get("status") == "error"
    assert "not found" in res.get("error", "").lower()


def test_mcp_get_account_history(mcp_client):
    """Test invoking 'get_account_history' tool through MCP client."""
    res = mcp_client.get_account_history("ACC-1001")
    assert res.get("status") == "success"
    assert res.get("account_id") == "ACC-1001"
    assert "history" in res
    assert isinstance(res["history"], list)
    assert res.get("record_count") >= 1


def test_mcp_calculate_transaction_deviation(mcp_client):
    """Test invoking 'calculate_transaction_deviation' tool through MCP client."""
    res = mcp_client.calculate_transaction_deviation("TXN-10198")
    assert res.get("status") == "success"
    assert "deviation" in res
    dev = res["deviation"]
    assert "z_score" in dev
    assert "mean_amount" in dev


def test_mcp_identify_transaction_indicators(mcp_client):
    """Test invoking 'identify_transaction_indicators' tool through MCP client."""
    res = mcp_client.identify_transaction_indicators("TXN-10198")
    assert res.get("status") == "success"
    assert "indicators" in res
    inds = res["indicators"]
    assert "high_amount" in inds
    assert "new_merchant" in inds


def test_mcp_validate_investigation_matching(mcp_client):
    """Test invoking 'validate_investigation' tool with accurate calculations."""
    res = mcp_client.validate_investigation(
        transaction_id="TXN-10078",
        reported_score=0,
        priority="Low",
        triggered_rule_names=[],
    )
    assert res.get("status") == "VALIDATED"
    assert res.get("score_matches") is True
    assert res.get("priority_matches") is True


def test_mcp_validate_investigation_discrepancy(mcp_client):
    """Test invoking 'validate_investigation' tool with tampered score."""
    res = mcp_client.validate_investigation(
        transaction_id="TXN-10078",
        reported_score=85,  # Deliberate tamper
        priority="Critical",
        triggered_rule_names=[],
    )
    assert res.get("status") == "VALIDATION_FAILED"
    assert res.get("score_matches") is False


def test_mcp_client_telemetry_recording(mcp_client):
    """Test that MCP client records execution duration and arguments."""
    mcp_client.get_transaction("TXN-10078")
    log = mcp_client.get_telemetry_history()
    assert len(log) > 0
    latest = log[-1]
    assert latest["tool_name"] == "get_transaction"
    assert "duration_ms" in latest
    assert latest["protocol"] == "MCP/2.0"
