"""BankShield AI — Standardized Local Model Context Protocol (MCP) Server.

Exposes specialized banking transaction investigation and validation tools
adhering to the official Model Context Protocol (MCP) specification.
Operates strictly on synthetic demonstration data and deterministic logic.
"""

import asyncio
from datetime import datetime, timezone
import json
import logging
from pathlib import Path
import sys
from typing import Any, Dict, List, Optional

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from mcp.server.mcpserver import MCPServer
from services.data_service import DataService
from tools.risk_rules import create_default_engine
from tools.transaction_tools import (
    calculate_transaction_deviation as calc_dev_tool,
    identify_transaction_indicators as id_ind_tool,
)

logger = logging.getLogger("BankShieldMCPServer")


def create_mcp_server(data_service: Optional[DataService] = None) -> MCPServer:
    """Create and configure the BankShield AI MCP Server with all 5 standard investigation tools.

    Args:
        data_service: Optional DataService instance. If None, instantiates a default DataService.

    Returns:
        Configured MCPServer instance with registered tools.
    """
    ds = data_service or DataService()
    server = MCPServer("BankShieldInvestigationServer")

    # --------------------------------------------------------------------------
    # MCP Tool 1: get_transaction
    # --------------------------------------------------------------------------
    @server.tool(
        name="get_transaction",
        description="Retrieve verified synthetic transaction record and merchant details by Transaction ID."
    )
    def get_transaction(transaction_id: str) -> Dict[str, Any]:
        """Fetch transaction record by ID.

        Args:
            transaction_id: Target transaction identifier (e.g. 'TXN-10198').

        Returns:
            Dictionary containing transaction fields or error status.
        """
        if not transaction_id or not isinstance(transaction_id, str):
            return {
                "status": "error",
                "error": "Invalid transaction_id: must be a non-empty string.",
                "data": None,
            }

        tx = ds.get_transaction(transaction_id.strip())
        if not tx:
            return {
                "status": "error",
                "error": f"Transaction '{transaction_id}' not found in synthetic database.",
                "data": None,
            }

        return {
            "status": "success",
            "transaction_id": transaction_id,
            "data": tx,
        }

    # --------------------------------------------------------------------------
    # MCP Tool 2: get_account_history
    # --------------------------------------------------------------------------
    @server.tool(
        name="get_account_history",
        description="Retrieve chronological transaction records and historical baseline for an account."
    )
    def get_account_history(account_id: str) -> Dict[str, Any]:
        """Fetch account transaction history and profile baseline.

        Args:
            account_id: Target account identifier (e.g. 'ACC-1001').

        Returns:
            Dictionary containing history list, profile baseline, and summary counts.
        """
        if not account_id or not isinstance(account_id, str):
            return {
                "status": "error",
                "error": "Invalid account_id: must be a non-empty string.",
                "history": [],
            }

        acc_clean = account_id.strip()
        history = ds.get_account_history(acc_clean)
        profile = ds.get_account_profile(acc_clean)

        return {
            "status": "success",
            "account_id": acc_clean,
            "record_count": len(history),
            "profile": profile,
            "history": history,
        }

    # --------------------------------------------------------------------------
    # MCP Tool 3: calculate_transaction_deviation
    # --------------------------------------------------------------------------
    @server.tool(
        name="calculate_transaction_deviation",
        description="Calculate statistical amount baseline deviation (z-score, ratio) against account history."
    )
    def calculate_transaction_deviation(transaction_id: str) -> Dict[str, Any]:
        """Compute statistical deviation for a transaction.

        Args:
            transaction_id: Transaction identifier to evaluate.

        Returns:
            Dictionary with mean, standard deviation, z-score, and deviation flag.
        """
        tx = ds.get_transaction(transaction_id.strip())
        if not tx:
            return {
                "status": "error",
                "error": f"Transaction '{transaction_id}' not found.",
            }

        acc_id = tx.get("account_id")
        history = ds.get_account_history(acc_id)
        dev = calc_dev_tool(tx, history)

        return {
            "status": "success",
            "transaction_id": transaction_id,
            "account_id": acc_id,
            "deviation": dev,
        }

    # --------------------------------------------------------------------------
    # MCP Tool 4: identify_transaction_indicators
    # --------------------------------------------------------------------------
    @server.tool(
        name="identify_transaction_indicators",
        description="Evaluate 5 empirical risk indicators (amount deviation, velocity, unseen merchant, failed attempts, nocturnal hours)."
    )
    def identify_transaction_indicators(transaction_id: str) -> Dict[str, Any]:
        """Extract empirical anomaly indicators for a target transaction.

        Args:
            transaction_id: Target transaction ID.

        Returns:
            Dictionary of evaluated indicators with detection flags and evidence values.
        """
        tx = ds.get_transaction(transaction_id.strip())
        if not tx:
            return {
                "status": "error",
                "error": f"Transaction '{transaction_id}' not found.",
                "indicators": {},
            }

        acc_id = tx.get("account_id")
        history = ds.get_account_history(acc_id)
        profile = ds.get_account_profile(acc_id)

        ind_bundle = id_ind_tool(tx, history, profile or {})
        return {
            "status": "success",
            "transaction_id": transaction_id,
            "account_id": acc_id,
            "indicators": ind_bundle,
        }

    # --------------------------------------------------------------------------
    # MCP Tool 5: validate_investigation
    # --------------------------------------------------------------------------
    @server.tool(
        name="validate_investigation",
        description="Perform independent Critic mathematical recalculation and audit of risk scores and priority tiers."
    )
    def validate_investigation(
        transaction_id: str,
        reported_score: int,
        priority: str,
        triggered_rule_names: List[str],
    ) -> Dict[str, Any]:
        """Audit risk assessment calculations independently.

        Args:
            transaction_id: Target transaction ID.
            reported_score: Risk score reported by Risk Assessment agent.
            priority: Priority classification reported.
            triggered_rule_names: List of rule identifiers claimed to have triggered.

        Returns:
            Audit result with independent recalculation, equality match, and audit status.
        """
        tx = ds.get_transaction(transaction_id.strip())
        if not tx:
            return {
                "status": "VALIDATION_FAILED",
                "score_matches": False,
                "error": f"Transaction ID '{transaction_id}' does not exist in records.",
                "recalculated_score": 0,
                "checks_passed": 0,
            }

        acc_id = tx.get("account_id")
        history = ds.get_account_history(acc_id)
        profile = ds.get_account_profile(acc_id)

        # Independent recomputation
        ind_bundle = id_ind_tool(tx, history, profile or {})
        engine = create_default_engine()
        recalc_result = engine.evaluate(ind_bundle)

        recalc_score = recalc_result.total_score
        recalc_priority = recalc_result.priority
        score_matches = (int(reported_score) == int(recalc_score))
        priority_matches = (priority.capitalize() == recalc_priority.capitalize())

        checks = [
            {"gate": "record_exists", "passed": True},
            {"gate": "score_range", "passed": 0 <= int(reported_score) <= 100},
            {"gate": "mathematical_recalculation", "passed": score_matches},
            {"gate": "priority_mapping", "passed": priority_matches},
        ]
        checks_passed = sum(1 for c in checks if c["passed"])

        val_status = "VALIDATED" if (score_matches and priority_matches) else "VALIDATION_FAILED"

        return {
            "status": val_status,
            "transaction_id": transaction_id,
            "reported_score": reported_score,
            "recalculated_score": recalc_score,
            "score_matches": score_matches,
            "reported_priority": priority,
            "recalculated_priority": recalc_priority,
            "priority_matches": priority_matches,
            "checks": checks,
            "checks_passed": checks_passed,
            "total_checks": len(checks),
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }

    return server


# Global default instance
mcp_server = create_mcp_server()


def get_mcp_server() -> MCPServer:
    """Return singleton MCPServer instance."""
    global mcp_server
    return mcp_server


if __name__ == "__main__":
    # When run directly, start stdio server
    print("Starting BankShield AI MCP Server on stdio...", file=sys.stderr)
    asyncio.run(mcp_server.run_stdio_async())
