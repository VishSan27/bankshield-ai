"""Standardized Model Context Protocol (MCP) Client for BankShield AI.

Communicates with the local BankShield MCP Server using the official Model Context Protocol.
Enforces schema validation, handles protocol responses, logs telemetry, and provides
resilient error recovery when tools or servers encounter exceptions.
"""

import asyncio
from datetime import datetime, timezone
import inspect
import json
import logging
import time
from typing import Any, Dict, List, Optional

from mcp_server.server import get_mcp_server

logger = logging.getLogger("BankShieldMCPClient")


class BankShieldMCPClient:
    """Standardized MCP Client for invoking local banking investigation tools."""

    def __init__(self, mcp_server_instance: Optional[Any] = None) -> None:
        """Initialize MCP Client with target local MCPServer.

        Args:
            mcp_server_instance: Optional MCPServer instance. If None, uses singleton server.
        """
        self.server = mcp_server_instance or get_mcp_server()
        self.telemetry_log: List[Dict[str, Any]] = []

    def list_tools(self) -> List[Dict[str, Any]]:
        """Query server for list of registered tools and their MCP schemas.

        Returns:
            List of tool schema dictionaries.
        """
        try:
            if hasattr(self.server, "_tool_manager"):
                tools = self.server._tool_manager.list_tools()
                return [
                    {
                        "name": t.name,
                        "description": getattr(t, "description", ""),
                        "input_schema": getattr(t, "parameters", getattr(t, "input_schema", {})),
                        "output_schema": getattr(t, "output_schema", {}),
                    }
                    for t in tools
                ]
            else:
                async def _list():
                    tools = await self.server.list_tools()
                    return [
                        {
                            "name": t.name,
                            "description": t.description,
                            "input_schema": getattr(t, "input_schema", getattr(t, "inputSchema", {})),
                            "output_schema": getattr(t, "output_schema", getattr(t, "outputSchema", {})),
                        }
                        for t in tools
                    ]
                return asyncio.run(_list())
        except Exception as exc:
            logger.error("Failed to list MCP tools: %s", exc)
            return []

    def call_tool(self, tool_name: str, arguments: Dict[str, Any]) -> Dict[str, Any]:
        """Invoke a tool via the Model Context Protocol.

        Validates arguments against MCP tool schema, executes through the MCP server,
        and records execution telemetry.

        Args:
            tool_name: Registered MCP tool name.
            arguments: Dictionary of arguments conforming to tool schema.

        Returns:
            Structured dictionary returned by the tool.
        """
        start_time = time.perf_counter()
        call_entry: Dict[str, Any] = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "tool_name": tool_name,
            "arguments": arguments,
            "status": "pending",
            "protocol": "MCP/2.0",
        }

        # Input validation
        if not tool_name or not isinstance(tool_name, str):
            err_msg = "Invalid tool_name: must be a non-empty string"
            call_entry.update({"status": "error", "error": err_msg, "duration_ms": 0.0})
            self.telemetry_log.append(call_entry)
            return {"status": "error", "error": err_msg}

        if not isinstance(arguments, dict):
            err_msg = "Invalid arguments: must be a dictionary"
            call_entry.update({"status": "error", "error": err_msg, "duration_ms": 0.0})
            self.telemetry_log.append(call_entry)
            return {"status": "error", "error": err_msg}

        try:
            if hasattr(self.server, "_tool_manager"):
                tool = self.server._tool_manager.get_tool(tool_name)
                if not tool:
                    err_msg = f"Unknown MCP tool: '{tool_name}'"
                    call_entry.update({"status": "tool_error", "error": err_msg, "duration_ms": 0.0})
                    self.telemetry_log.append(call_entry)
                    return {"status": "error", "error": err_msg}

                validated_args = tool.fn_metadata.validate_arguments(arguments)
                if inspect.iscoroutinefunction(tool.fn):
                    parsed_data = asyncio.run(tool.fn(**validated_args))
                else:
                    parsed_data = tool.fn(**validated_args)
            else:
                async def _call():
                    return await self.server.call_tool(tool_name, arguments)
                mcp_result = asyncio.run(_call())
                parsed_data = {}
                if mcp_result.content and len(mcp_result.content) > 0:
                    raw_text = getattr(mcp_result.content[0], "text", "")
                    try:
                        parsed_data = json.loads(raw_text)
                    except Exception:
                        parsed_data = {"raw_output": raw_text}

            elapsed_ms = round((time.perf_counter() - start_time) * 1000, 2)
            call_entry.update({
                "status": "success",
                "duration_ms": elapsed_ms,
                "output_preview": str(parsed_data)[:120],
            })
            self.telemetry_log.append(call_entry)
            return parsed_data

        except Exception as exc:
            elapsed_ms = round((time.perf_counter() - start_time) * 1000, 2)
            err_msg = f"MCP Protocol Communication Failure: {exc}"
            logger.error("Error executing MCP tool '%s': %s", tool_name, exc, exc_info=True)
            call_entry.update({
                "status": "server_unavailable",
                "error": err_msg,
                "duration_ms": elapsed_ms,
            })
            self.telemetry_log.append(call_entry)
            return {"status": "error", "error": err_msg}

    # --------------------------------------------------------------------------
    # High-level typed client helpers
    # --------------------------------------------------------------------------
    def get_transaction(self, transaction_id: str) -> Dict[str, Any]:
        """Fetch transaction via MCP tool 'get_transaction'."""
        return self.call_tool("get_transaction", {"transaction_id": transaction_id})

    def get_account_history(self, account_id: str) -> Dict[str, Any]:
        """Fetch account history via MCP tool 'get_account_history'."""
        return self.call_tool("get_account_history", {"account_id": account_id})

    def calculate_transaction_deviation(self, transaction_id: str) -> Dict[str, Any]:
        """Compute statistical deviation via MCP tool 'calculate_transaction_deviation'."""
        return self.call_tool("calculate_transaction_deviation", {"transaction_id": transaction_id})

    def identify_transaction_indicators(self, transaction_id: str) -> Dict[str, Any]:
        """Evaluate empirical indicators via MCP tool 'identify_transaction_indicators'."""
        return self.call_tool("identify_transaction_indicators", {"transaction_id": transaction_id})

    def validate_investigation(
        self,
        transaction_id: str,
        reported_score: int,
        priority: str,
        triggered_rule_names: List[str],
    ) -> Dict[str, Any]:
        """Audit risk assessment via MCP tool 'validate_investigation'."""
        return self.call_tool(
            "validate_investigation",
            {
                "transaction_id": transaction_id,
                "reported_score": reported_score,
                "priority": priority,
                "triggered_rule_names": triggered_rule_names,
            },
        )

    def get_telemetry_history(self) -> List[Dict[str, Any]]:
        """Return full history of all MCP tool calls executed during session."""
        return list(self.telemetry_log)
