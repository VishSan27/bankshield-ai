"""Model Context Protocol (MCP) Server for BankShield AI.

Exposes standardized banking risk investigation and validation tools over MCP.
"""

from .server import create_mcp_server, get_mcp_server, mcp_server

__all__ = ["create_mcp_server", "get_mcp_server", "mcp_server"]
