"""Orchestration package for BankShield AI.

Exports the LangGraphInvestigationOrchestrator and InvestigationState.
"""

from .investigation_state import InvestigationState
from .langgraph_orchestrator import (
    LangGraphInvestigationOrchestrator,
    build_investigation_graph,
)
from .orchestrator import InvestigationOrchestrator

# Default orchestrator uses LangGraph
DefaultOrchestrator = LangGraphInvestigationOrchestrator

__all__ = [
    "InvestigationState",
    "LangGraphInvestigationOrchestrator",
    "InvestigationOrchestrator",
    "DefaultOrchestrator",
    "build_investigation_graph",
]
