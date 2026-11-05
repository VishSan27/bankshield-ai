# BankShield AI: LangGraph and Model Context Protocol (MCP) Architecture

This document details the multi-agent system architecture for **BankShield AI**, featuring **LangGraph** for typed state machine orchestration and the **Model Context Protocol (MCP)** for standardized banking tool access.

---

## 1. System Architecture: LangGraph State Machine & MCP Server

```mermaid
flowchart TD
    %% User and UI Layer
    User([Compliance Analyst]) <--> UI[Streamlit Enterprise Interface]
    
    %% LangGraph Orchestration Layer
    UI -->|1. Submit Transaction ID / Batch| Orch[LangGraph Investigation Orchestrator]
    
    subgraph LangGraphStateMachine [LangGraph State Machine: StateGraph]
        direction TB
        
        StartNode([__start__]) --> PlannerNode[planner_node<br/><i>PlannerAgent: Dynamic 6-Step Roadmap</i>]
        PlannerNode --> ResearchNode[research_node<br/><i>ResearchAgent: Tool Operator</i>]
        ResearchNode --> RiskNode[risk_assessment_node<br/><i>RiskAssessmentAgent: 0-100 Scoring</i>]
        RiskNode --> CriticNode[critic_node<br/><i>CriticAgent: 9-Gate Quality Audit</i>]
        
        %% Conditional Branching
        CriticNode -->|VALIDATED| ReportNode[report_generation_node<br/><i>Compile 16-Field Dossier</i>]
        CriticNode -->|VALIDATION_FAILED & retry < max_retries| CorrectNode[correction_node<br/><i>Self-Correction Recalibration</i>]
        CriticNode -->|VALIDATION_FAILED & retry >= max_retries| ReportNode
        
        CorrectNode --> CriticNode
        ReportNode --> EndNode([__end__])
    end
    
    Orch <--> LangGraphStateMachine
    
    %% Model Context Protocol Layer
    subgraph MCPLayer [Model Context Protocol: Local Client & Server Architecture]
        direction TB
        MCPClient[BankShieldMCPClient<br/><i>services/mcp_client.py</i>]
        MCPServer[Local MCPServer<br/><i>mcp_server/server.py</i>]
        
        MCPClient <-->|Official MCP JSON-RPC Protocol / MCP 2.0| MCPServer
        
        subgraph MCPTools [Standardized MCP Tool Suite]
            T1[get_transaction<br/><i>Fetch transaction details & amount</i>]
            T2[get_account_history<br/><i>Retrieve account baseline & profile</i>]
            T3[calculate_transaction_deviation<br/><i>Compute sample mean, std dev, z-score</i>]
            T4[identify_transaction_indicators<br/><i>Evaluate 5 empirical risk indicators</i>]
            T5[validate_investigation<br/><i>Independent mathematical recalculation</i>]
        end
        MCPServer <--> MCPTools
        MCPTools <--> CSV[(Synthetic Banking CSVs<br/>transactions.csv & account_profiles.csv)]
    end
    
    ResearchNode <-->|Invoke MCP Tools 1, 2, 3, 4| MCPClient
    CriticNode <-->|Invoke MCP Tool 5| MCPClient
```

---

## 2. Sequence Diagram: LangGraph State Transitions & MCP Tool Calls

```mermaid
sequenceDiagram
    autonumber
    actor Analyst as Compliance Analyst
    participant UI as Streamlit UI
    participant Orch as LangGraphOrchestrator
    participant State as InvestigationState
    participant MCP as BankShieldMCPClient
    participant Server as MCPServer (Local)
    participant Rules as DeterministicRulesEngine

    Analyst->>UI: Select Transaction (e.g. TXN-10198) & Click Run
    UI->>Orch: investigate_transaction(transaction_id)
    Orch->>State: Initialize Typed InvestigationState (retry_count=0, max_retries=1)

    %% Node 1: Planner
    Note over Orch,State: LangGraph Node: planner_node
    Orch->>State: Run planner_node -> Create 6-step roadmap in state['plan']

    %% Node 2: Research via MCP
    Note over Orch,Server: LangGraph Node: research_node (MCP Invocations)
    Orch->>MCP: get_transaction(tx_id)
    MCP->>Server: MCP call_tool('get_transaction')
    Server-->>MCP: CallToolResult (data, amount, merchant)
    
    Orch->>MCP: get_account_history(acc_id)
    MCP->>Server: MCP call_tool('get_account_history')
    Server-->>MCP: CallToolResult (history, profile)
    
    Orch->>MCP: calculate_transaction_deviation(tx_id)
    MCP->>Server: MCP call_tool('calculate_transaction_deviation')
    Server-->>MCP: CallToolResult (mean, std dev, z-score=24.19)
    
    Orch->>MCP: identify_transaction_indicators(tx_id)
    MCP->>Server: MCP call_tool('identify_transaction_indicators')
    Server-->>MCP: CallToolResult (high_amount=True, new_merchant=True)
    Orch->>State: Store evidence bundle & tool telemetry in state

    %% Node 3: Risk Assessment
    Note over Orch,Rules: LangGraph Node: risk_assessment_node
    Orch->>Rules: Evaluate 5 deterministic weighted rules (Max 100)
    Rules-->>Orch: RiskResult (score=60/100, priority=High, 3 rules triggered)
    Orch->>State: Update state['risk_score'], state['priority'], state['triggered_rules']

    %% Node 4: Critic & Audit Gate
    Note over Orch,Server: LangGraph Node: critic_node (Independent Recalculation)
    Orch->>MCP: validate_investigation(tx_id, score=60, priority=High, rules)
    MCP->>Server: MCP call_tool('validate_investigation')
    Server-->>MCP: Audit Result (recalculated_score=60, score_matches=True)
    Orch->>State: Update state['validation_status'] = 'VALIDATED'

    %% Conditional Routing
    alt Validation Status == VALIDATED
        Note over Orch,State: Conditional Route: report_generation_node
        Orch->>State: Run report_generation_node -> Compile 16-field dossier
    else Validation Status == VALIDATION_FAILED (Discrepancy) and retry < 1
        Note over Orch,State: Conditional Route: correction_node (Self-Correction Loop)
        Orch->>State: Run correction_node -> Recalibrate score & increment retry_count
        Orch->>State: Route back to critic_node for re-audit
    end

    Orch-->>UI: Return final state bundle + node history + MCP call logs
    UI-->>Analyst: Render Executive Dossier, Gauge Chart & Node Trace
```

---

## 3. LangGraph State Machine Specification

### Shared Typed State (`orchestration/investigation_state.py`)
All graph nodes communicate strictly via the typed `InvestigationState` schema:

```python
class InvestigationState(TypedDict, total=False):
    investigation_id: str
    transaction_id: str
    objective: str
    created_at: str

    # Planning
    plan: Dict[str, Any]
    current_step_index: int

    # Research & MCP Tool Output
    transaction_data: Optional[Dict[str, Any]]
    account_profile: Optional[Dict[str, Any]]
    account_history: List[Dict[str, Any]]
    deviation_analysis: Dict[str, Any]
    indicators: Dict[str, Any]
    evidence: Dict[str, Any]
    mcp_tool_calls: List[Dict[str, Any]]

    # Risk Assessment
    risk_result: Dict[str, Any]
    risk_score: int
    raw_score: int
    priority: str
    triggered_rules: List[Dict[str, Any]]
    recommendation: str

    # Critic & Validation
    validation_result: Dict[str, Any]
    validation_status: str  # 'VALIDATED', 'NEEDS_REVIEW', 'VALIDATION_FAILED'
    score_matches: bool
    recalculated_score: int
    recalculated_priority: str

    # Self-Correction Governance
    retry_count: int
    max_retries: int
    correction_applied: bool
    correction_history: List[Dict[str, Any]]

    # Final Dossier & Trace
    final_report: Dict[str, Any]
    status: str
    errors: List[str]
    node_history: List[str]
    execution_trace: List[Dict[str, Any]]
    total_duration_ms: float
```

### Graph Nodes and Responsibilities:
1. **`planner_node`:** Decomposes request into an ordered 6-step plan (`validate_input`, `retrieve_data`, `calculate_indicators`, `assess_risk`, `validate_results`, `generate_report`).
2. **`research_node`:** Connects to the local MCP server via `BankShieldMCPClient`. Executes `get_transaction`, `get_account_history`, `calculate_transaction_deviation`, and `identify_transaction_indicators`. Populates the evidence bundle and logs telemetry.
3. **`risk_assessment_node`:** Evaluates 5 deterministic rules with configurable weights (Amount: 30, Velocity: 25, Unseen Merchant: 20, Failed Authorizations: 15, Nocturnal Window: 10). Caps score at 100.
4. **`critic_node`:** Independently audits calculations using the Critic Agent and invokes the MCP tool `validate_investigation`. Sets `validation_status` to `VALIDATED` or `VALIDATION_FAILED`.
5. **`correction_node`:** Self-correction node entered only when validation fails. Adjusts score to the independently verified recalculation, logs a correction event, increments `retry_count`, and routes back to `critic_node`.
6. **`report_generation_node`:** Compiles the complete 16-field standardized dossier via `tools/report_tools.py`.

---

## 4. Model Context Protocol (MCP) Server & Client Specification

### Local Server Architecture (`mcp_server/server.py`)
Implemented using the official Python MCP SDK (`mcp.server.mcpserver.MCPServer`).

| Tool Name | Input Schema | Output Schema | Purpose |
|---|---|---|---|
| `get_transaction` | `transaction_id: string` (required) | `status: string, data: object` | Fetches transaction amount, timestamp, merchant, and channel from synthetic CSV. |
| `get_account_history` | `account_id: string` (required) | `status: string, record_count: int, profile: object, history: array` | Retrieves chronological records and profile baseline for account. |
| `calculate_transaction_deviation` | `transaction_id: string` (required) | `status: string, deviation: object (mean, std, z_score)` | Computes statistical spending deviation against historical mean. |
| `identify_transaction_indicators` | `transaction_id: string` (required) | `status: string, indicators: object` | Evaluates the 5 empirical risk indicators, separating missing data from absent risk. |
| `validate_investigation` | `transaction_id: string, reported_score: int, priority: string, triggered_rule_names: array` | `status: string, score_matches: bool, recalculated_score: int, checks: array` | Independent mathematical audit tool that recalculates scores and checks priority mapping. |

### MCP Client (`services/mcp_client.py`)
- Adheres to Model Context Protocol standards.
- Enforces argument schema validation before dispatch.
- Measures round-trip duration in milliseconds.
- Captures protocol logs with timestamps, argument payloads, and protocol versions (`MCP/2.0`).
- Recovers gracefully from tool errors and missing records.
