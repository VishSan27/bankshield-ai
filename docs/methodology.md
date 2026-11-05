# BankShield AI: Methodology, LangGraph State Machine & MCP Protocol

This document outlines the mathematical foundation, LangGraph state machine formalization, Model Context Protocol (MCP) specification, scoring formulas, and ethical guardrails utilized by **BankShield AI**.

---

## 1. LangGraph State Machine Formalization

The transaction investigation lifecycle is modeled as a directed acyclic graph with a single bounded self-correction cycle:

$$\mathcal{G} = (\mathcal{V}, \mathcal{E}, \mathcal{S})$$

Where $\mathcal{S}$ is the shared typed state (`InvestigationState`), $\mathcal{V}$ is the set of graph nodes, and $\mathcal{E}$ represents directed edges.

### Graph Nodes ($\mathcal{V}$):
1. $v_{\text{planner}}$: Decomposes target goal into 6-step roadmap.
2. $v_{\text{research}}$: Connects to local MCP server to execute empirical tools.
3. $v_{\text{risk}}$: Evaluates deterministic rule engine ($S \le 100$).
4. $v_{\text{critic}}$: Independent mathematical audit gate.
5. $v_{\text{correct}}$: Self-correction recalculation node (activated upon validation failure).
6. $v_{\text{report}}$: Dossier synthesis node producing 16-field compliance reports.

### Conditional Routing Function ($\delta$):
Let $V_{\text{status}} \in \{\text{VALIDATED}, \text{VALIDATION\_FAILED}\}$, $k_{\text{retry}} \in \mathbb{N}$, and $k_{\text{max}} = 1$. The transition function from $v_{\text{critic}}$ is defined as:

$$\delta(v_{\text{critic}}, \mathcal{S}) = \begin{cases}
v_{\text{correct}}, & \text{if } V_{\text{status}} = \text{VALIDATION\_FAILED} \land k_{\text{retry}} < k_{\text{max}} \\
v_{\text{report}}, & \text{otherwise}
\end{cases}$$

This guarantees that self-correction is strictly bounded ($k_{\text{retry}} \le 1$), eliminating infinite loops and ensuring deterministic termination.

---

## 2. Model Context Protocol (MCP) Tool Integration

The system adheres to the official **Model Context Protocol (MCP/2.0)** specification via an in-process and stdio-compatible local server (`mcp_server/server.py`) and client (`services/mcp_client.py`).

### Standardized MCP Tools:
1. `get_transaction(transaction_id: string)`:
   - Queries `transactions.csv` and returns transaction attributes.
2. `get_account_history(account_id: string)`:
   - Queries `account_profiles.csv` and chronological transaction logs.
3. `calculate_transaction_deviation(transaction_id: string)`:
   - Computes historical mean $\mu_{\text{amt}}$, sample standard deviation $\sigma_{\text{amt}}$, and $z$-score deviation.
4. `identify_transaction_indicators(transaction_id: string)`:
   - Extracts binary indicators: $I = [I_{\text{amt}}, I_{\text{freq}}, I_{\text{merch}}, I_{\text{fail}}, I_{\text{time}}]$.
5. `validate_investigation(transaction_id: string, reported_score: int, priority: string, triggered_rule_names: array)`:
   - Performs independent Critic recalculation across the MCP boundary.

---

## 3. Mathematical Formulation of Risk Indicators

### A. Amount Baseline Deviation ($z$-Score)
$$\mu_{\text{amt}} = \frac{1}{N} \sum_{i=1}^{N} x_i, \quad \sigma_{\text{amt}} = \sqrt{\frac{1}{N-1} \sum_{i=1}^{N} (x_i - \mu_{\text{amt}})^2}$$

$$z = \frac{x_{\text{target}} - \mu_{\text{amt}}}{\sigma_{\text{amt}}}$$

$$\text{Trigger}_{\text{amount}} = (z \ge 3.0) \lor (x_{\text{target}} \ge 5.0 \times \mu_{\text{amt}})$$

### B. Velocity Burst Multiplier
$$R_{\text{freq}} = \frac{f_{24\text{h}}}{\max(f_{\text{hist}}, 0.5)}$$

$$\text{Trigger}_{\text{frequency}} = (R_{\text{freq}} \ge 3.0)$$

### C. Unseen Merchant Indicator
$$\text{Trigger}_{\text{merchant}} = (m_{\text{target}} \notin \mathcal{M}_{\text{hist}})$$

### D. Failed Authorization Attempts
$$\text{Trigger}_{\text{failed}} = (k_{\text{failed}} \ge 3)$$

### E. Off-Hours Nocturnal Execution Window
$$\text{Trigger}_{\text{time}} = (H(t_{\text{txn}}) \ge 2) \land (H(t_{\text{txn}}) \le 5)$$

---

## 4. Score Aggregation & Priority Mapping

Let $W = [30, 25, 20, 15, 10]$ represent indicator weights summing to 100. The risk score $S$ is capped at 100:

$$S = \min\left(100, \sum_{j=1}^{5} w_j \cdot I_j\right)$$

$$\mathcal{P}(S) = \begin{cases} 
\text{Critical}, & \text{if } S \ge 80 \\
\text{High}, & \text{if } 60 \le S < 80 \\
\text{Medium}, & \text{if } 30 \le S < 60 \\
\text{Low}, & \text{if } 0 \le S < 30 
\end{cases}$$

---

## 5. Human Oversight and Ethical Guardrails

1. **Four-Eyes Separation of Duties:** The entity scoring risk (`risk_assessment_node`) is isolated from the entity auditing risk (`critic_node`), mirroring regulatory compliance standards.
2. **Decision Support, Not Autonomous Sanction:** The system directs human analysts to review specific evidence points. It never executes automated card cancellation or fund clawbacks.
3. **Non-Prejudicial Phrasing:** The word "fraud" is never applied as a definitive label. Objective terminology ("elevated risk score", "baseline deviation") is strictly enforced.
4. **Data Privacy by Design:** 100% synthetic dataset; zero real customer credentials or PII exist in the system.
