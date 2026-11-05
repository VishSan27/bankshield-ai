"""Optional Local LLM Integration Service for BankShield AI.

Provides optional local language model synthesis via Ollama (e.g. llama3, mistral, qwen)
for generating natural-language investigation narratives from already validated evidence.

Operating Principles:
1. Purely optional: Works 100% offline in deterministic mode when Ollama is absent.
2. Zero hallucinated evidence: Strictly summarizes structured evidence output from agents.
3. Deterministic scores: LLM never computes or alters risk scores or priority tiers.
4. Transparent mode reporting: Accurately declares 'LOCAL_LLM_MODE' or 'DETERMINISTIC_RULES_MODE'.
"""

import json
import logging
from typing import Any, Dict, Optional, Tuple
import urllib.request
import urllib.error

logger = logging.getLogger(__name__)

DEFAULT_OLLAMA_URL = "http://localhost:11434"
DEFAULT_MODEL = "llama3:latest"


def check_ollama_status(base_url: str = DEFAULT_OLLAMA_URL) -> Tuple[bool, list[str]]:
    """Check if local Ollama daemon is reachable and list available local models.

    Args:
        base_url: Base HTTP endpoint for Ollama server.

    Returns:
        Tuple of (is_available, list_of_installed_model_names).
    """
    try:
        req = urllib.request.Request(f"{base_url}/api/tags", headers={"User-Agent": "BankShield-AI"})
        with urllib.request.urlopen(req, timeout=1.0) as resp:
            if resp.status == 200:
                data = json.loads(resp.read().decode("utf-8"))
                models = [m.get("name", "") for m in data.get("models", [])]
                return True, models
    except Exception:
        pass
    return False, []


def generate_executive_narrative(
    report: Dict[str, Any],
    use_llm_if_available: bool = True,
    model_name: Optional[str] = None,
    base_url: str = DEFAULT_OLLAMA_URL,
) -> Dict[str, Any]:
    """Generate executive narrative summary for an investigation report.

    If Ollama is available and enabled, queries the local model to draft a human-analyst
    briefing based strictly on validated facts. Otherwise, uses deterministic template synthesis.

    Args:
        report: Validated investigation dossier dictionary.
        use_llm_if_available: Flag indicating whether to attempt Ollama invocation.
        model_name: Specific model name to target. If None, picks first available or default.
        base_url: Ollama API endpoint.

    Returns:
        Dictionary with 'mode', 'model', 'narrative', and 'success'.
    """
    txn_id = report.get("transaction_id", "UNKNOWN")
    acc_id = report.get("account_id", "UNKNOWN")
    risk_score = report.get("risk_score", 0)
    priority = report.get("priority", "Low")
    status = report.get("status", "VALIDATED")
    recommendation = report.get("recommendation", "Review transaction.")
    triggered_rules = report.get("triggered_rules", [])

    rule_details = []
    for r in triggered_rules:
        name = r.get("rule_name", r.get("name", "Unknown"))
        exp = r.get("explanation", "")
        pts = r.get("score_contribution", r.get("weight", 0))
        rule_details.append(f"- {name} (+{pts} pts): {exp}")
    rules_text = "\n".join(rule_details) if rule_details else "No risk indicators triggered."

    # Deterministic fallback text
    fallback_narrative = (
        f"BankShield AI Decision Support Briefing for Transaction {txn_id} (Account: {acc_id}).\n\n"
        f"Risk Score: {risk_score}/100 | Priority Tier: {priority} | Critic Status: {status}\n\n"
        f"Triggered Evidence-Based Rules:\n{rules_text}\n\n"
        f"Operational Recommendation: {recommendation}\n\n"
        f"[Note: Generated deterministically via rule-based analytical agents.]"
    )

    if not use_llm_if_available:
        return {
            "mode": "DETERMINISTIC_RULES_MODE",
            "model": "None (Deterministic Engine)",
            "narrative": fallback_narrative,
            "success": True,
        }

    is_online, installed_models = check_ollama_status(base_url)
    if not is_online:
        return {
            "mode": "DETERMINISTIC_RULES_MODE (Ollama daemon offline)",
            "model": "None (Fallback Engine)",
            "narrative": fallback_narrative,
            "success": True,
        }

    chosen_model = model_name or (installed_models[0] if installed_models else DEFAULT_MODEL)

    prompt = (
        f"You are BankShield AI, a compliance and risk investigation assistant for banking analysts.\n"
        f"Summarize the following verified investigation evidence into an objective 2-paragraph decision-support briefing.\n"
        f"CRITICAL RULES: Do NOT invent transactions, do NOT claim definitive fraud, refer only to provided indicators.\n\n"
        f"Transaction ID: {txn_id}\n"
        f"Account ID: {acc_id}\n"
        f"Risk Score: {risk_score}/100\n"
        f"Priority Tier: {priority}\n"
        f"Critic Validation Status: {status}\n"
        f"Triggered Rules:\n{rules_text}\n"
        f"Recommended Action: {recommendation}\n\n"
        f"Draft the briefing concisely for the investigating analyst:"
    )

    try:
        payload = json.dumps({
            "model": chosen_model,
            "prompt": prompt,
            "stream": False,
        }).encode("utf-8")

        req = urllib.request.Request(
            f"{base_url}/api/generate",
            data=payload,
            headers={"Content-Type": "application/json", "User-Agent": "BankShield-AI"},
        )
        with urllib.request.urlopen(req, timeout=10.0) as resp:
            if resp.status == 200:
                resp_data = json.loads(resp.read().decode("utf-8"))
                llm_response = resp_data.get("response", "").strip()
                if llm_response:
                    return {
                        "mode": "LOCAL_LLM_MODE",
                        "model": chosen_model,
                        "narrative": llm_response,
                        "success": True,
                    }
    except Exception as exc:
        logger.debug("Ollama inference bypassed due to: %s", exc)

    return {
        "mode": "DETERMINISTIC_RULES_MODE (Ollama model unavailable)",
        "model": "Fallback Engine",
        "narrative": fallback_narrative,
        "success": True,
    }
