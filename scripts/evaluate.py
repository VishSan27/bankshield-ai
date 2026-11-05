"""Evaluation Runner for BankShield AI.

Executes structured synthetic evaluation scenarios against the InvestigationOrchestrator,
computes rule-based scenario agreement, critic audit pass rates, invalid-input handling,
and latency metrics. Saves results to JSON and CSV formats.
"""

import csv
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from orchestration.langgraph_orchestrator import LangGraphInvestigationOrchestrator
from services.data_service import DataService
from services.evaluation_service import TEST_SCENARIOS, run_evaluation


def run_benchmark():
    """Run evaluation benchmarks and generate reports."""
    print("=" * 70)
    print("  BANKSHIELD AI — MULTI-AGENT EVALUATION BENCHMARK SUITE")
    print("=" * 70)
    print(f"Timestamp: {datetime.now(timezone.utc).isoformat()}")
    print("Initializing DataService and InvestigationOrchestrator...")

    data_service = DataService()
    orchestrator = LangGraphInvestigationOrchestrator(data_service=data_service)

    print(f"Loaded {data_service.get_transaction_count()} synthetic transactions.")
    print(f"Running LangGraph evaluation across {len(TEST_SCENARIOS)} benchmark scenarios...\n")

    # Run evaluation
    results = run_evaluation(orchestrator, scenarios=TEST_SCENARIOS)

    # Print summary table
    print("-" * 70)
    print(f"{'Scenario Name':<45} | {'Exp.':<6} | {'Act.':<6} | {'Result':<6}")
    print("-" * 70)
    for res in results.get("results", []):
        name = res["scenario_name"][:43]
        exp_p = res.get("expected_min_priority", "-")[:4]
        act_p = res.get("actual_priority", "-")[:4]
        status = "[PASS]" if res.get("passed") else "[FAIL]"
        print(f"{name:<45} | {exp_p:<6} | {act_p:<6} | {status:<6}")

    print("-" * 70)
    print("\nSUMMARY METRICS:")
    print(f"  Total Scenarios Evaluated:     {results['total_scenarios']}")
    print(f"  Passed Scenarios:              {results['passed']}")
    print(f"  Failed Scenarios:              {results['failed']}")
    print(f"  Scenario Agreement Rate:       {results['validation_pass_rate']}%")
    print(f"  Average Execution Time:        {results['average_execution_time']} ms")
    print(f"  Priority Distribution:         {results['priority_distribution']}")
    print(f"  Invalid-Input Handled:         {results['error_handling_results'].get('handled_gracefully', False)}")

    print("\n" + "=" * 70)
    print("BENCHMARK METHODOLOGY NOTE:")
    print("  This benchmark measures RULE-BASED SCENARIO AGREEMENT against designed")
    print("  synthetic patterns and CRITIC AUDIT INTEGRITY. Because the dataset is")
    print("  synthetic and designed for decision support rather than production fraud")
    print("  detection, real-world precision/recall/F1 metrics are not claimed.")
    print("=" * 70 + "\n")

    # Ensure output directories exist
    output_dir = PROJECT_ROOT / "data"
    output_dir.mkdir(parents=True, exist_ok=True)

    json_path = output_dir / "evaluation_results.json"
    csv_path = output_dir / "evaluation_results.csv"

    # Save JSON report
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print(f"Saved evaluation JSON report to: {json_path}")

    # Save CSV summary
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([
            "scenario_name",
            "transaction_id",
            "passed",
            "duration_ms",
            "expected_min_priority",
            "actual_priority",
            "actual_score",
            "expected_indicators",
            "actual_indicators",
            "failure_reasons",
        ])
        for r in results.get("results", []):
            writer.writerow([
                r["scenario_name"],
                r["transaction_id"],
                r["passed"],
                r["duration_ms"],
                r["expected_min_priority"],
                r["actual_priority"],
                r["actual_score"],
                ";".join(r["expected_indicators"]),
                ";".join(r["actual_indicators"]),
                ";".join(r["failure_reasons"]),
            ])
    print(f"Saved evaluation CSV report to:  {csv_path}\n")

    return results


if __name__ == "__main__":
    benchmark_results = run_benchmark()
    if benchmark_results.get("failed", 0) > 0:
        sys.exit(1)
    sys.exit(0)
