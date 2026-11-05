"""Tests for RiskEngine and RiskRule logic in BankShield AI."""

import pytest
from tools.risk_rules import RiskEngine, RiskRule, create_default_engine


def test_default_engine_creation():
    """Test creating default risk engine and verifying registered rules."""
    engine = create_default_engine()
    assert len(engine.rules) >= 5
    expected_rules = {"high_amount", "unusual_frequency", "new_merchant", "failed_attempts", "unusual_time"}
    assert expected_rules.issubset(set(engine.rules.keys()))


def test_priority_tier_mapping():
    """Test priority tier boundaries."""
    engine = create_default_engine()
    assert engine.get_priority(0) == "Low"
    assert engine.get_priority(29) == "Low"
    assert engine.get_priority(30) == "Medium"
    assert engine.get_priority(59) == "Medium"
    assert engine.get_priority(60) == "High"
    assert engine.get_priority(79) == "High"
    assert engine.get_priority(80) == "Critical"
    assert engine.get_priority(100) == "Critical"


def test_score_capping_at_100():
    """Test that total_score is capped at 100 even if raw_score exceeds 100."""
    engine = create_default_engine()
    # Mock all indicators triggering
    indicators = {
        "high_amount": {"detected": True, "value": 50000, "threshold": 5000, "evidence": "Amount 10x avg"},
        "unusual_frequency": {"detected": True, "value": 8, "threshold": 3, "evidence": "8 txns in 1 hr"},
        "new_merchant": {"detected": True, "value": True, "threshold": True, "evidence": "New merchant"},
        "failed_attempts": {"detected": True, "value": 5, "threshold": 3, "evidence": "5 failed attempts"},
        "unusual_time": {"detected": True, "value": "03:30", "threshold": "02:00-05:00", "evidence": "3:30 AM"},
    }
    result = engine.evaluate(indicators)
    assert result.total_score <= 100
    assert result.raw_score == 100
    assert result.priority == "Critical"
    assert len(result.triggered_rules) == 5


def test_missing_data_discrimination():
    """Test that missing indicators are separated from untriggered indicators."""
    engine = create_default_engine()
    # Only provide 2 indicators out of 5
    indicators = {
        "high_amount": {"detected": True, "value": 15000, "threshold": 5000, "evidence": "High amount"},
        "new_merchant": {"detected": False, "value": False, "threshold": True, "evidence": "Known merchant"},
    }
    result = engine.evaluate(indicators)

    assert result.total_score == 30
    assert result.priority == "Medium"
    assert len(result.triggered_rules) == 1
    assert result.triggered_rules[0]["rule_name"] == "high_amount"

    # untriggered
    untriggered_names = [r["rule_name"] for r in result.untriggered_rules]
    assert "new_merchant" in untriggered_names

    # missing
    missing_names = [r["rule_name"] for r in result.missing_data_rules]
    assert "unusual_frequency" in missing_names
    assert "failed_attempts" in missing_names
    assert "unusual_time" in missing_names


def test_clean_zero_risk_evaluation():
    """Test when no rules are triggered."""
    engine = create_default_engine()
    indicators = {
        "high_amount": {"detected": False, "value": 100, "threshold": 5000},
        "unusual_frequency": {"detected": False, "value": 1, "threshold": 3},
        "new_merchant": {"detected": False, "value": False, "threshold": True},
        "failed_attempts": {"detected": False, "value": 0, "threshold": 3},
        "unusual_time": {"detected": False, "value": "14:00", "threshold": "02:00-05:00"},
    }
    result = engine.evaluate(indicators)
    assert result.total_score == 0
    assert result.priority == "Low"
    assert len(result.triggered_rules) == 0
    assert len(result.untriggered_rules) == 5
    assert len(result.missing_data_rules) == 0
