"""Configuration package for BankShield AI."""

from .settings import (
    AMOUNT_DEVIATION_THRESHOLD,
    FAILED_ATTEMPTS_THRESHOLD,
    FREQUENCY_MULTIPLIER_THRESHOLD,
    PRIORITY_THRESHOLDS,
    RISK_WEIGHTS,
    TRANSACTIONS_PATH,
    ACCOUNT_PROFILES_PATH,
    UNUSUAL_HOURS_END,
    UNUSUAL_HOURS_START,
    DetectionThresholds,
    PriorityLevel,
    PriorityThresholds,
    RiskWeights,
    Settings,
    settings,
)

__all__ = [
    "AMOUNT_DEVIATION_THRESHOLD",
    "FAILED_ATTEMPTS_THRESHOLD",
    "FREQUENCY_MULTIPLIER_THRESHOLD",
    "PRIORITY_THRESHOLDS",
    "RISK_WEIGHTS",
    "TRANSACTIONS_PATH",
    "ACCOUNT_PROFILES_PATH",
    "UNUSUAL_HOURS_END",
    "UNUSUAL_HOURS_START",
    "DetectionThresholds",
    "PriorityLevel",
    "PriorityThresholds",
    "RiskWeights",
    "Settings",
    "settings",
]

