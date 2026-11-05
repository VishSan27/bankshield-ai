"""BankShield AI Configuration Settings.

Provides centralized configuration for risk scoring weights, priority thresholds,
anomaly detection thresholds, data paths, and application constants.
"""

from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Dict, Tuple


class PriorityLevel(str, Enum):
    """Investigation priority levels for flagged transactions."""

    LOW = "Low"
    MEDIUM = "Medium"
    HIGH = "High"
    CRITICAL = "Critical"


@dataclass(frozen=True)
class RiskWeights:
    """Weights allocated to various risk indicators (must sum to 100).

    Attributes:
        amount: Weight assigned to amount deviation anomalies.
        frequency: Weight assigned to rapid transaction velocity.
        new_merchant: Weight assigned to transactions with previously unseen merchants.
        failed_attempts: Weight assigned to prior consecutive failed authorization attempts.
        unusual_time: Weight assigned to transactions occurring outside typical hours.
    """

    amount: float = 30.0
    frequency: float = 25.0
    new_merchant: float = 20.0
    failed_attempts: float = 15.0
    unusual_time: float = 10.0

    def __post_init__(self) -> None:
        """Validate that risk weights sum to 100.0."""
        total = (
            self.amount
            + self.frequency
            + self.new_merchant
            + self.failed_attempts
            + self.unusual_time
        )
        if abs(total - 100.0) > 1e-6:
            raise ValueError(f"Risk weights must sum to 100.0, got {total}")

    def as_dict(self) -> Dict[str, float]:
        """Return risk weights as a dictionary."""
        return {
            "amount": self.amount,
            "frequency": self.frequency,
            "new_merchant": self.new_merchant,
            "failed_attempts": self.failed_attempts,
            "unusual_time": self.unusual_time,
        }


@dataclass(frozen=True)
class PriorityThresholds:
    """Score thresholds defining investigation priority tiers.

    Tiers:
        Low: 0 - 29
        Medium: 30 - 59
        High: 60 - 79
        Critical: 80 - 100
    """

    low_min: float = 0.0
    low_max: float = 29.0
    medium_min: float = 30.0
    medium_max: float = 59.0
    high_min: float = 60.0
    high_max: float = 79.0
    critical_min: float = 80.0
    critical_max: float = 100.0

    def get_priority(self, score: float) -> PriorityLevel:
        """Map a numeric risk score (0-100) to a PriorityLevel.

        Args:
            score: Computed risk score between 0 and 100.

        Returns:
            PriorityLevel representing investigation urgency.
        """
        if score >= self.critical_min:
            return PriorityLevel.CRITICAL
        elif score >= self.high_min:
            return PriorityLevel.HIGH
        elif score >= self.medium_min:
            return PriorityLevel.MEDIUM
        else:
            return PriorityLevel.LOW


@dataclass(frozen=True)
class DetectionThresholds:
    """Configurable numerical thresholds for anomaly detection rules.

    Attributes:
        amount_deviation_threshold: Number of standard deviations above mean to flag amount.
        high_amount_multiplier: Multiplier over historical average to consider an amount extreme.
        frequency_threshold: Multiplier over historical daily frequency to flag burst velocity.
        unusual_hours_start: Starting hour (inclusive, 24-hr format) for nocturnal window.
        unusual_hours_end: Ending hour (inclusive, 24-hr format) for nocturnal window.
        failed_attempts_threshold: Minimum failed authorization attempts to trigger alert.
    """

    amount_deviation_threshold: float = 3.0
    high_amount_multiplier: float = 5.0
    frequency_threshold: float = 3.0
    unusual_hours_start: int = 2
    unusual_hours_end: int = 5
    failed_attempts_threshold: int = 3

    @property
    def unusual_hours_range(self) -> Tuple[int, int]:
        """Return (start, end) tuple representing nocturnal window."""
        return (self.unusual_hours_start, self.unusual_hours_end)

    def is_unusual_hour(self, hour: int) -> bool:
        """Check if an hour falls within the unusual nocturnal window [start, end]."""
        return self.unusual_hours_start <= hour <= self.unusual_hours_end


@dataclass
class PathSettings:
    """Filesystem paths for data files and project directories."""

    root_dir: Path = field(
        default_factory=lambda: Path(__file__).resolve().parent.parent
    )

    @property
    def data_dir(self) -> Path:
        """Path to data directory."""
        return self.root_dir / "data"

    @property
    def transactions_file(self) -> Path:
        """Path to transactions CSV file."""
        return self.data_dir / "transactions.csv"

    @property
    def account_profiles_file(self) -> Path:
        """Path to account profiles CSV file."""
        return self.data_dir / "account_profiles.csv"


@dataclass
class AppConfig:
    """General application settings."""

    app_name: str = "BankShield AI"
    version: str = "1.0.0"
    random_seed: int = 42
    debug: bool = False
    log_level: str = "INFO"


@dataclass
class Settings:
    """Global configuration aggregator for BankShield AI."""

    app: AppConfig = field(default_factory=AppConfig)
    paths: PathSettings = field(default_factory=PathSettings)
    weights: RiskWeights = field(default_factory=RiskWeights)
    priorities: PriorityThresholds = field(default_factory=PriorityThresholds)
    detection: DetectionThresholds = field(default_factory=DetectionThresholds)


# Global singleton instance for convenient imports across modules
settings = Settings()

# Direct configuration constants
RISK_WEIGHTS = {
    'high_amount': 30,
    'unusual_frequency': 25,
    'new_merchant': 20,
    'failed_attempts': 15,
    'unusual_time': 10,
}

PRIORITY_THRESHOLDS = [
    (80, 'Critical'),
    (60, 'High'),
    (30, 'Medium'),
    (0, 'Low'),
]

AMOUNT_DEVIATION_THRESHOLD = 3.0
FREQUENCY_MULTIPLIER_THRESHOLD = 3.0
FAILED_ATTEMPTS_THRESHOLD = 3
UNUSUAL_HOURS_START = 0
UNUSUAL_HOURS_END = 5

TRANSACTIONS_PATH = 'data/transactions.csv'
ACCOUNT_PROFILES_PATH = 'data/account_profiles.csv'
