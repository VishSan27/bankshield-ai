"""Data Service for BankShield AI.

Provides resilient access, schema validation, and caching for transactions
and account profiles from underlying CSV datasets.
"""

from datetime import datetime, timedelta, timezone
import logging
from pathlib import Path
import random
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd

# Set fixed seed for reproducibility
random.seed(42)

logger = logging.getLogger(__name__)

# Required schemas for dataset validation
REQUIRED_TRANSACTION_COLUMNS: List[str] = [
    "transaction_id",
    "account_id",
    "timestamp",
    "amount",
]

REQUIRED_ACCOUNT_PROFILE_COLUMNS: List[str] = [
    "account_id",
]


class DataService:
    """Service for loading, validating, caching, and querying banking transaction datasets."""

    def __init__(
        self,
        transactions_path: Optional[str | Path] = None,
        account_profiles_path: Optional[str | Path] = None,
        auto_load: bool = True,
    ) -> None:
        """Initialize the DataService with paths to data files.

        Args:
            transactions_path: Optional path to transactions.csv. Defaults to project data dir.
            account_profiles_path: Optional path to account_profiles.csv. Defaults to project data dir.
            auto_load: Whether to immediately load and validate data into cache on instantiation.
        """
        project_root = Path(__file__).resolve().parent.parent

        if transactions_path is not None:
            self.transactions_path = Path(transactions_path)
        else:
            self.transactions_path = project_root / "data" / "transactions.csv"

        if account_profiles_path is not None:
            self.account_profiles_path = Path(account_profiles_path)
        else:
            self.account_profiles_path = project_root / "data" / "account_profiles.csv"

        self._transactions_df: Optional[pd.DataFrame] = None
        self._profiles_df: Optional[pd.DataFrame] = None

        # In-memory caches for fast lookup
        self._transactions_cache: Dict[str, Dict[str, Any]] = {}
        self._profiles_cache: Dict[str, Dict[str, Any]] = {}
        self._account_history_cache: Dict[str, List[Dict[str, Any]]] = {}
        self._is_loaded: bool = False

        if auto_load:
            self.load_data()

    def _clean_record(self, record: Dict[str, Any]) -> Dict[str, Any]:
        """Convert NaN / pd.NA values to None and ensure Python native types."""
        cleaned: Dict[str, Any] = {}
        for k, v in record.items():
            if pd.isna(v):
                cleaned[k] = None
            elif hasattr(v, "item"):  # numpy scalars like np.int64, np.float64, np.bool_
                cleaned[k] = v.item()
            elif isinstance(v, (pd.Timestamp, datetime)):
                cleaned[k] = v.strftime("%Y-%m-%d %H:%M:%S")
            else:
                cleaned[k] = v
        return cleaned

    def _validate_schema(
        self,
        df: pd.DataFrame,
        required_columns: List[str],
        dataset_name: str,
    ) -> None:
        """Validate that a DataFrame contains all required columns.

        Args:
            df: DataFrame to validate.
            required_columns: List of column names that must exist.
            dataset_name: Name of dataset for error reporting.

        Raises:
            ValueError: If any required column is missing.
        """
        missing_columns = [col for col in required_columns if col not in df.columns]
        if missing_columns:
            error_msg = (
                f"Schema validation failed for {dataset_name}: "
                f"missing required columns {missing_columns}"
            )
            logger.error(error_msg)
            raise ValueError(error_msg)

    def load_data(self) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Load and validate transactions and account profile CSV files into memory.

        Handles missing files, corrupt rows, and schema discrepancies gracefully.

        Returns:
            Tuple of (transactions_df, account_profiles_df).
        """
        logger.info(
            "Loading datasets from '%s' and '%s'",
            self.transactions_path,
            self.account_profiles_path,
        )

        # 1. Load Transactions
        if not self.transactions_path.exists():
            logger.warning(
                "Transactions file '%s' does not exist. Initializing empty DataFrame.",
                self.transactions_path,
            )
            self._transactions_df = pd.DataFrame(columns=REQUIRED_TRANSACTION_COLUMNS)
        else:
            try:
                self._transactions_df = pd.read_csv(self.transactions_path)
                self._validate_schema(
                    self._transactions_df,
                    REQUIRED_TRANSACTION_COLUMNS,
                    "transactions",
                )
                logger.info(
                    "Successfully loaded %d transactions from '%s'",
                    len(self._transactions_df),
                    self.transactions_path,
                )
            except Exception as exc:
                logger.error(
                    "Error reading transactions file '%s': %s. Falling back to empty DataFrame.",
                    self.transactions_path,
                    exc,
                )
                self._transactions_df = pd.DataFrame(columns=REQUIRED_TRANSACTION_COLUMNS)

        # 2. Load Account Profiles
        if not self.account_profiles_path.exists():
            logger.warning(
                "Account profiles file '%s' does not exist. Initializing empty DataFrame.",
                self.account_profiles_path,
            )
            self._profiles_df = pd.DataFrame(columns=REQUIRED_ACCOUNT_PROFILE_COLUMNS)
        else:
            try:
                self._profiles_df = pd.read_csv(self.account_profiles_path)
                self._validate_schema(
                    self._profiles_df,
                    REQUIRED_ACCOUNT_PROFILE_COLUMNS,
                    "account_profiles",
                )
                logger.info(
                    "Successfully loaded %d account profiles from '%s'",
                    len(self._profiles_df),
                    self.account_profiles_path,
                )
            except Exception as exc:
                logger.error(
                    "Error reading account profiles file '%s': %s. Falling back to empty DataFrame.",
                    self.account_profiles_path,
                    exc,
                )
                self._profiles_df = pd.DataFrame(columns=REQUIRED_ACCOUNT_PROFILE_COLUMNS)

        # 3. Populate In-Memory Caches
        self._transactions_cache.clear()
        self._profiles_cache.clear()
        self._account_history_cache.clear()

        if self._transactions_df is not None and not self._transactions_df.empty:
            for record in self._transactions_df.to_dict(orient="records"):
                cleaned = self._clean_record(record)
                txn_id = str(cleaned.get("transaction_id", "")).strip()
                if txn_id:
                    self._transactions_cache[txn_id] = cleaned

                acc_id = str(cleaned.get("account_id", "")).strip()
                if acc_id:
                    if acc_id not in self._account_history_cache:
                        self._account_history_cache[acc_id] = []
                    self._account_history_cache[acc_id].append(cleaned)

            # Sort account histories chronologically by timestamp
            for acc_id, hist in self._account_history_cache.items():
                hist.sort(key=lambda t: str(t.get("timestamp", "")))

        if self._profiles_df is not None and not self._profiles_df.empty:
            for record in self._profiles_df.to_dict(orient="records"):
                cleaned = self._clean_record(record)
                acc_id = str(cleaned.get("account_id", "")).strip()
                if acc_id:
                    self._profiles_cache[acc_id] = cleaned

        self._is_loaded = True
        return self._transactions_df, self._profiles_df

    def get_transaction(self, transaction_id: str) -> Optional[Dict[str, Any]]:
        """Retrieve a transaction record by its unique identifier.

        Args:
            transaction_id: The unique transaction identifier (e.g. 'TXN-10001').

        Returns:
            Dict containing transaction attributes, or None if not found.
        """
        if not self._is_loaded:
            self.load_data()

        if not isinstance(transaction_id, str) or not transaction_id.strip():
            return None

        record = self._transactions_cache.get(transaction_id.strip())
        return dict(record) if record is not None else None

    def get_account_profile(self, account_id: str) -> Optional[Dict[str, Any]]:
        """Retrieve an account profile by its unique account identifier.

        Args:
            account_id: The unique account identifier (e.g. 'ACC-1001').

        Returns:
            Dict containing account profile attributes, or None if not found.
        """
        if not self._is_loaded:
            self.load_data()

        if not isinstance(account_id, str) or not account_id.strip():
            return None

        record = self._profiles_cache.get(account_id.strip())
        return dict(record) if record is not None else None

    def get_account_history(self, account_id: str) -> List[Dict[str, Any]]:
        """Retrieve the historical transaction records for an account sorted chronologically.

        Args:
            account_id: The account identifier.

        Returns:
            List of transaction dictionaries for the account. Returns empty list if none exist.
        """
        if not self._is_loaded:
            self.load_data()

        if not isinstance(account_id, str) or not account_id.strip():
            return []

        history = self._account_history_cache.get(account_id.strip(), [])
        return [dict(tx) for tx in history]

    def get_recent_transactions(
        self,
        account_id: str,
        hours: int = 24,
        reference_time: Optional[datetime | str] = None,
    ) -> List[Dict[str, Any]]:
        """Retrieve transactions for an account within a specified recent time window.

        If reference_time is not provided, evaluates relative to the latest transaction timestamp
        observed in the account's history (or current UTC time).

        Args:
            account_id: The account identifier.
            hours: Lookback window in hours (default: 24).
            reference_time: Optional anchor timestamp for evaluation.

        Returns:
            List of recent transaction dictionaries.
        """
        history = self.get_account_history(account_id)
        if not history:
            return []

        parsed_txns: List[Tuple[datetime, Dict[str, Any]]] = []
        for tx in history:
            ts_str = str(tx.get("timestamp", "")).strip()
            if not ts_str:
                continue
            try:
                # Handle space or ISO format
                dt = datetime.fromisoformat(ts_str.replace(" ", "T"))
                parsed_txns.append((dt, tx))
            except Exception:
                continue

        if not parsed_txns:
            return []

        # Determine reference anchor time
        anchor_dt: datetime
        if reference_time is not None:
            if isinstance(reference_time, str):
                anchor_dt = datetime.fromisoformat(reference_time.replace(" ", "T"))
            else:
                anchor_dt = reference_time
        else:
            # Use latest transaction timestamp in account history as reference
            anchor_dt = max(dt for dt, _ in parsed_txns)

        cutoff = anchor_dt - timedelta(hours=hours)

        recent = [
            dict(tx)
            for dt, tx in parsed_txns
            if cutoff <= dt <= anchor_dt
        ]
        return recent

    def get_all_transaction_ids(self) -> List[str]:
        """Retrieve a list of all known transaction IDs in the dataset.

        Returns:
            List of transaction identifier strings.
        """
        if not self._is_loaded:
            self.load_data()

        return list(self._transactions_cache.keys())

    def get_transaction_count(self) -> int:
        """Retrieve the total count of transactions loaded in the dataset.

        Returns:
            Integer total transaction count.
        """
        if not self._is_loaded:
            self.load_data()

        return len(self._transactions_cache)

    def reload(self) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Force reload data from disk and refresh cache."""
        self._is_loaded = False
        return self.load_data()
