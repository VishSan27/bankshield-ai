"""Tests for DataService in BankShield AI."""

import os
from pathlib import Path
import pytest
import pandas as pd

from services.data_service import DataService


@pytest.fixture
def data_service():
    """Create a DataService instance."""
    return DataService()


def test_data_service_initialization(data_service):
    """Test that DataService loads transactions and account profiles."""
    tx_df, acc_df = data_service.load_data()
    assert isinstance(tx_df, pd.DataFrame)
    assert isinstance(acc_df, pd.DataFrame)
    assert len(tx_df) >= 400
    assert len(acc_df) >= 80


def test_schema_validation(data_service):
    """Test that required columns exist in both datasets."""
    tx_df, acc_df = data_service.load_data()
    required_tx_cols = {"transaction_id", "account_id", "timestamp", "amount"}
    required_acc_cols = {"account_id", "account_type", "historical_average_amount"}

    assert required_tx_cols.issubset(set(tx_df.columns))
    assert required_acc_cols.issubset(set(acc_df.columns))


def test_get_transaction_valid(data_service):
    """Test retrieving an existing transaction."""
    tx_ids = data_service.get_all_transaction_ids()
    assert len(tx_ids) > 0
    sample_id = tx_ids[0]

    txn = data_service.get_transaction(sample_id)
    assert txn is not None
    assert txn["transaction_id"] == sample_id
    assert "amount" in txn
    assert "account_id" in txn


def test_get_transaction_invalid(data_service):
    """Test retrieving a non-existent transaction."""
    txn = data_service.get_transaction("TXN-NONEXISTENT-999")
    assert txn is None


def test_get_account_profile_valid(data_service):
    """Test retrieving an existing account profile."""
    tx_ids = data_service.get_all_transaction_ids()
    sample_id = tx_ids[0]
    txn = data_service.get_transaction(sample_id)
    acc_id = txn["account_id"]

    profile = data_service.get_account_profile(acc_id)
    assert profile is not None
    assert profile["account_id"] == acc_id
    assert "historical_average_amount" in profile


def test_get_account_profile_invalid(data_service):
    """Test retrieving a non-existent account profile."""
    profile = data_service.get_account_profile("ACC-NONEXISTENT-000")
    assert profile is None


def test_get_account_history(data_service):
    """Test retrieving chronological account history."""
    tx_ids = data_service.get_all_transaction_ids()
    sample_id = tx_ids[0]
    txn = data_service.get_transaction(sample_id)
    acc_id = txn["account_id"]

    history = data_service.get_account_history(acc_id)
    assert isinstance(history, list)
    assert len(history) >= 1
    # Check that all records belong to the account
    for item in history:
        assert item["account_id"] == acc_id


def test_get_recent_transactions(data_service):
    """Test retrieving recent transactions within a time window."""
    tx_ids = data_service.get_all_transaction_ids()
    sample_id = tx_ids[0]
    txn = data_service.get_transaction(sample_id)
    acc_id = txn["account_id"]

    recent = data_service.get_recent_transactions(acc_id, hours=720)  # 30 days
    assert isinstance(recent, list)
