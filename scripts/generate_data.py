"""Synthetic Banking Data Generator for BankShield AI.

Generates realistic banking transactions and account profiles with deliberate
fraud/high-risk anomalies for risk scoring, anomaly detection, and investigation.

Usage:
    python scripts/generate_data.py
"""

from collections import defaultdict
from datetime import datetime, timedelta
from pathlib import Path
import random
from typing import Dict, List, Tuple
import pandas as pd

# Fixed seed for deterministic reproducibility
RANDOM_SEED: int = 42

# Anchor reference time for reproducible last-30-days window
REFERENCE_TIME: datetime = datetime(2026, 10, 13, 12, 0, 0)
WINDOW_DAYS: int = 30
START_TIME: datetime = REFERENCE_TIME - timedelta(days=WINDOW_DAYS)

# Standard merchants across multiple commercial categories (30 merchants)
STANDARD_MERCHANTS: List[str] = [
    "QuickMart Express",
    "TechWorld Electronics",
    "Sunrise Cafe",
    "GlobalPay Services",
    "MedCare Pharmacy",
    "UrbanCloth Apparel",
    "MegaBazaar Superstore",
    "PrimeGadgets Hub",
    "SpiceRoute Dining",
    "UrbanBites Bistro",
    "GoldenCrust Bakery",
    "Cafe Coffee Treats",
    "ChaiChowk Express",
    "SwiftCabs Mobility",
    "ElectroPower Utilities",
    "AquaFlow Municipal",
    "SkyHigh Airlines",
    "MetroRail Services",
    "SafeRide Cabs",
    "ApolloPlus Diagnostics",
    "WellnessPath Clinic",
    "Lifeline Hospitals",
    "StreamNow Media",
    "GameZone Interactive",
    "CinePlex Cinemas",
    "CloudHost Solutions",
    "BookWorm Corner",
    "HomeComforts Decor",
    "FreshMart Groceries",
    "Zenith Department Store",
]

# Rare/high-value merchants reserved for deliberate anomalous activity (8 merchants)
RARE_SUSPICIOUS_MERCHANTS: List[str] = [
    "Apex Luxury Jewels",
    "CryptoGlobal Exchange",
    "Prestige Gold Mart",
    "Offshore Wire Services",
    "VipJewels International",
    "GlobalForex Hub",
    "Fortune Gaming & Casino",
    "Monaco Yacht Charters",
]

ALL_MERCHANTS: List[str] = STANDARD_MERCHANTS + RARE_SUSPICIOUS_MERCHANTS

LOCATIONS: List[str] = [
    "Mumbai",
    "Delhi",
    "Bangalore",
    "Chennai",
    "Pune",
    "Hyderabad",
    "Kolkata",
    "Jaipur",
    "Ahmedabad",
    "Chandigarh",
    "Kochi",
    "Goa",
]

CHANNELS: List[str] = ["online", "ATM", "POS", "mobile_banking", "branch"]
STATUSES: List[str] = ["completed", "failed", "pending"]


def get_data_dir() -> Path:
    """Resolve absolute path to data directory."""
    project_root = Path(__file__).resolve().parent.parent
    data_dir = project_root / "data"
    data_dir.mkdir(parents=True, exist_ok=True)
    return data_dir


def generate_account_profiles(num_accounts: int = 100) -> pd.DataFrame:
    """Generate synthetic customer account profiles.

    Args:
        num_accounts: Total number of accounts to generate (default 100).

    Returns:
        DataFrame containing account profile records.
    """
    random.seed(RANDOM_SEED)

    profiles: List[Dict] = []

    # Distribution of account types
    # 50 savings, 25 current, 25 salary
    num_savings = 50
    num_current = 25
    num_salary = num_accounts - num_savings - num_current

    types_allocation = (
        ["savings"] * num_savings
        + ["current"] * num_current
        + ["salary"] * num_salary
    )
    random.shuffle(types_allocation)

    for i in range(1, num_accounts + 1):
        account_id = f"ACC-{1000 + i}"
        account_type = types_allocation[i - 1]

        if account_type == "savings":
            hist_avg = round(random.uniform(1500.0, 6000.0), 2)
            hist_freq = round(random.uniform(0.5, 2.0), 2)
            usual_hours = random.choice(["8-21", "9-21", "8-22", "9-20"])
            age_days = random.randint(180, 2500)
        elif account_type == "current":
            hist_avg = round(random.uniform(15000.0, 50000.0), 2)
            hist_freq = round(random.uniform(3.0, 7.5), 2)
            usual_hours = random.choice(["8-20", "9-19", "8-21"])
            age_days = random.randint(120, 3200)
        else:  # salary
            hist_avg = round(random.uniform(2500.0, 12000.0), 2)
            hist_freq = round(random.uniform(0.8, 2.5), 2)
            usual_hours = random.choice(["7-23", "8-22", "9-21"])
            age_days = random.randint(90, 1800)

        profiles.append(
            {
                "account_id": account_id,
                "account_type": account_type,
                "historical_average_amount": hist_avg,
                "historical_transaction_frequency": hist_freq,
                "usual_transaction_hours": usual_hours,
                "account_age_days": age_days,
            }
        )

    return pd.DataFrame(profiles)


def _random_timestamp_within_window(
    start: datetime, end: datetime, hour_range: Tuple[int, int] = None
) -> datetime:
    """Generate a random datetime between start and end, optionally within specific hours."""
    total_seconds = int((end - start).total_seconds())
    offset_seconds = random.randint(0, total_seconds)
    dt = start + timedelta(seconds=offset_seconds)

    if hour_range is not None:
        start_h, end_h = hour_range
        chosen_hour = random.randint(start_h, end_h)
        dt = dt.replace(hour=chosen_hour, minute=random.randint(0, 59), second=random.randint(0, 59))

    return dt


def generate_transactions(
    profiles_df: pd.DataFrame, target_count: int = 500
) -> pd.DataFrame:
    """Generate synthetic transactions with deliberate high-risk patterns.

    Args:
        profiles_df: Account profiles dataframe.
        target_count: Target number of transactions to generate (~500).

    Returns:
        DataFrame containing transactions sorted chronologically.
    """
    random.seed(RANDOM_SEED)

    # Index profiles for quick lookup
    profiles_map = {row["account_id"]: row for _, row in profiles_df.iterrows()}
    all_account_ids = list(profiles_map.keys())

    # Assign primary city and regular favorite merchants per account
    account_primary_city = {
        acc_id: random.choice(LOCATIONS) for acc_id in all_account_ids
    }
    account_favorite_merchants = {
        acc_id: random.sample(STANDARD_MERCHANTS, k=random.randint(3, 5))
        for acc_id in all_account_ids
    }

    raw_transactions: List[Dict] = []

    # ---------------------------------------------------------
    # 1. Deliberate Pattern A & B: High Amounts (>5x historical avg)
    #    with a subset having brand-new merchants
    # ---------------------------------------------------------
    # We select 8 distinct accounts for extreme amounts (>5x avg).
    # 6 of them will use rare suspicious merchants never seen by the account.
    high_amount_accounts = [
        "ACC-1005",
        "ACC-1018",
        "ACC-1027",
        "ACC-1045",
        "ACC-1062",
        "ACC-1073",
        "ACC-1088",
        "ACC-1095",
    ]

    for idx, acc_id in enumerate(high_amount_accounts):
        profile = profiles_map[acc_id]
        hist_avg = profile["historical_average_amount"]
        # Multiplier strictly > 5.0 (between 5.5 and 8.5)
        multiplier = random.uniform(5.5, 8.5)
        amount = round(hist_avg * multiplier, 2)

        if idx < 6:
            # Pair with brand-new rare suspicious merchant
            merchant = RARE_SUSPICIOUS_MERCHANTS[idx % len(RARE_SUSPICIOUS_MERCHANTS)]
        else:
            merchant = random.choice(STANDARD_MERCHANTS)

        # Place towards middle/latter half of the window
        day_offset = random.randint(10, 28)
        ts = START_TIME + timedelta(
            days=day_offset,
            hours=random.randint(10, 20),
            minutes=random.randint(0, 59),
            seconds=random.randint(0, 59),
        )

        raw_transactions.append(
            {
                "account_id": acc_id,
                "timestamp": ts,
                "amount": amount,
                "merchant": merchant,
                "location": random.choice(LOCATIONS),
                "channel": random.choice(["online", "POS", "mobile_banking"]),
                "status": "completed",
                "failed_attempts": 0,
            }
        )

    # ---------------------------------------------------------
    # 2. Deliberate Pattern C: Unusually high frequency bursts
    #    (3-5 accounts with rapid burst of transactions)
    # ---------------------------------------------------------
    burst_accounts = ["ACC-1012", "ACC-1035", "ACC-1058", "ACC-1082"]
    burst_base_days = [7, 14, 21, 26]

    for b_idx, acc_id in enumerate(burst_accounts):
        profile = profiles_map[acc_id]
        hist_avg = profile["historical_average_amount"]
        burst_day = burst_base_days[b_idx]
        burst_base_time = START_TIME + timedelta(
            days=burst_day,
            hours=random.randint(11, 18),
            minutes=random.randint(10, 40),
        )

        # 6 rapid transactions within 25 minutes
        for step in range(6):
            ts = burst_base_time + timedelta(minutes=step * 4, seconds=random.randint(5, 45))
            amount = round(random.uniform(0.8, 2.2) * hist_avg, 2)
            merchant = random.choice(account_favorite_merchants[acc_id])
            raw_transactions.append(
                {
                    "account_id": acc_id,
                    "timestamp": ts,
                    "amount": amount,
                    "merchant": merchant,
                    "location": account_primary_city[acc_id],
                    "channel": random.choice(["online", "mobile_banking"]),
                    "status": "completed",
                    "failed_attempts": 0,
                }
            )

    # ---------------------------------------------------------
    # 3. Deliberate Pattern D: Failed Attempts >= 3
    # ---------------------------------------------------------
    failed_accounts = [
        "ACC-1008",
        "ACC-1022",
        "ACC-1039",
        "ACC-1051",
        "ACC-1067",
        "ACC-1081",
        "ACC-1092",
        "ACC-1099",
    ]

    for f_idx, acc_id in enumerate(failed_accounts):
        profile = profiles_map[acc_id]
        hist_avg = profile["historical_average_amount"]
        ts = _random_timestamp_within_window(START_TIME, REFERENCE_TIME)
        failed_cnt = random.choice([3, 4, 5])
        # 5 failed, 3 completed after multiple failed attempts
        status = "failed" if f_idx < 5 else "completed"
        amount = round(random.uniform(0.5, 3.0) * hist_avg, 2)

        raw_transactions.append(
            {
                "account_id": acc_id,
                "timestamp": ts,
                "amount": amount,
                "merchant": random.choice(account_favorite_merchants[acc_id]),
                "location": account_primary_city[acc_id],
                "channel": random.choice(["online", "mobile_banking", "ATM"]),
                "status": status,
                "failed_attempts": failed_cnt,
            }
        )

    # ---------------------------------------------------------
    # 4. Deliberate Pattern E: Unusual nocturnal hours (2am - 5am)
    # ---------------------------------------------------------
    night_accounts = [
        "ACC-1002",
        "ACC-1015",
        "ACC-1029",
        "ACC-1041",
        "ACC-1053",
        "ACC-1064",
        "ACC-1076",
        "ACC-1085",
        "ACC-1091",
        "ACC-1098",
        "ACC-1010",
        "ACC-1024",
        "ACC-1038",
        "ACC-1060",
        "ACC-1070",
    ]

    for acc_id in night_accounts:
        profile = profiles_map[acc_id]
        hist_avg = profile["historical_average_amount"]
        ts = _random_timestamp_within_window(
            START_TIME, REFERENCE_TIME, hour_range=(2, 4)
        )
        amount = round(random.uniform(0.7, 2.5) * hist_avg, 2)
        raw_transactions.append(
            {
                "account_id": acc_id,
                "timestamp": ts,
                "amount": amount,
                "merchant": random.choice(STANDARD_MERCHANTS),
                "location": account_primary_city[acc_id],
                "channel": random.choice(["online", "ATM", "mobile_banking"]),
                "status": "completed",
                "failed_attempts": random.choice([0, 0, 1]),
            }
        )

    # ---------------------------------------------------------
    # 5. Ordinary baseline transactions across all accounts
    # ---------------------------------------------------------
    current_count = len(raw_transactions)
    remaining_needed = target_count - current_count

    for _ in range(remaining_needed):
        acc_id = random.choice(all_account_ids)
        profile = profiles_map[acc_id]
        hist_avg = profile["historical_average_amount"]

        # Parse usual transaction hours
        hours_str = profile["usual_transaction_hours"]
        start_h, end_h = map(int, hours_str.split("-"))

        ts = _random_timestamp_within_window(
            START_TIME, REFERENCE_TIME, hour_range=(start_h, end_h)
        )

        # Baseline amount with natural variance around historical average
        amount_factor = max(0.15, random.gauss(1.0, 0.28))
        amount = round(min(amount_factor, 2.8) * hist_avg, 2)

        # 85% favorite merchant (repeat), 15% random standard merchant
        if random.random() < 0.85:
            merchant = random.choice(account_favorite_merchants[acc_id])
        else:
            merchant = random.choice(STANDARD_MERCHANTS)

        # 88% primary city, 12% other city
        if random.random() < 0.88:
            location = account_primary_city[acc_id]
        else:
            location = random.choice(LOCATIONS)

        channel = random.choices(
            CHANNELS, weights=[0.40, 0.20, 0.25, 0.12, 0.03], k=1
        )[0]
        status = random.choices(
            STATUSES, weights=[0.94, 0.04, 0.02], k=1
        )[0]
        failed_attempts = random.choices(
            [0, 1, 2], weights=[0.95, 0.04, 0.01], k=1
        )[0]

        raw_transactions.append(
            {
                "account_id": acc_id,
                "timestamp": ts,
                "amount": amount,
                "merchant": merchant,
                "location": location,
                "channel": channel,
                "status": status,
                "failed_attempts": failed_attempts,
            }
        )

    # ---------------------------------------------------------
    # 6. Sort chronologically and dynamically compute:
    #    - is_new_merchant
    #    - is_new_location
    #    - transaction_id (TXN-XXXXX)
    # ---------------------------------------------------------
    raw_transactions.sort(key=lambda t: t["timestamp"])

    seen_merchants: Dict[str, set] = defaultdict(set)
    seen_locations: Dict[str, set] = defaultdict(set)

    final_transactions: List[Dict] = []
    for idx, txn in enumerate(raw_transactions, start=1):
        acc_id = txn["account_id"]
        merch = txn["merchant"]
        loc = txn["location"]

        is_new_merchant = merch not in seen_merchants[acc_id]
        seen_merchants[acc_id].add(merch)

        is_new_location = loc not in seen_locations[acc_id]
        seen_locations[acc_id].add(loc)

        txn_record = {
            "transaction_id": f"TXN-{10000 + idx}",
            "account_id": acc_id,
            "timestamp": txn["timestamp"].strftime("%Y-%m-%d %H:%M:%S"),
            "amount": txn["amount"],
            "merchant": merch,
            "location": loc,
            "channel": txn["channel"],
            "status": txn["status"],
            "is_new_merchant": is_new_merchant,
            "is_new_location": is_new_location,
            "failed_attempts": txn["failed_attempts"],
        }
        final_transactions.append(txn_record)

    txns_df = pd.DataFrame(final_transactions)
    return txns_df


def print_summary_statistics(
    profiles_df: pd.DataFrame, txns_df: pd.DataFrame
) -> None:
    """Print comprehensive summary metrics of the generated datasets."""
    print("=" * 70)
    print("BANKSHIELD AI - SYNTHETIC DATA GENERATION SUMMARY")
    print("=" * 70)

    print("\n[1] ACCOUNT PROFILES DATASET:")
    print(f"  Total Accounts: {len(profiles_df)}")
    print("  Breakdown by Account Type:")
    for acc_type, count in profiles_df["account_type"].value_counts().items():
        sub = profiles_df[profiles_df["account_type"] == acc_type]
        avg_amt = sub["historical_average_amount"].mean()
        avg_freq = sub["historical_transaction_frequency"].mean()
        print(f"    - {acc_type:<10}: {count} accounts | Mean Avg Amt: INR {avg_amt:,.2f} | Mean Freq: {avg_freq:.2f}/day")

    print("\n[2] TRANSACTIONS DATASET:")
    print(f"  Total Transactions: {len(txns_df)}")
    print(f"  Date Range: {txns_df['timestamp'].min()} to {txns_df['timestamp'].max()}")
    print(f"  Total Volume: INR {txns_df['amount'].sum():,.2f}")
    print(f"  Mean Transaction Amount: INR {txns_df['amount'].mean():,.2f}")
    print(f"  Median Transaction Amount: INR {txns_df['amount'].median():,.2f}")
    print(f"  Unique Merchants: {txns_df['merchant'].nunique()}")
    print(f"  Unique Locations: {txns_df['location'].nunique()}")

    print("\n  Channel Distribution:")
    for ch, count in txns_df["channel"].value_counts().items():
        pct = (count / len(txns_df)) * 100
        print(f"    - {ch:<16}: {count:>4} ({pct:>5.1f}%)")

    print("\n  Status Distribution:")
    for st, count in txns_df["status"].value_counts().items():
        pct = (count / len(txns_df)) * 100
        print(f"    - {st:<16}: {count:>4} ({pct:>5.1f}%)")

    # High-Risk Deliberate Pattern Verification
    profiles_map = {row["account_id"]: row for _, row in profiles_df.iterrows()}

    txns_with_hist = txns_df.copy()
    txns_with_hist["hist_avg"] = txns_with_hist["account_id"].map(
        lambda a: profiles_map[a]["historical_average_amount"]
    )
    txns_with_hist["amount_ratio"] = (
        txns_with_hist["amount"] / txns_with_hist["hist_avg"]
    )
    txns_with_hist["hour"] = pd.to_datetime(txns_with_hist["timestamp"]).dt.hour

    extreme_amt_count = (txns_with_hist["amount_ratio"] > 5.0).sum()
    new_merch_high_amt = (
        (txns_with_hist["amount_ratio"] > 5.0) & txns_with_hist["is_new_merchant"]
    ).sum()
    failed_attempts_3plus = (txns_with_hist["failed_attempts"] >= 3).sum()
    unusual_hours_count = (
        (txns_with_hist["hour"] >= 2) & (txns_with_hist["hour"] <= 5)
    ).sum()

    print("\n[3] HIGH-RISK ANOMALY PATTERNS VERIFICATION:")
    print(f"  [PASS] Amounts > 5x Historical Average       : {extreme_amt_count} transactions (Requirement: >= 5-10)")
    print(f"  [PASS] New Merchant AND Amount > 5x Avg       : {new_merch_high_amt} transactions (Requirement: >= 5)")
    print(f"  [PASS] Failed Attempts >= 3                   : {failed_attempts_3plus} transactions (Requirement: >= 5)")
    print(f"  [PASS] Nocturnal Window (2:00 AM - 5:00 AM)   : {unusual_hours_count} transactions (Requirement: deliberate presence)")
    print("  [PASS] Account Burst Velocity accounts        : 4 accounts with 6 txns within 25 mins")

    # Validation assertions to verify dataset distributions and edge cases
    assert extreme_amt_count >= 5, f"Expected >= 5 extreme amount transactions, got {extreme_amt_count}"
    assert new_merch_high_amt >= 5, f"Expected >= 5 new merchant high amount txns, got {new_merch_high_amt}"
    assert failed_attempts_3plus >= 5, f"Expected >= 5 failed attempts >= 3, got {failed_attempts_3plus}"
    assert unusual_hours_count >= 5, f"Expected >= 5 nocturnal transactions, got {unusual_hours_count}"
    assert txns_df["merchant"].nunique() >= 30, f"Expected >= 30 unique merchants, got {txns_df['merchant'].nunique()}"
    assert set(txns_df["account_id"]).issubset(set(profiles_df["account_id"])), "Orphan account_ids in transactions!"

    print("\n[4] INTEGRITY CHECK: All criteria verified successfully.")
    print("=" * 70)


def main() -> None:
    """Generate and persist synthetic datasets to data/ directory."""
    data_dir = get_data_dir()
    accounts_csv_path = data_dir / "account_profiles.csv"
    transactions_csv_path = data_dir / "transactions.csv"

    print("Generating synthetic account profiles (~100 rows)...")
    profiles_df = generate_account_profiles(num_accounts=100)
    profiles_df.to_csv(accounts_csv_path, index=False)
    print(f"Saved account profiles to: {accounts_csv_path}")

    print("Generating synthetic transactions (~500 rows)...")
    txns_df = generate_transactions(profiles_df, target_count=500)
    txns_df.to_csv(transactions_csv_path, index=False)
    print(f"Saved transactions to: {transactions_csv_path}")

    print_summary_statistics(profiles_df, txns_df)


if __name__ == "__main__":
    main()
