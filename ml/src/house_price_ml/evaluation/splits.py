"""Time-aware data splitting."""

from __future__ import annotations

import pandas as pd


def walk_forward_splits(
    df: pd.DataFrame,
    date_col: str = "sale_date",
    n_splits: int = 3,
    test_size_quarters: int = 1,
) -> list[tuple[pd.DataFrame, pd.DataFrame]]:
    """Generate walk-forward train/validation splits ordered by time."""
    data = df.copy()
    data[date_col] = pd.to_datetime(data[date_col])
    data = data.sort_values(date_col)
    data["quarter"] = data[date_col].dt.to_period("Q")
    quarters = sorted(data["quarter"].unique())

    if len(quarters) < n_splits + test_size_quarters + 1:
        split_idx = int(len(data) * 0.8)
        return [(data.iloc[:split_idx], data.iloc[split_idx:])]

    splits: list[tuple[pd.DataFrame, pd.DataFrame]] = []
    for i in range(n_splits):
        val_end_idx = len(quarters) - test_size_quarters - n_splits + i + 1
        train_quarters = quarters[:val_end_idx]
        val_quarter = quarters[val_end_idx]
        train = data[data["quarter"].isin(train_quarters)]
        val = data[data["quarter"] == val_quarter]
        if len(train) > 0 and len(val) > 0:
            splits.append((train, val))
    return splits


def holdout_test_split(
    df: pd.DataFrame,
    date_col: str = "sale_date",
    test_quarters: int = 2,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Hold out final quarters as untouched test set."""
    data = df.copy()
    data[date_col] = pd.to_datetime(data[date_col])
    data = data.sort_values(date_col)
    data["quarter"] = data[date_col].dt.to_period("Q")
    quarters = sorted(data["quarter"].unique())
    if len(quarters) <= test_quarters:
        split_idx = int(len(data) * 0.85)
        return data.iloc[:split_idx], data.iloc[split_idx:]
    test_q = set(quarters[-test_quarters:])
    train = data[~data["quarter"].isin(test_q)]
    test = data[data["quarter"].isin(test_q)]
    return train, test
