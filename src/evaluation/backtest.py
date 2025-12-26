"""
Generic rolling backtest utilities.

Provides functions for creating time-series cross-validation splits
and ensuring chronological ordering of data for backtesting.
"""

from typing import List, Tuple, Union
import pandas as pd
import numpy as np
from datetime import datetime


def ensure_chronological_order(
    data: Union[pd.DataFrame, pd.Series, np.ndarray],
    dates: Union[pd.Series, np.ndarray, List] = None,
    date_col: str = None
) -> Tuple[Union[pd.DataFrame, pd.Series, np.ndarray], np.ndarray]:
    """
    Ensure data is sorted in chronological order by dates.
    
    Args:
        data: Input data (DataFrame, Series, or array).
        dates: Array of dates. If None, will try to extract from data.
        date_col: Name of date column if data is DataFrame. Defaults to 'quarter_end_date'.
    
    Returns:
        Tuple of (sorted_data, sorted_dates) where:
        - sorted_data: Data sorted by dates
        - sorted_dates: Sorted dates array
    
    Examples:
        >>> dates = pd.to_datetime(['2020-01-01', '2019-01-01', '2021-01-01'])
        >>> data = np.array([1, 2, 3])
        >>> sorted_data, sorted_dates = ensure_chronological_order(data, dates)
        >>> np.array_equal(sorted_data, np.array([2, 1, 3]))
        True
    """
    # Extract dates if not provided
    if dates is None:
        if isinstance(data, pd.DataFrame):
            if date_col is None:
                # Try common date column names
                for col in ['quarter_end_date', 'date', 'Date', 'DATE']:
                    if col in data.columns:
                        date_col = col
                        break
                
                if date_col is None:
                    raise ValueError(
                        f"Date column not found in DataFrame. "
                        f"Available columns: {list(data.columns)}. "
                        f"Please specify date_col parameter."
                    )
            
            dates = data[date_col].values
        elif isinstance(data, pd.Series):
            if data.index.name in ['quarter_end_date', 'date', 'Date', 'DATE']:
                dates = data.index.values
            else:
                raise ValueError(
                    "Cannot infer dates from Series. Please provide dates parameter."
                )
        else:
            raise ValueError(
                "Dates must be provided when data is not a DataFrame with date column."
            )
    
    # Convert dates to numpy array and ensure datetime type
    dates = np.asarray(dates)
    
    # Convert to datetime if not already
    if not isinstance(dates[0], (datetime, pd.Timestamp)):
        try:
            dates = pd.to_datetime(dates).values
        except Exception:
            # If conversion fails, assume numeric (e.g., quarter indices)
            dates = dates.astype(float)
    
    # Get sort indices
    sort_indices = np.argsort(dates)
    
    # Sort data
    if isinstance(data, pd.DataFrame):
        sorted_data = data.iloc[sort_indices].reset_index(drop=True)
    elif isinstance(data, pd.Series):
        sorted_data = data.iloc[sort_indices].reset_index(drop=True)
    else:
        # Numpy array
        sorted_data = np.asarray(data)[sort_indices]
    
    # Sort dates
    sorted_dates = dates[sort_indices]
    
    return sorted_data, sorted_dates


def rolling_origin_splits(
    dates: Union[pd.Series, np.ndarray, List],
    min_train_periods: int,
    horizon: int = 1,
    step: int = 1
) -> List[Tuple[np.ndarray, np.ndarray]]:
    """
    Generate rolling origin time-series cross-validation splits.
    
    Creates train/test splits where:
    - Training set: all data up to time t
    - Test set: data from time t+1 to t+horizon
    
    Each subsequent split moves forward by `step` periods.
    
    Args:
        dates: Array of dates (will be sorted chronologically).
        min_train_periods: Minimum number of periods required for training.
        horizon: Number of periods ahead to forecast. Defaults to 1.
        step: Number of periods to step forward between splits. Defaults to 1.
    
    Returns:
        List of tuples (train_indices, test_indices) where each tuple contains
        numpy arrays of indices for training and testing.
    
    Examples:
        >>> dates = pd.date_range('2019-01-01', periods=20, freq='Q')
        >>> splits = rolling_origin_splits(dates, min_train_periods=8, horizon=1)
        >>> len(splits) > 0
        True
        >>> train_idx, test_idx = splits[0]
        >>> len(train_idx) >= 8
        True
        >>> len(test_idx) == 1
        True
    """
    # Convert to numpy array and ensure chronological order
    dates = np.asarray(dates)
    
    # Convert to datetime if not already
    if len(dates) > 0 and not isinstance(dates[0], (datetime, pd.Timestamp)):
        try:
            dates = pd.to_datetime(dates).values
        except Exception:
            # If conversion fails, assume numeric (e.g., quarter indices)
            dates = dates.astype(float)
    
    # Get unique sorted dates
    unique_dates = np.unique(dates)
    unique_dates = np.sort(unique_dates)
    
    n_periods = len(unique_dates)
    
    if n_periods < min_train_periods + horizon:
        raise ValueError(
            f"Insufficient data: need at least {min_train_periods + horizon} periods, "
            f"got {n_periods}"
        )
    
    splits = []
    
    # Generate splits
    for i in range(min_train_periods, n_periods - horizon + 1, step):
        # Training: all periods up to (but not including) period i
        train_end_period = unique_dates[i - 1]
        train_mask = dates <= train_end_period
        train_indices = np.where(train_mask)[0]
        
        # Testing: periods from i to i+horizon-1
        test_start_period = unique_dates[i]
        test_end_period = unique_dates[min(i + horizon - 1, n_periods - 1)]
        test_mask = (dates >= test_start_period) & (dates <= test_end_period)
        test_indices = np.where(test_mask)[0]
        
        # Only add split if we have valid train and test sets
        if len(train_indices) >= min_train_periods and len(test_indices) > 0:
            splits.append((train_indices, test_indices))
    
    return splits


def expanding_window_splits(
    dates: Union[pd.Series, np.ndarray, List],
    min_train_periods: int,
    horizon: int = 1,
    step: int = 1
) -> List[Tuple[np.ndarray, np.ndarray]]:
    """
    Generate expanding window time-series cross-validation splits.
    
    Similar to rolling_origin_splits, but training set always starts from the
    first available period (expanding window rather than rolling window).
    
    Args:
        dates: Array of dates (will be sorted chronologically).
        min_train_periods: Minimum number of periods required for training.
        horizon: Number of periods ahead to forecast. Defaults to 1.
        step: Number of periods to step forward between splits. Defaults to 1.
    
    Returns:
        List of tuples (train_indices, test_indices).
    """
    # Convert to numpy array and ensure chronological order
    dates = np.asarray(dates)
    
    # Convert to datetime if not already
    if len(dates) > 0 and not isinstance(dates[0], (datetime, pd.Timestamp)):
        try:
            dates = pd.to_datetime(dates).values
        except Exception:
            dates = dates.astype(float)
    
    # Get unique sorted dates
    unique_dates = np.unique(dates)
    unique_dates = np.sort(unique_dates)
    
    n_periods = len(unique_dates)
    
    if n_periods < min_train_periods + horizon:
        raise ValueError(
            f"Insufficient data: need at least {min_train_periods + horizon} periods, "
            f"got {n_periods}"
        )
    
    splits = []
    
    # Generate splits (training always starts from first period)
    for i in range(min_train_periods, n_periods - horizon + 1, step):
        # Training: all periods from start up to (but not including) period i
        train_end_period = unique_dates[i - 1]
        train_mask = dates <= train_end_period
        train_indices = np.where(train_mask)[0]
        
        # Testing: periods from i to i+horizon-1
        test_start_period = unique_dates[i]
        test_end_period = unique_dates[min(i + horizon - 1, n_periods - 1)]
        test_mask = (dates >= test_start_period) & (dates <= test_end_period)
        test_indices = np.where(test_mask)[0]
        
        # Only add split if we have valid train and test sets
        if len(train_indices) >= min_train_periods and len(test_indices) > 0:
            splits.append((train_indices, test_indices))
    
    return splits


if __name__ == "__main__":
    """Test backtest utilities with example data."""
    import doctest
    
    # Run doctests
    doctest.testmod(verbose=True)
    
    # Example usage
    print("\n" + "=" * 60)
    print("Example Usage")
    print("=" * 60)
    
    # Example 1: Ensure chronological order
    print("\n1. Ensuring chronological order:")
    dates = pd.to_datetime(['2020-01-01', '2019-01-01', '2021-01-01', '2018-01-01'])
    data = np.array([1, 2, 3, 4])
    sorted_data, sorted_dates = ensure_chronological_order(data, dates)
    print(f"   Original dates: {dates}")
    print(f"   Sorted dates: {sorted_dates}")
    print(f"   Sorted data: {sorted_data}")
    
    # Example 2: Rolling origin splits
    print("\n2. Rolling origin splits:")
    dates = pd.date_range('2019-01-01', periods=20, freq='Q')
    splits = rolling_origin_splits(dates, min_train_periods=8, horizon=1, step=1)
    print(f"   Total periods: {len(dates)}")
    print(f"   Number of splits: {len(splits)}")
    for i, (train_idx, test_idx) in enumerate(splits[:3]):
        print(f"   Split {i+1}: train={len(train_idx)} periods, test={len(test_idx)} periods")
    
    # Example 3: Expanding window splits
    print("\n3. Expanding window splits:")
    splits_exp = expanding_window_splits(dates, min_train_periods=8, horizon=1, step=1)
    print(f"   Number of splits: {len(splits_exp)}")
    for i, (train_idx, test_idx) in enumerate(splits_exp[:3]):
        print(f"   Split {i+1}: train={len(train_idx)} periods, test={len(test_idx)} periods")
        if i == 0:
            print(f"      First train period: {dates[train_idx[0]]}")
        if i == 1:
            print(f"      Second train period: {dates[train_idx[0]]} (should be same as first)")

