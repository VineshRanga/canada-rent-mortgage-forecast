"""
Evaluation metrics for forecasting models.

Provides implementations of MAE, sMAPE, and utility functions
for handling missing values in metric calculations.
"""

from typing import Tuple
import numpy as np
import pandas as pd


def safe_dropna_pair(y_true: np.ndarray, y_pred: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Safely drop NaN pairs from y_true and y_pred arrays.
    
    Handles object dtype arrays by coercing to numeric floats.
    Removes pairs where either y_true or y_pred is NaN or inf, ensuring
    both arrays have the same length and no NaN/inf values.
    
    Args:
        y_true: True values array (can be object dtype, will be coerced to float).
        y_pred: Predicted values array (can be object dtype, will be coerced to float).
    
    Returns:
        Tuple of (y_true_clean, y_pred_clean) with NaN/inf pairs removed.
        Both arrays are float dtype.
    
    Raises:
        ValueError: If arrays have different lengths or no valid numeric pairs after coercion.
    
    Examples:
        >>> y_true = np.array([1.0, 2.0, np.nan, 4.0])
        >>> y_pred = np.array([1.1, np.nan, 3.0, 4.1])
        >>> y_true_clean, y_pred_clean = safe_dropna_pair(y_true, y_pred)
        >>> len(y_true_clean) == len(y_pred_clean)
        True
        >>> np.isfinite(y_true_clean).all()
        True
    """
    # Convert to 1D numpy arrays
    y_true = np.asarray(y_true).reshape(-1)
    y_pred = np.asarray(y_pred).reshape(-1)
    
    # Check lengths match
    if len(y_true) != len(y_pred):
        raise ValueError(
            f"y_true and y_pred must have the same length. "
            f"Got {len(y_true)} and {len(y_pred)}"
        )
    
    # Coerce to numeric floats safely (handles strings/objects)
    # pd.to_numeric may return ndarray, so use np.asarray instead of .to_numpy()
    y_true_num = np.asarray(pd.to_numeric(y_true, errors="coerce"), dtype=float).reshape(-1)
    y_pred_num = np.asarray(pd.to_numeric(y_pred, errors="coerce"), dtype=float).reshape(-1)
    
    # Use finite mask (handles nan and inf)
    valid_mask = np.isfinite(y_true_num) & np.isfinite(y_pred_num)
    
    # Guard: check if we have any valid pairs
    if valid_mask.sum() == 0:
        # Return empty arrays and let mae/smape raise a ValueError
        return np.array([], dtype=float), np.array([], dtype=float)
    
    # Return cleaned arrays
    return y_true_num[valid_mask], y_pred_num[valid_mask]


def mae(y_true: np.ndarray, y_pred: np.ndarray, dropna: bool = True) -> float:
    """
    Calculate Mean Absolute Error.
    
    MAE = mean(|y_true - y_pred|)
    
    Args:
        y_true: True values array.
        y_pred: Predicted values array.
        dropna: Whether to drop NaN pairs before calculation. Defaults to True.
    
    Returns:
        MAE value (float).
    
    Raises:
        ValueError: If arrays have different lengths or no valid pairs after dropping NaN.
    
    Examples:
        >>> y_true = np.array([1.0, 2.0, 3.0, 4.0])
        >>> y_pred = np.array([1.1, 2.2, 2.9, 4.1])
        >>> mae(y_true, y_pred)
        0.1
    """
    # Convert to numpy arrays if not already
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    
    # Drop NaN pairs if requested
    if dropna:
        y_true, y_pred = safe_dropna_pair(y_true, y_pred)
    
    # Check we have valid data
    if len(y_true) == 0:
        raise ValueError(
            "No valid numeric pairs after coercion. "
            "All values in y_true and/or y_pred are NaN, inf, or non-numeric."
        )
    
    # Calculate MAE
    return np.mean(np.abs(y_true - y_pred))


def smape(y_true: np.ndarray, y_pred: np.ndarray, dropna: bool = True) -> float:
    """
    Calculate Symmetric Mean Absolute Percentage Error.
    
    sMAPE = 100 * mean(|y_true - y_pred| / ((|y_true| + |y_pred|) / 2))
    
    The denominator is the average of absolute true and predicted values,
    which makes sMAPE symmetric and bounded between 0% and 200%.
    
    Args:
        y_true: True values array.
        y_pred: Predicted values array.
        dropna: Whether to drop NaN pairs before calculation. Defaults to True.
    
    Returns:
        sMAPE value as percentage (float).
    
    Raises:
        ValueError: If arrays have different lengths or no valid pairs after dropping NaN.
    
    Examples:
        >>> y_true = np.array([100.0, 200.0, 300.0])
        >>> y_pred = np.array([110.0, 190.0, 310.0])
        >>> smape(y_true, y_pred)  # Should be around 3-4%
        3.33...
    """
    # Convert to numpy arrays if not already
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    
    # Drop NaN pairs if requested
    if dropna:
        y_true, y_pred = safe_dropna_pair(y_true, y_pred)
    
    # Check we have valid data
    if len(y_true) == 0:
        raise ValueError(
            "No valid numeric pairs after coercion. "
            "All values in y_true and/or y_pred are NaN, inf, or non-numeric."
        )
    
    # Calculate denominator: (|y_true| + |y_pred|) / 2
    denominator = (np.abs(y_true) + np.abs(y_pred)) / 2
    
    # Handle zero denominators (when both y_true and y_pred are zero)
    # In this case, the error is zero, so we set denominator to avoid division by zero
    zero_mask = denominator == 0
    denominator = np.where(zero_mask, np.nan, denominator)
    
    # Calculate sMAPE
    errors = np.abs(y_true - y_pred) / denominator
    
    # Return as percentage
    return 100 * np.nanmean(errors)


if __name__ == "__main__":
    """Test metrics with example data."""
    import doctest
    
    # Run doctests
    doctest.testmod(verbose=True)
    
    # Example usage
    print("\n" + "=" * 60)
    print("Example Usage")
    print("=" * 60)
    
    # Example 1: Perfect predictions
    y_true = np.array([100.0, 200.0, 300.0, 400.0])
    y_pred = np.array([100.0, 200.0, 300.0, 400.0])
    print(f"\nPerfect predictions:")
    print(f"  MAE: {mae(y_true, y_pred):.4f}")
    print(f"  sMAPE: {smape(y_true, y_pred):.4f}%")
    
    # Example 2: Some error
    y_true = np.array([100.0, 200.0, 300.0, 400.0])
    y_pred = np.array([110.0, 190.0, 310.0, 390.0])
    print(f"\nWith errors:")
    print(f"  MAE: {mae(y_true, y_pred):.4f}")
    print(f"  sMAPE: {smape(y_true, y_pred):.4f}%")
    
    # Example 3: With NaN values
    y_true = np.array([100.0, 200.0, np.nan, 400.0])
    y_pred = np.array([110.0, np.nan, 310.0, 390.0])
    print(f"\nWith NaN values (auto-dropped):")
    print(f"  MAE: {mae(y_true, y_pred):.4f}")
    print(f"  sMAPE: {smape(y_true, y_pred):.4f}%")
    
    # Example 4: Zero values (test sMAPE handling)
    y_true = np.array([0.0, 100.0, 200.0])
    y_pred = np.array([0.0, 110.0, 190.0])
    print(f"\nWith zero values:")
    print(f"  MAE: {mae(y_true, y_pred):.4f}")
    print(f"  sMAPE: {smape(y_true, y_pred):.4f}%")

