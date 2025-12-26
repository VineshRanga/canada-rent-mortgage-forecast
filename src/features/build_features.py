"""
Build modeling datasets for rent and mortgage forecasting.

Creates feature-engineered datasets with lags, seasonality, and merged
exogenous features for both rent (panel) and mortgage (time series) models.
"""

from pathlib import Path
from typing import Optional
import pandas as pd
import numpy as np

from src.config import PROCESSED_DIR, PROJECT_WINDOW_START, PROJECT_WINDOW_END
from src.io.statcan_read import filter_window


def build_rent_model_dataset(
    rent_target_path: Optional[Path] = None,
    exog_path: Optional[Path] = None,
    output_path: Optional[Path] = None
) -> pd.DataFrame:
    """
    Build rent modeling dataset with lags, seasonality, and exogenous features.
    
    Loads rent target data and exogenous features, creates lag features
    grouped by (cma, unit_type), adds seasonality, and merges everything.
    
    Args:
        rent_target_path: Path to rent target parquet file.
                          Defaults to PROCESSED_DIR / "rent_target_quarterly.parquet".
        exog_path: Path to exogenous features parquet file.
                   Defaults to PROCESSED_DIR / "exog_quarterly.parquet".
        output_path: Path to save output parquet file.
                     Defaults to PROCESSED_DIR / "rent_model_dataset.parquet".
    
    Returns:
        DataFrame with columns:
        - quarter_end_date: Quarter-end Timestamp
        - cma: CMA name (categorical)
        - unit_type: Unit type (categorical)
        - y: Target rent value
        - y_lag_1, y_lag_2, y_lag_3, y_lag_4: Lagged target values
        - y_yoy_change: Year-over-year change (y - y_lag_4)
        - quarter: Quarter number (1-4, as int)
        - All exogenous features from exog_quarterly
    
    Raises:
        FileNotFoundError: If input files do not exist.
    """
    if rent_target_path is None:
        rent_target_path = PROCESSED_DIR / "rent_target_quarterly.parquet"
    if exog_path is None:
        exog_path = PROCESSED_DIR / "exog_quarterly.parquet"
    if output_path is None:
        output_path = PROCESSED_DIR / "rent_model_dataset.parquet"
    
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    print("Building rent modeling dataset...")
    print("=" * 60)
    
    # Load rent target data
    print(f"Loading rent target from: {rent_target_path}")
    if not rent_target_path.exists():
        raise FileNotFoundError(f"Rent target file not found: {rent_target_path}")
    
    rent_df = pd.read_parquet(rent_target_path)
    print(f"  Rent target shape: {rent_df.shape}")
    print(f"  Columns: {list(rent_df.columns)}")
    
    # Strong diagnostics after loading rent_target
    print("\n  Rent target diagnostics:")
    print(f"    Shape: {rent_df.shape}")
    if 'cma' in rent_df.columns:
        print(f"    Unique CMAs: {rent_df['cma'].nunique()}")
    if 'unit_type' in rent_df.columns:
        print(f"    Unique unit types: {rent_df['unit_type'].nunique()}")
    if 'quarter_end_date' in rent_df.columns:
        rent_df['quarter_end_date'] = pd.to_datetime(rent_df['quarter_end_date'])
        print(f"    Date range: {rent_df['quarter_end_date'].min()} to {rent_df['quarter_end_date'].max()}")
    
    # Identify target column (could be 'y', 'y_level', 'rent', etc.)
    target_col = None
    for col in ['y', 'y_level', 'rent', 'value']:
        if col in rent_df.columns:
            target_col = col
            break
    
    if target_col is None:
        raise ValueError(
            f"Could not identify target column in rent data. "
            f"Available columns: {list(rent_df.columns)}"
        )
    
    # Rename target column to 'y' for consistency
    if target_col != 'y':
        rent_df = rent_df.rename(columns={target_col: 'y'})
    
    # Ensure required columns exist
    required_cols = ['quarter_end_date', 'cma', 'unit_type', 'y']
    missing_cols = [col for col in required_cols if col not in rent_df.columns]
    if missing_cols:
        raise ValueError(
            f"Missing required columns in rent data: {missing_cols}. "
            f"Available columns: {list(rent_df.columns)}"
        )
    
    # Convert quarter_end_date to datetime if needed (if not already done)
    if rent_df['quarter_end_date'].dtype != 'datetime64[ns]':
        rent_df['quarter_end_date'] = pd.to_datetime(rent_df['quarter_end_date'])
    
    # Sort by (cma, unit_type, quarter_end_date) for lag calculation
    rent_df = rent_df.sort_values(['cma', 'unit_type', 'quarter_end_date']).reset_index(drop=True)
    
    # Create lag features grouped by (cma, unit_type)
    print("Creating lag features...")
    for lag in [1, 2, 3, 4]:
        rent_df[f'y_lag_{lag}'] = rent_df.groupby(['cma', 'unit_type'])['y'].shift(lag)
    
    # Create year-over-year change (optional: y - y_lag_4)
    rent_df['y_yoy_change'] = rent_df['y'] - rent_df['y_lag_4']
    
    # Diagnostics after creating lags
    print("  Lag feature diagnostics:")
    print(f"    Rows with non-null y_lag_1: {rent_df['y_lag_1'].notna().sum()} / {len(rent_df)}")
    print(f"    Rows with non-null y_lag_4: {rent_df['y_lag_4'].notna().sum()} / {len(rent_df)}")
    
    # Create quarter feature (as int, one-hot encoding will be done in model)
    print("Adding quarter feature...")
    rent_df['quarter'] = rent_df['quarter_end_date'].dt.quarter.astype(int)
    
    # Ensure cma and unit_type are categorical
    rent_df['cma'] = rent_df['cma'].astype('category')
    rent_df['unit_type'] = rent_df['unit_type'].astype('category')
    
    # Load exogenous features
    print(f"Loading exogenous features from: {exog_path}")
    if not exog_path.exists():
        raise FileNotFoundError(f"Exogenous features file not found: {exog_path}")
    
    exog_df = pd.read_parquet(exog_path)
    print(f"  Exogenous features shape: {exog_df.shape}")
    
    # Ensure quarter_end_date is datetime
    exog_df['quarter_end_date'] = pd.to_datetime(exog_df['quarter_end_date'])
    
    # Merge with exogenous features
    print("Merging exogenous features...")
    rent_model_df = rent_df.merge(
        exog_df,
        on='quarter_end_date',
        how='left'
    )
    
    # Diagnostics after merging exog
    print(f"  Merged shape: {rent_model_df.shape}")
    print("  Missing rate of key columns:")
    key_cols = ['y', 'y_lag_1', 'y_lag_2', 'y_lag_3', 'y_lag_4']
    exog_key_cols = ['rate_bank', 'labour_unemployment_rate']
    
    for col in key_cols + exog_key_cols:
        if col in rent_model_df.columns:
            missing_pct = rent_model_df[col].isna().sum() / len(rent_model_df) * 100
            print(f"    {col}: {missing_pct:.1f}% missing")
        else:
            print(f"    {col}: column not found")
    
    # Handle missing exog: drop columns with < 90% non-null coverage
    print("Checking exogenous feature coverage...")
    exog_cols = [col for col in rent_model_df.columns 
                 if col.startswith(('rate_', 'labour_', 'starts_', 'pop_', 'mig_', 'npr_'))]
    
    dropped_cols = []
    for col in exog_cols:
        non_null_pct = rent_model_df[col].notna().sum() / len(rent_model_df) * 100
        if non_null_pct < 90.0:
            dropped_cols.append((col, non_null_pct))
            rent_model_df = rent_model_df.drop(columns=[col])
    
    if dropped_cols:
        print(f"  Dropped {len(dropped_cols)} exog columns with < 90% coverage:")
        for col, pct in dropped_cols:
            print(f"    - {col}: {pct:.1f}% coverage")
    else:
        print("  All exog columns have >= 90% coverage")
    
    # Drop rows where y_lag_1 is missing (so the model has at least 1 lag)
    print("Dropping rows with missing y_lag_1...")
    before_drop = len(rent_model_df)
    rent_model_df = rent_model_df[rent_model_df['y_lag_1'].notna()].copy()
    after_drop = len(rent_model_df)
    print(f"  Dropped {before_drop - after_drop} rows ({before_drop} -> {after_drop})")
    
    # Filter to project window
    print("Filtering to project window...")
    print(f"  Configured window: {PROJECT_WINDOW_START} to {PROJECT_WINDOW_END}")
    
    # Capture date range before filtering
    before_filter = len(rent_model_df)
    df_date_min_before = None
    df_date_max_before = None
    if 'quarter_end_date' in rent_model_df.columns:
        df_date_min_before = rent_model_df['quarter_end_date'].min()
        df_date_max_before = rent_model_df['quarter_end_date'].max()
        print(f"  DataFrame date range before filter: {df_date_min_before} to {df_date_max_before}")
    
    rent_model_df = filter_window(rent_model_df, date_col="quarter_end_date")
    after_filter = len(rent_model_df)
    
    # Check if filter_window made the dataset empty
    if len(rent_model_df) == 0:
        error_msg_parts = [
            f"Dataset became empty after filter_window.",
            f"Before filter: {before_filter} rows, after filter: {after_filter} rows.",
            f"Configured window: {PROJECT_WINDOW_START} to {PROJECT_WINDOW_END}"
        ]
        if df_date_min_before is not None and df_date_max_before is not None:
            error_msg_parts.append(f"DataFrame date range before filter: {df_date_min_before} to {df_date_max_before}")
        raise ValueError("\n".join(error_msg_parts))
    
    print(f"  After filter_window: {before_filter} -> {after_filter} rows")
    
    # Sort final dataset
    rent_model_df = rent_model_df.sort_values(
        ['cma', 'unit_type', 'quarter_end_date']
    ).reset_index(drop=True)
    
    # Strong assertion before saving
    if rent_model_df.shape[0] == 0:
        # Gather diagnostic info for error message
        error_parts = ["Rent model dataset is empty after processing."]
        
        if 'unit_type' in rent_df.columns:
            unit_counts = rent_df['unit_type'].value_counts()
            error_parts.append(f"\nCounts by unit_type in rent_target:")
            for unit_type, count in unit_counts.items():
                error_parts.append(f"  {unit_type}: {count} rows")
        
        if 'cma' in rent_df.columns:
            cma_counts = rent_df['cma'].value_counts()
            error_parts.append(f"\nTop 10 CMAs by row count in rent_target:")
            for i, (cma, count) in enumerate(cma_counts.head(10).items(), 1):
                error_parts.append(f"  {i:2d}. {cma}: {count} rows")
        
        if 'quarter_end_date' in rent_df.columns:
            error_parts.append(f"\nDate range in rent_target: {rent_df['quarter_end_date'].min()} to {rent_df['quarter_end_date'].max()}")
        
        raise ValueError("\n".join(error_parts))
    
    # Save to parquet
    print(f"Saving rent model dataset to: {output_path}")
    rent_model_df.to_parquet(output_path, index=False, engine='pyarrow')
    
    print(f"\nRent model dataset created:")
    print(f"  Shape: {rent_model_df.shape}")
    print(f"  Date range: {rent_model_df['quarter_end_date'].min()} to {rent_model_df['quarter_end_date'].max()}")
    print(f"  Unique CMAs: {rent_model_df['cma'].nunique()}")
    print(f"  Unique unit types: {rent_model_df['unit_type'].nunique()}")
    print(f"  Total features: {len(rent_model_df.columns)}")
    
    return rent_model_df


def build_mortgage_model_dataset(
    mortgage_target_path: Optional[Path] = None,
    exog_path: Optional[Path] = None,
    output_path: Optional[Path] = None
) -> pd.DataFrame:
    """
    Build mortgage modeling dataset with exogenous features.
    
    Loads mortgage target data and exogenous features, merges them,
    and prepares target variables for SARIMAX modeling.
    
    Args:
        mortgage_target_path: Path to mortgage target parquet file.
                              Defaults to PROCESSED_DIR / "mortgage_target_quarterly.parquet".
        exog_path: Path to exogenous features parquet file.
                   Defaults to PROCESSED_DIR / "exog_quarterly.parquet".
        output_path: Path to save output parquet file.
                     Defaults to PROCESSED_DIR / "mortgage_model_dataset.parquet".
    
    Returns:
        DataFrame with columns:
        - quarter_end_date: Quarter-end Timestamp
        - y_level: Target mortgage level (for SARIMAX)
        - y_yoy_pct: Year-over-year percentage change (alternate target)
        - y_qoq_pct: Quarter-over-quarter percentage change
        - All exogenous features from exog_quarterly
    
    Raises:
        FileNotFoundError: If input files do not exist.
    """
    if mortgage_target_path is None:
        mortgage_target_path = PROCESSED_DIR / "mortgage_target_quarterly.parquet"
    if exog_path is None:
        exog_path = PROCESSED_DIR / "exog_quarterly.parquet"
    if output_path is None:
        output_path = PROCESSED_DIR / "mortgage_model_dataset.parquet"
    
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    print("Building mortgage modeling dataset...")
    print("=" * 60)
    
    # Load mortgage target data
    print(f"Loading mortgage target from: {mortgage_target_path}")
    if not mortgage_target_path.exists():
        raise FileNotFoundError(f"Mortgage target file not found: {mortgage_target_path}")
    
    mortgage_df = pd.read_parquet(mortgage_target_path)
    print(f"  Mortgage target shape: {mortgage_df.shape}")
    print(f"  Columns: {list(mortgage_df.columns)}")
    
    # Ensure required columns exist
    required_cols = ['quarter_end_date', 'y_level']
    missing_cols = [col for col in required_cols if col not in mortgage_df.columns]
    if missing_cols:
        raise ValueError(
            f"Missing required columns in mortgage data: {missing_cols}. "
            f"Available columns: {list(mortgage_df.columns)}"
        )
    
    # Convert quarter_end_date to datetime if needed
    mortgage_df['quarter_end_date'] = pd.to_datetime(mortgage_df['quarter_end_date'])
    
    # Sort by date
    mortgage_df = mortgage_df.sort_values('quarter_end_date').reset_index(drop=True)
    
    # Ensure y_yoy_pct and y_qoq_pct exist (they should from mortgage_target.py)
    if 'y_yoy_pct' not in mortgage_df.columns:
        mortgage_df['y_yoy_pct'] = mortgage_df['y_level'].pct_change(periods=4) * 100
    if 'y_qoq_pct' not in mortgage_df.columns:
        mortgage_df['y_qoq_pct'] = mortgage_df['y_level'].pct_change(periods=1) * 100
    
    # Load exogenous features
    print(f"Loading exogenous features from: {exog_path}")
    if not exog_path.exists():
        raise FileNotFoundError(f"Exogenous features file not found: {exog_path}")
    
    exog_df = pd.read_parquet(exog_path)
    print(f"  Exogenous features shape: {exog_df.shape}")
    
    # Ensure quarter_end_date is datetime
    exog_df['quarter_end_date'] = pd.to_datetime(exog_df['quarter_end_date'])
    
    # Merge with exogenous features
    print("Merging exogenous features...")
    mortgage_model_df = mortgage_df.merge(
        exog_df,
        on='quarter_end_date',
        how='left'
    )
    
    # Filter to project window
    mortgage_model_df = filter_window(mortgage_model_df, date_col="quarter_end_date")
    
    # Sort final dataset
    mortgage_model_df = mortgage_model_df.sort_values('quarter_end_date').reset_index(drop=True)
    
    # Save to parquet
    print(f"Saving mortgage model dataset to: {output_path}")
    mortgage_model_df.to_parquet(output_path, index=False, engine='pyarrow')
    
    print(f"\nMortgage model dataset created:")
    print(f"  Shape: {mortgage_model_df.shape}")
    print(f"  Date range: {mortgage_model_df['quarter_end_date'].min()} to {mortgage_model_df['quarter_end_date'].max()}")
    print(f"  Total features: {len(mortgage_model_df.columns)}")
    
    return mortgage_model_df


def build_all_datasets() -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Build both rent and mortgage modeling datasets.
    
    Returns:
        Tuple of (rent_model_df, mortgage_model_df)
    """
    print("Building all modeling datasets...")
    print("=" * 60)
    
    rent_df = build_rent_model_dataset()
    print("\n")
    mortgage_df = build_mortgage_model_dataset()
    
    return rent_df, mortgage_df


if __name__ == "__main__":
    """Run dataset building as standalone script."""
    rent_df, mortgage_df = build_all_datasets()
    
    print("\n" + "=" * 60)
    print("RENT MODEL DATASET SUMMARY")
    print("=" * 60)
    print(f"Shape: {rent_df.shape}")
    print(f"\nColumns ({len(rent_df.columns)}):")
    for col in rent_df.columns:
        non_null = rent_df[col].notna().sum()
        print(f"  {col}: {non_null}/{len(rent_df)} non-null")
    print(f"\nFirst few rows:")
    print(rent_df.head())
    
    print("\n" + "=" * 60)
    print("MORTGAGE MODEL DATASET SUMMARY")
    print("=" * 60)
    print(f"Shape: {mortgage_df.shape}")
    print(f"\nColumns ({len(mortgage_df.columns)}):")
    for col in mortgage_df.columns:
        non_null = mortgage_df[col].notna().sum()
        print(f"  {col}: {non_null}/{len(mortgage_df)} non-null")
    print(f"\nFirst few rows:")
    print(mortgage_df.head())

