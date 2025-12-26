"""
Extract and preprocess mortgage target series.

Loads chartered bank mortgage loans data and extracts a Canada-wide
quarterly target series for forecasting.
"""

from pathlib import Path
import pandas as pd
import numpy as np
from typing import Optional

from src.config import RAW_DIR, RAW_FILES, PROCESSED_DIR, quarter_end
from src.io.statcan_read import load_wide_pivot_csv, filter_window


def extract_mortgage_target(
    input_path: Optional[Path] = None,
    output_path: Optional[Path] = None
) -> pd.DataFrame:
    """
    Extract Canada-wide quarterly mortgage target series.
    
    Loads the mortgage CSV file and extracts the "Total, mortgages" series
    under "Mortgages in Canada outstanding" for Canada. Uses contains-based
    matching for robust series selection.
    
    Args:
        input_path: Path to input CSV file. Defaults to RAW_DIR / RAW_FILES["mortgage"].
        output_path: Path to save output parquet file. 
                     Defaults to PROCESSED_DIR / "mortgage_target_quarterly.parquet".
    
    Returns:
        DataFrame with columns:
        - quarter_end_date: Quarter-end Timestamp
        - y_level: Mortgage loan value (original level)
        - y_qoq_pct: Quarter-over-quarter percentage change
        - y_yoy_pct: Year-over-year percentage change
    
    Raises:
        FileNotFoundError: If input file does not exist.
        ValueError: If target series cannot be found.
    """
    # Set default paths
    if input_path is None:
        input_path = RAW_DIR / RAW_FILES["mortgage"]
    
    if output_path is None:
        output_path = PROCESSED_DIR / "mortgage_target_quarterly.parquet"
    
    # Ensure output directory exists
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Load wide pivot CSV
    print(f"Loading mortgage data from: {input_path}")
    df = load_wide_pivot_csv(input_path)
    
    # Filter for Canada if geo column exists and contains "Canada"
    if "geo" in df.columns:
        df_filtered = df[df["geo"] == "Canada"].copy()
        if df_filtered.empty:
            # Try case-insensitive contains as fallback
            df_filtered = df[df["geo"].str.contains("Canada", case=False, na=False)].copy()
    else:
        df_filtered = df.copy()
    
    # Remove junk rows (series containing "how to cite")
    df_filtered = df_filtered[
        ~df_filtered["series"].str.contains("how to cite", case=False, na=False)
    ].copy()
    
    # Preferred exact match
    target_series_exact = "Mortgages in Canada outstanding | Total, mortgages"
    mortgage_series = df_filtered[df_filtered["series"] == target_series_exact].copy()
    
    if mortgage_series.empty:
        # Fallback: contains-based matching
        print(f"Exact match not found, using fallback selection...")
        mortgage_series = df_filtered[
            df_filtered["series"].str.contains("mortgages in canada outstanding", case=False, na=False) &
            df_filtered["series"].str.contains("total, mortgages", case=False, na=False)
        ].copy()
    
    # Assertion: at least 10 rows after filtering
    if len(mortgage_series) < 10:
        # Get mortgage-like series for error message
        mortgage_like = df_filtered[
            df_filtered["series"].str.contains("mortgage", case=False, na=False)
        ]["series"].unique()[:30]
        
        raise ValueError(
            f"Insufficient data after filtering: found {len(mortgage_series)} rows, need at least 10. "
            f"Top 30 mortgage-like series found:\n{list(mortgage_like)}"
        )
    
    # If multiple series match, take the first one (shouldn't happen with exact match)
    if mortgage_series["series"].nunique() > 1:
        selected_series = mortgage_series["series"].iloc[0]
        mortgage_series = mortgage_series[mortgage_series["series"] == selected_series].copy()
        print(f"Warning: Multiple series matched. Using: {selected_series}")
    else:
        print(f"Selected series: {mortgage_series['series'].iloc[0]}")
    
    # Rename date to quarter_end_date
    if "date" in mortgage_series.columns:
        mortgage_series = mortgage_series.rename(columns={"date": "quarter_end_date"})
    
    # Ensure quarter_end_date exists and convert to quarter-end dates
    if "quarter_end_date" not in mortgage_series.columns:
        raise ValueError("Date column not found in mortgage series data")
    
    mortgage_series["quarter_end_date"] = mortgage_series["quarter_end_date"].apply(quarter_end)
    
    # Ensure values are numeric float
    mortgage_series["value"] = pd.to_numeric(mortgage_series["value"], errors="coerce")
    
    # Aggregate to quarterly (in case of multiple observations per quarter)
    # Use the last value in each quarter
    df_quarterly = mortgage_series.groupby("quarter_end_date").agg({
        "value": "last"  # Take last observation in quarter
    }).reset_index()
    
    # Rename value to y_level
    df_quarterly = df_quarterly.rename(columns={"value": "y_level"})
    
    # Sort by date
    df_quarterly = df_quarterly.sort_values("quarter_end_date").reset_index(drop=True)
    
    # Remove any rows with missing values
    df_quarterly = df_quarterly.dropna(subset=["y_level"])
    
    # Calculate quarter-over-quarter percentage change
    df_quarterly["y_qoq_pct"] = df_quarterly["y_level"].pct_change(periods=1) * 100
    
    # Calculate year-over-year percentage change (4 quarters)
    df_quarterly["y_yoy_pct"] = df_quarterly["y_level"].pct_change(periods=4) * 100
    
    # Filter to project window
    df_quarterly = filter_window(df_quarterly, date_col="quarter_end_date")
    
    # Select and order columns
    result = df_quarterly[["quarter_end_date", "y_level", "y_qoq_pct", "y_yoy_pct"]].copy()
    
    # Save to parquet
    print(f"Saving mortgage target to: {output_path}")
    result.to_parquet(output_path, index=False, engine="pyarrow")
    
    print(f"Mortgage target extracted: {len(result)} quarters")
    print(f"Date range: {result['quarter_end_date'].min()} to {result['quarter_end_date'].max()}")
    
    return result


if __name__ == "__main__":
    """Run extraction as standalone script."""
    df = extract_mortgage_target()
    print("\nFirst few rows:")
    print(df.head(10))
    print("\nLast few rows:")
    print(df.tail(10))
    print(f"\nSummary statistics:")
    print(df[["y_level", "y_qoq_pct", "y_yoy_pct"]].describe())

