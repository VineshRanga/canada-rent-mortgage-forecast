"""
Extract and aggregate exogenous features from multiple StatCan data sources.

Builds a quarterly features table with rates, labour market, housing starts,
population, migration, and non-permanent residents data.
"""

from pathlib import Path
from typing import Optional
import pandas as pd
import numpy as np
import re
import csv as _csv

from src.config import (
    RAW_DIR,
    RAW_FILES,
    PROCESSED_DIR,
    quarter_end,
)
from src.io.statcan_read import load_wide_pivot_csv, load_tidy_statcan_csv, filter_window

# Regex for month-name + year token (for searching in text, case-insensitive)
MONTH_NAME_YEAR_TOKEN = re.compile(
    r"\b(January|February|March|April|May|June|July|August|September|October|November|December)\s+\d{4}\b",
    re.IGNORECASE
)


def extract_rates_features() -> pd.DataFrame:
    """
    Extract quarterly rate features from weekly financial markets data.
    
    Extracts Bank rate, Prime rate, and GoC 5-year yield (or closest 5y benchmark).
    Uses quarter-end (last observation in quarter) aggregation.
    
    Returns:
        DataFrame with columns: quarter_end_date, rate_bank, rate_prime, rate_goc_5y
    """
    print("Extracting rate features...")
    filepath = RAW_DIR / RAW_FILES["financial_markets"]
    
    df = load_wide_pivot_csv(filepath)
    
    # Clean data
    df["series"] = df["series"].astype(str).str.strip()
    df = df[df["series"].ne("")]  # remove blank series
    df = df.dropna(subset=["value"])
    
    # Restrict geo to Canada if present
    if "geo" in df.columns:
        df = df[df["geo"].str.strip().eq("Canada")]
    
    df = filter_window(df)
    
    # Select series using case-insensitive contains
    # Bank rate: exact match (case-insensitive)
    bank_rate = df[df["series"].str.strip().str.lower().eq("bank rate")].copy()
    
    # Prime rate: contains "Prime rate" AND contains "Chartered bank administered interest rates"
    prime_rate = df[
        df["series"].str.contains("Prime rate", case=False, na=False) &
        df["series"].str.contains("Chartered bank administered interest rates", case=False, na=False)
    ].copy()
    
    # GoC 5-year: contains "Government of Canada" AND contains "5 year" AND contains ("bond yield" or "benchmark")
    goc_5y = df[
        df["series"].str.contains("Government of Canada", case=False, na=False) &
        df["series"].str.contains("5 year", case=False, na=False) &
        (df["series"].str.contains("bond yield", case=False, na=False) |
         df["series"].str.contains("benchmark", case=False, na=False))
    ].copy()
    
    # Create dataframe with columns: date, bank_rate, prime_rate, goc_5y
    rates_list = []
    
    # Process bank rate
    if not bank_rate.empty:
        bank_rate["quarter_end_date"] = bank_rate["date"].apply(quarter_end)
        bank_quarterly = bank_rate.groupby("quarter_end_date")["value"].last().reset_index()
        bank_quarterly = bank_quarterly.rename(columns={"value": "bank_rate"})
        rates_list.append(bank_quarterly[["quarter_end_date", "bank_rate"]])
    else:
        print("  Warning: Bank rate not found")
    
    # Process prime rate
    if not prime_rate.empty:
        prime_rate["quarter_end_date"] = prime_rate["date"].apply(quarter_end)
        prime_quarterly = prime_rate.groupby("quarter_end_date")["value"].last().reset_index()
        prime_quarterly = prime_quarterly.rename(columns={"value": "prime_rate"})
        rates_list.append(prime_quarterly[["quarter_end_date", "prime_rate"]])
    else:
        print("  Warning: Prime rate not found")
    
    # Process GoC 5-year
    if not goc_5y.empty:
        goc_5y["quarter_end_date"] = goc_5y["date"].apply(quarter_end)
        goc_quarterly = goc_5y.groupby("quarter_end_date")["value"].last().reset_index()
        goc_quarterly = goc_quarterly.rename(columns={"value": "goc_5y"})
        rates_list.append(goc_quarterly[["quarter_end_date", "goc_5y"]])
    else:
        print("  Warning: GoC 5-year yield not found")
    
    # Merge all rates on quarter_end_date
    if rates_list:
        rates_df = rates_list[0]
        for df_part in rates_list[1:]:
            rates_df = rates_df.merge(df_part, on="quarter_end_date", how="outer")
    else:
        # If no rates found, create empty dataframe with expected columns
        rates_df = pd.DataFrame(columns=["quarter_end_date", "bank_rate", "prime_rate", "goc_5y"])
    
    # Rename columns to add rate_ prefix
    rates_df = rates_df.rename(columns={
        "bank_rate": "rate_bank",
        "prime_rate": "rate_prime",
        "goc_5y": "rate_goc_5y"
    })
    
    # Ensure all expected columns exist
    for col in ["rate_bank", "rate_prime", "rate_goc_5y"]:
        if col not in rates_df.columns:
            rates_df[col] = np.nan
    
    # Sort by quarter_end_date
    rates_df = rates_df.sort_values("quarter_end_date").reset_index(drop=True)
    
    # Filter to project window
    rates_df = filter_window(rates_df, date_col="quarter_end_date")
    
    print(f"  Extracted rates: {len(rates_df)} quarters")
    return rates_df


def extract_labour_features() -> pd.DataFrame:
    """
    Extract quarterly labour market features.
    
    Extracts unemployment rate (seasonally adjusted if available) and employment count.
    Uses quarterly average aggregation.
    
    Returns:
        DataFrame with columns: quarter_end_date, labour_unemployment_rate
        # TODO: Re-add labour_employment_count once we confirm a clean "total employees" series exists
    """
    print("Extracting labour features...")
    
    # Unemployment rate - special handling for labour force file
    unemp_filepath = RAW_DIR / RAW_FILES["labour_force"]
    
    # Detect header row by scanning for month-name-year tokens
    header_line_index = None
    with unemp_filepath.open("r", encoding="utf-8-sig", errors="replace") as f:
        for i, line in enumerate(f):
            if i >= 5000:
                break
            n = len(MONTH_NAME_YEAR_TOKEN.findall(line))
            if n >= 10:
                header_line_index = i
                break
    
    if header_line_index is None:
        raise ValueError(f"Could not find header row with month-name-year tokens in {unemp_filepath}")
    
    # Read CSV manually
    df_unemp = pd.read_csv(
        unemp_filepath,
        skiprows=header_line_index,
        engine="python",
        sep=",",
        quotechar='"',
        on_bad_lines="skip",
    )
    
    # Sanitize column names
    df_unemp.columns = [str(c).strip().strip('"') for c in df_unemp.columns]
    
    # Expected structure: first two columns are "Labour force characteristics" and "Data type"
    if len(df_unemp.columns) < 2:
        raise ValueError(f"Expected at least 2 columns in {unemp_filepath}, got {len(df_unemp.columns)}")
    
    col0_name = df_unemp.columns[0]
    col1_name = df_unemp.columns[1]
    
    # Filter rows where:
    # - "Labour force characteristics" contains "Unemployment rate" (case-insensitive)
    # - and "Data type" contains "Seasonally adjusted" (case-insensitive)
    mask = (
        df_unemp[col0_name].astype(str).str.contains("Unemployment rate", case=False, na=False) &
        df_unemp[col1_name].astype(str).str.contains("Seasonally adjusted", case=False, na=False)
    )
    
    unemp_rows = df_unemp[mask].copy()
    
    if unemp_rows.empty:
        raise ValueError(f"No unemployment rate (seasonally adjusted) found in {unemp_filepath}")
    
    # If multiple match, prefer row containing "15 years and over" or "Total population"
    if len(unemp_rows) > 1:
        preferred_mask = (
            unemp_rows[col0_name].astype(str).str.contains("15 years and over", case=False, na=False) |
            unemp_rows[col0_name].astype(str).str.contains("Total population", case=False, na=False)
        )
        if preferred_mask.any():
            unemp_rows = unemp_rows[preferred_mask].copy()
        # Otherwise take first match
        if len(unemp_rows) > 1:
            unemp_rows = unemp_rows.iloc[[0]].copy()
    
    # Get the single row
    unemp_row = unemp_rows.iloc[0]
    
    # Identify month columns (columns that match MONTH_NAME_YEAR_TOKEN)
    month_cols = []
    for col in df_unemp.columns[2:]:  # Skip first two stub columns
        col_str = str(col).strip()
        if MONTH_NAME_YEAR_TOKEN.search(col_str):
            month_cols.append(col)
    
    if not month_cols:
        raise ValueError(f"No month-name-year columns found in {unemp_filepath}")
    
    # Melt month columns to long format
    unemp_long = []
    for col in month_cols:
        value = unemp_row[col]
        if pd.notna(value):
            date_str = str(col).strip()
            date = pd.to_datetime(date_str, errors="coerce")
            if pd.notna(date):
                # Convert value to float
                try:
                    value_float = float(value)
                    unemp_long.append({
                        "date": date,
                        "geo": "Canada",
                        "series": "unemployment_rate_sa",
                        "value": value_float
                    })
                except (ValueError, TypeError):
                    # Skip non-numeric values
                    continue
    
    df_unemp_long = pd.DataFrame(unemp_long)
    
    # Drop NaN values
    df_unemp_long = df_unemp_long.dropna(subset=["value"])
    
    if df_unemp_long.empty:
        raise ValueError(f"No valid unemployment rate data found in {unemp_filepath}")
    
    # Aggregate to quarterly average
    df_unemp_long["quarter_end_date"] = df_unemp_long["date"].apply(quarter_end)
    unemp_quarterly = df_unemp_long.groupby("quarter_end_date")["value"].mean().reset_index()
    unemp_quarterly = unemp_quarterly.rename(columns={"value": "labour_unemployment_rate"})
    
    # TODO: Re-add employees extraction once we confirm a clean "total employees" series exists
    # Employment count extraction temporarily disabled for v1
    # emp_filepath = RAW_DIR / RAW_FILES["employees"]
    # df_emp = load_wide_pivot_csv(emp_filepath)
    # df_emp = filter_window(df_emp)
    # 
    # df_emp_canada = df_emp[
    #     df_emp["geo"].str.contains("Canada", case=False, na=False) &
    #     df_emp["series"].str.contains("employment", case=False, na=False)
    # ].copy()
    # 
    # # Helper function to aggregate to quarterly (average)
    # def agg_to_quarterly_avg(df_series: pd.DataFrame) -> pd.DataFrame:
    #     if df_series.empty:
    #         return pd.DataFrame(columns=["quarter_end_date", "value"])
    #     
    #     df_series["quarter_end_date"] = df_series["date"].apply(quarter_end)
    #     result = df_series.groupby("quarter_end_date")["value"].mean().reset_index()
    #     return result
    
    # Build quarterly DataFrame
    labour_df = pd.DataFrame({"quarter_end_date": pd.date_range(
        start=pd.to_datetime("2019-01-01"),
        end=pd.to_datetime("2025-12-31"),
        freq="QE"
    )})
    labour_df["quarter_end_date"] = labour_df["quarter_end_date"].apply(quarter_end)
    
    # Merge unemployment
    if not unemp_quarterly.empty:
        labour_df = labour_df.merge(
            unemp_quarterly[["quarter_end_date", "labour_unemployment_rate"]],
            on="quarter_end_date",
            how="left"
        )
    else:
        labour_df["labour_unemployment_rate"] = np.nan
        print("  Warning: Unemployment rate not found")
    
    # TODO: Re-add employment merge once employees extraction is re-enabled
    # # Merge employment
    # emp_df = agg_to_quarterly_avg(df_emp_canada)
    # if not emp_df.empty:
    #     labour_df = labour_df.merge(
    #         emp_df[["quarter_end_date", "value"]],
    #         on="quarter_end_date",
    #         how="left"
    #     )
    #     labour_df = labour_df.rename(columns={"value": "labour_employment_count"})
    # else:
    #     labour_df["labour_employment_count"] = np.nan
    #     print("  Warning: Employment count not found")
    
    # Filter to project window
    labour_df = filter_window(labour_df, date_col="quarter_end_date")
    
    print(f"  Extracted labour: {len(labour_df)} quarters")
    return labour_df


def extract_housing_starts_features() -> pd.DataFrame:
    """
    Extract quarterly housing starts features.
    
    Computes quarterly sum for Canada and national total across all geos.
    
    Returns:
        DataFrame with columns: quarter_end_date, starts_canada, starts_national_total
    """
    print("Extracting housing starts features...")
    filepath = RAW_DIR / RAW_FILES["housing_starts"]
    
    df = load_tidy_statcan_csv(filepath)
    df = filter_window(df)
    
    # Filter for "Total units" series
    df_total = df[df["series"].str.contains("total", case=False, na=False)].copy()
    
    # Canada-level
    df_canada = df_total[df_total["geo"].str.contains("Canada", case=False, na=False)].copy()
    
    # National sum across all geos
    df_all_geos = df_total.copy()
    
    # Aggregate to quarterly (sum)
    def agg_to_quarterly_sum(df_series: pd.DataFrame) -> pd.DataFrame:
        if df_series.empty:
            return pd.DataFrame(columns=["quarter_end_date", "value"])
        
        df_series["quarter_end_date"] = df_series["date"].apply(quarter_end)
        result = df_series.groupby("quarter_end_date")["value"].sum().reset_index()
        return result
    
    # Build quarterly DataFrame
    starts_df = pd.DataFrame({"quarter_end_date": pd.date_range(
        start=pd.to_datetime("2019-01-01"),
        end=pd.to_datetime("2025-12-31"),
        freq="QE"
    )})
    starts_df["quarter_end_date"] = starts_df["quarter_end_date"].apply(quarter_end)
    
    # Merge Canada
    canada_df = agg_to_quarterly_sum(df_canada)
    if not canada_df.empty:
        starts_df = starts_df.merge(
            canada_df[["quarter_end_date", "value"]],
            on="quarter_end_date",
            how="left"
        )
        starts_df = starts_df.rename(columns={"value": "starts_canada"})
    else:
        starts_df["starts_canada"] = np.nan
    
    # Merge national total (sum across all geos)
    national_df = agg_to_quarterly_sum(df_all_geos)
    if not national_df.empty:
        starts_df = starts_df.merge(
            national_df[["quarter_end_date", "value"]],
            on="quarter_end_date",
            how="left"
        )
        starts_df = starts_df.rename(columns={"value": "starts_national_total"})
    else:
        starts_df["starts_national_total"] = np.nan
    
    # Filter to project window
    starts_df = filter_window(starts_df, date_col="quarter_end_date")
    
    print(f"  Extracted housing starts: {len(starts_df)} quarters")
    return starts_df


def extract_population_features() -> pd.DataFrame:
    """
    Extract quarterly population features.
    
    Extracts population level and QoQ/YoY growth for Canada.
    
    Returns:
        DataFrame with columns: quarter_end_date, pop_canada_level, pop_canada_qoq_pct, pop_canada_yoy_pct
    """
    print("Extracting population features...")
    filepath = RAW_DIR / RAW_FILES["population"]
    
    df = load_tidy_statcan_csv(filepath)
    df = filter_window(df)
    
    # Filter for Canada
    df_canada = df[df["geo"].str.contains("Canada", case=False, na=False)].copy()
    
    # Try to find total population series
    df_pop = df_canada[
        df_canada["series"].str.contains("total", case=False, na=False) |
        df_canada["series"].str.contains("population", case=False, na=False)
    ].copy()
    
    if df_pop.empty:
        # If no match, use first series
        df_pop = df_canada.copy()
    
    # If multiple series, take the one with highest average value (likely total)
    if df_pop["series"].nunique() > 1:
        series_means = df_pop.groupby("series")["value"].mean().sort_values(ascending=False)
        selected_series = series_means.index[0]
        df_pop = df_pop[df_pop["series"] == selected_series].copy()
    
    # Aggregate to quarterly (take last observation in quarter)
    df_pop["quarter_end_date"] = df_pop["date"].apply(quarter_end)
    pop_df = df_pop.groupby("quarter_end_date")["value"].last().reset_index()
    pop_df = pop_df.rename(columns={"value": "pop_canada_level"})
    
    # Sort by date
    pop_df = pop_df.sort_values("quarter_end_date").reset_index(drop=True)
    
    # Calculate QoQ and YoY growth
    pop_df["pop_canada_qoq_pct"] = pop_df["pop_canada_level"].pct_change(periods=1) * 100
    pop_df["pop_canada_yoy_pct"] = pop_df["pop_canada_level"].pct_change(periods=4) * 100
    
    # Filter to project window
    pop_df = filter_window(pop_df, date_col="quarter_end_date")
    
    print(f"  Extracted population: {len(pop_df)} quarters")
    return pop_df


def extract_migration_features() -> pd.DataFrame:
    """
    Extract quarterly migration features.
    
    Extracts migration components and builds aggregates like total immigrants
    and net international migration.
    
    Returns:
        DataFrame with columns: quarter_end_date, mig_immigrants, mig_emigrants, 
        mig_net_international, mig_total_components
    """
    print("Extracting migration features...")
    filepath = RAW_DIR / RAW_FILES["migration"]
    
    df = load_tidy_statcan_csv(filepath)
    df = filter_window(df)
    
    # Filter for Canada
    df_canada = df[df["geo"].str.contains("Canada", case=False, na=False)].copy()
    
    # Extract components
    immigrants = df_canada[
        df_canada["series"].str.contains("immigrant", case=False, na=False)
    ].copy()
    
    emigrants = df_canada[
        df_canada["series"].str.contains("emigrant", case=False, na=False)
    ].copy()
    
    # Net international migration
    net_migration = df_canada[
        df_canada["series"].str.contains("net", case=False, na=False) &
        df_canada["series"].str.contains("international", case=False, na=False)
    ].copy()
    
    # Aggregate to quarterly (sum for counts, average for rates)
    def agg_to_quarterly_sum(df_series: pd.DataFrame) -> pd.DataFrame:
        if df_series.empty:
            return pd.DataFrame(columns=["quarter_end_date", "value"])
        
        df_series["quarter_end_date"] = df_series["date"].apply(quarter_end)
        result = df_series.groupby("quarter_end_date")["value"].sum().reset_index()
        return result
    
    # Build quarterly DataFrame
    mig_df = pd.DataFrame({"quarter_end_date": pd.date_range(
        start=pd.to_datetime("2019-01-01"),
        end=pd.to_datetime("2025-12-31"),
        freq="QE"
    )})
    mig_df["quarter_end_date"] = mig_df["quarter_end_date"].apply(quarter_end)
    
    # Merge immigrants
    imm_df = agg_to_quarterly_sum(immigrants)
    if not imm_df.empty:
        mig_df = mig_df.merge(
            imm_df[["quarter_end_date", "value"]],
            on="quarter_end_date",
            how="left"
        )
        mig_df = mig_df.rename(columns={"value": "mig_immigrants"})
    else:
        mig_df["mig_immigrants"] = np.nan
    
    # Merge emigrants
    emi_df = agg_to_quarterly_sum(emigrants)
    if not emi_df.empty:
        mig_df = mig_df.merge(
            emi_df[["quarter_end_date", "value"]],
            on="quarter_end_date",
            how="left"
        )
        mig_df = mig_df.rename(columns={"value": "mig_emigrants"})
    else:
        mig_df["mig_emigrants"] = np.nan
    
    # Merge net international migration
    net_df = agg_to_quarterly_sum(net_migration)
    if not net_df.empty:
        mig_df = mig_df.merge(
            net_df[["quarter_end_date", "value"]],
            on="quarter_end_date",
            how="left"
        )
        mig_df = mig_df.rename(columns={"value": "mig_net_international"})
    else:
        # Calculate if components exist
        if "mig_immigrants" in mig_df.columns and "mig_emigrants" in mig_df.columns:
            mig_df["mig_net_international"] = (
                mig_df["mig_immigrants"] - mig_df["mig_emigrants"]
            )
        else:
            mig_df["mig_net_international"] = np.nan
    
    # Total components (sum of all migration components for reference)
    all_components = agg_to_quarterly_sum(df_canada)
    if not all_components.empty:
        mig_df = mig_df.merge(
            all_components[["quarter_end_date", "value"]],
            on="quarter_end_date",
            how="left"
        )
        mig_df = mig_df.rename(columns={"value": "mig_total_components"})
    else:
        mig_df["mig_total_components"] = np.nan
    
    # Filter to project window
    mig_df = filter_window(mig_df, date_col="quarter_end_date")
    
    print(f"  Extracted migration: {len(mig_df)} quarters")
    return mig_df


def extract_npr_features() -> pd.DataFrame:
    """
    Extract quarterly non-permanent residents features.
    
    Note: Data starts around 2021Q3, so earlier periods will be NaN.
    
    Returns:
        DataFrame with columns: quarter_end_date, npr_canada_total
    """
    print("Extracting NPR features...")
    filepath = RAW_DIR / RAW_FILES["non_permanent_residents"]
    
    df = load_tidy_statcan_csv(filepath)
    df = filter_window(df)
    
    # Filter for Canada
    df_canada = df[df["geo"].str.contains("Canada", case=False, na=False)].copy()
    
    # Try to find total series
    df_npr = df_canada[
        df_canada["series"].str.contains("total", case=False, na=False)
    ].copy()
    
    if df_npr.empty:
        df_npr = df_canada.copy()
    
    # If multiple series, take the one with highest average value
    if df_npr["series"].nunique() > 1:
        series_means = df_npr.groupby("series")["value"].mean().sort_values(ascending=False)
        selected_series = series_means.index[0]
        df_npr = df_npr[df_npr["series"] == selected_series].copy()
    
    # Aggregate to quarterly (take last observation in quarter)
    df_npr["quarter_end_date"] = df_npr["date"].apply(quarter_end)
    npr_df = df_npr.groupby("quarter_end_date")["value"].last().reset_index()
    npr_df = npr_df.rename(columns={"value": "npr_canada_total"})
    
    # Filter to project window
    npr_df = filter_window(npr_df, date_col="quarter_end_date")
    
    print(f"  Extracted NPR: {len(npr_df)} quarters")
    return npr_df


def build_exogenous_features(
    output_path: Optional[Path] = None
) -> pd.DataFrame:
    """
    Build quarterly exogenous features table from all data sources.
    
    Merges rates, labour, housing starts, population, migration, and NPR features
    into a single quarterly DataFrame.
    
    Args:
        output_path: Path to save output parquet file.
                    Defaults to PROCESSED_DIR / "exog_quarterly.parquet".
    
    Returns:
        DataFrame with all exogenous features keyed by quarter_end_date.
    """
    if output_path is None:
        output_path = PROCESSED_DIR / "exog_quarterly.parquet"
    
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    print("Building exogenous features table...")
    print("=" * 60)
    
    # Extract all feature groups
    rates_df = extract_rates_features()
    labour_df = extract_labour_features()
    starts_df = extract_housing_starts_features()
    pop_df = extract_population_features()
    mig_df = extract_migration_features()
    npr_df = extract_npr_features()
    
    # Create base quarterly index
    quarter_dates = pd.date_range(
        start=pd.to_datetime("2019-01-01"),
        end=pd.to_datetime("2025-12-31"),
        freq="QE"
    )
    quarter_dates = pd.Series(quarter_dates).apply(quarter_end).unique()
    quarter_dates = sorted(quarter_dates)
    
    # Start with base DataFrame
    features_df = pd.DataFrame({"quarter_end_date": quarter_dates})
    
    # Merge all feature groups
    print("\nMerging feature groups...")
    for name, df in [
        ("rates", rates_df),
        ("labour", labour_df),
        ("housing starts", starts_df),
        ("population", pop_df),
        ("migration", mig_df),
        ("NPR", npr_df),
    ]:
        if not df.empty:
            features_df = features_df.merge(
                df,
                on="quarter_end_date",
                how="left"
            )
            print(f"  Merged {name}: {df.shape[1] - 1} features")
    
    # Filter to project window
    features_df = filter_window(features_df, date_col="quarter_end_date")
    
    # Sort by date
    features_df = features_df.sort_values("quarter_end_date").reset_index(drop=True)
    
    # Save to parquet
    print(f"\nSaving exogenous features to: {output_path}")
    features_df.to_parquet(output_path, index=False, engine="pyarrow")
    
    print(f"\nExogenous features table created:")
    print(f"  Shape: {features_df.shape}")
    print(f"  Date range: {features_df['quarter_end_date'].min()} to {features_df['quarter_end_date'].max()}")
    print(f"  Features: {len(features_df.columns) - 1}")
    print(f"\nColumns:")
    for col in features_df.columns:
        if col != "quarter_end_date":
            non_null = features_df[col].notna().sum()
            print(f"  {col}: {non_null}/{len(features_df)} non-null")
    
    return features_df


if __name__ == "__main__":
    """Run extraction as standalone script."""
    df = build_exogenous_features()
    print("\nFirst few rows:")
    print(df.head(10))
    print("\nSummary statistics:")
    print(df.describe())

