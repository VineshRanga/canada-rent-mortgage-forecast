"""
Generate matplotlib "rates overlay" story charts.

Creates dual-axis charts comparing rent changes (YoY/QoQ) with interest rates.
"""

from pathlib import Path
from typing import List, Optional
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from src.config import OUTPUT_DIR, PROCESSED_DIR


def format_quarter_label(date: pd.Timestamp) -> str:
    """
    Format a quarter-end date as "YYYYQ#" string.
    
    Args:
        date: Quarter-end date (pd.Timestamp or datetime).
        
    Returns:
        String in format "YYYYQ#" (e.g., "2022Q3").
    """
    year = date.year
    quarter = (date.month - 1) // 3 + 1
    return f"{year}Q{quarter}"


def prepare_region_data(
    rent_df: pd.DataFrame,
    cma_list: List[str]
) -> pd.DataFrame:
    """
    Prepare aggregated data for a region (Toronto, GTA proxy, etc.).
    
    Args:
        rent_df: DataFrame with columns: quarter_end_date, cma, unit_type, y
        cma_list: List of CMA names to include in the region.
        
    Returns:
        DataFrame with aggregated rent data by quarter_end_date and unit_type,
        including YoY and QoQ percentage changes.
    """
    # Filter to specified CMAs
    df_region = rent_df[rent_df['cma'].isin(cma_list)].copy()
    
    if len(df_region) == 0:
        return pd.DataFrame()
    
    # Ensure numeric
    df_region['y'] = pd.to_numeric(df_region['y'], errors='coerce')
    df_region = df_region.dropna(subset=['y'])
    
    # Aggregate by quarter_end_date and unit_type (mean across CMAs)
    df_agg = df_region.groupby(['quarter_end_date', 'unit_type']).agg({
        'y': 'mean'
    }).reset_index()
    
    # Sort by quarter_end_date and unit_type
    df_agg = df_agg.sort_values(['unit_type', 'quarter_end_date']).reset_index(drop=True)
    
    # Compute lags within each unit_type group
    df_agg['y_lag1'] = df_agg.groupby('unit_type')['y'].shift(1)
    df_agg['y_lag4'] = df_agg.groupby('unit_type')['y'].shift(4)
    
    # Compute YoY % and QoQ %
    df_agg['yoy_pct'] = (df_agg['y'] / df_agg['y_lag4'] - 1) * 100
    df_agg['qoq_pct'] = (df_agg['y'] / df_agg['y_lag1'] - 1) * 100
    
    return df_agg


def plot_yoy_vs_bank_rate(
    rent_df: pd.DataFrame,
    rates_df: pd.DataFrame,
    region_name: str,
    cma_list: List[str],
    save_path: Path
) -> None:
    """
    Plot YoY rent change vs Bank rate (dual axis).
    
    Args:
        rent_df: Rent target DataFrame.
        rates_df: Exogenous features DataFrame with rate_bank.
        region_name: Name of the region (for title).
        cma_list: List of CMA names for the region.
        save_path: Path to save the PNG file.
    """
    # Prepare region data
    region_data = prepare_region_data(rent_df, cma_list)
    
    if len(region_data) == 0:
        print(f"[WARN] No data for {region_name}, skipping chart")
        return
    
    # Aggregate across unit types (mean)
    region_agg = region_data.groupby('quarter_end_date').agg({
        'yoy_pct': 'mean'
    }).reset_index()
    
    # Merge with rates
    plot_df = region_agg.merge(
        rates_df[['quarter_end_date', 'rate_bank']],
        on='quarter_end_date',
        how='inner'
    )
    
    # Drop rows with missing values
    plot_df = plot_df.dropna(subset=['yoy_pct', 'rate_bank'])
    plot_df = plot_df.sort_values('quarter_end_date')
    
    if len(plot_df) == 0:
        print(f"[WARN] No overlapping data for {region_name} YoY vs Bank rate, skipping")
        return
    
    # Format quarter labels
    plot_df['quarter_label'] = plot_df['quarter_end_date'].apply(format_quarter_label)
    
    # Create figure with dual axes
    fig, ax1 = plt.subplots(figsize=(12, 6))
    
    # Left axis: YoY %
    ax1.set_xlabel('Quarter (period end) — aligned quarterly reporting')
    ax1.set_ylabel('Rent YoY change (%) — captures inflation-like dynamics')
    line1 = ax1.plot(plot_df['quarter_label'], plot_df['yoy_pct'], 
                     marker='o', markersize=4, linewidth=1.5, label='Rent YoY %')
    # Get color from line for axis label
    color1 = line1[0].get_color()
    ax1.tick_params(axis='y', labelcolor=color1)
    ax1.grid(True, alpha=0.3)
    
    # Right axis: Bank rate
    ax2 = ax1.twinx()
    ax2.set_ylabel('Bank rate (%) — monetary policy signal')
    line2 = ax2.plot(plot_df['quarter_label'], plot_df['rate_bank'],
                     marker='s', markersize=4, linewidth=1.5, label='Bank rate', linestyle='--')
    # Get color from line for axis label
    color2 = line2[0].get_color()
    ax2.tick_params(axis='y', labelcolor=color2)
    
    # Title
    plt.title(f'{region_name}: Rent YoY Change vs Bank Rate')
    
    # Rotate x ticks
    plt.xticks(rotation=35, ha='right')
    
    # Legend (outside plot area)
    lines = line1 + line2
    labels = [l.get_label() for l in lines]
    ax1.legend(lines, labels, loc='upper left', bbox_to_anchor=(0, 1))
    
    # Use tight layout
    plt.tight_layout()
    
    # Save
    save_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"[OK] Saved: {save_path}")


def plot_qoq_vs_goc5y(
    rent_df: pd.DataFrame,
    rates_df: pd.DataFrame,
    region_name: str,
    cma_list: List[str],
    save_path: Path
) -> None:
    """
    Plot QoQ rent change vs 5-year GoC yield (dual axis).
    
    Args:
        rent_df: Rent target DataFrame.
        rates_df: Exogenous features DataFrame with rate_goc_5y.
        region_name: Name of the region (for title).
        cma_list: List of CMA names for the region.
        save_path: Path to save the PNG file.
    """
    # Prepare region data
    region_data = prepare_region_data(rent_df, cma_list)
    
    if len(region_data) == 0:
        print(f"[WARN] No data for {region_name}, skipping chart")
        return
    
    # Aggregate across unit types (mean)
    region_agg = region_data.groupby('quarter_end_date').agg({
        'qoq_pct': 'mean'
    }).reset_index()
    
    # Merge with rates
    plot_df = region_agg.merge(
        rates_df[['quarter_end_date', 'rate_goc_5y']],
        on='quarter_end_date',
        how='inner'
    )
    
    # Drop rows with missing values
    plot_df = plot_df.dropna(subset=['qoq_pct', 'rate_goc_5y'])
    plot_df = plot_df.sort_values('quarter_end_date')
    
    if len(plot_df) == 0:
        print(f"[WARN] No overlapping data for {region_name} QoQ vs GoC 5y, skipping")
        return
    
    # Format quarter labels
    plot_df['quarter_label'] = plot_df['quarter_end_date'].apply(format_quarter_label)
    
    # Create figure with dual axes
    fig, ax1 = plt.subplots(figsize=(12, 6))
    
    # Left axis: QoQ %
    ax1.set_xlabel('Quarter (period end) — aligned quarterly reporting')
    ax1.set_ylabel('Rent QoQ change (%) — short-horizon momentum')
    line1 = ax1.plot(plot_df['quarter_label'], plot_df['qoq_pct'],
                     marker='o', markersize=4, linewidth=1.5, label='Rent QoQ %')
    # Get color from line for axis label
    color1 = line1[0].get_color()
    ax1.tick_params(axis='y', labelcolor=color1)
    ax1.grid(True, alpha=0.3)
    
    # Right axis: GoC 5y
    ax2 = ax1.twinx()
    ax2.set_ylabel('5-year GoC yield (%) — medium-term rate benchmark')
    line2 = ax2.plot(plot_df['quarter_label'], plot_df['rate_goc_5y'],
                     marker='s', markersize=4, linewidth=1.5, label='GoC 5y yield', linestyle='--')
    # Get color from line for axis label
    color2 = line2[0].get_color()
    ax2.tick_params(axis='y', labelcolor=color2)
    
    # Title
    plt.title(f'{region_name}: Rent QoQ Change vs 5-Year GoC Yield')
    
    # Rotate x ticks
    plt.xticks(rotation=35, ha='right')
    
    # Legend (outside plot area)
    lines = line1 + line2
    labels = [l.get_label() for l in lines]
    ax1.legend(lines, labels, loc='upper left', bbox_to_anchor=(0, 1))
    
    # Use tight layout
    plt.tight_layout()
    
    # Save
    save_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"[OK] Saved: {save_path}")


def generate_rate_story_charts(
    out_dir: str = "outputs/plots/mpl_story"
) -> None:
    """
    Generate all rate story charts (rent changes vs interest rates).
    
    Creates 4 charts:
    - Toronto YoY vs Bank rate
    - GTA proxy YoY vs Bank rate
    - Toronto QoQ vs 5-year GoC yield
    - GTA proxy QoQ vs 5-year GoC yield
    
    Args:
        out_dir: Output directory for PNG files.
    """
    # Convert to Path object
    out_path = Path(out_dir)
    if not out_path.is_absolute():
        out_path = OUTPUT_DIR.parent / out_dir
    
    # Create output directory
    out_path.mkdir(parents=True, exist_ok=True)
    
    # Load data
    rent_path = PROCESSED_DIR / "rent_target_quarterly.parquet"
    exog_path = PROCESSED_DIR / "exog_quarterly.parquet"
    
    if not rent_path.exists():
        raise FileNotFoundError(f"Rent target file not found: {rent_path}")
    if not exog_path.exists():
        raise FileNotFoundError(f"Exogenous features file not found: {exog_path}")
    
    print(f"Loading rent target from: {rent_path}")
    rent_df = pd.read_parquet(rent_path)
    rent_df['quarter_end_date'] = pd.to_datetime(rent_df['quarter_end_date'])
    
    print(f"Loading exogenous features from: {exog_path}")
    rates_df = pd.read_parquet(exog_path)
    rates_df['quarter_end_date'] = pd.to_datetime(rates_df['quarter_end_date'])
    
    print(f"\nGenerating rate story charts...")
    print("=" * 80)
    
    # Define regions
    regions = {
        "Toronto": ["Toronto"],
        "GTA proxy": ["Toronto", "Oshawa"]
    }
    
    # Generate charts
    # 1. Toronto YoY vs Bank rate
    print("\n1. Toronto YoY vs Bank rate...")
    plot_yoy_vs_bank_rate(
        rent_df, rates_df, "Toronto", ["Toronto"],
        out_path / "toronto_rent_yoy_vs_bank_rate.png"
    )
    
    # 2. GTA proxy YoY vs Bank rate
    print("\n2. GTA proxy YoY vs Bank rate...")
    plot_yoy_vs_bank_rate(
        rent_df, rates_df, "GTA proxy", ["Toronto", "Oshawa"],
        out_path / "gta_rent_yoy_vs_bank_rate.png"
    )
    
    # 3. Toronto QoQ vs GoC 5y
    print("\n3. Toronto QoQ vs GoC 5y...")
    plot_qoq_vs_goc5y(
        rent_df, rates_df, "Toronto", ["Toronto"],
        out_path / "toronto_rent_qoq_vs_goc5y.png"
    )
    
    # 4. GTA proxy QoQ vs GoC 5y
    print("\n4. GTA proxy QoQ vs GoC 5y...")
    plot_qoq_vs_goc5y(
        rent_df, rates_df, "GTA proxy", ["Toronto", "Oshawa"],
        out_path / "gta_rent_qoq_vs_goc5y.png"
    )
    
    print("\n" + "=" * 80)
    print(f"All rate story charts saved to: {out_path}")
    print("=" * 80)


if __name__ == "__main__":
    generate_rate_story_charts()

