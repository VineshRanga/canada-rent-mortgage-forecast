"""Rates overlay story charts: rent changes vs interest rates."""

from pathlib import Path
from typing import List, Optional, Tuple
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker

from src.config import OUTPUT_DIR, PROCESSED_DIR


def format_quarter_label(date: pd.Timestamp) -> str:
    """Format quarter-end date as YYYYQ#."""
    year = date.year
    quarter = (date.month - 1) // 3 + 1
    return f"{year}Q{quarter}"


def build_quarter_axis(df_dates: pd.Series) -> Tuple[np.ndarray, List[str]]:
    """Build integer x positions and quarter labels from sorted dates."""
    n = len(df_dates)
    x = np.arange(n)
    labels = [format_quarter_label(date) for date in df_dates]
    return x, labels


def apply_quarter_ticks(ax, x: np.ndarray, labels: List[str], max_ticks: int = 8) -> None:
    """Force fixed tick locations with downsampled labels."""
    n = len(labels)
    if n == 0:
        return
    
    ticks = np.linspace(0, n - 1, num=min(max_ticks, n), dtype=int)
    ticks = sorted(set(ticks.tolist()))
    
    ax.xaxis.set_major_locator(matplotlib.ticker.FixedLocator(ticks))
    ax.set_xticklabels([labels[i] for i in ticks], rotation=35, ha='right')
    ax.tick_params(axis='x', labelsize=9)
    ax.minorticks_off()


def prepare_region_data(
    rent_df: pd.DataFrame,
    cma_list: List[str]
) -> pd.DataFrame:
    """Aggregate rent data for a region, compute YoY/QoQ changes."""
    df_region = rent_df[rent_df['cma'].isin(cma_list)].copy()
    
    if len(df_region) == 0:
        return pd.DataFrame()
    
    df_region['y'] = pd.to_numeric(df_region['y'], errors='coerce')
    df_region = df_region.dropna(subset=['y'])
    
    df_agg = df_region.groupby(['quarter_end_date', 'unit_type']).agg({
        'y': 'mean'
    }).reset_index()
    
    df_agg = df_agg.sort_values(['unit_type', 'quarter_end_date']).reset_index(drop=True)
    
    df_agg['y_lag1'] = df_agg.groupby('unit_type')['y'].shift(1)
    df_agg['y_lag4'] = df_agg.groupby('unit_type')['y'].shift(4)
    
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
    """Plot YoY rent change vs Bank rate (dual axis)."""
    region_data = prepare_region_data(rent_df, cma_list)
    
    if len(region_data) == 0:
        print(f"[WARN] No data for {region_name}, skipping chart")
        return
    
    region_agg = region_data.groupby('quarter_end_date').agg({
        'yoy_pct': 'mean'
    }).reset_index()
    
    plot_df = region_agg.merge(
        rates_df[['quarter_end_date', 'rate_bank']],
        on='quarter_end_date',
        how='inner'
    )
    
    plot_df = plot_df.dropna(subset=['yoy_pct', 'rate_bank'])
    plot_df = plot_df.sort_values('quarter_end_date')
    
    if len(plot_df) == 0:
        print(f"[WARN] No overlapping data for {region_name} YoY vs Bank rate, skipping")
        return
    
    x, qlabels = build_quarter_axis(plot_df['quarter_end_date'])
    
    fig, ax1 = plt.subplots()
    fig.set_size_inches(16, 5)
    
    ax1.set_xlabel('Quarter (period end) - aligned quarterly reporting')
    ax1.set_ylabel('Rent YoY change (%) — captures inflation-like dynamics')
    line1 = ax1.plot(x, plot_df['yoy_pct'], 
                     marker='o', markersize=4, linewidth=1.5, label='Rent YoY (%)')
    color1 = line1[0].get_color()
    ax1.tick_params(axis='y', labelcolor=color1)
    ax1.grid(True, alpha=0.3)
    
    ax2 = ax1.twinx()
    ax2.set_ylabel('Bank rate (%) — monetary policy signal')
    line2 = ax2.plot(x, plot_df['rate_bank'],
                     marker='s', markersize=4, linewidth=1.5, label='Bank rate (%)', linestyle='--')
    color2 = line2[0].get_color()
    ax2.tick_params(axis='y', labelcolor=color2)
    
    plt.title(f'{region_name}: Rent YoY Change vs Bank Rate')
    
    lines = line1 + line2
    labels = [l.get_label() for l in lines]
    ax1.legend(lines, labels, loc='upper left', bbox_to_anchor=(0, 1.15), ncol=2)
    
    apply_quarter_ticks(ax1, x, qlabels, max_ticks=8)
    
    fig.subplots_adjust(bottom=0.28, top=0.85)
    
    save_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=200, bbox_inches='tight')
    plt.close()
    print(f"[OK] Saved: {save_path}")


def plot_qoq_vs_goc5y(
    rent_df: pd.DataFrame,
    rates_df: pd.DataFrame,
    region_name: str,
    cma_list: List[str],
    save_path: Path
) -> None:
    """Plot QoQ rent change vs 5-year GoC yield (dual axis)."""
    region_data = prepare_region_data(rent_df, cma_list)
    
    if len(region_data) == 0:
        print(f"[WARN] No data for {region_name}, skipping chart")
        return
    
    region_agg = region_data.groupby('quarter_end_date').agg({
        'qoq_pct': 'mean'
    }).reset_index()
    
    plot_df = region_agg.merge(
        rates_df[['quarter_end_date', 'rate_goc_5y']],
        on='quarter_end_date',
        how='inner'
    )
    
    plot_df = plot_df.dropna(subset=['qoq_pct', 'rate_goc_5y'])
    plot_df = plot_df.sort_values('quarter_end_date')
    
    if len(plot_df) == 0:
        print(f"[WARN] No overlapping data for {region_name} QoQ vs GoC 5y, skipping")
        return
    
    x, qlabels = build_quarter_axis(plot_df['quarter_end_date'])
    
    fig, ax1 = plt.subplots()
    fig.set_size_inches(16, 5)
    
    ax1.set_xlabel('Quarter (period end) - aligned quarterly reporting')
    ax1.set_ylabel('Rent QoQ change (%) — short-horizon momentum')
    line1 = ax1.plot(x, plot_df['qoq_pct'],
                     marker='o', markersize=4, linewidth=1.5, label='Rent QoQ (%)')
    color1 = line1[0].get_color()
    ax1.tick_params(axis='y', labelcolor=color1)
    ax1.grid(True, alpha=0.3)
    
    ax2 = ax1.twinx()
    ax2.set_ylabel('5-year GoC yield (%) — medium-term rate benchmark')
    line2 = ax2.plot(x, plot_df['rate_goc_5y'],
                     marker='s', markersize=4, linewidth=1.5, label='5-year GoC yield (%)', linestyle='--')
    color2 = line2[0].get_color()
    ax2.tick_params(axis='y', labelcolor=color2)
    
    plt.title(f'{region_name}: Rent QoQ Change vs 5-Year GoC Yield')
    
    lines = line1 + line2
    labels = [l.get_label() for l in lines]
    ax1.legend(lines, labels, loc='upper left', bbox_to_anchor=(0, 1.15), ncol=2)
    
    apply_quarter_ticks(ax1, x, qlabels, max_ticks=8)
    
    fig.subplots_adjust(bottom=0.28, top=0.85)
    
    save_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=200, bbox_inches='tight')
    plt.close()
    print(f"[OK] Saved: {save_path}")


def generate_rate_story_charts(
    out_dir: str = "outputs/plots/mpl_story"
) -> None:
    """Generate all rate story charts (rent changes vs interest rates)."""
    out_path = Path(out_dir)
    if not out_path.is_absolute():
        out_path = OUTPUT_DIR.parent / out_dir
    
    out_path.mkdir(parents=True, exist_ok=True)
    
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
    
    print("\n1. Toronto QoQ vs GoC 5y...")
    plot_qoq_vs_goc5y(
        rent_df, rates_df, "Toronto", ["Toronto"],
        out_path / "toronto_rent_qoq_vs_goc5y.png"
    )
    
    print("\n2. GTA proxy QoQ vs GoC 5y...")
    plot_qoq_vs_goc5y(
        rent_df, rates_df, "GTA proxy", ["Toronto", "Oshawa"],
        out_path / "gta_rent_qoq_vs_goc5y.png"
    )
    
    print("\n" + "=" * 80)
    print(f"All rate story charts saved to: {out_path}")
    print("=" * 80)


if __name__ == "__main__":
    generate_rate_story_charts()

