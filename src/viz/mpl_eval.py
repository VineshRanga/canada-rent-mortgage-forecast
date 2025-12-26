"""Evaluation charts: heatmaps, bar charts, residual time series."""

from pathlib import Path
from typing import Optional
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

from src.config import OUTPUT_DIR
from src.evaluation.metrics import smape


def format_quarter_label(date: pd.Timestamp) -> str:
    """Format quarter-end date as YYYYQ#."""
    year = date.year
    quarter = (date.month - 1) // 3 + 1
    return f"{year}Q{quarter}"


def compute_smape_heatmap(df: pd.DataFrame) -> pd.DataFrame:
    """Compute sMAPE per (cma, unit_type), pivot to matrix."""
    df = df.copy()
    df['y_true'] = pd.to_numeric(df['y_true'], errors='coerce')
    df['y_pred'] = pd.to_numeric(df['y_pred'], errors='coerce')
    
    df = df.dropna(subset=['cma', 'unit_type', 'y_true', 'y_pred'])
    
    results = []
    for (cma, unit_type), group in df.groupby(['cma', 'unit_type']):
        if len(group) > 0:
            try:
                smape_val = smape(group['y_true'].values, group['y_pred'].values)
                results.append({
                    'cma': cma,
                    'unit_type': unit_type,
                    'smape': smape_val
                })
            except Exception:
                continue
    
    if len(results) == 0:
        return pd.DataFrame()
    
    results_df = pd.DataFrame(results)
    pivot_df = results_df.pivot(index='cma', columns='unit_type', values='smape')
    pivot_df = pivot_df.sort_index()
    
    return pivot_df


def compute_uplift_heatmap(df: pd.DataFrame) -> pd.DataFrame:
    """Compute uplift vs Lag-4 baseline per (cma, unit_type), pivot to matrix."""
    df = df.copy()
    df['y_true'] = pd.to_numeric(df['y_true'], errors='coerce')
    df['y_pred'] = pd.to_numeric(df['y_pred'], errors='coerce')
    
    if 'y_pred_lag4' not in df.columns:
        return pd.DataFrame()
    
    df['y_pred_lag4'] = pd.to_numeric(df['y_pred_lag4'], errors='coerce')
    df = df.dropna(subset=['cma', 'unit_type', 'y_true', 'y_pred', 'y_pred_lag4'])
    
    results = []
    for (cma, unit_type), group in df.groupby(['cma', 'unit_type']):
        if len(group) > 0:
            try:
                smape_model = smape(group['y_true'].values, group['y_pred'].values)
                smape_lag4 = smape(group['y_true'].values, group['y_pred_lag4'].values)
                
                if smape_lag4 > 0:
                    uplift = (smape_lag4 - smape_model) / smape_lag4 * 100
                else:
                    uplift = np.nan
                
                results.append({
                    'cma': cma,
                    'unit_type': unit_type,
                    'uplift': uplift
                })
            except Exception:
                continue
    
    if len(results) == 0:
        return pd.DataFrame()
    
    results_df = pd.DataFrame(results)
    pivot_df = results_df.pivot(index='cma', columns='unit_type', values='uplift')
    pivot_df = pivot_df.sort_index()
    
    return pivot_df


def plot_smape_heatmap(pivot_df: pd.DataFrame, save_path: Path) -> None:
    """Plot sMAPE heatmap: CMA × unit_type."""
    if len(pivot_df) == 0:
        print(f"Warning: No data for sMAPE heatmap, skipping")
        return
    
    n_cmas = len(pivot_df)
    n_unit_types = len(pivot_df.columns)
    
    fig_height = max(10, n_cmas * 0.5)
    fig_width = max(10, n_unit_types * 3.5)
    
    fig, ax = plt.subplots(figsize=(fig_width, fig_height))
    
    im = ax.imshow(pivot_df.values, aspect='auto', cmap='viridis_r')
    
    ax.set_xticks(np.arange(len(pivot_df.columns)))
    ax.set_yticks(np.arange(len(pivot_df.index)))
    ax.set_xticklabels(pivot_df.columns, rotation=15, ha='right')
    ax.set_yticklabels(pivot_df.index)
    
    ax.set_xlabel('Unit type — segmenting rental market')
    ax.set_ylabel('CMA — geographic segmentation')
    ax.set_title('Rent Forecast Error (sMAPE %) by CMA and Unit Type')
    
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label('sMAPE (%) — scale-free error metric')
    
    plt.tight_layout()
    
    save_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved: {save_path}")


def plot_uplift_heatmap(pivot_df: pd.DataFrame, save_path: Path) -> None:
    """Plot uplift vs Lag-4 heatmap: CMA × unit_type."""
    if len(pivot_df) == 0:
        print(f"Warning: No data for uplift heatmap, skipping")
        return
    
    n_cmas = len(pivot_df)
    n_unit_types = len(pivot_df.columns)
    
    fig_height = max(10, n_cmas * 0.5)
    fig_width = max(10, n_unit_types * 3.5)
    
    fig, ax = plt.subplots(figsize=(fig_width, fig_height))
    
    im = ax.imshow(pivot_df.values, aspect='auto', cmap='RdYlGn')
    
    ax.set_xticks(np.arange(len(pivot_df.columns)))
    ax.set_yticks(np.arange(len(pivot_df.index)))
    ax.set_xticklabels(pivot_df.columns, rotation=15, ha='right')
    ax.set_yticklabels(pivot_df.index)
    
    ax.set_xlabel('Unit type — segmenting rental market')
    ax.set_ylabel('CMA — geographic segmentation')
    ax.set_title('Model Improvement vs Seasonal Baseline (Lag-4)')
    
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label('Uplift vs Lag-4 baseline (sMAPE improvement, %)')
    
    plt.tight_layout()
    
    save_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved: {save_path}")


def plot_top15_uplift_bar(df: pd.DataFrame, save_path: Path) -> None:
    """Plot top 15 CMAs by uplift vs Lag-4 (averaged over unit types)."""
    uplift_df = compute_uplift_heatmap(df)
    
    if len(uplift_df) == 0:
        print(f"Warning: No data for uplift bar chart, skipping")
        return
    
    cma_uplift = uplift_df.mean(axis=1)
    cma_uplift_sorted = cma_uplift.sort_values(ascending=False).head(15)
    
    fig, ax = plt.subplots(figsize=(12, 8))
    
    y_pos = np.arange(len(cma_uplift_sorted))
    ax.barh(y_pos, cma_uplift_sorted.values)
    
    ax.set_yticks(y_pos)
    ax.set_yticklabels(cma_uplift_sorted.index)
    
    ax.set_xlabel('Uplift vs Lag-4 (sMAPE improvement, %) — higher is better')
    ax.set_ylabel('CMA — markets ranked by model improvement')
    ax.set_title('Top CMAs by Improvement vs Seasonal Baseline')
    
    ax.invert_yaxis()
    
    plt.tight_layout()
    
    save_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved: {save_path}")


def plot_residual_time_series(
    df: pd.DataFrame,
    region_name: str,
    cma_list: list,
    unit_type: str,
    save_path: Path
) -> None:
    """Plot residual time series for a specific region and unit type."""
    df_region = df[df['cma'].isin(cma_list) & (df['unit_type'] == unit_type)].copy()
    
    if len(df_region) == 0:
        print(f"Warning: No data for {region_name} - {unit_type}, skipping")
        return
    
    df_region['y_true'] = pd.to_numeric(df_region['y_true'], errors='coerce')
    df_region['y_pred'] = pd.to_numeric(df_region['y_pred'], errors='coerce')
    
    df_region = df_region.dropna(subset=['quarter_end_date', 'y_true', 'y_pred'])
    
    if len(cma_list) > 1:
        df_region = df_region.groupby('quarter_end_date').agg({
            'y_true': 'mean',
            'y_pred': 'mean'
        }).reset_index()
    
    df_region = df_region.sort_values('quarter_end_date')
    df_region['residual'] = df_region['y_true'] - df_region['y_pred']
    df_region['quarter_label'] = df_region['quarter_end_date'].apply(format_quarter_label)
    
    fig, ax = plt.subplots(figsize=(12, 6))
    
    ax.plot(df_region['quarter_label'], df_region['residual'], 
            marker='o', markersize=4, linewidth=1.5)
    
    ax.axhline(y=0, color='gray', linestyle='--', linewidth=1, alpha=0.5)
    
    ax.set_xlabel('Quarter (period end) — aligned quarterly reporting')
    ax.set_ylabel('Residual (CAD) = Actual − Predicted — signed error')
    ax.set_title(f'{region_name}: {unit_type} — Residual Time Series')
    
    plt.xticks(rotation=35, ha='right')
    
    plt.tight_layout()
    
    save_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved: {save_path}")


def generate_rent_eval_charts(
    pred_csv_path: str = "outputs/rent_predictions.csv",
    out_dir: str = "outputs/plots/mpl_eval"
) -> None:
    """Generate all rent evaluation charts."""
    pred_path = Path(pred_csv_path)
    if not pred_path.is_absolute():
        pred_path = OUTPUT_DIR.parent / pred_csv_path
    
    out_path = Path(out_dir)
    if not out_path.is_absolute():
        out_path = OUTPUT_DIR.parent / out_dir
    
    out_path.mkdir(parents=True, exist_ok=True)
    
    if not pred_path.exists():
        raise FileNotFoundError(f"Predictions file not found: {pred_path}")
    
    print(f"Reading predictions from: {pred_path}")
    df = pd.read_csv(pred_path)
    df['quarter_end_date'] = pd.to_datetime(df['quarter_end_date'])
    
    print(f"\nGenerating evaluation charts...")
    print("=" * 80)
    
    print("\n1. Generating sMAPE heatmap...")
    smape_pivot = compute_smape_heatmap(df)
    plot_smape_heatmap(smape_pivot, out_path / "rent_smape_heatmap.png")
    
    print("\n2. Generating uplift vs Lag-4 heatmap...")
    uplift_pivot = compute_uplift_heatmap(df)
    plot_uplift_heatmap(uplift_pivot, out_path / "rent_uplift_lag4_heatmap.png")
    
    print("\n3. Generating top 15 CMAs by uplift bar chart...")
    plot_top15_uplift_bar(df, out_path / "rent_uplift_top15_cmas.png")
    
    print("\n4. Generating residual time series plots...")
    
    plot_residual_time_series(
        df, "Toronto", ["Toronto"], "Apartment - 1 bedroom",
        out_path / "rent_residuals_toronto_1bed.png"
    )
    
    plot_residual_time_series(
        df, "Toronto", ["Toronto"], "Apartment - 2 bedrooms",
        out_path / "rent_residuals_toronto_2bed.png"
    )
    
    plot_residual_time_series(
        df, "GTA proxy", ["Toronto", "Oshawa"], "Apartment - 1 bedroom",
        out_path / "rent_residuals_gta_1bed.png"
    )
    
    plot_residual_time_series(
        df, "GTA proxy", ["Toronto", "Oshawa"], "Apartment - 2 bedrooms",
        out_path / "rent_residuals_gta_2bed.png"
    )
    
    print("\n" + "=" * 80)
    print(f"All evaluation charts saved to: {out_path}")
    print("=" * 80)


if __name__ == "__main__":
    generate_rent_eval_charts()

