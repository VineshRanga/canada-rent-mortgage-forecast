"""
Visualization functions for forecasting results.

Creates matplotlib plots for actual vs predicted values, model performance,
and feature importance.
"""

from pathlib import Path
from typing import Optional, List, Union
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

from src.config import OUTPUT_DIR


def plot_actual_vs_predicted(
    df: pd.DataFrame,
    date_col: str,
    actual_col: str,
    pred_col: str,
    group_cols: Optional[List[str]] = None,
    title: Optional[str] = None,
    save_path: Optional[Path] = None,
    figsize: tuple = (12, 6)
) -> None:
    """
    Plot actual vs predicted values over time.
    
    Args:
        df: DataFrame with date, actual, and predicted columns.
        date_col: Name of date column.
        actual_col: Name of actual values column.
        pred_col: Name of predicted values column.
        group_cols: Optional list of columns to group by (creates subplots).
        title: Plot title. If None, auto-generates from column names.
        save_path: Path to save plot. If None, saves to OUTPUT_DIR.
        figsize: Figure size tuple (width, height).
    """
    # Ensure date column is datetime
    df = df.copy()
    df[date_col] = pd.to_datetime(df[date_col])
    
    # Remove rows with missing dates or values
    df = df.dropna(subset=[date_col, actual_col, pred_col])
    
    if len(df) == 0:
        print("Warning: No valid data to plot")
        return
    
    # Sort by date
    df = df.sort_values(date_col)
    
    # Create figure
    if group_cols is None or len(group_cols) == 0:
        # Single plot
        fig, ax = plt.subplots(figsize=figsize)
        
        ax.plot(df[date_col], df[actual_col], label='Actual', marker='o', markersize=3, linewidth=1.5)
        ax.plot(df[date_col], df[pred_col], label='Predicted', marker='s', markersize=3, linewidth=1.5, linestyle='--')
        
        ax.set_xlabel('Date')
        ax.set_ylabel('Value')
        ax.set_title(title if title else f'Actual vs Predicted: {actual_col}')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Format x-axis dates
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
        ax.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
        plt.xticks(rotation=45)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Saved plot to: {save_path}")
        else:
            plt.show()
        
        plt.close()
        
    else:
        # Multiple subplots by group
        groups = df.groupby(group_cols)
        n_groups = len(groups)
        
        # Determine subplot layout
        n_cols = min(3, n_groups)
        n_rows = (n_groups + n_cols - 1) // n_cols
        
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(figsize[0] * n_cols, figsize[1] * n_rows))
        
        if n_groups == 1:
            axes = [axes]
        else:
            axes = axes.flatten() if n_rows > 1 else [axes] if n_cols == 1 else axes
        
        for idx, ((group_vals, group_df), ax) in enumerate(zip(groups, axes)):
            if isinstance(group_vals, tuple):
                group_label = ' | '.join(str(v) for v in group_vals)
            else:
                group_label = str(group_vals)
            
            group_df = group_df.sort_values(date_col)
            
            ax.plot(group_df[date_col], group_df[actual_col], label='Actual', marker='o', markersize=2, linewidth=1)
            ax.plot(group_df[date_col], group_df[pred_col], label='Predicted', marker='s', markersize=2, linewidth=1, linestyle='--')
            
            ax.set_title(group_label, fontsize=10)
            ax.set_xlabel('Date')
            ax.set_ylabel('Value')
            ax.legend(fontsize=8)
            ax.grid(True, alpha=0.3)
            
            # Format x-axis dates
            ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
            ax.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
            plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha='right')
        
        # Hide unused subplots
        for idx in range(n_groups, len(axes)):
            axes[idx].set_visible(False)
        
        if title:
            fig.suptitle(title, fontsize=14, y=1.02)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Saved plot to: {save_path}")
        else:
            plt.show()
        
        plt.close()


def plot_rent_overall(
    predictions_df: pd.DataFrame,
    save_path: Optional[Path] = None
) -> None:
    """
    Plot overall rent predictions (all CMAs and unit types combined).
    
    Args:
        predictions_df: DataFrame with rent predictions (from rent_predictions.csv).
        save_path: Path to save plot. Defaults to OUTPUT_DIR / "plots" / "rent_overall.png".
    """
    if save_path is None:
        save_path = OUTPUT_DIR / "plots" / "rent_overall.png"
    
    save_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Aggregate to overall
    df_agg = predictions_df.groupby('quarter_end_date').agg({
        'y_true': 'mean',
        'y_pred': 'mean'
    }).reset_index()
    
    plot_actual_vs_predicted(
        df_agg,
        date_col='quarter_end_date',
        actual_col='y_true',
        pred_col='y_pred',
        title='Rent Forecast: Overall (All CMAs and Unit Types)',
        save_path=save_path
    )


def plot_rent_by_unit_type(
    predictions_df: pd.DataFrame,
    save_path: Optional[Path] = None
) -> None:
    """
    Plot rent predictions by unit type.
    
    Args:
        predictions_df: DataFrame with rent predictions.
        save_path: Path to save plot. Defaults to OUTPUT_DIR / "plots" / "rent_by_unit_type.png".
    """
    if save_path is None:
        save_path = OUTPUT_DIR / "plots" / "rent_by_unit_type.png"
    
    save_path.parent.mkdir(parents=True, exist_ok=True)
    
    if 'unit_type' not in predictions_df.columns:
        print("Warning: unit_type column not found in predictions DataFrame")
        return
    
    plot_actual_vs_predicted(
        predictions_df,
        date_col='quarter_end_date',
        actual_col='y_true',
        pred_col='y_pred',
        group_cols=['unit_type'],
        title='Rent Forecast: By Unit Type',
        save_path=save_path
    )


def plot_rent_top_cmas(
    predictions_df: pd.DataFrame,
    n_cmas: int = 6,
    save_path: Optional[Path] = None
) -> None:
    """
    Plot rent predictions for top N CMAs by volume of data.
    
    Args:
        predictions_df: DataFrame with rent predictions.
        n_cmas: Number of top CMAs to plot. Defaults to 6.
        save_path: Path to save plot. Defaults to OUTPUT_DIR / "plots" / "rent_top_cmas.png".
    """
    if save_path is None:
        save_path = OUTPUT_DIR / "plots" / "rent_top_cmas.png"
    
    save_path.parent.mkdir(parents=True, exist_ok=True)
    
    if 'cma' not in predictions_df.columns:
        print("Warning: cma column not found in predictions DataFrame")
        return
    
    # Find top CMAs by data volume
    cma_counts = predictions_df['cma'].value_counts()
    top_cmas = cma_counts.head(n_cmas).index.tolist()
    
    # Filter to top CMAs
    df_top = predictions_df[predictions_df['cma'].isin(top_cmas)].copy()
    
    plot_actual_vs_predicted(
        df_top,
        date_col='quarter_end_date',
        actual_col='y_true',
        pred_col='y_pred',
        group_cols=['cma'],
        title=f'Rent Forecast: Top {n_cmas} CMAs by Data Volume',
        save_path=save_path
    )


def plot_mortgage_actual_vs_predicted(
    predictions_df: pd.DataFrame,
    save_path: Optional[Path] = None
) -> None:
    """
    Plot mortgage actual vs predicted level.
    
    Args:
        predictions_df: DataFrame with mortgage predictions (from mortgage_predictions.csv).
        save_path: Path to save plot. Defaults to OUTPUT_DIR / "plots" / "mortgage_actual_vs_predicted.png".
    """
    if save_path is None:
        save_path = OUTPUT_DIR / "plots" / "mortgage_actual_vs_predicted.png"
    
    save_path.parent.mkdir(parents=True, exist_ok=True)
    
    plot_actual_vs_predicted(
        predictions_df,
        date_col='quarter_end_date',
        actual_col='y_true',
        pred_col='y_pred',
        title='Mortgage Forecast: Actual vs Predicted Level',
        save_path=save_path
    )


def plot_rent_actual_vs_predicted(preds_df: pd.DataFrame) -> None:
    """
    Generate rent actual vs predicted plots.
    
    Creates three plots:
    1. Overall plot (aggregate mean by quarter)
    2. By unit_type (two lines per plot: actual and predicted)
    3. Top 6 CMAs by row count (one combined plot with 6 lines)
    
    Args:
        preds_df: DataFrame with rent predictions (columns: quarter_end_date, cma, unit_type, y_true, y_pred).
    """
    print("Generating rent actual vs predicted plots...")
    
    # Ensure date column is datetime
    preds_df = preds_df.copy()
    preds_df['quarter_end_date'] = pd.to_datetime(preds_df['quarter_end_date'])
    
    # Remove rows with missing values
    preds_df = preds_df.dropna(subset=['quarter_end_date', 'y_true', 'y_pred'])
    
    if len(preds_df) == 0:
        print("Warning: No valid data to plot")
        return
    
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    plots_dir = OUTPUT_DIR / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)
    
    # 1. Overall plot (aggregate mean by quarter)
    print("  Creating overall plot...")
    df_overall = preds_df.groupby('quarter_end_date').agg({
        'y_true': 'mean',
        'y_pred': 'mean'
    }).reset_index()
    df_overall = df_overall.sort_values('quarter_end_date')
    
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(df_overall['quarter_end_date'], df_overall['y_true'], 
            label='Actual', marker='o', markersize=4, linewidth=2)
    ax.plot(df_overall['quarter_end_date'], df_overall['y_pred'], 
            label='Predicted', marker='s', markersize=4, linewidth=2, linestyle='--')
    ax.set_xlabel('Date')
    ax.set_ylabel('Rent ($)')
    ax.set_title('Rent Forecast: Overall (All CMAs and Unit Types)')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
    plt.xticks(rotation=45)
    plt.tight_layout()
    
    overall_path = plots_dir / "rent_actual_vs_predicted_overall.png"
    plt.savefig(overall_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"    Saved: {overall_path}")
    
    # 2. By unit_type plot (two lines per plot: actual and predicted)
    print("  Creating by unit_type plot...")
    if 'unit_type' not in preds_df.columns:
        print("    Warning: unit_type column not found, skipping by unit_type plot")
    else:
        df_by_unit = preds_df.groupby(['quarter_end_date', 'unit_type']).agg({
            'y_true': 'mean',
            'y_pred': 'mean'
        }).reset_index()
        df_by_unit = df_by_unit.sort_values(['unit_type', 'quarter_end_date'])
        
        fig, ax = plt.subplots(figsize=(12, 6))
        unit_types = df_by_unit['unit_type'].unique()
        colors = plt.cm.tab10(np.linspace(0, 1, len(unit_types)))
        
        for i, unit_type in enumerate(unit_types):
            unit_df = df_by_unit[df_by_unit['unit_type'] == unit_type].sort_values('quarter_end_date')
            ax.plot(unit_df['quarter_end_date'], unit_df['y_true'], 
                   label=f'{unit_type} - Actual', marker='o', markersize=3, 
                   linewidth=1.5, color=colors[i])
            ax.plot(unit_df['quarter_end_date'], unit_df['y_pred'], 
                   label=f'{unit_type} - Predicted', marker='s', markersize=3, 
                   linewidth=1.5, linestyle='--', color=colors[i], alpha=0.7)
        
        ax.set_xlabel('Date')
        ax.set_ylabel('Rent ($)')
        ax.set_title('Rent Forecast: By Unit Type')
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
        ax.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
        plt.xticks(rotation=45)
        plt.tight_layout()
        
        by_unit_path = plots_dir / "rent_actual_vs_predicted_by_unit_type.png"
        plt.savefig(by_unit_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"    Saved: {by_unit_path}")
    
    # 3. Top 6 CMAs by row count (one combined plot with 6 lines)
    print("  Creating top CMAs plot...")
    if 'cma' not in preds_df.columns:
        print("    Warning: cma column not found, skipping top CMAs plot")
    else:
        # Find top 6 CMAs by row count
        cma_counts = preds_df['cma'].value_counts()
        top_cmas = cma_counts.head(6).index.tolist()
        
        # Aggregate by quarter and CMA
        df_by_cma = preds_df[preds_df['cma'].isin(top_cmas)].copy()
        df_by_cma = df_by_cma.groupby(['quarter_end_date', 'cma']).agg({
            'y_true': 'mean',
            'y_pred': 'mean'
        }).reset_index()
        df_by_cma = df_by_cma.sort_values(['cma', 'quarter_end_date'])
        
        fig, ax = plt.subplots(figsize=(14, 8))
        colors = plt.cm.tab10(np.linspace(0, 1, len(top_cmas)))
        
        for i, cma in enumerate(top_cmas):
            cma_df = df_by_cma[df_by_cma['cma'] == cma].sort_values('quarter_end_date')
            ax.plot(cma_df['quarter_end_date'], cma_df['y_true'], 
                   label=f'{cma} - Actual', marker='o', markersize=3, 
                   linewidth=1.5, color=colors[i])
            ax.plot(cma_df['quarter_end_date'], cma_df['y_pred'], 
                   label=f'{cma} - Predicted', marker='s', markersize=3, 
                   linewidth=1.5, linestyle='--', color=colors[i], alpha=0.7)
        
        ax.set_xlabel('Date')
        ax.set_ylabel('Rent ($)')
        ax.set_title('Rent Forecast: Top 6 CMAs by Data Volume')
        ax.legend(fontsize=8, ncol=2)
        ax.grid(True, alpha=0.3)
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
        ax.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
        plt.xticks(rotation=45)
        plt.tight_layout()
        
        top_cmas_path = plots_dir / "rent_actual_vs_predicted_top_cmas.png"
        plt.savefig(top_cmas_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"    Saved: {top_cmas_path}")
    
    print("Rent actual vs predicted plots generated.")


def plot_all_rent_plots(predictions_df: pd.DataFrame) -> None:
    """
    Generate all rent plots.
    
    Args:
        predictions_df: DataFrame with rent predictions.
    """
    print("Generating rent plots...")
    plot_rent_overall(predictions_df)
    plot_rent_by_unit_type(predictions_df)
    plot_rent_top_cmas(predictions_df, n_cmas=6)
    print("All rent plots generated.")


if __name__ == "__main__":
    """Generate plots from saved prediction files."""
    # Load rent predictions
    rent_pred_path = OUTPUT_DIR / "rent_predictions.csv"
    if rent_pred_path.exists():
        print(f"Loading rent predictions from: {rent_pred_path}")
        rent_df = pd.read_csv(rent_pred_path)
        rent_df['quarter_end_date'] = pd.to_datetime(rent_df['quarter_end_date'])
        plot_all_rent_plots(rent_df)
    else:
        print(f"Rent predictions file not found: {rent_pred_path}")
    
    # Load mortgage predictions
    mortgage_pred_path = OUTPUT_DIR / "mortgage_predictions.csv"
    if mortgage_pred_path.exists():
        print(f"\nLoading mortgage predictions from: {mortgage_pred_path}")
        mortgage_df = pd.read_csv(mortgage_pred_path)
        mortgage_df['quarter_end_date'] = pd.to_datetime(mortgage_df['quarter_end_date'])
        plot_mortgage_actual_vs_predicted(mortgage_df)
    else:
        print(f"Mortgage predictions file not found: {mortgage_pred_path}")

