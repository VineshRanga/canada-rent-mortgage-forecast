"""
Generate LinkedIn-ready Plotly charts for rent forecasts focused on Toronto and the GTA.

Creates interactive Plotly charts with proper formatting for professional sharing.
"""

from pathlib import Path
from typing import List
import pandas as pd
import plotly.graph_objects as go
import warnings

from src.config import OUTPUT_DIR


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


def prepare_data_for_region(
    df: pd.DataFrame,
    cma_list: List[str]
) -> pd.DataFrame:
    """
    Prepare data for a region by filtering and aggregating.
    
    Args:
        df: DataFrame with rent predictions.
        cma_list: List of CMA names to include in the region.
        
    Returns:
        Aggregated DataFrame with mean values by quarter_end_date and unit_type.
    """
    # Filter to specified CMAs
    df_region = df[df['cma'].isin(cma_list)].copy()
    
    if len(df_region) == 0:
        return pd.DataFrame()
    
    # Coerce numeric columns
    numeric_cols = ['y_true', 'y_pred', 'y_pred_lag1', 'y_pred_lag4']
    for col in numeric_cols:
        if col in df_region.columns:
            df_region[col] = pd.to_numeric(df_region[col], errors='coerce')
    
    # Drop rows missing y_true or y_pred
    df_region = df_region.dropna(subset=['y_true', 'y_pred'])
    
    if len(df_region) == 0:
        return pd.DataFrame()
    
    # Aggregate by quarter_end_date and unit_type
    agg_dict = {
        'y_true': 'mean',
        'y_pred': 'mean'
    }
    
    # Add baseline columns if they exist
    if 'y_pred_lag1' in df_region.columns:
        agg_dict['y_pred_lag1'] = 'mean'
    if 'y_pred_lag4' in df_region.columns:
        agg_dict['y_pred_lag4'] = 'mean'
    
    df_agg = df_region.groupby(['quarter_end_date', 'unit_type']).agg(agg_dict).reset_index()
    
    # Sort by quarter_end_date
    df_agg = df_agg.sort_values('quarter_end_date')
    
    return df_agg


def create_rent_chart(
    df: pd.DataFrame,
    region_name: str,
    unit_type: str,
    output_dir: Path
) -> None:
    """
    Create a Plotly chart for a specific region and unit type.
    
    Args:
        df: DataFrame filtered to the region and unit type.
        region_name: Name of the region (for title and filename).
        unit_type: Unit type name (for title and filename).
        output_dir: Directory to save the chart.
    """
    # Filter to unit type
    df_unit = df[df['unit_type'] == unit_type].copy()
    
    if len(df_unit) == 0:
        print(f"Warning: No data for {region_name} - {unit_type}, skipping chart")
        return
    
    # Sort by quarter_end_date
    df_unit = df_unit.sort_values('quarter_end_date')
    
    # Format quarter labels for x-axis
    df_unit['quarter_label'] = df_unit['quarter_end_date'].apply(format_quarter_label)
    
    # Create figure
    fig = go.Figure()
    
    # Add traces for each series (only if data exists)
    if 'y_true' in df_unit.columns and df_unit['y_true'].notna().any():
        fig.add_trace(go.Scatter(
            x=df_unit['quarter_label'],
            y=df_unit['y_true'],
            mode='lines+markers',
            name='Actual',
            line=dict(width=2.5),
            marker=dict(size=6)
        ))
    
    if 'y_pred' in df_unit.columns and df_unit['y_pred'].notna().any():
        fig.add_trace(go.Scatter(
            x=df_unit['quarter_label'],
            y=df_unit['y_pred'],
            mode='lines+markers',
            name='Model',
            line=dict(width=2.5, dash='dash'),
            marker=dict(size=6)
        ))
    
    if 'y_pred_lag1' in df_unit.columns and df_unit['y_pred_lag1'].notna().any():
        fig.add_trace(go.Scatter(
            x=df_unit['quarter_label'],
            y=df_unit['y_pred_lag1'],
            mode='lines+markers',
            name='Baseline lag1',
            line=dict(width=2, dash='dot'),
            marker=dict(size=5)
        ))
    
    if 'y_pred_lag4' in df_unit.columns and df_unit['y_pred_lag4'].notna().any():
        fig.add_trace(go.Scatter(
            x=df_unit['quarter_label'],
            y=df_unit['y_pred_lag4'],
            mode='lines+markers',
            name='Baseline lag4',
            line=dict(width=2, dash='dot'),
            marker=dict(size=5)
        ))
    
    # Update layout
    fig.update_layout(
        template='plotly_white',
        width=1200,
        height=650,
        title={
            'text': f"{region_name}: {unit_type} — Actual vs Predicted (with baselines)",
            'x': 0.5,
            'xanchor': 'center',
            'font': {'size': 18}
        },
        xaxis=dict(
            title='Quarter (period end) — quarterly alignment across all datasets',
            tickangle=-35,
            automargin=True,
            type='category'  # Categorical axis to prevent overlap
        ),
        yaxis=dict(
            title='Asking rent (CAD) — target variable being forecast'
        ),
        hovermode='x unified',
        legend=dict(
            orientation='h',
            yanchor='bottom',
            y=1.02,
            xanchor='right',
            x=1
        ),
        margin=dict(l=80, r=80, t=100, b=120)  # Generous margins
    )
    
    # Create filename based on exact requirements
    # Map region names to filename components
    region_map = {
        "Toronto": "toronto",
        "GTA proxy": "gta_proxy",
        "GTHA proxy": "gtha_proxy"
    }
    region_safe = region_map.get(region_name, region_name.lower().replace(' ', '_'))
    
    # Map unit types to filename components
    if "1 bedroom" in unit_type:
        unit_safe = "1bed"
    elif "2 bedrooms" in unit_type:
        unit_safe = "2bed"
    else:
        unit_safe = unit_type.lower().replace(' ', '_').replace('-', '')
    
    filename_base = f"rent_{region_safe}_{unit_safe}"
    
    # Save HTML
    html_path = output_dir / f"{filename_base}.html"
    fig.write_html(str(html_path))
    print(f"Saved HTML: {html_path}")
    
    # Try to save PNG (warn if fails, don't crash)
    try:
        png_path = output_dir / f"{filename_base}.png"
        fig.write_image(str(png_path), width=1200, height=650, scale=2)
        print(f"Saved PNG: {png_path}")
    except Exception as e:
        warnings.warn(f"Failed to save PNG for {filename_base}: {e}. HTML saved successfully.")


def generate_toronto_gta_rent_charts() -> None:
    """
    Generate LinkedIn-ready Plotly charts for rent forecasts focused on Toronto and the GTA.
    
    Reads outputs/rent_predictions.csv and creates charts for:
    - Toronto (CMA == "Toronto")
    - GTA proxy aggregate: mean across CMAs ["Toronto", "Oshawa"]
    - Optional GTHA proxy aggregate: mean across ["Toronto", "Oshawa", "Hamilton"]
    
    Creates separate charts for:
    - "Apartment - 1 bedroom"
    - "Apartment - 2 bedrooms"
    
    Saves HTML and PNG files to outputs/plots/plotly/
    """
    # Read predictions
    predictions_path = OUTPUT_DIR / "rent_predictions.csv"
    if not predictions_path.exists():
        raise FileNotFoundError(f"Predictions file not found: {predictions_path}")
    
    print(f"Reading predictions from: {predictions_path}")
    df = pd.read_csv(predictions_path)
    
    # Convert quarter_end_date to datetime
    df['quarter_end_date'] = pd.to_datetime(df['quarter_end_date'])
    
    # Create output directory
    output_dir = OUTPUT_DIR / "plots" / "plotly"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Define regions
    regions = {
        "Toronto": ["Toronto"],
        "GTA proxy": ["Toronto", "Oshawa"],
        "GTHA proxy": ["Toronto", "Oshawa", "Hamilton"]
    }
    
    # Unit types
    unit_types = [
        "Apartment - 1 bedroom",
        "Apartment - 2 bedrooms"
    ]
    
    # Generate charts for each region and unit type
    for region_name, cma_list in regions.items():
        print(f"\nProcessing {region_name}...")
        
        # Prepare data for region
        df_region = prepare_data_for_region(df, cma_list)
        
        if len(df_region) == 0:
            print(f"  Warning: No data found for {region_name}, skipping")
            continue
        
        # Create chart for each unit type
        for unit_type in unit_types:
            print(f"  Creating chart for {unit_type}...")
            create_rent_chart(df_region, region_name, unit_type, output_dir)
    
    print(f"\nAll charts generated and saved to: {output_dir}")


if __name__ == "__main__":
    generate_toronto_gta_rent_charts()
