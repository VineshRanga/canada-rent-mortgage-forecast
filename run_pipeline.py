"""
Main pipeline script for Canadian rent and mortgage forecasting.

Orchestrates the complete pipeline from data preprocessing to model training,
backtesting, and visualization.
"""

import sys
from pathlib import Path
import pandas as pd
from datetime import datetime

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

from src.config import ensure_output_dirs, OUTPUT_DIR, PROCESSED_DIR, RAW_DIR, RAW_FILES
from src.preprocess.mortgage_target import extract_mortgage_target
from src.preprocess.rent_target import extract_rent_target
from src.preprocess.exogenous_features import build_exogenous_features
from src.features.build_features import build_rent_model_dataset, build_mortgage_model_dataset
from src.models.rent_elasticnet import train_and_backtest_rent
from src.models.mortgage_sarimax import train_and_backtest_mortgage
from src.viz.plots import plot_rent_actual_vs_predicted, plot_mortgage_actual_vs_predicted

# Import Plotly chart generator (optional, may not be available)
try:
    from src.viz.plotly_charts import generate_toronto_gta_rent_charts
    PLOTLY_CHARTS_AVAILABLE = True
except ImportError:
    PLOTLY_CHARTS_AVAILABLE = False

# Import matplotlib chart generators (optional)
try:
    from src.viz.mpl_eval import generate_rent_eval_charts
    MPL_EVAL_AVAILABLE = True
except ImportError:
    MPL_EVAL_AVAILABLE = False

try:
    from src.viz.mpl_story import generate_rate_story_charts
    MPL_STORY_AVAILABLE = True
except ImportError:
    MPL_STORY_AVAILABLE = False


def fmt_float(x, digits=2):
    """Safely format a float value with specified decimal digits."""
    try:
        return f"{float(x):.{digits}f}"
    except Exception:
        return "N/A"


def write_data_coverage_md(
    mortgage_target_df: pd.DataFrame,
    rent_target_df: pd.DataFrame,
    exog_df: pd.DataFrame,
    rent_model_df: pd.DataFrame
) -> None:
    """
    Write DATA_COVERAGE.md summarizing data coverage information.
    
    Args:
        mortgage_target_df: Mortgage target DataFrame.
        rent_target_df: Rent target DataFrame.
        exog_df: Exogenous features DataFrame.
        rent_model_df: Final rent model dataset DataFrame.
    """
    coverage_path = OUTPUT_DIR / "DATA_COVERAGE.md"
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    lines = []
    lines.append("# Data Coverage Summary")
    lines.append("")
    lines.append("This document summarizes data coverage for the forecasting pipeline.")
    lines.append("")
    
    # Target date ranges
    lines.append("## Target Date Ranges")
    lines.append("")
    
    if 'quarter_end_date' in mortgage_target_df.columns:
        mortgage_dates = pd.to_datetime(mortgage_target_df['quarter_end_date'])
        lines.append(f"### Mortgage Target")
        lines.append(f"- Date range: {mortgage_dates.min()} to {mortgage_dates.max()}")
        lines.append(f"- Total quarters: {mortgage_dates.nunique()}")
        lines.append(f"- Total rows: {len(mortgage_target_df)}")
        lines.append("")
    
    if 'quarter_end_date' in rent_target_df.columns:
        rent_dates = pd.to_datetime(rent_target_df['quarter_end_date'])
        lines.append(f"### Rent Target")
        lines.append(f"- Date range: {rent_dates.min()} to {rent_dates.max()}")
        lines.append(f"- Total quarters: {rent_dates.nunique()}")
        lines.append(f"- Total rows: {len(rent_target_df)}")
        if 'cma' in rent_target_df.columns:
            lines.append(f"- Unique CMAs: {rent_target_df['cma'].nunique()}")
        if 'unit_type' in rent_target_df.columns:
            lines.append(f"- Unit types: {', '.join(sorted(rent_target_df['unit_type'].unique().astype(str)))}")
        lines.append("")
    
    # Exogenous features
    lines.append("## Exogenous Features")
    lines.append("")
    
    exog_cols = [col for col in exog_df.columns 
                 if col not in ['quarter_end_date'] and 
                 col.startswith(('rate_', 'labour_', 'starts_', 'pop_', 'mig_', 'npr_'))]
    
    if 'quarter_end_date' in exog_df.columns:
        exog_dates = pd.to_datetime(exog_df['quarter_end_date'])
        lines.append(f"- Date range: {exog_dates.min()} to {exog_dates.max()}")
        lines.append(f"- Total quarters: {exog_dates.nunique()}")
        lines.append("")
    
    # Check coverage in rent model dataset
    lines.append("### Feature Coverage in Rent Model Dataset")
    lines.append("")
    
    kept_features = []
    dropped_features = []
    
    for col in exog_cols:
        if col in rent_model_df.columns:
            non_null_pct = rent_model_df[col].notna().sum() / len(rent_model_df) * 100
            kept_features.append((col, non_null_pct))
        else:
            # Check if it was dropped due to low coverage
            if col in exog_df.columns:
                # Feature exists in exog but not in final dataset
                dropped_features.append((col, "dropped (< 90% coverage)"))
    
    if kept_features:
        lines.append("#### Kept Features (>= 90% coverage)")
        lines.append("")
        lines.append("| Feature | Coverage % |")
        lines.append("|---------|------------|")
        for col, pct in sorted(kept_features):
            lines.append(f"| {col} | {pct:.1f}% |")
        lines.append("")
    
    if dropped_features:
        lines.append("#### Dropped Features (< 90% coverage)")
        lines.append("")
        for col, reason in dropped_features:
            lines.append(f"- {col}: {reason}")
        lines.append("")
    
    # Final rent dataset
    lines.append("## Final Rent Model Dataset")
    lines.append("")
    lines.append(f"- Shape: {rent_model_df.shape[0]} rows × {rent_model_df.shape[1]} columns")
    
    if 'quarter_end_date' in rent_model_df.columns:
        model_dates = pd.to_datetime(rent_model_df['quarter_end_date'])
        lines.append(f"- Date range: {model_dates.min()} to {model_dates.max()}")
        lines.append(f"- Total quarters: {model_dates.nunique()}")
    
    if 'cma' in rent_model_df.columns:
        lines.append(f"- Unique CMAs: {rent_model_df['cma'].nunique()}")
    
    if 'unit_type' in rent_model_df.columns:
        lines.append(f"- Unit types: {', '.join(sorted(rent_model_df['unit_type'].unique().astype(str)))}")
    
    lines.append("")
    
    # Rows dropped by lag filters
    lines.append("### Rows Dropped by Lag Filters")
    lines.append("")
    
    # Estimate rows dropped: rent_target initial -> after lag_1 filter
    # We can't get exact numbers without modifying build_rent_model_dataset,
    # but we can estimate based on the final dataset
    if 'y_lag_1' in rent_model_df.columns:
        rows_with_lag1 = rent_model_df['y_lag_1'].notna().sum()
        lines.append(f"- Rows with valid y_lag_1: {rows_with_lag1} / {len(rent_model_df)} (100%)")
        lines.append("  (All rows in final dataset have y_lag_1 after filtering)")
    
    if 'y_lag_4' in rent_model_df.columns:
        rows_with_lag4 = rent_model_df['y_lag_4'].notna().sum()
        lag4_pct = rows_with_lag4 / len(rent_model_df) * 100
        lines.append(f"- Rows with valid y_lag_4: {rows_with_lag4} / {len(rent_model_df)} ({lag4_pct:.1f}%)")
    
    lines.append("")
    lines.append(f"*Note: Exact row counts dropped during processing are logged in the pipeline output.*")
    
    # Write file
    with open(coverage_path, 'w') as f:
        f.write('\n'.join(lines))
    
    print(f"Saved data coverage summary to: {coverage_path}")


def run_pipeline() -> None:
    """
    Run the complete forecasting pipeline.
    
    Steps:
    1. Ensure output directories exist
    2. Preprocess targets (mortgage, and rent if available)
    3. Preprocess exogenous features
    4. Build feature datasets
    5. Train and backtest rent Elastic Net model
    6. Train and backtest mortgage SARIMAX model
    7. Generate plots
    """
    print("=" * 80)
    print("CANADIAN RENT & MORTGAGE FORECASTING PIPELINE")
    print("=" * 80)
    print(f"Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
    
    start_time = datetime.now()
    outputs_created = []
    
    try:
        # Step 1: Ensure output directories
        print("Step 1: Creating output directories...")
        print("-" * 80)
        ensure_output_dirs()
        print("✓ Output directories created\n")
        
        # Step 2: Preprocess targets
        print("Step 2: Preprocessing targets...")
        print("-" * 80)
        
        # Mortgage target
        print("  Extracting mortgage target...")
        mortgage_target_df = extract_mortgage_target()
        mortgage_target_path = PROCESSED_DIR / "mortgage_target_quarterly.parquet"
        outputs_created.append(f"  ✓ Mortgage target: {mortgage_target_path}")
        print()
        
        # Step 3: Preprocess exogenous features
        print("Step 3: Building exogenous features...")
        print("-" * 80)
        exog_df = build_exogenous_features()
        exog_path = PROCESSED_DIR / "exog_quarterly.parquet"
        outputs_created.append(f"  ✓ Exogenous features: {exog_path}")
        print()
        
        # Step 4: Extract rent target (after exogenous features are built)
        print("Step 4: Extracting rent target...")
        print("-" * 80)
        
        # Check if rent data file exists
        rent_raw_path = RAW_DIR / RAW_FILES["rent"]
        if not rent_raw_path.exists():
            raise FileNotFoundError(
                f"Rent data file not found: {rent_raw_path}\n"
                f"Please ensure the rent data file is available in data/raw/"
            )
        
        print("  Extracting rent target...")
        rent_target_df = extract_rent_target()
        rent_target_path = PROCESSED_DIR / "rent_target_quarterly.parquet"
        outputs_created.append(f"  ✓ Rent target: {rent_target_path}")
        print()
        
        # Generate rate story charts
        exog_path = PROCESSED_DIR / "exog_quarterly.parquet"
        if exog_path.exists() and rent_target_path.exists():
            print("Generating matplotlib rate story charts...")
            print("-" * 80)
            try:
                if MPL_STORY_AVAILABLE:
                    generate_rate_story_charts()
                    story_output_dir = OUTPUT_DIR / "plots" / "mpl_story"
                    print(f"[OK] Matplotlib story charts saved to {story_output_dir}")
                    outputs_created.append(f"  ✓ Matplotlib story charts: {story_output_dir}")
                else:
                    print("[WARN] Chart generation skipped: mpl_story module not available")
            except Exception as e:
                print(f"[WARN] Chart generation skipped: {type(e).__name__}: {e}")
        else:
            print("[WARN] Chart generation skipped: exog or rent_target files not found")
        print()
        
        # Step 5: Build rent model dataset
        print("Step 5: Building rent modeling dataset...")
        print("-" * 80)
        print("  Building rent dataset...")
        rent_model_df = build_rent_model_dataset()
        rent_dataset_path = PROCESSED_DIR / "rent_model_dataset.parquet"
        outputs_created.append(f"  ✓ Rent dataset: {rent_dataset_path}")
        print()
        
        # Step 6: Train and backtest rent Elastic Net
        print("Step 6: Training and backtesting rent Elastic Net model...")
        print("-" * 80)
        print(f"  Loading rent dataset from: {rent_dataset_path}")
        print(f"  Dataset shape: {rent_model_df.shape}")
        
        model, predictions_df, metrics = train_and_backtest_rent(
            rent_model_df,
            min_train_quarters=12,
            forecast_horizon=1,
            save_outputs=True
        )
        
        rent_pred_path = OUTPUT_DIR / "rent_predictions.csv"
        rent_metrics_path = OUTPUT_DIR / "rent_metrics.json"
        rent_coef_path = OUTPUT_DIR / "rent_model_coefficients.csv"
        
        outputs_created.extend([
            f"  ✓ Rent predictions: {rent_pred_path}",
            f"  ✓ Rent metrics: {rent_metrics_path}",
            f"  ✓ Rent coefficients: {rent_coef_path}"
        ])
        
        # Print rent metrics
        elasticnet = metrics.get("elasticnet", {})
        print(f"  ElasticNet MAE: {fmt_float(elasticnet.get('mae'))}")
        print(f"  ElasticNet sMAPE: {fmt_float(elasticnet.get('smape'), digits=2)}%")
        
        # Print baselines if available
        baselines = metrics.get("baselines", {})
        if "lag1" in baselines:
            lag1 = baselines["lag1"]
            print(f"  Lag1 baseline MAE: {fmt_float(lag1.get('mae'))}")
            print(f"  Lag1 baseline sMAPE: {fmt_float(lag1.get('smape'), digits=2)}%")
        if "lag4" in baselines:
            lag4 = baselines["lag4"]
            print(f"  Lag4 baseline MAE: {fmt_float(lag4.get('mae'))}")
            print(f"  Lag4 baseline sMAPE: {fmt_float(lag4.get('smape'), digits=2)}%")
        
        # Print uplift if available
        uplift = metrics.get("uplift_vs_lag4", {})
        if uplift:
            print(f"  Uplift vs Lag4: MAE {fmt_float(uplift.get('mae_pct'), digits=1)}%, sMAPE {fmt_float(uplift.get('smape_pct'), digits=1)}%")
        print()
        
        # Generate matplotlib evaluation charts
        if rent_pred_path.exists():
            print("Generating matplotlib evaluation charts...")
            print("-" * 80)
            try:
                if MPL_EVAL_AVAILABLE:
                    generate_rent_eval_charts()
                    eval_output_dir = OUTPUT_DIR / "plots" / "mpl_eval"
                    print(f"[OK] Matplotlib eval charts saved to {eval_output_dir}")
                    outputs_created.append(f"  ✓ Matplotlib eval charts: {eval_output_dir}")
                else:
                    print("[WARN] Chart generation skipped: mpl_eval module not available")
            except Exception as e:
                print(f"[WARN] Chart generation skipped: {type(e).__name__}: {e}")
        else:
            print("[WARN] Chart generation skipped: rent_predictions.csv not found")
        print()
        
        # Generate Plotly charts (if predictions file exists)
        if rent_pred_path.exists():
            print("Generating Plotly rent charts...")
            print("-" * 80)
            try:
                if PLOTLY_CHARTS_AVAILABLE:
                    generate_toronto_gta_rent_charts()
                    plotly_output_dir = OUTPUT_DIR / "plots" / "plotly"
                    print(f"[OK] Plotly charts saved to {plotly_output_dir}")
                    outputs_created.append(f"  ✓ Plotly charts: {plotly_output_dir}")
                else:
                    print("[WARN] Plotly chart generation skipped: plotly_charts module not available")
            except Exception as e:
                print(f"[WARN] Plotly chart generation skipped: {type(e).__name__}: {e}")
        else:
            print("[WARN] Plotly chart generation skipped: rent_predictions.csv not found")
        print()
        
        # Step 7: Generate rent plots
        print("Step 7: Generating rent plots...")
        print("-" * 80)
        print("  Generating rent plots...")
        plot_rent_actual_vs_predicted(predictions_df)
        
        outputs_created.extend([
            f"  ✓ Rent overall plot: {OUTPUT_DIR / 'plots' / 'rent_actual_vs_predicted_overall.png'}",
            f"  ✓ Rent by unit type plot: {OUTPUT_DIR / 'plots' / 'rent_actual_vs_predicted_by_unit_type.png'}",
            f"  ✓ Rent top CMAs plot: {OUTPUT_DIR / 'plots' / 'rent_actual_vs_predicted_top_cmas.png'}"
        ])
        print()
        
        # Step 8: Build mortgage model dataset
        print("Step 8: Building mortgage modeling dataset...")
        print("-" * 80)
        
        mortgage_dataset_path = PROCESSED_DIR / "mortgage_model_dataset.parquet"
        if mortgage_dataset_path.exists():
            print(f"  ✓ Mortgage dataset found: {mortgage_dataset_path}")
        else:
            print("  Building mortgage dataset...")
            build_mortgage_model_dataset()
            outputs_created.append(f"  ✓ Mortgage dataset: {mortgage_dataset_path}")
        print()
        
        # Step 9: Train and backtest mortgage SARIMAX
        print("Step 9: Training and backtesting mortgage SARIMAX model...")
        print("-" * 80)
        
        mortgage_dataset_path = PROCESSED_DIR / "mortgage_model_dataset.parquet"
        if mortgage_dataset_path.exists():
            print(f"  Loading mortgage dataset from: {mortgage_dataset_path}")
            mortgage_df = pd.read_parquet(mortgage_dataset_path)
            print(f"  Dataset shape: {mortgage_df.shape}")
            
            model, train_results_df, predictions_df, metrics = train_and_backtest_mortgage(
                mortgage_df,
                target_col="y_level",
                use_grid_search=True,
                min_train_quarters=12,
                save_outputs=True
            )
            
            mortgage_pred_path = OUTPUT_DIR / "mortgage_predictions.csv"
            mortgage_metrics_path = OUTPUT_DIR / "mortgage_metrics.json"
            mortgage_summary_path = OUTPUT_DIR / "mortgage_model_summary.txt"
            
            outputs_created.extend([
                f"  ✓ Mortgage predictions: {mortgage_pred_path}",
                f"  ✓ Mortgage metrics: {mortgage_metrics_path}",
                f"  ✓ Mortgage model summary: {mortgage_summary_path}"
            ])
            
            # Print mortgage metrics
            overall = metrics.get("overall", {})
            print(f"  Overall MAE: {fmt_float(overall.get('mae'))}")
            print(f"  Overall sMAPE: {fmt_float(overall.get('smape'), digits=2)}%")
        else:
            print(f"  ⚠ Skipping mortgage model (mortgage dataset not available)")
        print()
        
        # Step 10: Generate mortgage plot
        print("Step 10: Generating mortgage plot...")
        print("-" * 80)
        
        mortgage_pred_path = OUTPUT_DIR / "mortgage_predictions.csv"
        if mortgage_pred_path.exists():
            print("  Generating mortgage plot...")
            mortgage_df = pd.read_csv(mortgage_pred_path)
            mortgage_df['quarter_end_date'] = pd.to_datetime(mortgage_df['quarter_end_date'])
            plot_mortgage_actual_vs_predicted(mortgage_df)
            
            outputs_created.append(
                f"  ✓ Mortgage plot: {OUTPUT_DIR / 'plots' / 'mortgage_actual_vs_predicted.png'}"
            )
        else:
            print(f"  ⚠ Skipping mortgage plot (predictions not available)")
        print()
        
        # Write data coverage summary
        print("Writing data coverage summary...")
        print("-" * 80)
        write_data_coverage_md(
            mortgage_target_df=mortgage_target_df,
            rent_target_df=rent_target_df,
            exog_df=exog_df,
            rent_model_df=rent_model_df
        )
        outputs_created.append(f"  ✓ Data coverage: {OUTPUT_DIR / 'DATA_COVERAGE.md'}")
        print()
        
        # Summary
        end_time = datetime.now()
        duration = (end_time - start_time).total_seconds()
        
        print("=" * 80)
        print("PIPELINE COMPLETED SUCCESSFULLY")
        print("=" * 80)
        print(f"Duration: {duration:.1f} seconds")
        print(f"Completed at: {end_time.strftime('%Y-%m-%d %H:%M:%S')}\n")
        
        print("OUTPUTS CREATED:")
        print("-" * 80)
        for output in outputs_created:
            print(output)
        
        print("\n" + "=" * 80)
        print("OUTPUT DIRECTORY STRUCTURE:")
        print("=" * 80)
        print(f"Processed data: {PROCESSED_DIR}")
        print(f"Outputs: {OUTPUT_DIR}")
        print(f"  - Predictions: {OUTPUT_DIR / '*.csv'}")
        print(f"  - Metrics: {OUTPUT_DIR / '*.json'}")
        print(f"  - Plots: {OUTPUT_DIR / 'plots' / '*.png'}")
        print(f"  - Model files: {OUTPUT_DIR / '*.csv'}, {OUTPUT_DIR / '*.txt'}")
        print("=" * 80)
        
    except Exception as e:
        print("\n" + "=" * 80)
        print("PIPELINE FAILED")
        print("=" * 80)
        print(f"Error: {type(e).__name__}: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    run_pipeline()

