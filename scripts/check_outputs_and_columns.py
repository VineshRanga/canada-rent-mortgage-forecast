"""
Validation script for checking required output files and columns.

Usage:
    Run from repo root:
    python scripts/check_outputs_and_columns.py

This script validates that all required output files exist and have the expected
columns for plotting and analysis.
"""

import sys
from pathlib import Path
from typing import Tuple
import pandas as pd
import json

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.config import OUTPUT_DIR, PROCESSED_DIR


def check_file_exists(filepath: Path) -> bool:
    """Check if a file exists."""
    return filepath.exists()


def check_csv_columns(filepath: Path, required_cols: list, optional_cols: list = None) -> Tuple[bool, list]:
    """
    Check if CSV file has required and optional columns.
    
    Returns:
        (has_required, warnings) where warnings is a list of missing optional columns
    """
    if optional_cols is None:
        optional_cols = []
    
    try:
        df = pd.read_csv(filepath, nrows=0)  # Read only header
        existing_cols = set(df.columns)
        required_set = set(required_cols)
        optional_set = set(optional_cols)
        
        missing_required = required_set - existing_cols
        missing_optional = optional_set - existing_cols
        
        has_required = len(missing_required) == 0
        warnings = []
        
        if missing_optional:
            warnings = [f"Missing optional column: {col}" for col in missing_optional]
        
        return has_required, warnings
    except Exception as e:
        print(f"  [ERROR] Failed to read CSV: {e}")
        return False, []


def check_parquet_file(filepath: Path, date_col: str = None) -> dict:
    """
    Check parquet file and return metadata.
    
    Returns:
        dict with keys: exists, row_count, date_min, date_max, date_col_found
    """
    result = {
        'exists': False,
        'row_count': 0,
        'date_min': None,
        'date_max': None,
        'date_col_found': None
    }
    
    if not filepath.exists():
        return result
    
    result['exists'] = True
    
    try:
        df = pd.read_parquet(filepath)
        result['row_count'] = len(df)
        
        # Detect date column
        date_cols = ['quarter_end_date', 'date', 'REF_DATE', 'ref_date']
        found_date_col = None
        
        if date_col and date_col in df.columns:
            found_date_col = date_col
        else:
            for col in date_cols:
                if col in df.columns:
                    found_date_col = col
                    break
        
        if found_date_col:
            result['date_col_found'] = found_date_col
            df[found_date_col] = pd.to_datetime(df[found_date_col], errors='coerce')
            valid_dates = df[found_date_col].dropna()
            if len(valid_dates) > 0:
                result['date_min'] = valid_dates.min()
                result['date_max'] = valid_dates.max()
    
    except Exception as e:
        print(f"  [ERROR] Failed to read parquet: {e}")
    
    return result


def main():
    """Main validation function."""
    print("=" * 80)
    print("OUTPUT FILES AND COLUMNS VALIDATION")
    print("=" * 80)
    print()
    
    missing_files = []
    warnings = []
    errors = []
    
    # Files to check
    files_to_check = {
        'outputs/rent_predictions.csv': {
            'path': OUTPUT_DIR / 'rent_predictions.csv',
            'required_cols': ['quarter_end_date', 'cma', 'unit_type', 'y_true', 'y_pred'],
            'optional_cols': ['y_pred_lag1', 'y_pred_lag4']
        },
        'outputs/rent_metrics.json': {
            'path': OUTPUT_DIR / 'rent_metrics.json',
            'required_cols': None,  # JSON file, check separately
            'optional_cols': None
        },
        'outputs/rent_model_coefficients.csv': {
            'path': OUTPUT_DIR / 'rent_model_coefficients.csv',
            'required_cols': ['feature', 'coef'],
            'optional_cols': ['family']
        },
        'outputs/mortgage_predictions.csv': {
            'path': OUTPUT_DIR / 'mortgage_predictions.csv',
            'required_cols': ['quarter_end_date', 'y_true', 'y_pred'],
            'optional_cols': None
        },
        'outputs/mortgage_metrics.json': {
            'path': OUTPUT_DIR / 'mortgage_metrics.json',
            'required_cols': None,  # JSON file, check separately
            'optional_cols': None
        },
        'data/processed/rent_target_quarterly.parquet': {
            'path': PROCESSED_DIR / 'rent_target_quarterly.parquet',
            'required_cols': None,  # Parquet file, check separately
            'optional_cols': None,
            'date_col': 'quarter_end_date'
        },
        'data/processed/mortgage_target_quarterly.parquet': {
            'path': PROCESSED_DIR / 'mortgage_target_quarterly.parquet',
            'required_cols': None,  # Parquet file, check separately
            'optional_cols': None,
            'date_col': None  # Detect automatically
        },
        'data/processed/exog_quarterly.parquet': {
            'path': PROCESSED_DIR / 'exog_quarterly.parquet',
            'required_cols': None,  # Parquet file, check separately
            'optional_cols': None,
            'date_col': 'quarter_end_date'
        }
    }
    
    # Check each file
    for file_key, file_info in files_to_check.items():
        filepath = file_info['path']
        print(f"Checking: {file_key}")
        
        if not check_file_exists(filepath):
            print(f"  [MISSING] {filepath}")
            missing_files.append(file_key)
            print()
            continue
        
        print(f"  [OK] File exists")
        
        # Check CSV files
        if filepath.suffix == '.csv' and file_info['required_cols']:
            has_required, csv_warnings = check_csv_columns(
                filepath,
                file_info['required_cols'],
                file_info.get('optional_cols', [])
            )
            
            if not has_required:
                missing_cols = set(file_info['required_cols']) - set(pd.read_csv(filepath, nrows=0).columns)
                print(f"  [ERROR] Missing required columns: {', '.join(missing_cols)}")
                errors.append(f"{file_key}: missing required columns")
            else:
                print(f"  [OK] All required columns present")
            
            if csv_warnings:
                for warning in csv_warnings:
                    print(f"  [WARN] {warning}")
                    warnings.append(f"{file_key}: {warning}")
        
        # Check JSON files
        elif filepath.suffix == '.json':
            try:
                with open(filepath, 'r') as f:
                    json.load(f)
                print(f"  [OK] Valid JSON file")
            except Exception as e:
                print(f"  [ERROR] Invalid JSON: {e}")
                errors.append(f"{file_key}: invalid JSON")
        
        # Check parquet files
        elif filepath.suffix == '.parquet':
            date_col = file_info.get('date_col')
            parquet_info = check_parquet_file(filepath, date_col)
            
            if parquet_info['exists']:
                print(f"  [OK] Row count: {parquet_info['row_count']:,}")
                
                if parquet_info['date_col_found']:
                    print(f"  [OK] Date column: {parquet_info['date_col_found']}")
                    if parquet_info['date_min'] and parquet_info['date_max']:
                        print(f"  [OK] Date range: {parquet_info['date_min']} to {parquet_info['date_max']}")
                    else:
                        print(f"  [WARN] No valid dates found")
                        warnings.append(f"{file_key}: no valid dates")
                else:
                    print(f"  [WARN] No date column found")
                    warnings.append(f"{file_key}: no date column")
        
        print()
    
    # Summary
    print("=" * 80)
    print("VALIDATION SUMMARY")
    print("=" * 80)
    print(f"Missing files: {len(missing_files)}")
    if missing_files:
        for f in missing_files:
            print(f"  - {f}")
    
    print(f"Warnings: {len(warnings)}")
    if warnings:
        for w in warnings[:10]:  # Show first 10 warnings
            print(f"  - {w}")
        if len(warnings) > 10:
            print(f"  ... and {len(warnings) - 10} more warnings")
    
    print(f"Errors: {len(errors)}")
    if errors:
        for e in errors:
            print(f"  - {e}")
    
    # Final status
    can_plot = len(missing_files) == 0 and len(errors) == 0
    print()
    print(f"OK to generate Plotly charts: {'YES' if can_plot else 'NO'}")
    
    # Exit with appropriate status code
    if len(missing_files) > 0 or len(errors) > 0:
        sys.exit(1)
    else:
        sys.exit(0)


if __name__ == "__main__":
    main()

