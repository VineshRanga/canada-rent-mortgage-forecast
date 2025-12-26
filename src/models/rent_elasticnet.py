"""
Elastic Net model for rent forecasting.

Implements ElasticNetCV with time-series cross-validation and rolling backtest
for forecasting apartment rents by CMA and unit type.
"""

from pathlib import Path
from typing import Optional, Tuple, Dict, List
import pandas as pd
import numpy as np
import json
from datetime import datetime

from sklearn.linear_model import ElasticNetCV
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import TimeSeriesSplit

from src.config import OUTPUT_DIR, PROCESSED_DIR
from src.evaluation.metrics import mae, smape


def _categorize_feature(feature_name: str) -> str:
    """
    Categorize a feature name into a family.
    
    Args:
        feature_name: Name of the feature.
    
    Returns:
        Family name: lag, rates, labour, starts, pop, migration, cma, unit_type, quarter
    """
    feature_lower = feature_name.lower()
    
    if feature_lower.startswith('y_lag_') or 'lag' in feature_lower:
        return 'lag'
    elif feature_lower.startswith('rate_'):
        return 'rates'
    elif feature_lower.startswith('labour_'):
        return 'labour'
    elif feature_lower.startswith('starts_'):
        return 'starts'
    elif feature_lower.startswith('pop_'):
        return 'pop'
    elif feature_lower.startswith('mig_') or feature_lower.startswith('npr_'):
        return 'migration'
    elif feature_lower.startswith('cma_'):
        return 'cma'
    elif feature_lower.startswith('unit_type_'):
        return 'unit_type'
    elif feature_lower.startswith('quarter_'):
        return 'quarter'
    else:
        return 'other'




def prepare_features(df: pd.DataFrame) -> Tuple[pd.DataFrame, np.ndarray, List[str]]:
    """
    Prepare feature matrix with one-hot encoding and feature names.
    
    One-hot encodes: cma, unit_type, quarter (treat as categorical).
    Includes: y_lag_1..y_lag_4, exog columns, and categorical one-hots.
    
    Args:
        df: Input DataFrame with columns including cma, unit_type, quarter,
            lag features, and exogenous features.
    
    Returns:
        Tuple of (X, y, feature_names) where:
        - X: Feature matrix (DataFrame)
        - y: Target vector (numpy array)
        - feature_names: List of feature names
    """
    # Start with numeric features
    numeric_features = []
    
    # Lag features
    for lag in [1, 2, 3, 4]:
        col = f'y_lag_{lag}'
        if col in df.columns:
            numeric_features.append(col)
    
    # Exogenous features (all columns starting with rate_, labour_, starts_, pop_, mig_, npr_)
    exog_prefixes = ['rate_', 'labour_', 'starts_', 'pop_', 'mig_', 'npr_']
    for col in df.columns:
        if any(col.startswith(prefix) for prefix in exog_prefixes):
            if col not in numeric_features:
                numeric_features.append(col)
    
    # Build numeric feature matrix
    X_numeric = df[numeric_features].copy()
    
    # One-hot encode cma, unit_type, and quarter (treat quarter as categorical)
    X_cma = pd.get_dummies(df['cma'], prefix='cma', drop_first=False)
    X_unit = pd.get_dummies(df['unit_type'], prefix='unit_type', drop_first=False)
    
    # One-hot encode quarter (treat as categorical)
    if 'quarter' in df.columns:
        X_quarter = pd.get_dummies(df['quarter'], prefix='quarter', drop_first=False)
    else:
        # If quarter not in df, create it from quarter_end_date
        if 'quarter_end_date' in df.columns:
            quarter_vals = pd.to_datetime(df['quarter_end_date']).dt.quarter
            X_quarter = pd.get_dummies(quarter_vals, prefix='quarter', drop_first=False)
        else:
            X_quarter = pd.DataFrame()
    
    # Combine all features
    X = pd.concat([X_numeric, X_cma, X_unit, X_quarter], axis=1)
    
    # Extract target
    y = df['y'].values
    
    # Get feature names
    feature_names = list(X.columns)
    
    return X, y, feature_names


def train_elasticnet_rent(
    df: pd.DataFrame,
    min_train_quarters: int = 12,
    n_splits: int = 5,
    alphas: Optional[np.ndarray] = None,
    l1_ratios: Optional[List[float]] = None
) -> Tuple[ElasticNetCV, StandardScaler, List[str], pd.DataFrame]:
    """
    Train ElasticNet model for rent forecasting with time-series cross-validation.
    
    Uses ElasticNetCV with time-series split on quarters to avoid data leakage.
    
    Args:
        df: Input DataFrame with features and target.
        min_train_quarters: Minimum number of quarters for training.
        n_splits: Number of time-series splits for cross-validation.
        alphas: Array of alpha values to try. Defaults to logspace(-3, 1, 50).
        l1_ratios: Array of l1_ratio values to try. Defaults to [0.1, 0.5, 0.7, 0.9, 0.95, 0.99, 1.0].
    
    Returns:
        Tuple of (model, scaler, feature_names, train_df) where:
        - model: Fitted ElasticNetCV model
        - scaler: Fitted StandardScaler
        - feature_names: List of feature names
        - train_df: DataFrame with training data and predictions
    """
    print("Training ElasticNet model for rent forecasting...")
    print("=" * 60)
    
    # Prepare features
    X, y, feature_names = prepare_features(df)
    
    # Remove rows with missing target
    valid_mask = ~pd.isna(y)
    X = X[valid_mask]
    y = y[valid_mask]
    
    # Remove rows with too many missing features (more than 50% missing)
    missing_threshold = X.shape[1] * 0.5
    valid_rows = (X.isna().sum(axis=1) < missing_threshold)
    X = X[valid_rows]
    y = y[valid_rows]
    
    # Fill remaining missing values with median
    X = X.fillna(X.median())
    
    # Sort by date for time-series split
    if 'quarter_end_date' in df.columns:
        date_col = df.loc[valid_mask][valid_rows]['quarter_end_date'].values
        sort_idx = np.argsort(date_col)
        X = X.iloc[sort_idx].reset_index(drop=True)
        y = y[sort_idx]
        date_col = date_col[sort_idx]
    else:
        date_col = None
    
    # Guard: check if training set is empty
    if X.shape[0] == 0:
        raise ValueError(
            f"Empty training set after preprocessing. "
            f"X shape={X.shape}. Check rent_model_dataset filters/lags/window."
        )
    
    # Compact summary when training starts
    print("Training data summary:")
    print(f"  Rows: {X.shape[0]}")
    print(f"  Features: {len(feature_names)}")
    if 'cma' in df.columns:
        cma_vals = df.loc[valid_mask][valid_rows].iloc[sort_idx]['cma'].values if date_col is not None else df.loc[valid_mask][valid_rows]['cma'].values
        print(f"  Unique CMAs: {pd.Series(cma_vals).nunique()}")
    if 'unit_type' in df.columns:
        unit_vals = df.loc[valid_mask][valid_rows].iloc[sort_idx]['unit_type'].values if date_col is not None else df.loc[valid_mask][valid_rows]['unit_type'].values
        print(f"  Unique unit types: {pd.Series(unit_vals).nunique()}")
    if date_col is not None:
        print(f"  Date range: {pd.Series(date_col).min()} to {pd.Series(date_col).max()}")
    
    # Scale features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Set up time-series cross-validation
    if len(X) < min_train_quarters + n_splits:
        n_splits = max(1, len(X) - min_train_quarters)
        print(f"  Adjusting n_splits to {n_splits} due to data size")
    
    tscv = TimeSeriesSplit(n_splits=n_splits)
    
    # Set default hyperparameters if not provided
    # Use small grid for reasonable runtime
    if alphas is None:
        alphas = np.logspace(-3, 1, 20)  # Smaller grid for speed
    if l1_ratios is None:
        l1_ratios = [0.1, 0.5, 0.9]  # Small grid as per requirements
    
    # Train ElasticNetCV
    print(f"\nTraining ElasticNetCV with {n_splits} time-series splits...")
    model = ElasticNetCV(
        alphas=alphas,
        l1_ratio=l1_ratios,
        cv=tscv,
        max_iter=2000,
        random_state=42,
        n_jobs=-1
    )
    
    model.fit(X_scaled, y)
    
    print(f"Best alpha: {model.alpha_:.6f}")
    print(f"Best l1_ratio: {model.l1_ratio_:.4f}")
    print(f"Number of non-zero coefficients: {np.sum(model.coef_ != 0)}")
    
    # Get predictions on training data
    y_pred_train = model.predict(X_scaled)
    
    # Create training results DataFrame
    train_df = pd.DataFrame({
        'quarter_end_date': date_col if date_col is not None else np.arange(len(y)),
        'y_true': y,
        'y_pred': y_pred_train
    })
    
    # Add cma and unit_type if available
    if 'cma' in df.columns:
        train_df['cma'] = df.loc[valid_mask][valid_rows].iloc[sort_idx]['cma'].values if date_col is not None else df.loc[valid_mask][valid_rows]['cma'].values
    if 'unit_type' in df.columns:
        train_df['unit_type'] = df.loc[valid_mask][valid_rows].iloc[sort_idx]['unit_type'].values if date_col is not None else df.loc[valid_mask][valid_rows]['unit_type'].values
    
    return model, scaler, feature_names, train_df


def rolling_backtest(
    df: pd.DataFrame,
    model: Optional[ElasticNetCV] = None,
    scaler: Optional[StandardScaler] = None,
    min_train_quarters: int = 12,
    forecast_horizon: int = 1
) -> Tuple[pd.DataFrame, Dict]:
    """
    Perform rolling backtest for rent forecasting.
    
    For each step, trains up to time t and predicts t+1 (and optionally t+2).
    
    Args:
        df: Input DataFrame with features and target.
        model: Pre-trained model (optional). If None, trains new model at each step.
        scaler: Pre-fitted scaler (optional). If None, fits new scaler at each step.
        min_train_quarters: Minimum number of quarters for training.
        forecast_horizon: Number of quarters ahead to forecast (1 or 2).
    
    Returns:
        Tuple of (predictions_df, metrics_dict) where:
        - predictions_df: DataFrame with predictions and actuals
        - metrics_dict: Dictionary with overall and by-unit-type metrics
    """
    print("Performing rolling backtest...")
    print("=" * 60)
    
    # Prepare features
    X, y, feature_names = prepare_features(df)
    
    # Remove rows with missing target
    valid_mask = ~pd.isna(y)
    X = X[valid_mask]
    y = y[valid_mask]
    
    # Get dates and grouping columns
    if 'quarter_end_date' in df.columns:
        dates = df.loc[valid_mask]['quarter_end_date'].values
        sort_idx = np.argsort(dates)
        X = X.iloc[sort_idx].reset_index(drop=True)
        y = y[sort_idx]
        dates = dates[sort_idx]
    else:
        dates = np.arange(len(y))
        sort_idx = np.arange(len(y))
    
    # Remove rows with too many missing features
    missing_threshold = X.shape[1] * 0.5
    valid_rows = (X.isna().sum(axis=1) < missing_threshold)
    X = X[valid_rows]
    y = y[valid_rows]
    dates = dates[valid_rows]
    
    # Get cma and unit_type for grouping
    cma_values = None
    unit_type_values = None
    y_lag1_values = None
    y_lag4_values = None
    if 'cma' in df.columns:
        cma_values = df.loc[valid_mask][valid_rows].iloc[sort_idx[valid_rows] if len(sort_idx) > 0 else valid_rows]['cma'].values
    if 'unit_type' in df.columns:
        unit_type_values = df.loc[valid_mask][valid_rows].iloc[sort_idx[valid_rows] if len(sort_idx) > 0 else valid_rows]['unit_type'].values
    # Get lag features for baseline predictions
    if 'y_lag_1' in df.columns:
        y_lag1_values = df.loc[valid_mask][valid_rows].iloc[sort_idx[valid_rows] if len(sort_idx) > 0 else valid_rows]['y_lag_1'].values
    if 'y_lag_4' in df.columns:
        y_lag4_values = df.loc[valid_mask][valid_rows].iloc[sort_idx[valid_rows] if len(sort_idx) > 0 else valid_rows]['y_lag_4'].values
    
    # Fill missing values with median
    X = X.fillna(X.median())
    
    # Rolling backtest: rolling origin, 1-step ahead
    # For each t: train on <=t, predict t+1
    predictions = []
    
    # Get unique quarters sorted
    if 'quarter_end_date' in df.columns:
        unique_quarters = pd.Series(dates).unique()
        unique_quarters = sorted(unique_quarters)
    else:
        unique_quarters = np.arange(len(y))
    
    # Start from min_train_quarters
    # For each quarter t starting from min_train_quarters, train on <=t and predict t+1
    for t_idx in range(min_train_quarters, len(unique_quarters)):
        # Training: all data up to and including quarter t
        train_end_quarter = unique_quarters[t_idx]
        train_mask = pd.Series(dates) <= train_end_quarter
        
        # Testing: quarter t+1 (1-step ahead)
        if t_idx + 1 < len(unique_quarters):
            test_quarter = unique_quarters[t_idx + 1]
            test_mask = pd.Series(dates) == test_quarter
        else:
            # No more quarters to predict
            break
        
        X_train = X[train_mask].values
        y_train = y[train_mask]
        X_test = X[test_mask].values
        y_test = y[test_mask]
        train_dates = dates[train_mask]
        test_dates = dates[test_mask]
        test_cma = cma_values[test_mask] if cma_values is not None else None
        test_unit_type = unit_type_values[test_mask] if unit_type_values is not None else None
        test_y_lag1 = y_lag1_values[test_mask] if y_lag1_values is not None else None
        test_y_lag4 = y_lag4_values[test_mask] if y_lag4_values is not None else None
        
        if len(X_test) == 0:
            continue
        
        # Assert no data leakage: max training date must be < test date
        if len(train_dates) > 0 and len(test_dates) > 0:
            max_train_date = pd.Series(train_dates).max()
            test_date = pd.Series(test_dates).min()  # Use min in case of multiple test dates
            if max_train_date >= test_date:
                raise ValueError("data leakage detected")
        
        # Train model at each step (rolling origin)
        # Scale features
        scaler_step = StandardScaler()
        X_train_scaled = scaler_step.fit_transform(X_train)
        X_test_scaled = scaler_step.transform(X_test)
        
        # Train model with small grid for reasonable runtime
        model_step = ElasticNetCV(
            alphas=np.logspace(-3, 1, 20),  # Smaller grid
            l1_ratio=[0.1, 0.5, 0.9],  # Small grid as per requirements
            cv=TimeSeriesSplit(n_splits=min(5, max(2, len(X_train) - 1))),
            max_iter=2000,
            random_state=42,
            n_jobs=-1
        )
        model_step.fit(X_train_scaled, y_train)
        
        # Predict
        y_pred = model_step.predict(X_test_scaled)
        
        # Store predictions with baseline predictions
        for j in range(len(y_test)):
            pred_dict = {
                'quarter_end_date': test_dates[j],
                'y_true': y_test[j],
                'y_pred': y_pred[j]
            }
            # Add baseline predictions
            if test_y_lag1 is not None:
                pred_dict['y_pred_lag1'] = test_y_lag1[j]
            if test_y_lag4 is not None:
                pred_dict['y_pred_lag4'] = test_y_lag4[j]
            if test_cma is not None:
                pred_dict['cma'] = test_cma[j]
            if test_unit_type is not None:
                pred_dict['unit_type'] = test_unit_type[j]
            predictions.append(pred_dict)
        
        if (t_idx - min_train_quarters + 1) % 5 == 0:
            print(f"  Completed {t_idx - min_train_quarters + 1} backtest steps...")
    
    predictions_df = pd.DataFrame(predictions)
    
    # Print backtest coverage statistics
    print("\nBacktest coverage statistics:")
    print(f"  Total predictions: {len(predictions_df)}")
    if 'quarter_end_date' in predictions_df.columns:
        print(f"  Unique quarters: {predictions_df['quarter_end_date'].nunique()}")
        print(f"  Date range: {predictions_df['quarter_end_date'].min()} to {predictions_df['quarter_end_date'].max()}")
        print(f"  Sample counts per quarter (head):")
        quarter_counts = predictions_df['quarter_end_date'].value_counts().sort_index().head(10)
        for quarter, count in quarter_counts.items():
            print(f"    {quarter}: {count} predictions")
    
    # Save backtest coverage JSON
    coverage_stats = {
        'total_predictions': int(len(predictions_df)),
        'unique_quarters': int(predictions_df['quarter_end_date'].nunique()) if 'quarter_end_date' in predictions_df.columns else 0,
        'date_min': str(predictions_df['quarter_end_date'].min()) if 'quarter_end_date' in predictions_df.columns else None,
        'date_max': str(predictions_df['quarter_end_date'].max()) if 'quarter_end_date' in predictions_df.columns else None,
        'quarter_counts': {}
    }
    
    if 'quarter_end_date' in predictions_df.columns:
        quarter_counts_dict = predictions_df['quarter_end_date'].value_counts().sort_index().to_dict()
        coverage_stats['quarter_counts'] = {str(k): int(v) for k, v in quarter_counts_dict.items()}
    
    coverage_path = OUTPUT_DIR / "rent_backtest_coverage.json"
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    with open(coverage_path, 'w') as f:
        json.dump(coverage_stats, f, indent=2)
    print(f"  Saved coverage stats to: {coverage_path}")
    
    # Calculate metrics
    print("\nCalculating metrics...")
    metrics = {}
    
    # Overall metrics
    if len(predictions_df) > 0:
        # ElasticNet metrics
        elasticnet_mae = float(mae(predictions_df['y_true'].values, predictions_df['y_pred'].values))
        elasticnet_smape = float(smape(predictions_df['y_true'].values, predictions_df['y_pred'].values))
        metrics['elasticnet'] = {
            'mae': elasticnet_mae,
            'smape': elasticnet_smape,
            'n_predictions': int(len(predictions_df))
        }
        
        # Baseline metrics
        metrics['baselines'] = {}
        
        # Lag1 baseline
        if 'y_pred_lag1' in predictions_df.columns:
            # Drop rows where baseline is missing
            lag1_df = predictions_df[['y_true', 'y_pred_lag1']].dropna()
            if len(lag1_df) > 0:
                lag1_mae = float(mae(lag1_df['y_true'].values, lag1_df['y_pred_lag1'].values))
                lag1_smape = float(smape(lag1_df['y_true'].values, lag1_df['y_pred_lag1'].values))
                metrics['baselines']['lag1'] = {
                    'mae': lag1_mae,
                    'smape': lag1_smape,
                    'n_predictions': int(len(lag1_df))
                }
        
        # Lag4 baseline
        if 'y_pred_lag4' in predictions_df.columns:
            # Drop rows where baseline is missing (only for this baseline's metric)
            lag4_df = predictions_df[['y_true', 'y_pred_lag4', 'y_pred']].dropna(subset=['y_true', 'y_pred_lag4'])
            if len(lag4_df) > 0:
                lag4_mae = float(mae(lag4_df['y_true'].values, lag4_df['y_pred_lag4'].values))
                lag4_smape = float(smape(lag4_df['y_true'].values, lag4_df['y_pred_lag4'].values))
                metrics['baselines']['lag4'] = {
                    'mae': lag4_mae,
                    'smape': lag4_smape,
                    'n_predictions': int(len(lag4_df))
                }
                
                # Uplift vs lag4 baseline (compute elasticnet on same subset for fair comparison)
                if 'y_pred' in lag4_df.columns and lag4_df['y_pred'].notna().sum() > 0:
                    elasticnet_mae_subset = float(mae(lag4_df['y_true'].values, lag4_df['y_pred'].values))
                    elasticnet_smape_subset = float(smape(lag4_df['y_true'].values, lag4_df['y_pred'].values))
                    metrics['uplift_vs_lag4'] = {
                        'mae_pct': float((lag4_mae - elasticnet_mae_subset) / lag4_mae * 100),
                        'smape_pct': float((lag4_smape - elasticnet_smape_subset) / lag4_smape * 100)
                    }
        
        # Metrics by unit_type (model + baselines + uplift)
        if 'unit_type' in predictions_df.columns:
            metrics['by_unit_type'] = {}
            for unit_type in predictions_df['unit_type'].unique():
                unit_mask = predictions_df['unit_type'] == unit_type
                unit_df = predictions_df[unit_mask]
                if len(unit_df) > 0:
                    unit_metrics = {
                        'n_predictions': int(len(unit_df))
                    }
                    
                    # Model metrics
                    unit_metrics['elasticnet'] = {
                        'mae': float(mae(unit_df['y_true'].values, unit_df['y_pred'].values)),
                        'smape': float(smape(unit_df['y_true'].values, unit_df['y_pred'].values))
                    }
                    
                    # Baseline metrics
                    unit_metrics['baselines'] = {}
                    
                    # Lag1 baseline
                    if 'y_pred_lag1' in unit_df.columns:
                        lag1_unit_df = unit_df[['y_true', 'y_pred_lag1']].dropna()
                        if len(lag1_unit_df) > 0:
                            unit_metrics['baselines']['lag1'] = {
                                'mae': float(mae(lag1_unit_df['y_true'].values, lag1_unit_df['y_pred_lag1'].values)),
                                'smape': float(smape(lag1_unit_df['y_true'].values, lag1_unit_df['y_pred_lag1'].values)),
                                'n_predictions': int(len(lag1_unit_df))
                            }
                    
                    # Lag4 baseline
                    if 'y_pred_lag4' in unit_df.columns:
                        lag4_unit_df = unit_df[['y_true', 'y_pred_lag4', 'y_pred']].dropna(subset=['y_true', 'y_pred_lag4'])
                        if len(lag4_unit_df) > 0:
                            lag4_mae = float(mae(lag4_unit_df['y_true'].values, lag4_unit_df['y_pred_lag4'].values))
                            lag4_smape = float(smape(lag4_unit_df['y_true'].values, lag4_unit_df['y_pred_lag4'].values))
                            unit_metrics['baselines']['lag4'] = {
                                'mae': lag4_mae,
                                'smape': lag4_smape,
                                'n_predictions': int(len(lag4_unit_df))
                            }
                            
                            # Uplift vs lag4
                            if 'y_pred' in lag4_unit_df.columns and lag4_unit_df['y_pred'].notna().sum() > 0:
                                elasticnet_mae_subset = float(mae(lag4_unit_df['y_true'].values, lag4_unit_df['y_pred'].values))
                                elasticnet_smape_subset = float(smape(lag4_unit_df['y_true'].values, lag4_unit_df['y_pred'].values))
                                unit_metrics['uplift_vs_lag4'] = {
                                    'mae_pct': float((lag4_mae - elasticnet_mae_subset) / lag4_mae * 100),
                                    'smape_pct': float((lag4_smape - elasticnet_smape_subset) / lag4_smape * 100)
                                }
                    
                    metrics['by_unit_type'][str(unit_type)] = unit_metrics
        
        # Metrics by top CMAs (top 10 by sample count) - model + baselines + uplift
        if 'cma' in predictions_df.columns:
            cma_counts = predictions_df['cma'].value_counts()
            top_cmas = cma_counts.head(10).index.tolist()
            metrics['by_top_cmas'] = {}
            for cma_name in top_cmas:
                cma_mask = predictions_df['cma'] == cma_name
                cma_df = predictions_df[cma_mask]
                if len(cma_df) > 0:
                    cma_metrics = {
                        'n_predictions': int(len(cma_df))
                    }
                    
                    # Model metrics
                    cma_metrics['elasticnet'] = {
                        'mae': float(mae(cma_df['y_true'].values, cma_df['y_pred'].values)),
                        'smape': float(smape(cma_df['y_true'].values, cma_df['y_pred'].values))
                    }
                    
                    # Baseline metrics
                    cma_metrics['baselines'] = {}
                    
                    # Lag1 baseline
                    if 'y_pred_lag1' in cma_df.columns:
                        lag1_cma_df = cma_df[['y_true', 'y_pred_lag1']].dropna()
                        if len(lag1_cma_df) > 0:
                            cma_metrics['baselines']['lag1'] = {
                                'mae': float(mae(lag1_cma_df['y_true'].values, lag1_cma_df['y_pred_lag1'].values)),
                                'smape': float(smape(lag1_cma_df['y_true'].values, lag1_cma_df['y_pred_lag1'].values)),
                                'n_predictions': int(len(lag1_cma_df))
                            }
                    
                    # Lag4 baseline
                    if 'y_pred_lag4' in cma_df.columns:
                        lag4_cma_df = cma_df[['y_true', 'y_pred_lag4', 'y_pred']].dropna(subset=['y_true', 'y_pred_lag4'])
                        if len(lag4_cma_df) > 0:
                            lag4_mae = float(mae(lag4_cma_df['y_true'].values, lag4_cma_df['y_pred_lag4'].values))
                            lag4_smape = float(smape(lag4_cma_df['y_true'].values, lag4_cma_df['y_pred_lag4'].values))
                            cma_metrics['baselines']['lag4'] = {
                                'mae': lag4_mae,
                                'smape': lag4_smape,
                                'n_predictions': int(len(lag4_cma_df))
                            }
                            
                            # Uplift vs lag4
                            if 'y_pred' in lag4_cma_df.columns and lag4_cma_df['y_pred'].notna().sum() > 0:
                                elasticnet_mae_subset = float(mae(lag4_cma_df['y_true'].values, lag4_cma_df['y_pred'].values))
                                elasticnet_smape_subset = float(smape(lag4_cma_df['y_true'].values, lag4_cma_df['y_pred'].values))
                                cma_metrics['uplift_vs_lag4'] = {
                                    'mae_pct': float((lag4_mae - elasticnet_mae_subset) / lag4_mae * 100),
                                    'smape_pct': float((lag4_smape - elasticnet_smape_subset) / lag4_smape * 100)
                                }
                    
                    metrics['by_top_cmas'][str(cma_name)] = cma_metrics
        
        # Distribution across CMAs: compute per-CMA sMAPE for elasticnet
        if 'cma' in predictions_df.columns:
            cma_smape_list = []
            for cma_name in predictions_df['cma'].unique():
                cma_mask = predictions_df['cma'] == cma_name
                cma_df = predictions_df[cma_mask]
                if len(cma_df) > 0 and 'y_pred' in cma_df.columns:
                    try:
                        cma_smape_val = float(smape(cma_df['y_true'].values, cma_df['y_pred'].values))
                        cma_smape_list.append(cma_smape_val)
                    except Exception:
                        # Skip CMAs where sMAPE cannot be computed
                        continue
            
            if len(cma_smape_list) > 0:
                cma_smape_array = np.array(cma_smape_list)
                metrics['cma_smape_distribution'] = {
                    'median_smape': float(np.median(cma_smape_array)),
                    'p25_smape': float(np.percentile(cma_smape_array, 25)),
                    'p75_smape': float(np.percentile(cma_smape_array, 75)),
                    'n_cmas': int(len(cma_smape_list))
                }
        
    
    print(f"  ElasticNet MAE: {metrics.get('elasticnet', {}).get('mae', 'N/A'):.2f}")
    print(f"  ElasticNet sMAPE: {metrics.get('elasticnet', {}).get('smape', 'N/A'):.2f}%")
    
    # Print baseline metrics
    if 'baselines' in metrics:
        if 'lag1' in metrics['baselines']:
            lag1_mae = metrics['baselines']['lag1']['mae']
            lag1_smape = metrics['baselines']['lag1']['smape']
            print(f"  Lag1 baseline MAE: {lag1_mae:.2f}, sMAPE: {lag1_smape:.2f}%")
        if 'lag4' in metrics['baselines']:
            lag4_mae = metrics['baselines']['lag4']['mae']
            lag4_smape = metrics['baselines']['lag4']['smape']
            print(f"  Lag4 baseline MAE: {lag4_mae:.2f}, sMAPE: {lag4_smape:.2f}%")
    
    # Print uplift vs lag4
    if 'uplift_vs_lag4' in metrics:
        uplift = metrics['uplift_vs_lag4']
        print(f"  Uplift vs Lag4: MAE {uplift['mae_pct']:.1f}%, sMAPE {uplift['smape_pct']:.1f}%")
    
    return predictions_df, metrics


def train_and_backtest_rent(
    df: pd.DataFrame,
    min_train_quarters: int = 12,
    forecast_horizon: int = 1,
    save_outputs: bool = True
) -> Tuple[ElasticNetCV, StandardScaler, List[str], pd.DataFrame, pd.DataFrame, Dict]:
    """
    Train ElasticNet model and perform rolling backtest.
    
    Args:
        df: Input DataFrame with features and target.
        min_train_quarters: Minimum number of quarters for training.
        forecast_horizon: Number of quarters ahead to forecast.
        save_outputs: Whether to save outputs to files.
    
    Returns:
        Tuple of (model, scaler, feature_names, train_df, predictions_df, metrics_dict)
    """
    # Train model
    model, scaler, feature_names, train_df = train_elasticnet_rent(
        df,
        min_train_quarters=min_train_quarters
    )
    
    # Perform backtest
    predictions_df, metrics = rolling_backtest(
        df,
        model=model,
        scaler=scaler,
        min_train_quarters=min_train_quarters,
        forecast_horizon=forecast_horizon
    )
    
    # Save outputs
    if save_outputs:
        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        
        # Save predictions with required columns: quarter_end_date, cma, unit_type, y_true, y_pred, y_pred_lag1, y_pred_lag4
        predictions_path = OUTPUT_DIR / "rent_predictions.csv"
        # Include all available columns (required + optional lag predictions)
        predictions_df_save = predictions_df.copy()
        predictions_df_save.to_csv(predictions_path, index=False)
        print(f"\nSaved predictions to: {predictions_path}")
        
        # Save metrics
        metrics_path = OUTPUT_DIR / "rent_metrics.json"
        with open(metrics_path, 'w') as f:
            json.dump(metrics, f, indent=2)
        print(f"Saved metrics to: {metrics_path}")
        
        # Save model coefficients (from final trained model)
        coefficients_df = pd.DataFrame({
            'feature': feature_names,
            'coef': model.coef_
        })
        
        # Add family column
        coefficients_df['family'] = coefficients_df['feature'].apply(_categorize_feature)
        
        # Sort by absolute coefficient value
        coefficients_df = coefficients_df.sort_values('coef', key=lambda x: np.abs(x), ascending=False)
        
        coefficients_path = OUTPUT_DIR / "rent_model_coefficients.csv"
        coefficients_df.to_csv(coefficients_path, index=False)
        print(f"Saved coefficients to: {coefficients_path}")
        
        # Save top coefficients (top 20 positive and top 20 negative by abs(coef))
        # Sort positive coefficients by absolute value descending
        pos_df = coefficients_df[coefficients_df['coef'] > 0].copy()
        pos_df = pos_df.sort_values('coef', key=lambda x: np.abs(x), ascending=False)
        top_pos = pos_df.head(20)
        
        # Sort negative coefficients by absolute value descending
        neg_df = coefficients_df[coefficients_df['coef'] < 0].copy()
        neg_df = neg_df.sort_values('coef', key=lambda x: np.abs(x), ascending=False)
        top_neg = neg_df.head(20)
        
        # Combine and sort by absolute value
        top_coefficients_df = pd.concat([top_pos, top_neg], ignore_index=True)
        top_coefficients_df = top_coefficients_df.sort_values('coef', key=lambda x: np.abs(x), ascending=False)
        
        top_coefficients_path = OUTPUT_DIR / "rent_top_coefficients.csv"
        top_coefficients_df.to_csv(top_coefficients_path, index=False)
        print(f"Saved top coefficients to: {top_coefficients_path}")
    
    return model, predictions_df, metrics


if __name__ == "__main__":
    """Run training and backtest as standalone script."""
    # Load data
    data_path = PROCESSED_DIR / "rent_model_dataset.parquet"
    
    if not data_path.exists():
        print(f"Error: Data file not found: {data_path}")
        print("Please run build_features.py first to create the dataset.")
    else:
        print(f"Loading data from: {data_path}")
        df = pd.read_parquet(data_path)
        print(f"Data shape: {df.shape}")
        
        # Train and backtest
        model, predictions_df, metrics = train_and_backtest_rent(
            df,
            min_train_quarters=12,
            forecast_horizon=1
        )
        
        print("\n" + "=" * 60)
        print("TRAINING SUMMARY")
        print("=" * 60)
        print(f"Model: ElasticNetCV")
        print(f"Best alpha: {model.alpha_:.6f}")
        print(f"Best l1_ratio: {model.l1_ratio_:.4f}")
        print(f"Non-zero coefficients: {np.sum(model.coef_ != 0)}/{len(model.coef_)}")
        
        print("\n" + "=" * 60)
        print("BACKTEST RESULTS")
        print("=" * 60)
        print(f"Total predictions: {len(predictions_df)}")
        print(f"ElasticNet MAE: {metrics.get('elasticnet', {}).get('mae', 'N/A'):.2f}")
        print(f"ElasticNet sMAPE: {metrics.get('elasticnet', {}).get('smape', 'N/A'):.2f}%")
        
        if 'by_unit_type' in metrics:
            print("\nBy Unit Type:")
            for unit_type, unit_metrics in metrics['by_unit_type'].items():
                print(f"  {unit_type}:")
                print(f"    MAE: {unit_metrics['mae']:.2f}")
                print(f"    sMAPE: {unit_metrics['smape']:.2f}%")

