"""
SARIMAX model for mortgage forecasting.

Implements SARIMAX with exogenous regressors and rolling backtest
for forecasting chartered bank mortgage loans.
"""

from pathlib import Path
from typing import Optional, Tuple, Dict, List
import pandas as pd
import numpy as np
import json
import warnings
from itertools import product

from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tools.sm_exceptions import ConvergenceWarning

from src.config import OUTPUT_DIR, PROCESSED_DIR
from src.models.rent_elasticnet import mae, smape

warnings.filterwarnings('ignore', category=ConvergenceWarning)
warnings.filterwarnings('ignore', category=UserWarning)


def _to_1d(x):
    """
    Convert array-like to 1D array.
    
    Handles Series, DataFrame, ndarray, and list inputs.
    If 2D with one column, squeezes to 1D.
    
    Args:
        x: Array-like input (can be None, Series, DataFrame, ndarray, list)
    
    Returns:
        1D numpy array or None
    
    Raises:
        ValueError: If array cannot be converted to 1D
    """
    if x is None:
        return None
    if hasattr(x, "to_numpy"):
        arr = x.to_numpy()
    else:
        arr = np.asarray(x)
    # If 2D with one column, squeeze
    if arr.ndim == 2 and arr.shape[1] == 1:
        arr = arr[:, 0]
    if arr.ndim != 1:
        raise ValueError(f"Expected 1D array, got shape {arr.shape}")
    return arr


def _as_float_scalar(x):
    """
    Convert value to float scalar, handling arrays/Series/objects.
    
    Args:
        x: Input value (can be scalar, array, Series, object, None)
    
    Returns:
        float scalar (or np.nan if conversion fails)
    """
    if x is None:
        return np.nan
    arr = np.asarray(x)
    # Squeeze arrays/Series like (1,), (1,1)
    arr = np.squeeze(arr)
    if arr.shape == ():  # scalar
        try:
            return float(arr)
        except Exception:
            return np.nan
    # If still not scalar, try first element
    try:
        return float(arr.flat[0])
    except Exception:
        return np.nan


def fit_sarimax(
    df: pd.DataFrame,
    target_col: str = "y_level",
    exog_cols: Optional[List[str]] = None,
    order: Optional[Tuple[int, int, int]] = None,
    seasonal_order: Optional[Tuple[int, int, int, int]] = None,
    enforce_stationarity: bool = True,
    enforce_invertibility: bool = True
) -> Tuple[SARIMAX, pd.DataFrame]:
    """
    Fit SARIMAX model for mortgage forecasting.
    
    Args:
        df: Input DataFrame with target and exogenous variables.
        target_col: Name of target column. Defaults to "y_level".
        exog_cols: List of exogenous variable column names. If None, auto-detects
                   columns with prefixes: rate_, labour_, starts_, pop_, mig_, npr_.
        order: ARIMA order (p, d, q). If None, will use grid search.
        seasonal_order: Seasonal order (P, D, Q, s). If None, uses (1, 1, 1, 4).
        enforce_stationarity: Whether to enforce stationarity.
        enforce_invertibility: Whether to enforce invertibility.
    
    Returns:
        Tuple of (fitted_model, results_df) where:
        - fitted_model: Fitted SARIMAX model
        - results_df: DataFrame with fitted values and residuals
    """
    print("Fitting SARIMAX model for mortgage forecasting...")
    print("=" * 60)
    
    # Prepare target series
    if target_col not in df.columns:
        raise ValueError(f"Target column '{target_col}' not found in DataFrame. Available columns: {list(df.columns)}")
    
    # Sort by date
    if 'quarter_end_date' in df.columns:
        df = df.sort_values('quarter_end_date').reset_index(drop=True)
        dates = df['quarter_end_date'].values
    else:
        dates = np.arange(len(df))
    
    y = df[target_col].values
    
    # Remove NaN values
    valid_mask = ~pd.isna(y)
    y = y[valid_mask]
    dates = dates[valid_mask]
    df_valid = df.loc[valid_mask].reset_index(drop=True)
    
    print(f"Target series length: {len(y)}")
    print(f"Date range: {dates[0]} to {dates[-1]}")
    
    # Prepare exogenous variables
    if exog_cols is None:
        # Auto-detect exogenous columns
        exog_prefixes = ['rate_', 'labour_', 'starts_', 'pop_', 'mig_', 'npr_']
        exog_cols = [
            col for col in df_valid.columns
            if any(col.startswith(prefix) for prefix in exog_prefixes)
        ]
    
    exog = None
    if exog_cols:
        exog = df_valid[exog_cols].copy()
        
        # Fill missing values with forward fill then backward fill
        exog = exog.ffill().bfill()
        
        # If still missing, fill with median
        exog = exog.fillna(exog.median())
        
        print(f"Exogenous variables: {len(exog_cols)}")
        print(f"  {exog_cols[:5]}{'...' if len(exog_cols) > 5 else ''}")
    else:
        print("No exogenous variables found")
    
    # Set default orders if not provided
    if order is None:
        # Will use grid search
        order = (1, 1, 1)  # Default for grid search start
    
    if seasonal_order is None:
        seasonal_order = (1, 1, 1, 4)  # Quarterly seasonality
    
    # Fit model
    print(f"\nFitting SARIMAX({order[0]},{order[1]},{order[2]})x{seasonal_order}...")
    
    try:
        model = SARIMAX(
            y,
            exog=exog.values if exog is not None else None,
            order=order,
            seasonal_order=seasonal_order,
            enforce_stationarity=enforce_stationarity,
            enforce_invertibility=enforce_invertibility
        )
        
        fitted_model = model.fit(disp=False, maxiter=200)
        
        print(f"Model fitted successfully")
        print(f"AIC: {fitted_model.aic:.2f}")
        print(f"BIC: {fitted_model.bic:.2f}")
        
        # Get fitted values
        fitted_values = fitted_model.fittedvalues
        residuals = fitted_model.resid
        
        # Create results DataFrame
        results_df = pd.DataFrame({
            'quarter_end_date': dates,
            'y_true': y,
            'y_fitted': fitted_values,
            'residual': residuals
        })
        
        return fitted_model, results_df
        
    except Exception as e:
        print(f"Error fitting model: {e}")
        raise


def grid_search_sarimax(
    df: pd.DataFrame,
    target_col: str = "y_level",
    exog_cols: Optional[List[str]] = None,
    p_values: List[int] = [0, 1, 2],
    d_values: List[int] = [0, 1, 2],
    q_values: List[int] = [0, 1, 2],
    P_values: List[int] = [0, 1],
    D_value: int = 1,
    Q_values: List[int] = [0, 1],
    seasonal_period: int = 4,
    max_models: int = 50
) -> Tuple[SARIMAX, Tuple[int, int, int], Tuple[int, int, int, int], pd.DataFrame]:
    """
    Perform grid search over SARIMAX orders to find best AIC.
    
    Args:
        df: Input DataFrame with target and exogenous variables.
        target_col: Name of target column.
        exog_cols: List of exogenous variable column names.
        p_values: List of p (AR) values to try.
        d_values: List of d (differencing) values to try.
        q_values: List of q (MA) values to try.
        P_values: List of P (seasonal AR) values to try.
        D_value: D (seasonal differencing) value (fixed).
        Q_values: List of Q (seasonal MA) values to try.
        seasonal_period: Seasonal period (default 4 for quarterly).
        max_models: Maximum number of models to try (for speed).
    
    Returns:
        Tuple of (best_model, best_order, best_seasonal_order, results_df)
    """
    print("Performing grid search for best SARIMAX order...")
    print("=" * 60)
    
    # Prepare target series
    if target_col not in df.columns:
        raise ValueError(f"Target column '{target_col}' not found in DataFrame.")
    
    # Sort by date
    if 'quarter_end_date' in df.columns:
        df = df.sort_values('quarter_end_date').reset_index(drop=True)
    
    y = df[target_col].values
    valid_mask = ~pd.isna(y)
    y = y[valid_mask]
    df_valid = df.loc[valid_mask].reset_index(drop=True)
    
    # Prepare exogenous variables
    if exog_cols is None:
        exog_prefixes = ['rate_', 'labour_', 'starts_', 'pop_', 'mig_', 'npr_']
        exog_cols = [
            col for col in df_valid.columns
            if any(col.startswith(prefix) for prefix in exog_prefixes)
        ]
    
    exog = None
    if exog_cols:
        exog = df_valid[exog_cols].copy()
        exog = exog.ffill().bfill()
        exog = exog.fillna(exog.median())
    
    # Generate all combinations
    orders = list(product(p_values, d_values, q_values))
    seasonal_orders = list(product(P_values, [D_value], Q_values, [seasonal_period]))
    
    # Limit number of combinations for speed
    total_combinations = len(orders) * len(seasonal_orders)
    if total_combinations > max_models:
        # Sample combinations
        import random
        random.seed(42)
        order_sample = random.sample(orders, min(len(orders), max_models // len(seasonal_orders)))
        seasonal_sample = random.sample(seasonal_orders, min(len(seasonal_orders), max_models // len(order_sample)))
        orders = order_sample
        seasonal_orders = seasonal_sample
        total_combinations = len(orders) * len(seasonal_orders)
    
    print(f"Trying {total_combinations} model combinations...")
    
    best_aic = np.inf
    best_model = None
    best_order = None
    best_seasonal_order = None
    results = []
    
    for order in orders:
        for seasonal_order in seasonal_orders:
            try:
                model = SARIMAX(
                    y,
                    exog=exog.values if exog is not None else None,
                    order=order,
                    seasonal_order=seasonal_order,
                    enforce_stationarity=True,
                    enforce_invertibility=True
                )
                
                fitted_model = model.fit(disp=False, maxiter=200)
                aic = fitted_model.aic
                
                results.append({
                    'order': order,
                    'seasonal_order': seasonal_order,
                    'aic': aic,
                    'bic': fitted_model.bic
                })
                
                if aic < best_aic:
                    best_aic = aic
                    best_model = fitted_model
                    best_order = order
                    best_seasonal_order = seasonal_order
                    print(f"  New best: {order}x{seasonal_order}, AIC={aic:.2f}")
                
            except Exception as e:
                # Skip models that don't converge
                continue
    
    if best_model is None:
        raise ValueError("No models converged in grid search. Try different parameter ranges.")
    
    print(f"\nBest model: SARIMAX{best_order}x{best_seasonal_order}")
    print(f"Best AIC: {best_aic:.2f}")
    
    # Create results DataFrame
    results_df = pd.DataFrame(results).sort_values('aic')
    
    return best_model, best_order, best_seasonal_order, results_df


def rolling_backtest_sarimax(
    df: pd.DataFrame,
    target_col: str = "y_level",
    exog_cols: Optional[List[str]] = None,
    order: Optional[Tuple[int, int, int]] = None,
    seasonal_order: Optional[Tuple[int, int, int, int]] = None,
    min_train_quarters: int = 12,
    use_grid_search: bool = False
) -> Tuple[pd.DataFrame, Dict]:
    """
    Perform rolling backtest for SARIMAX model (1-step ahead).
    
    For each step, trains on data up to time t and predicts t+1.
    
    Args:
        df: Input DataFrame with target and exogenous variables.
        target_col: Name of target column.
        exog_cols: List of exogenous variable column names.
        order: ARIMA order (p, d, q). If None and use_grid_search=False, uses (1,1,1).
        seasonal_order: Seasonal order (P, D, Q, s). If None, uses (1,1,1,4).
        min_train_quarters: Minimum number of quarters for training.
        use_grid_search: Whether to use grid search at each step (slower but better).
    
    Returns:
        Tuple of (predictions_df, metrics_dict)
    """
    print("Performing rolling backtest for SARIMAX...")
    print("=" * 60)
    
    # Prepare data
    if target_col not in df.columns:
        raise ValueError(f"Target column '{target_col}' not found.")
    
    # Sort by date
    if 'quarter_end_date' in df.columns:
        df = df.sort_values('quarter_end_date').reset_index(drop=True)
        dates = df['quarter_end_date'].values
    else:
        dates = np.arange(len(df))
    
    y = df[target_col].values
    
    # Prepare exogenous variables
    if exog_cols is None:
        exog_prefixes = ['rate_', 'labour_', 'starts_', 'pop_', 'mig_', 'npr_']
        exog_cols = [
            col for col in df.columns
            if any(col.startswith(prefix) for prefix in exog_prefixes)
        ]
    
    predictions = []
    
    # Rolling backtest
    for i in range(min_train_quarters, len(y)):
        train_end = i
        test_idx = i
        
        # Training data
        y_train = y[:train_end]
        dates_train = dates[:train_end]
        
        # Test data
        y_test = y[test_idx]
        date_test = dates[test_idx]
        
        # Exogenous variables
        exog_train = None
        exog_test = None
        if exog_cols:
            exog_train_df = df.loc[:train_end-1, exog_cols].copy()
            exog_train_df = exog_train_df.ffill().bfill()
            exog_train_df = exog_train_df.fillna(exog_train_df.median())
            exog_train = exog_train_df.values
            
            exog_test_df = df.loc[[test_idx], exog_cols].copy()
            exog_test_df = exog_test_df.ffill().bfill()
            exog_test_df = exog_test_df.fillna(exog_train_df.median())
            exog_test = exog_test_df.values
        
        # Skip if too many NaN in training
        if np.sum(np.isnan(y_train)) > len(y_train) * 0.3:
            continue
        
        # Fit model
        try:
            if use_grid_search:
                # Use grid search (slower)
                model, order_used, seasonal_order_used, _ = grid_search_sarimax(
                    df.iloc[:train_end],
                    target_col=target_col,
                    exog_cols=exog_cols,
                    max_models=20  # Limit for speed
                )
            else:
                # Use provided or default orders
                if order is None:
                    order = (1, 1, 1)
                if seasonal_order is None:
                    seasonal_order = (1, 1, 1, 4)
                
                model_fit = SARIMAX(
                    y_train,
                    exog=exog_train,
                    order=order,
                    seasonal_order=seasonal_order,
                    enforce_stationarity=True,
                    enforce_invertibility=True
                )
                model = model_fit.fit(disp=False, maxiter=200)
            
            # Forecast 1 step ahead
            forecast = model.forecast(steps=1, exog=exog_test if exog_test is not None else None)
            y_pred = _to_1d(forecast)  # Ensure 1D
            
            # Enforce numeric scalar values
            predictions.append({
                'quarter_end_date': date_test,
                'y_true': _as_float_scalar(y_test),
                'y_pred': _as_float_scalar(y_pred),
                'horizon': 1
            })
            
        except Exception as e:
            # Skip if model doesn't converge
            continue
        
        if (i - min_train_quarters + 1) % 5 == 0:
            print(f"  Completed {i - min_train_quarters + 1} backtest steps...")
    
    predictions_df = pd.DataFrame(predictions)
    
    # Ensure numeric types and drop NaN rows
    if len(predictions_df) > 0:
        # Convert to numeric
        predictions_df["y_true"] = pd.to_numeric(predictions_df["y_true"], errors="coerce")
        predictions_df["y_pred"] = pd.to_numeric(predictions_df["y_pred"], errors="coerce")
        
        # Drop rows where y_true or y_pred is NaN
        predictions_df = predictions_df.dropna(subset=["y_true", "y_pred"])
    
    # Check if predictions_df is empty after cleaning
    if len(predictions_df) == 0:
        raise ValueError(
            "Mortgage backtest produced no valid predictions; "
            "check model fit/prediction extraction."
        )
    
    # Calculate metrics
    print("\nCalculating metrics...")
    metrics = {}
    
    if len(predictions_df) > 0:
        metrics['overall'] = {
            'mae': float(mae(predictions_df['y_true'].values, predictions_df['y_pred'].values)),
            'smape': float(smape(predictions_df['y_true'].values, predictions_df['y_pred'].values)),
            'n_predictions': int(len(predictions_df))
        }
        
        print(f"  Overall MAE: {metrics['overall']['mae']:.2f}")
        print(f"  Overall sMAPE: {metrics['overall']['smape']:.2f}%")
    
    return predictions_df, metrics


def train_and_backtest_mortgage(
    df: pd.DataFrame,
    target_col: str = "y_level",
    use_grid_search: bool = True,
    min_train_quarters: int = 12,
    save_outputs: bool = True
) -> Tuple[SARIMAX, pd.DataFrame, pd.DataFrame, Dict]:
    """
    Train SARIMAX model and perform rolling backtest.
    
    Args:
        df: Input DataFrame with features and target.
        target_col: Name of target column.
        use_grid_search: Whether to use grid search for order selection.
        min_train_quarters: Minimum number of quarters for training.
        save_outputs: Whether to save outputs to files.
    
    Returns:
        Tuple of (model, train_results_df, predictions_df, metrics_dict)
    """
    # Fit model (with or without grid search)
    if use_grid_search:
        model, order, seasonal_order, grid_results = grid_search_sarimax(
            df,
            target_col=target_col,
            max_models=50
        )
        print(f"\nBest order: {order}x{seasonal_order}")
    else:
        model, train_results_df = fit_sarimax(
            df,
            target_col=target_col
        )
        # Extract order from model (approximate - actual order may differ)
        order = (1, 1, 1)  # Default fallback
        seasonal_order = (1, 1, 1, 4)  # Default fallback
    
    # Perform backtest
    predictions_df, metrics = rolling_backtest_sarimax(
        df,
        target_col=target_col,
        order=order if not use_grid_search else None,
        seasonal_order=seasonal_order if not use_grid_search else None,
        min_train_quarters=min_train_quarters,
        use_grid_search=False  # Don't grid search at each backtest step
    )
    
    # Get training results
    if use_grid_search:
        # Refit on full data for training results
        y = df[target_col].values
        valid_mask = ~pd.isna(y)
        y = y[valid_mask]
        
        exog_prefixes = ['rate_', 'labour_', 'starts_', 'pop_', 'mig_', 'npr_']
        exog_cols = [
            col for col in df.columns
            if any(col.startswith(prefix) for prefix in exog_prefixes)
        ]
        
        exog = None
        if exog_cols:
            exog_df = df.loc[valid_mask, exog_cols].copy()
            exog_df = exog_df.ffill().bfill()
            exog_df = exog_df.fillna(exog_df.median())
            exog = exog_df.values
        
        model_fit = SARIMAX(y, exog=exog, order=order, seasonal_order=seasonal_order)
        model = model_fit.fit(disp=False, maxiter=200)
    
    # Normalize arrays to 1D before constructing DataFrame
    date_col = 'quarter_end_date' if 'quarter_end_date' in df.columns else None
    
    if date_col:
        dates_1d = pd.to_datetime(df.loc[~pd.isna(df[target_col]), date_col]).to_list()
    else:
        dates_1d = np.arange(len(model.fittedvalues)).tolist()
    
    y_true_1d = _to_1d(model.model.endog)
    pred_1d = _to_1d(model.fittedvalues)  # Fitted values (predictions on training data)
    
    # Handle confidence intervals if available
    # For fitted values, we can get prediction intervals using get_prediction()
    # For now, set to None (can be added later if needed)
    lower_1d = None
    upper_1d = None
    
    # If confidence intervals are provided, extract them:
    # conf = conf_int  # Would come from model.get_prediction().conf_int() or similar
    # if conf is not None:
    #     if isinstance(conf, pd.DataFrame) and conf.shape[1] >= 2:
    #         lower_1d = _to_1d(conf.iloc[:, 0])
    #         upper_1d = _to_1d(conf.iloc[:, 1])
    #     elif isinstance(conf, np.ndarray) and conf.ndim == 2 and conf.shape[1] >= 2:
    #         lower_1d = _to_1d(conf[:, 0])
    #         upper_1d = _to_1d(conf[:, 1])
    
    # Build DataFrame with available data
    train_results_dict = {
        'quarter_end_date': dates_1d,
        'y_true': y_true_1d,
        'y_pred': pred_1d,
    }
    
    # Add confidence intervals if available
    if lower_1d is not None and upper_1d is not None:
        train_results_dict['y_pred_lower'] = lower_1d
        train_results_dict['y_pred_upper'] = upper_1d
    
    train_results_df = pd.DataFrame(train_results_dict)
    
    # Save outputs
    if save_outputs:
        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        
        # Save predictions
        predictions_path = OUTPUT_DIR / "mortgage_predictions.csv"
        predictions_df.to_csv(predictions_path, index=False)
        print(f"\nSaved predictions to: {predictions_path}")
        
        # Save metrics
        metrics_path = OUTPUT_DIR / "mortgage_metrics.json"
        with open(metrics_path, 'w') as f:
            json.dump(metrics, f, indent=2)
        print(f"Saved metrics to: {metrics_path}")
        
        # Save model summary
        summary_path = OUTPUT_DIR / "mortgage_model_summary.txt"
        with open(summary_path, 'w') as f:
            f.write(str(model.summary()))
        print(f"Saved model summary to: {summary_path}")
    
    return model, train_results_df, predictions_df, metrics


if __name__ == "__main__":
    """Run training and backtest as standalone script."""
    # Load data
    data_path = PROCESSED_DIR / "mortgage_model_dataset.parquet"
    
    if not data_path.exists():
        print(f"Error: Data file not found: {data_path}")
        print("Please run build_features.py first to create the dataset.")
    else:
        print(f"Loading data from: {data_path}")
        df = pd.read_parquet(data_path)
        print(f"Data shape: {df.shape}")
        
        # Train and backtest
        model, train_results_df, predictions_df, metrics = train_and_backtest_mortgage(
            df,
            target_col="y_level",
            use_grid_search=True,
            min_train_quarters=12
        )
        
        print("\n" + "=" * 60)
        print("TRAINING SUMMARY")
        print("=" * 60)
        print(f"Model: SARIMAX")
        print(f"AIC: {model.aic:.2f}")
        print(f"BIC: {model.bic:.2f}")
        
        print("\n" + "=" * 60)
        print("BACKTEST RESULTS")
        print("=" * 60)
        print(f"Total predictions: {len(predictions_df)}")
        print(f"Overall MAE: {metrics.get('overall', {}).get('mae', 'N/A'):.2f}")
        print(f"Overall sMAPE: {metrics.get('overall', {}).get('smape', 'N/A'):.2f}%")

