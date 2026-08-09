#!/usr/bin/env python3
"""
Model Comparison: Random Forest vs ARIMA
Corn Price Forecasting - Phase 3 (Fixed Version)

FIX: Both models now predict the exact same time period.
- ARIMA: trained on weeks 0..N-20, forecasts weeks N-20..N-1
- Random Forest: features from weeks N-21..N-2, target from weeks N-20..N-1

This ensures RMSE, MAE, and R² are directly comparable.
"""

import os
import sys
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.preprocessing import StandardScaler
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.stattools import adfuller
import itertools

warnings.filterwarnings('ignore')

# ============================================================
# SECTION 1: DATA LOADING AND PREPROCESSING
# ============================================================

def load_data(filepath='dataset/integrated_corn_weekly.csv'):
    """Load and prepare corn dataset."""
    df = pd.read_csv(filepath)
    df['Date'] = pd.to_datetime(df['Date'])
    df = df.sort_values('Date').reset_index(drop=True)
    
    # Target: next week's closing price (forecast horizon = 1 week)
    df['target'] = df['Close'].shift(-1)
    
    # Price lags (for Random Forest)
    for lag in [1, 2, 3, 4]:
        df[f'Close_lag_{lag}'] = df['Close'].shift(lag)
    
    # Weather features (with lags)
    weather_cols = ['temperature_2m_max', 'precipitation_sum', 'et0_fao_evapotranspiration']
    for col in weather_cols:
        for lag in [1, 2]:
            df[f'{col}_lag_{lag}'] = df[col].shift(lag)
    
    # Technical indicators (if available)
    tech_cols = ['RSI_14', 'volatility_20d', 'MACD']
    for col in tech_cols:
        if col in df.columns:
            df[f'{col}_lag_1'] = df[col].shift(1)
    
    # Calendar features
    if 'cal_is_pollination' in df.columns:
        df['pollination'] = df['cal_is_pollination'].astype(int)
    
    # Drop rows with NaN (due to shifts)
    df = df.dropna()
    
    return df

# ============================================================
# SECTION 2: TRAIN-TEST SPLIT (CHRONOLOGICAL, ALIGNED)
# ============================================================

def aligned_train_test_split(df, test_size=20):
    """
    Split data chronologically with aligned test periods.
    
    Returns:
        - train: training data (all rows except last test_size+1)
        - test_features: features for RF (last test_size+1 rows, excluding the very last)
        - test_target: actual target values for both models (last test_size rows)
        - test_dates: dates for the test period
    """
    n = len(df)
    
    # Test period: last 'test_size' rows (weeks N-test_size to N-1)
    # RF needs features from week N-test_size-1 to predict week N-test_size
    # So features are from indices N-test_size-1 to N-2 (inclusive)
    # Target is from indices N-test_size to N-1 (inclusive)
    
    train = df.iloc[:-(test_size + 1)].copy()  # Exclude last test_size+1 rows
    
    # For Random Forest: features from week before test period up to second-last week
    rf_features_df = df.iloc[-(test_size + 1):-1].copy()  # N-test_size-1 to N-2
    
    # For both models: actual target values (last test_size weeks)
    y_true = df['target'].iloc[-test_size:].values  # N-test_size to N-1
    test_dates = df['Date'].iloc[-test_size:].values
    
    # For ARIMA: use Close prices from training set
    arima_train = train['Close'].values
    
    print(f"Train: {len(train)} rows ({train['Date'].min()} to {train['Date'].max()})")
    print(f"Test period: {test_dates[0]} to {test_dates[-1]} ({test_size} weeks)")
    print(f"RF features: {len(rf_features_df)} rows ({rf_features_df['Date'].min()} to {rf_features_df['Date'].max()})")
    
    return train, rf_features_df, y_true, test_dates

# ============================================================
# SECTION 3: ARIMA MODEL
# ============================================================

def find_best_arima_order(y, max_p=5, max_d=2, max_q=5):
    """Find best ARIMA order using AIC grid search."""
    best_aic = np.inf
    best_order = None
    
    # Test for stationarity
    result = adfuller(y)
    d = 0
    if result[1] > 0.05:
        d = 1
        result = adfuller(y.diff().dropna())
        if result[1] > 0.05:
            d = 2
    
    p_range = range(0, max_p + 1)
    q_range = range(0, max_q + 1)
    
    print(f"   Searching ARIMA orders (p: 0-{max_p}, d={d}, q: 0-{max_q})...")
    
    for p, q in itertools.product(p_range, q_range):
        if p == 0 and q == 0:
            continue
        try:
            model = ARIMA(y, order=(p, d, q))
            fitted = model.fit()
            if fitted.aic < best_aic:
                best_aic = fitted.aic
                best_order = (p, d, q)
        except:
            continue
    
    if best_order is None:
        best_order = (1, d, 1)
        print(f"   No valid order found, using default (1,{d},1)")
    else:
        print(f"   Best order: {best_order}, AIC: {best_aic:.2f}")
    
    return best_order

def train_arima(arima_train, forecast_steps):
    """Train ARIMA and predict."""
    print("\n" + "=" * 60)
    print("ARIMA MODEL")
    print("=" * 60)
    
    # Find best order
    best_order = find_best_arima_order(arima_train)
    
    # Train final model
    print(f"   Training ARIMA{best_order}...")
    model = ARIMA(arima_train, order=best_order)
    fitted = model.fit()
    print(f"   AIC: {fitted.aic:.2f}")
    
    # Predict (forecast_steps weeks ahead)
    predictions = fitted.forecast(steps=forecast_steps)
    
    return {
        'model': 'ARIMA',
        'order': best_order,
        'predictions': predictions,
        'aic': fitted.aic
    }

# ============================================================
# SECTION 4: RANDOM FOREST MODEL
# ============================================================

def get_rf_features(df):
    """Get feature columns for Random Forest."""
    exclude = ['Date', 'Close', 'target', 'Unnamed: 0']
    feature_cols = [col for col in df.columns if col not in exclude]
    return feature_cols

def train_random_forest(train, rf_features_df, y_true):
    """Train Random Forest and predict."""
    print("\n" + "=" * 60)
    print("RANDOM FOREST MODEL")
    print("=" * 60)
    
    # Features
    feature_cols = get_rf_features(train)
    print(f"   Features: {len(feature_cols)} columns")
    
    X_train = train[feature_cols].values
    y_train = train['target'].values
    
    X_test = rf_features_df[feature_cols].values
    
    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Train model
    print("   Training Random Forest...")
    model = RandomForestRegressor(
        n_estimators=100,
        max_depth=15,
        min_samples_split=10,
        min_samples_leaf=5,
        random_state=42,
        n_jobs=-1
    )
    model.fit(X_train_scaled, y_train)
    
    # Predict
    y_pred = model.predict(X_test_scaled)
    
    # Feature importance
    importance = pd.DataFrame({
        'feature': feature_cols,
        'importance': model.feature_importances_
    }).sort_values('importance', ascending=False)
    
    print("\n   Top 10 Features:")
    for i, row in importance.head(10).iterrows():
        print(f"      {row['feature']}: {row['importance']:.4f}")
    
    return {
        'model': 'Random Forest',
        'predictions': y_pred,
        'feature_importance': importance
    }

# ============================================================
# SECTION 5: EVALUATION AND COMPARISON
# ============================================================

def evaluate_and_compare(arima_result, rf_result, y_true, test_dates):
    """Calculate metrics and compare both models."""
    
    y_pred_arima = arima_result['predictions']
    y_pred_rf = rf_result['predictions']
    
    # Calculate metrics
    rmse_arima = np.sqrt(mean_squared_error(y_true, y_pred_arima))
    mae_arima = mean_absolute_error(y_true, y_pred_arima)
    r2_arima = r2_score(y_true, y_pred_arima)
    
    rmse_rf = np.sqrt(mean_squared_error(y_true, y_pred_rf))
    mae_rf = mean_absolute_error(y_true, y_pred_rf)
    r2_rf = r2_score(y_true, y_pred_rf)
    
    results = {
        'ARIMA': {
            'rmse': rmse_arima,
            'mae': mae_arima,
            'r2': r2_arima,
            'predictions': y_pred_arima
        },
        'Random Forest': {
            'rmse': rmse_rf,
            'mae': mae_rf,
            'r2': r2_rf,
            'predictions': y_pred_rf
        }
    }
    
    return results

# ============================================================
# SECTION 6: VISUALIZATION
# ============================================================

def plot_results(results, y_true, test_dates, output_dir='outputs'):
    """Plot actual vs predicted values for both models."""
    os.makedirs(output_dir, exist_ok=True)
    
    fig, axes = plt.subplots(2, 1, figsize=(14, 10))
    
    # ARIMA plot
    ax1 = axes[0]
    ax1.plot(test_dates, y_true, label='Actual', color='black', linewidth=2)
    ax1.plot(test_dates, results['ARIMA']['predictions'], label='ARIMA Predicted', 
             color='blue', linestyle='--', linewidth=2)
    ax1.set_title(f'ARIMA - RMSE: {results["ARIMA"]["rmse"]:.2f}, MAE: {results["ARIMA"]["mae"]:.2f}, R²: {results["ARIMA"]["r2"]:.4f}')
    ax1.set_ylabel('Corn Price (USD/bushel)')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Random Forest plot
    ax2 = axes[1]
    ax2.plot(test_dates, y_true, label='Actual', color='black', linewidth=2)
    ax2.plot(test_dates, results['Random Forest']['predictions'], label='Random Forest Predicted', 
             color='green', linestyle='--', linewidth=2)
    ax2.set_title(f'Random Forest - RMSE: {results["Random Forest"]["rmse"]:.2f}, MAE: {results["Random Forest"]["mae"]:.2f}, R²: {results["Random Forest"]["r2"]:.4f}')
    ax2.set_xlabel('Date')
    ax2.set_ylabel('Corn Price (USD/bushel)')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f'{output_dir}/model_comparison_fixed.png', dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"\n✅ Plot saved: {output_dir}/model_comparison_fixed.png")

# ============================================================
# SECTION 7: SUMMARY TABLE
# ============================================================

def print_summary(results):
    """Print comparison summary table."""
    print("\n" + "=" * 60)
    print("MODEL COMPARISON SUMMARY (ALIGNED TEST PERIOD)")
    print("=" * 60)
    
    print("\n{:<20} {:>12} {:>12} {:>12}".format('Metric', 'ARIMA', 'Random Forest', 'Difference'))
    print("-" * 60)
    
    metrics = ['rmse', 'mae', 'r2']
    for metric in metrics:
        arima_val = results['ARIMA'][metric]
        rf_val = results['Random Forest'][metric]
        
        if metric == 'r2':
            diff = rf_val - arima_val
        else:
            diff = arima_val - rf_val
        
        diff_pct = (diff / abs(arima_val) * 100) if arima_val != 0 else 0
        
        print("{:<20} {:>12.4f} {:>12.4f} {:>+12.2f}%".format(
            metric.upper(), arima_val, rf_val, diff_pct
        ))
    
    print("-" * 60)
    
    # Determine winner
    if results['Random Forest']['rmse'] < results['ARIMA']['rmse']:
        winner = "Random Forest"
    else:
        winner = "ARIMA"
    
    print(f"\n🏆 Winner: {winner}")
    print("\n" + "=" * 60)

# ============================================================
# SECTION 8: MAIN
# ============================================================

def main():
    print("=" * 60)
    print("PHASE 3: MODEL COMPARISON (FIXED)")
    print("Random Forest vs ARIMA - Aligned Test Period")
    print("=" * 60)
    
    # Load data
    print("\n📂 Loading data...")
    df = load_data()
    print(f"   Total records: {len(df)}")
    
    # Split data (aligned test period)
    print("\n📊 Splitting data chronologically (aligned)...")
    train, rf_features_df, y_true, test_dates = aligned_train_test_split(df, test_size=20)
    
    # Train models
    arima_result = train_arima(train['Close'].values, forecast_steps=len(y_true))
    rf_result = train_random_forest(train, rf_features_df, y_true)
    
    # Evaluate
    results = evaluate_and_compare(arima_result, rf_result, y_true, test_dates)
    
    # Visualize
    plot_results(results, y_true, test_dates)
    
    # Summary
    print_summary(results)
    
    # Save results to file
    output_file = 'model_comparison_fixed_results.txt'
    with open(output_file, 'w') as f:
        f.write("=" * 60 + "\n")
        f.write("MODEL COMPARISON RESULTS (ALIGNED TEST PERIOD)\n")
        f.write("Random Forest vs ARIMA\n")
        f.write("=" * 60 + "\n\n")
        
        f.write("Test Period: {} to {} ({} weeks)\n\n".format(
            test_dates[0], test_dates[-1], len(y_true)
        ))
        
        f.write("ARIMA:\n")
        f.write(f"  Order: {arima_result['order']}\n")
        f.write(f"  AIC: {arima_result['aic']:.2f}\n")
        f.write(f"  RMSE: {results['ARIMA']['rmse']:.4f}\n")
        f.write(f"  MAE:  {results['ARIMA']['mae']:.4f}\n")
        f.write(f"  R²:   {results['ARIMA']['r2']:.4f}\n\n")
        
        f.write("Random Forest:\n")
        f.write(f"  RMSE: {results['Random Forest']['rmse']:.4f}\n")
        f.write(f"  MAE:  {results['Random Forest']['mae']:.4f}\n")
        f.write(f"  R²:   {results['Random Forest']['r2']:.4f}\n\n")
        
        if results['Random Forest']['rmse'] < results['ARIMA']['rmse']:
            f.write("Winner: Random Forest\n")
        else:
            f.write("Winner: ARIMA\n")
    
    print(f"\n📁 Results saved to: {output_file}")

if __name__ == "__main__":
    main()
