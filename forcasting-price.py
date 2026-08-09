#!/usr/bin/env python3
"""
Phase 4: Model Interpretability with Random Forest
Corn Price Forecasting - SHAP, TreeSHAP, LIME, TreeInterpreter

This script:
1. Auto-installs required packages (shap, lime, treeinterpreter)
2. Trains a Random Forest model on corn price data
3. Generates interpretable explanations using:
   - SHAP (SHapley Additive exPlanations)
   - TreeSHAP (SHAP for tree-based models)
   - LIME (Local Interpretable Model-agnostic Explanations)
   - TreeInterpreter (Feature contributions per tree)
4. Visualizes feature importance and individual predictions
5. Saves results for paper inclusion
"""

import os
import sys
import subprocess
import warnings
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings('ignore')

# ============================================================
# AUTO-INSTALL DEPENDENCIES
# ============================================================

def install_package(pkg):
    """Install a Python package using pip."""
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", pkg, "--quiet"])
        return True
    except Exception as e:
        print(f"   ⚠️ Failed to install {pkg}: {e}")
        return False

def ensure_packages():
    """Ensure all required packages are installed."""
    print("\n" + "=" * 60)
    print("CHECKING DEPENDENCIES")
    print("=" * 60)
    
    required = {
        'shap': 'shap',
        'lime': 'lime',
        'treeinterpreter': 'treeinterpreter'
    }
    
    installed = {}
    for name, pkg in required.items():
        try:
            __import__(name)
            print(f"   ✅ {name} already installed")
            installed[name] = True
        except ImportError:
            print(f"   ⚠️ {name} not found. Installing...")
            success = install_package(pkg)
            installed[name] = success
            if success:
                print(f"   ✅ {name} installed successfully")
            else:
                print(f"   ❌ Failed to install {name}")
    
    # Import after installation
    try:
        import shap
        import lime
        import treeinterpreter
        print("\n   ✅ All dependencies ready")
        return True
    except ImportError as e:
        print(f"\n   ❌ Still missing dependencies: {e}")
        return False

# Try to import, if fails, install and retry
try:
    import shap
    import lime
    import treeinterpreter
except ImportError:
    print("⚠️ Some packages are missing. Attempting auto-install...")
    if ensure_packages():
        import shap
        import lime
        import treeinterpreter
    else:
        print("❌ Could not install required packages. Please install manually:")
        print("   pip install shap lime treeinterpreter")
        sys.exit(1)

# ============================================================
# SECTION 1: DATA LOADING AND PREPROCESSING
# ============================================================

def load_data(filepath='dataset/integrated_corn_weekly.csv'):
    """Load and prepare corn dataset."""
    df = pd.read_csv(filepath)
    df['Date'] = pd.to_datetime(df['Date'])
    df = df.sort_values('Date').reset_index(drop=True)
    
    # Target: next week's closing price
    df['target'] = df['Close'].shift(-1)
    
    # Price lags
    for lag in [1, 2, 3, 4]:
        df[f'Close_lag_{lag}'] = df['Close'].shift(lag)
    
    # Weather features
    weather_cols = ['temperature_2m_max', 'precipitation_sum', 'et0_fao_evapotranspiration']
    for col in weather_cols:
        for lag in [1, 2]:
            df[f'{col}_lag_{lag}'] = df[col].shift(lag)
    
    # Technical indicators
    tech_cols = ['RSI_14', 'volatility_20d', 'MACD']
    for col in tech_cols:
        if col in df.columns:
            df[f'{col}_lag_1'] = df[col].shift(1)
    
    # Calendar features
    if 'cal_is_pollination' in df.columns:
        df['pollination'] = df['cal_is_pollination'].astype(int)
    
    # Additional price features (for interpretation)
    if 'Open' in df.columns and 'High' in df.columns and 'Low' in df.columns:
        df['price_range'] = df['High'] - df['Low']
        df['price_range_pct'] = (df['High'] - df['Low']) / df['Open'] * 100
        df['close_to_high'] = (df['High'] - df['Close']) / df['Close'] * 100
        df['close_to_low'] = (df['Close'] - df['Low']) / df['Close'] * 100
    
    # Drop rows with NaN
    df = df.dropna()
    
    return df

def prepare_features(df):
    """Extract features and target."""
    exclude = ['Date', 'Close', 'target', 'Unnamed: 0']
    feature_cols = [col for col in df.columns if col not in exclude]
    X = df[feature_cols].values
    y = df['target'].values
    feature_names = feature_cols
    return X, y, feature_names, df['Date'].values

# ============================================================
# SECTION 2: TRAIN-TEST SPLIT
# ============================================================

def train_test_split_chronological(X, y, dates, test_size=20):
    """Split data chronologically."""
    n = len(X)
    split_idx = n - test_size
    X_train, X_test = X[:split_idx], X[split_idx:]
    y_train, y_test = y[:split_idx], y[split_idx:]
    dates_train, dates_test = dates[:split_idx], dates[split_idx:]
    
    print(f"Train: {len(X_train)} rows ({dates_train[0]} to {dates_train[-1]})")
    print(f"Test: {len(X_test)} rows ({dates_test[0]} to {dates_test[-1]})")
    
    return X_train, X_test, y_train, y_test, dates_train, dates_test

# ============================================================
# SECTION 3: TRAIN RANDOM FOREST
# ============================================================

def train_random_forest(X_train, y_train, X_test, y_test, feature_names):
    """Train Random Forest and return model + predictions."""
    print("\n" + "=" * 60)
    print("TRAINING RANDOM FOREST")
    print("=" * 60)
    
    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Train model
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
    y_pred_train = model.predict(X_train_scaled)
    y_pred_test = model.predict(X_test_scaled)
    
    # Metrics
    rmse_train = np.sqrt(mean_squared_error(y_train, y_pred_train))
    rmse_test = np.sqrt(mean_squared_error(y_test, y_pred_test))
    r2_train = r2_score(y_train, y_pred_train)
    r2_test = r2_score(y_test, y_pred_test)
    
    print(f"Train RMSE: {rmse_train:.4f}, Train R²: {r2_train:.4f}")
    print(f"Test RMSE:  {rmse_test:.4f}, Test R²:  {r2_test:.4f}")
    
    return {
        'model': model,
        'scaler': scaler,
        'X_train': X_train_scaled,
        'X_test': X_test_scaled,
        'y_train': y_train,
        'y_test': y_test,
        'y_pred_train': y_pred_train,
        'y_pred_test': y_pred_test,
        'rmse_train': rmse_train,
        'rmse_test': rmse_test,
        'r2_train': r2_train,
        'r2_test': r2_test,
        'feature_names': feature_names
    }

# ============================================================
# SECTION 4: SHAP ANALYSIS
# ============================================================

def run_shap_analysis(model, X_train, X_test, feature_names, output_dir='outputs'):
    """Run SHAP and TreeSHAP analysis."""
    print("\n" + "=" * 60)
    print("SHAP ANALYSIS")
    print("=" * 60)
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Create explainer
    print("   Creating TreeExplainer...")
    explainer = shap.TreeExplainer(model)
    
    # Compute SHAP values (sample for speed)
    print("   Computing SHAP values (using first 100 samples)...")
    X_sample = X_test[:100] if len(X_test) > 100 else X_test
    shap_values = explainer.shap_values(X_sample)
    
    # Summary plot
    print("   Generating summary plot...")
    fig, ax = plt.subplots(figsize=(12, 8))
    shap.summary_plot(shap_values, X_sample, feature_names=feature_names, show=False)
    plt.tight_layout()
    plt.savefig(f'{output_dir}/shap_summary.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"   ✅ shap_summary.png saved")
    
    # Bar plot (mean absolute SHAP)
    print("   Generating bar plot...")
    fig, ax = plt.subplots(figsize=(10, 8))
    shap.summary_plot(shap_values, X_sample, feature_names=feature_names, plot_type="bar", show=False)
    plt.tight_layout()
    plt.savefig(f'{output_dir}/shap_bar.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"   ✅ shap_bar.png saved")
    
    # Force plot for first prediction
    print("   Generating force plot for first test sample...")
    try:
        fig, ax = plt.subplots(figsize=(14, 4))
        shap.force_plot(explainer.expected_value, shap_values[0], X_sample[0], 
                       feature_names=feature_names, show=False, matplotlib=True)
        plt.tight_layout()
        plt.savefig(f'{output_dir}/shap_force_plot_sample1.png', dpi=150, bbox_inches='tight')
        plt.close()
        print(f"   ✅ shap_force_plot_sample1.png saved")
    except Exception as e:
        print(f"   ⚠️ Force plot failed: {e}")
    
    # Decision plot for multiple samples
    print("   Generating decision plot...")
    try:
        fig, ax = plt.subplots(figsize=(12, 8))
        shap.decision_plot(explainer.expected_value, shap_values[:20], X_sample[:20], 
                          feature_names=feature_names, show=False)
        plt.tight_layout()
        plt.savefig(f'{output_dir}/shap_decision_plot.png', dpi=150, bbox_inches='tight')
        plt.close()
        print(f"   ✅ shap_decision_plot.png saved")
    except Exception as e:
        print(f"   ⚠️ Decision plot failed: {e}")
    
    # Extract top features by SHAP importance
    shap_importance = np.abs(shap_values).mean(axis=0)
    top_features = sorted(
        [(feature_names[i], shap_importance[i]) for i in range(len(feature_names))],
        key=lambda x: x[1],
        reverse=True
    )[:15]
    
    print("\n   Top 15 Features by SHAP Importance:")
    for i, (name, val) in enumerate(top_features, 1):
        print(f"      {i}. {name}: {val:.4f}")
    
    # Save SHAP importance to JSON
    shap_data = {
        'top_features': [{'feature': name, 'importance': float(val)} for name, val in top_features],
        'expected_value': float(explainer.expected_value)
    }
    
    with open(f'{output_dir}/shap_importance.json', 'w') as f:
        json.dump(shap_data, f, indent=2)
    print(f"   ✅ shap_importance.json saved")
    
    return {
        'explainer': explainer,
        'shap_values': shap_values,
        'top_features': top_features
    }

# ============================================================
# SECTION 5: LIME ANALYSIS
# ============================================================

def run_lime_analysis(model, X_train, X_test, y_test, feature_names, output_dir='outputs'):
    """Run LIME analysis on selected samples."""
    print("\n" + "=" * 60)
    print("LIME ANALYSIS")
    print("=" * 60)
    
    os.makedirs(output_dir, exist_ok=True)
    
    try:
        from lime.lime_tabular import LimeTabularExplainer
        
        print("   Creating LIME explainer...")
        explainer = LimeTabularExplainer(
            X_train,
            feature_names=feature_names,
            mode='regression',
            random_state=42
        )
        
        # Explain first 3 test samples
        print("   Explaining 3 test samples...")
        for i in range(min(3, len(X_test))):
            exp = explainer.explain_instance(
                X_test[i],
                model.predict,
                num_features=10
            )
            
            # Save LIME explanation as HTML
            try:
                exp.save_to_file(f'{output_dir}/lime_explanation_sample_{i+1}.html')
                print(f"   ✅ lime_explanation_sample_{i+1}.html saved")
            except Exception as e:
                print(f"   ⚠️ Could not save HTML: {e}")
            
            # Print top features
            print(f"\n   Sample {i+1} (actual: {y_test[i]:.2f}, predicted: {model.predict([X_test[i]])[0]:.2f}):")
            for feat, weight in exp.as_list():
                print(f"      {feat}: {weight:.4f}")
        
        return True
        
    except Exception as e:
        print(f"   ⚠️ LIME error: {e}")
        return None

# ============================================================
# SECTION 6: TREEINTERPRETER ANALYSIS
# ============================================================

def run_treeinterpreter(model, X_test, y_test, feature_names, output_dir='outputs'):
    """Run TreeInterpreter analysis for feature contributions."""
    print("\n" + "=" * 60)
    print("TREEINTERPRETER ANALYSIS")
    print("=" * 60)
    
    os.makedirs(output_dir, exist_ok=True)
    
    try:
        from treeinterpreter import treeinterpreter
        
        print("   Computing feature contributions...")
        pred, bias, contributions = treeinterpreter.predict(model, X_test[:10])
        
        print(f"   Bias (mean prediction): {bias[0]:.4f}")
        print(f"   Number of features with contributions: {contributions.shape[1]}")
        
        # For the first sample, show top contributors
        sample_idx = 0
        contribs = contributions[sample_idx]
        contrib_df = pd.DataFrame({
            'feature': feature_names,
            'contribution': contribs
        }).sort_values('contribution', key=abs, ascending=False)
        
        print(f"\n   Sample {sample_idx+1} (actual: {y_test[sample_idx]:.2f}):")
        print(f"   Bias: {bias[sample_idx]:.4f}")
        print(f"   Predicted: {pred[sample_idx]:.4f}")
        print(f"   Sum of contributions: {contribs.sum():.4f}")
        print(f"   Top 10 contributors:")
        for i, row in contrib_df.head(10).iterrows():
            print(f"      {row['feature']}: {row['contribution']:+.4f}")
        
        # Save results
        all_contribs = []
        for i in range(min(20, len(X_test))):
            pred_i, bias_i, contribs_i = treeinterpreter.predict(model, X_test[i:i+1])
            all_contribs.append({
                'sample': i,
                'prediction': float(pred_i[0]),
                'bias': float(bias_i[0]),
                'contributions': {feature_names[j]: float(contribs_i[0][j]) for j in range(len(feature_names))}
            })
        
        with open(f'{output_dir}/treeinterpreter_results.json', 'w') as f:
            json.dump(all_contribs, f, indent=2)
        print(f"   ✅ treeinterpreter_results.json saved")
        
        return True
        
    except Exception as e:
        print(f"   ⚠️ TreeInterpreter error: {e}")
        return None

# ============================================================
# SECTION 7: VISUALIZATION - PREDICTION VS ACTUAL
# ============================================================

def plot_predictions(y_test, y_pred_test, dates_test, output_dir='outputs'):
    """Plot actual vs predicted values with confidence bands."""
    print("\n" + "=" * 60)
    print("PREDICTION VISUALIZATION")
    print("=" * 60)
    
    os.makedirs(output_dir, exist_ok=True)
    
    fig, ax = plt.subplots(figsize=(14, 8))
    
    ax.plot(dates_test, y_test, label='Actual', color='black', linewidth=2, marker='o', markersize=4)
    ax.plot(dates_test, y_pred_test, label='Predicted', color='blue', linestyle='--', linewidth=2, marker='s', markersize=4)
    
    # Add error bars (MAE)
    mae = mean_absolute_error(y_test, y_pred_test)
    ax.fill_between(dates_test, y_pred_test - mae, y_pred_test + mae, alpha=0.2, color='blue', label=f'±MAE ({mae:.2f})')
    
    ax.set_xlabel('Date')
    ax.set_ylabel('Corn Price (USD/bushel)')
    ax.set_title('Random Forest: Actual vs Predicted Corn Prices')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f'{output_dir}/rf_predictions_plot.png', dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"   ✅ rf_predictions_plot.png saved")

# ============================================================
# SECTION 8: SUMMARY JSON
# ============================================================

def save_summary(results, output_dir='outputs'):
    """Save summary metrics to JSON."""
    summary = {
        'model': 'Random Forest',
        'train_rmse': float(results['rmse_train']),
        'test_rmse': float(results['rmse_test']),
        'train_r2': float(results['r2_train']),
        'test_r2': float(results['r2_test']),
        'n_estimators': 100,
        'max_depth': 15,
        'test_samples': len(results['y_test'])
    }
    
    with open(f'{output_dir}/interpretability_summary.json', 'w') as f:
        json.dump(summary, f, indent=2)
    print(f"   ✅ interpretability_summary.json saved")

# ============================================================
# SECTION 9: MAIN
# ============================================================

def main():
    print("=" * 60)
    print("PHASE 4: MODEL INTERPRETABILITY")
    print("Random Forest with SHAP, LIME, TreeInterpreter")
    print("=" * 60)
    
    os.makedirs('outputs', exist_ok=True)
    
    # Load data
    print("\n📂 Loading data...")
    df = load_data()
    X, y, feature_names, dates = prepare_features(df)
    print(f"   Records: {len(X)}, Features: {len(feature_names)}")
    
    # Split
    print("\n📊 Splitting data chronologically...")
    X_train, X_test, y_train, y_test, dates_train, dates_test = train_test_split_chronological(X, y, dates, test_size=20)
    
    # Train model
    results = train_random_forest(X_train, y_train, X_test, y_test, feature_names)
    
    # Plot predictions
    plot_predictions(y_test, results['y_pred_test'], dates_test)
    
    # SHAP analysis
    shap_results = run_shap_analysis(
        results['model'], 
        results['X_train'], 
        results['X_test'], 
        feature_names
    )
    
    # LIME analysis
    lime_results = run_lime_analysis(
        results['model'],
        results['X_train'],
        results['X_test'],
        y_test,
        feature_names
    )
    
    # TreeInterpreter
    ti_results = run_treeinterpreter(
        results['model'],
        results['X_test'],
        y_test,
        feature_names
    )
    
    # Save summary
    save_summary(results)
    
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    
    print(f"Test RMSE: {results['rmse_test']:.4f}")
    print(f"Test MAE:  {mean_absolute_error(y_test, results['y_pred_test']):.4f}")
    print(f"Test R²:   {results['r2_test']:.4f}")
    
    print("\n📁 Outputs saved to 'outputs/':")
    print("   - rf_predictions_plot.png")
    print("   - shap_summary.png")
    print("   - shap_bar.png")
    print("   - shap_force_plot_sample1.png")
    print("   - shap_decision_plot.png")
    print("   - shap_importance.json")
    print("   - interpretability_summary.json")
    if lime_results:
        print("   - lime_explanation_sample_*.html")
    if ti_results:
        print("   - treeinterpreter_results.json")
    
    print("\n" + "=" * 60)
    print("✅ INTERPRETABILITY ANALYSIS COMPLETE")
    print("=" * 60)

if __name__ == "__main__":
    main()
