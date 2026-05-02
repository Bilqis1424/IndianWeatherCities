"""
================================================================================
STANDALONE SHAP ANALYSIS (FIXED – NO OBJECT DTYPES, CORRECT MODEL EXTRACTION)
================================================================================
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import warnings
import shap
import joblib
from sklearn.ensemble import StackingRegressor
from sklearn.multioutput import MultiOutputRegressor
from sklearn.linear_model import Ridge
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from catboost import CatBoostRegressor
from sklearn.preprocessing import LabelEncoder

warnings.filterwarnings('ignore')
np.random.seed(42)

# ============================================================================
# 1. DATA PREPARATION (ONLY KEEP NECESSARY COLUMNS)
# ============================================================================

def prepare_data_for_shap(filepath, train_ratio=0.75):
    df = pd.read_csv(filepath, low_memory=False)
    df = df.loc[:, ~df.columns.str.contains('^Unnamed', na=False)]
    
    needed_cols = ['timezone', 'last_updated_epoch', 'wind_mph', 'temperature_celsius', 'humidity', 'condition_text']
    for col in needed_cols:
        if col not in df.columns:
            raise ValueError(f"Required column '{col}' not found.")
    df = df[needed_cols].copy()
    
    df = df.rename(columns={
        'timezone': 'city',
        'last_updated_epoch': 'region',
        'wind_mph': 'datetime',
        'temperature_celsius': 'temp',
        'humidity': 'hum'
    })
    
    df['datetime'] = pd.to_datetime(df['datetime'], format='%m/%d/%Y %H:%M', errors='coerce')
    df['temp'] = pd.to_numeric(df['temp'], errors='coerce')
    df['hum'] = pd.to_numeric(df['hum'], errors='coerce')
    df = df.dropna(subset=['temp', 'hum', 'datetime', 'city', 'region'])
    df = df.sort_values(['city', 'datetime']).reset_index(drop=True)
    
    # Feature engineering
    df['temp_lag1'] = df.groupby('city')['temp'].shift(1)
    df['hum_lag1'] = df.groupby('city')['hum'].shift(1)
    df['hour_sin'] = np.sin(2 * np.pi * df['datetime'].dt.hour / 24)
    df['hour_cos'] = np.cos(2 * np.pi * df['datetime'].dt.hour / 24)
    df['month_sin'] = np.sin(2 * np.pi * df['datetime'].dt.month / 12)
    df['month_cos'] = np.cos(2 * np.pi * df['datetime'].dt.month / 12)
    
    # Encode categoricals
    le_city = LabelEncoder()
    le_region = LabelEncoder()
    le_cond = LabelEncoder()
    df['city_enc'] = le_city.fit_transform(df['city'].astype(str))
    df['region_enc'] = le_region.fit_transform(df['region'].astype(str))
    df['condition_enc'] = le_cond.fit_transform(df['condition_text'].astype(str))
    
    df = df.dropna().reset_index(drop=True)
    
    feature_cols = ['temp_lag1', 'hum_lag1', 'hour_sin', 'hour_cos', 'month_sin', 'month_cos',
                    'city_enc', 'region_enc', 'condition_enc']
    X = df[feature_cols].astype(np.float32)
    y = df[['temp', 'hum']].astype(np.float32)
    
    split_idx = int(len(X) * train_ratio)
    X_train, X_test = X[:split_idx], X[split_idx:]
    y_train, y_test = y[:split_idx], y[split_idx:]
    print(f"Data prepared: X_train {X_train.shape}, X_test {X_test.shape}")
    return X_train, X_test, y_train, y_test

# ============================================================================
# 2. STACKED ENSEMBLE MODEL
# ============================================================================

def get_stacked_model():
    base_models = [
        ('xgb', XGBRegressor(n_estimators=200, max_depth=5, learning_rate=0.05,
                             random_state=42, verbosity=0)),
        ('lgbm', LGBMRegressor(n_estimators=200, num_leaves=31, learning_rate=0.05,
                               random_state=42, verbose=-1)),
        ('cat', CatBoostRegressor(iterations=200, depth=6, learning_rate=0.05,
                                  random_seed=42, verbose=0))
    ]
    stack = StackingRegressor(estimators=base_models, final_estimator=Ridge(alpha=1.0), cv=5)
    return MultiOutputRegressor(stack, n_jobs=-1)

def train_or_load_model(X_train, y_train, model_path='stacked_model_shap.pkl'):
    try:
        model = joblib.load(model_path)
        print(f"Loaded model from {model_path}")
    except FileNotFoundError:
        print("Training stacked ensemble (may take a few minutes)...")
        model = get_stacked_model()
        model.fit(X_train, y_train)
        joblib.dump(model, model_path)
        print(f"Model saved to {model_path}")
    return model

# ============================================================================
# 3. SHAP ANALYSIS (FIXED EXTRACTION)
# ============================================================================

def shap_analysis(model, X_sample, feature_names):
    # MultiOutputRegressor -> StackingRegressor for temperature
    stacking_reg = model.estimators_[0]
    # Correct way to access the fitted XGBoost model
    xgb_model = stacking_reg.named_estimators_['xgb']
    
    print("Computing SHAP values...")
    explainer = shap.TreeExplainer(xgb_model)
    shap_values = explainer.shap_values(X_sample)
    
    # Summary plot
    plt.figure(figsize=(10, 6))
    shap.summary_plot(shap_values, X_sample, feature_names=feature_names, show=False)
    plt.title("SHAP Feature Importance (Stacked Ensemble – Temperature)")
    plt.tight_layout()
    plt.savefig('shap_summary.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("Saved: shap_summary.png")
    
    # Bar plot
    mean_abs_shap = np.abs(shap_values).mean(axis=0)
    imp_df = pd.DataFrame({'feature': feature_names, 'importance': mean_abs_shap})
    imp_df = imp_df.sort_values('importance', ascending=False).head(15)
    plt.figure(figsize=(10, 5))
    plt.barh(imp_df['feature'], imp_df['importance'])
    plt.xlabel('Mean |SHAP value|')
    plt.title('Top 15 Feature Importance (XGBoost in Stacked Ensemble)')
    plt.gca().invert_yaxis()
    plt.tight_layout()
    plt.savefig('shap_bar.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("Saved: shap_bar.png")
    
    imp_df.to_csv('shap_importance.csv', index=False)
    print("Saved: shap_importance.csv")
    return imp_df

# ============================================================================
# 4. MAIN
# ============================================================================

def main():
    print("Loading and preparing data...")
    X_train, X_test, y_train, y_test = prepare_data_for_shap('IndianWeatherRepository.csv')
    model = train_or_load_model(X_train, y_train)
    
    X_sample = X_test.sample(n=min(200, len(X_test)), random_state=42)
    feature_names = X_test.columns.tolist()
    shap_analysis(model, X_sample, feature_names)
    print("\nSHAP analysis completed. Files: shap_summary.png, shap_bar.png, shap_importance.csv")

if __name__ == "__main__":
    main()