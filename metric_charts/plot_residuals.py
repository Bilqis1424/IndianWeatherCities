"""
================================================================================
STANDALONE SCRIPT: RESIDUAL DIAGNOSTICS FOR STACKED ENSEMBLE
Generates residual plots for temperature and humidity predictions.
================================================================================
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import warnings
import joblib
from scipy import stats
from pandas.plotting import autocorrelation_plot
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
# 1. DATA PREPARATION (same as in the main pipeline)
# ============================================================================

def prepare_data(filepath, train_ratio=0.75):
    df = pd.read_csv(filepath, low_memory=False)
    df = df.loc[:, ~df.columns.str.contains('^Unnamed', na=False)]
    
    # Keep only necessary columns
    needed = ['timezone', 'last_updated_epoch', 'wind_mph', 'temperature_celsius', 'humidity', 'condition_text']
    df = df[needed].copy()
    
    # Rename
    df = df.rename(columns={
        'timezone': 'city',
        'last_updated_epoch': 'region',
        'wind_mph': 'datetime',
        'temperature_celsius': 'temp',
        'humidity': 'hum'
    })
    
    # Parse datetime
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
    
    # Encode categoricals (fit on whole data for simplicity in residual plotting)
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
    
    split = int(len(X) * train_ratio)
    X_train, X_test = X[:split], X[split:]
    y_train, y_test = y[:split], y[split:]
    return X_train, X_test, y_train, y_test

# ============================================================================
# 2. LOAD OR TRAIN STACKED MODEL
# ============================================================================

def get_stacked_model():
    base = [
        ('xgb', XGBRegressor(n_estimators=200, max_depth=5, learning_rate=0.05, random_state=42, verbosity=0)),
        ('lgbm', LGBMRegressor(n_estimators=200, num_leaves=31, learning_rate=0.05, random_state=42, verbose=-1)),
        ('cat', CatBoostRegressor(iterations=200, depth=6, learning_rate=0.05, random_seed=42, verbose=0))
    ]
    stack = StackingRegressor(estimators=base, final_estimator=Ridge(alpha=1.0), cv=5)
    return MultiOutputRegressor(stack, n_jobs=-1)

def load_or_train_model(X_train, y_train, model_path='stacked_model_residuals.pkl'):
    try:
        model = joblib.load(model_path)
        print(f"Loaded model from {model_path}")
    except FileNotFoundError:
        print("Training stacked ensemble (this may take a few minutes)...")
        model = get_stacked_model()
        model.fit(X_train, y_train)
        joblib.dump(model, model_path)
        print(f"Model saved to {model_path}")
    return model

# ============================================================================
# 3. RESIDUAL PLOTS
# ============================================================================

def plot_residuals(y_true, y_pred, target_name):
    residuals = y_true - y_pred
    
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    fig.suptitle(f'Residual Diagnostics – {target_name}', fontsize=14)
    
    # 1. Residuals vs predicted
    axes[0,0].scatter(y_pred, residuals, alpha=0.5, s=10, edgecolor='k')
    axes[0,0].axhline(0, color='r', linestyle='--', linewidth=1)
    axes[0,0].set_xlabel('Predicted')
    axes[0,0].set_ylabel('Residuals')
    axes[0,0].set_title('Residuals vs Predicted')
    
    # 2. Q-Q plot
    stats.probplot(residuals, dist="norm", plot=axes[0,1])
    axes[0,1].set_title('Q-Q Plot')
    
    # 3. Histogram
    axes[1,0].hist(residuals, bins=50, edgecolor='black', alpha=0.7, color='steelblue')
    axes[1,0].set_xlabel('Residuals')
    axes[1,0].set_ylabel('Frequency')
    axes[1,0].set_title('Distribution of Residuals')
    
    # 4. Autocorrelation
    autocorrelation_plot(pd.Series(residuals), ax=axes[1,1])
    axes[1,1].set_title('Autocorrelation of Residuals')
    axes[1,1].set_xlim([0, 40])
    axes[1,1].axhline(0, color='gray', linestyle='--', linewidth=0.8)
    
    plt.tight_layout()
    plt.savefig(f'residuals_{target_name}.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: residuals_{target_name}.png")

# ============================================================================
# 4. MAIN
# ============================================================================

def main():
    print("Preparing data...")
    X_train, X_test, y_train, y_test = prepare_data('IndianWeatherRepository.csv')
    print(f"Training set: {X_train.shape}, Test set: {X_test.shape}")
    
    print("Loading/training stacked ensemble...")
    model = load_or_train_model(X_train, y_train, model_path='stacked_model_residuals.pkl')
    
    print("Generating predictions on test set...")
    y_pred = model.predict(X_test)
    pred_temp = y_pred[:, 0]
    pred_hum = y_pred[:, 1]
    true_temp = y_test['temp'].values
    true_hum = y_test['hum'].values
    
    print("Plotting residuals for temperature...")
    plot_residuals(true_temp, pred_temp, 'temperature')
    
    print("Plotting residuals for humidity...")
    plot_residuals(true_hum, pred_hum, 'humidity')
    
    print("\nDone. Residual plots saved as 'residuals_temperature.png' and 'residuals_humidity.png'.")

if __name__ == "__main__":
    main()