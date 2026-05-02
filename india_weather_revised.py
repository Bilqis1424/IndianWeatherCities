import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import shap
import re
import warnings
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from catboost import CatBoostRegressor
from sklearn.ensemble import StackingRegressor
from sklearn.multioutput import MultiOutputRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import r2_score, mean_absolute_error
from statsmodels.tsa.arima.model import ARIMA

warnings.filterwarnings('ignore')

# ============================================================================
# 1. BULLETPROOF DATA PREPARATION
# ============================================================================

def sanitize_column_names(df):
    """
    Strips all JSON-unsafe characters from column names.
    LightGBM specifically dislikes: , [ ] { } : "
    """
    cols = []
    for col in df.columns:
        # Replace any non-alphanumeric character with an underscore
        new_col = re.sub(r'[^A-Za-z0-9_]+', '_', str(col))
        # Remove multiple underscores and trailing underscores
        new_col = re.sub(r'_+', '_', new_col).strip('_')
        cols.append(new_col)
    df.columns = cols
    return df

def prepare_robust_data(filepath):
    df = pd.read_csv(filepath, low_memory=False)
    
    # 1. Map Messy Columns
    df = df.rename(columns={
        'timezone': 'city', 
        'last_updated_epoch': 'region',
        'wind_mph': 'datetime', 
        'temperature_celsius': 'temp', 
        'humidity': 'hum'
    })
    
    # 2. Convert Types
    df['datetime'] = pd.to_datetime(df['datetime'], errors='coerce')
    for col in ['temp', 'hum']:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    
    # 3. Cyclical Encoding (R3 Request)
    df['hour_sin'] = np.sin(2 * np.pi * df['datetime'].dt.hour / 24)
    df['hour_cos'] = np.cos(2 * np.pi * df['datetime'].dt.hour / 24)
    
    df = df.sort_values(['city', 'datetime']).reset_index(drop=True)
    return df

def get_features_and_targets(df, train_idx):
    temp_df = df.copy()
    
    # Identify key columns post-renaming
    region_col = [c for c in temp_df.columns if 'region' in c.lower()][0]
    city_col = [c for c in temp_df.columns if 'city' in c.lower()][0]
    
    train_data = temp_df.iloc[:train_idx]
    
    # 1. Regional Spatial Features (R2: Point 7)
    reg_means_temp = train_data.groupby(region_col)['temp'].mean().to_dict()
    reg_means_hum = train_data.groupby(region_col)['hum'].mean().to_dict()
    
    temp_df['reg_temp_mean'] = temp_df[region_col].map(reg_means_temp).fillna(train_data['temp'].mean())
    temp_df['reg_hum_mean'] = temp_df[region_col].map(reg_means_hum).fillna(train_data['hum'].mean())

    # 2. Temporal Lags (R3: Point 1.3)
    for col in ['temp', 'hum']:
        temp_df[f'{col}_lag1'] = temp_df.groupby(city_col)[col].shift(1)
        temp_df[f'{col}_lag1'] = temp_df[f'{col}_lag1'].fillna(temp_df[col])

    # 3. Encoding & Cleaning
    for col in [region_col, city_col, 'condition_text', 'wind_direction']:
        if col in temp_df.columns:
            temp_df[col] = pd.factorize(temp_df[col].astype(str))[0]

    # Fill NaNs for the rest (Air quality, etc)
    temp_df = temp_df.fillna(temp_df.select_dtypes(include=[np.number]).median())

    # 4. FINAL SANITIZATION (The Fix for LightGBM)
    temp_df = sanitize_column_names(temp_df)

    X = temp_df.select_dtypes(include=[np.number]).drop(columns=['temp', 'hum'], errors='ignore')
    y = temp_df[['temp', 'hum']]
    
    return X, y

# ============================================================================
# 2. RUN EXPERIMENT
# ============================================================================

def run_experiment():
    # Load and Split
    df = prepare_robust_data('IndianWeatherRepository.csv')
    split_idx = int(len(df) * 0.85)
    
    X, y = get_features_and_targets(df, split_idx)
    
    X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
    y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]

    # Model Configuration
    base_models = [
        ('xgb', XGBRegressor(n_estimators=100, max_depth=5, learning_rate=0.05)),
        ('lgbm', LGBMRegressor(n_estimators=100, verbose=-1)),
        ('cat', CatBoostRegressor(iterations=100, verbose=0))
    ]
    
    stack = MultiOutputRegressor(StackingRegressor(
        estimators=base_models, 
        final_estimator=Ridge(), 
        cv=5
    ))
    
    print(f"Dataset Ready: {X.shape[0]} rows, {X.shape[1]} features.")
    print("Starting Training...")
    
    stack.fit(X_train, y_train)
    
    # Results
    preds = stack.predict(X_test)
    print(f"\nResults:\nTemp R2: {r2_score(y_test['temp'], preds[:,0]):.4f}")
    print(f"Hum R2:  {r2_score(y_test['hum'], preds[:,1]):.4f}")

    # SHAP (Reviewer Request)
    explainer = shap.Explainer(stack.estimators_[0].named_estimators_['xgb'], X_train)
    shap_values = explainer(X_test.iloc[:100])
    shap.summary_plot(shap_values, X_test.iloc[:100])

if __name__ == "__main__":
    run_experiment()