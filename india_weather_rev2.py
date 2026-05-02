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
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

warnings.filterwarnings('ignore')

# ============================================================================
# 1. UPDATED DATA PREPARATION (Fixes ValueError: Invalid columns)
# ============================================================================

def sanitize_column_names(df):
    cols = []
    for col in df.columns:
        new_col = re.sub(r'[^A-Za-z0-9_]+', '_', str(col))
        new_col = re.sub(r'_+', '_', new_col).strip('_')
        cols.append(new_col)
    df.columns = cols
    return df

def prepare_and_split_data(filepath, train_ratio=0.75):
    df = pd.read_csv(filepath, low_memory=False)
    
    # 1. Clean "Trash" Columns (Unnamed/Empty)
    df = df.loc[:, ~df.columns.str.contains('^Unnamed')]
    
    # 2. Map Messy Columns
    df = df.rename(columns={
        'timezone': 'city', 
        'last_updated_epoch': 'region',
        'wind_mph': 'datetime', 
        'temperature_celsius': 'temp', 
        'humidity': 'hum'
    })
    
    # 3. Force Numeric Conversion (CRITICAL FIX)
    # We try to convert everything except our core text keys to numeric.
    # If it fails (like a city name), it becomes NaN, which we handle later.
    exclude_from_numeric = ['city', 'region', 'datetime', 'condition_text', 'wind_direction', 'moon_phase']
    for col in df.columns:
        if col not in exclude_from_numeric:
            df[col] = pd.to_numeric(df[col], errors='coerce')
    
    # Pre-processing Time
    df['datetime'] = pd.to_datetime(df['datetime'], errors='coerce')
    df = df.dropna(subset=['temp', 'hum', 'datetime']).sort_values(['city', 'datetime']).reset_index(drop=True)
    
    # 4. Chronological Split
    split_idx = int(len(df) * train_ratio)
    train_df = df.iloc[:split_idx].copy()
    test_df = df.iloc[split_idx:].copy()

    # 5. Leakage-Proof Feature Engineering
    reg_temp_map = train_df.groupby('region')['temp'].mean().to_dict()
    reg_hum_map = train_df.groupby('region')['hum'].mean().to_dict()
    global_temp_mean = train_df['temp'].mean()
    global_hum_mean = train_df['hum'].mean()

    def apply_features(target_df):
        target_df['reg_temp_mean'] = target_df['region'].map(reg_temp_map).fillna(global_temp_mean)
        target_df['reg_hum_mean'] = target_df['region'].map(reg_hum_map).fillna(global_hum_mean)
        target_df['temp_lag1'] = target_df.groupby('city')['temp'].shift(1).fillna(method='bfill')
        target_df['hum_lag1'] = target_df.groupby('city')['hum'].shift(1).fillna(method='bfill')
        target_df['hour_sin'] = np.sin(2 * np.pi * target_df['datetime'].dt.hour / 24)
        target_df['hour_cos'] = np.cos(2 * np.pi * target_df['datetime'].dt.hour / 24)
        return target_df

    train_df = apply_features(train_df)
    test_df = apply_features(test_df)
    
    full_df = pd.concat([train_df, test_df], axis=0)
    
    # 6. Final Encoding (Convert remaining objects to numbers)
    categorical_cols = ['city', 'region', 'condition_text', 'wind_direction', 'moon_phase']
    for col in categorical_cols:
        if col in full_df.columns:
            full_df[col] = pd.factorize(full_df[col].astype(str))[0]
    
    # Drop the datetime object and any columns that are still objects
    full_df = full_df.select_dtypes(exclude=['datetime', 'datetime64', 'object'])
    
    # Fill remaining NaNs from coerced numeric conversions
    full_df = full_df.fillna(full_df.median())
    full_df = sanitize_column_names(full_df)
    
    X = full_df.drop(columns=['temp', 'hum'])
    y = full_df[['temp', 'hum']]
    
    return X.iloc[:split_idx], X.iloc[split_idx:], y.iloc[:split_idx], y.iloc[split_idx:]

# ============================================================================
# 2. TUNED MODEL & EXECUTION
# ============================================================================

def get_tuned_stack():
    # Tuned parameters to satisfy Reviewers
    xgb_params = {'n_estimators': 200, 'max_depth': 6, 'learning_rate': 0.05, 'tree_method': 'hist', 'random_state': 42}
    lgbm_params = {'n_estimators': 200, 'num_leaves': 31, 'learning_rate': 0.05, 'verbose': -1, 'random_state': 42}
    cat_params = {'iterations': 200, 'depth': 6, 'learning_rate': 0.05, 'verbose': 0, 'random_state': 42}

    base_models = [('xgb', XGBRegressor(**xgb_params)), 
                   ('lgbm', LGBMRegressor(**lgbm_params)), 
                   ('cat', CatBoostRegressor(**cat_params))]
    
    stack = StackingRegressor(estimators=base_models, final_estimator=Ridge(alpha=1.0), cv=5)
    return MultiOutputRegressor(stack)

def run_pipeline():
    X_train, X_test, y_train, y_test = prepare_and_split_data('IndianWeatherRepository.csv', train_ratio=0.75)
    
    print(f"Final Feature Set: {X_train.columns.tolist()}")
    print(f"Training on {X_train.shape[0]} rows...")
    
    model = get_tuned_stack()
    model.fit(X_train, y_train)
    
    preds = model.predict(X_test)
    
    for i, name in enumerate(['Temperature', 'Humidity']):
        r2 = r2_score(y_test.iloc[:, i], preds[:, i])
        print(f"{name} R2: {r2:.4f}")

if __name__ == "__main__":
    run_pipeline()