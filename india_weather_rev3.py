import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import re
import warnings
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import r2_score, mean_absolute_error
from sklearn.ensemble import StackingRegressor
from sklearn.multioutput import MultiOutputRegressor
from sklearn.linear_model import Ridge
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from catboost import CatBoostRegressor
from statsmodels.tsa.arima.model import ARIMA

# For LSTM
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout

warnings.filterwarnings('ignore')

# ============================================================================
# 1. DATA PREP & LEAKAGE-PROOF SPLITTING
# ============================================================================

def sanitize_column_names(df):
    df.columns = [re.sub(r'[^A-Za-z0-9_]+', '_', str(col)).strip('_') for col in df.columns]
    return df

def prepare_comparison_data(filepath):
    df = pd.read_csv(filepath, low_memory=False)
    df = df.loc[:, ~df.columns.str.contains('^Unnamed')]
    df = df.rename(columns={'timezone': 'city', 'last_updated_epoch': 'region', 
                            'wind_mph': 'datetime', 'temperature_celsius': 'temp', 'humidity': 'hum'})
    
    # Clean types
    df['datetime'] = pd.to_datetime(df['datetime'], errors='coerce')
    for col in ['temp', 'hum']:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    
    df = df.dropna(subset=['temp', 'hum', 'datetime']).sort_values(['city', 'datetime']).reset_index(drop=True)
    
    # 75/25 Split
    split_idx = int(len(df) * 0.75)
    train_df, test_df = df.iloc[:split_idx].copy(), df.iloc[split_idx:].copy()

    # Feature Engineering (Lags and Cyclical Time)
    for d in [train_df, test_df]:
        d['temp_lag1'] = d.groupby('city')['temp'].shift(1).fillna(method='bfill')
        d['hour_sin'] = np.sin(2 * np.pi * d['datetime'].dt.hour / 24)
        d['hour_cos'] = np.cos(2 * np.pi * d['datetime'].dt.hour / 24)

    # Encode and Sanitize
    full_df = pd.concat([train_df, test_df])
    for col in ['city', 'region', 'condition_text']:
        full_df[col] = pd.factorize(full_df[col].astype(str))[0]
    
    full_df = full_df.select_dtypes(exclude=['datetime', 'object']).fillna(0)
    full_df = sanitize_column_names(full_df)
    
    X = full_df.drop(columns=['temp', 'hum'])
    y = full_df[['temp', 'hum']]
    
    return X.iloc[:split_idx], X.iloc[split_idx:], y.iloc[:split_idx], y.iloc[split_idx:]

# ============================================================================
# 2. BASELINE 1: ARIMA (Classical)
# ============================================================================

def run_arima_baseline(y_train, y_test, samples=100):
    print("Running ARIMA Baseline (on sample)...")
    history = list(y_train['temp'].values[-200:])
    test_vals = y_test['temp'].values[:samples]
    predictions = []
    
    for t in range(len(test_vals)):
        model = ARIMA(history, order=(1,1,1))
        res = model.fit()
        predictions.append(res.forecast()[0])
        history.append(test_vals[t])
    
    return r2_score(test_vals, predictions)

# ============================================================================
# 3. BASELINE 2: LSTM (Deep Learning)
# ============================================================================

def run_lstm_baseline(X_train, X_test, y_train, y_test):
    print("Running LSTM Baseline...")
    scaler_x = MinMaxScaler()
    scaler_y = MinMaxScaler()
    
    # Scale and Reshape for LSTM [samples, time_steps, features]
    xt_scaled = scaler_x.fit_transform(X_train.values[:5000]) # Sampled for speed
    yt_scaled = scaler_y.fit_transform(y_train.values[:5000])
    xtest_scaled = scaler_x.transform(X_test.values[:1000])
    
    X_lstm = xt_scaled.reshape((xt_scaled.shape[0], 1, xt_scaled.shape[1]))
    
    model = Sequential([
        LSTM(50, activation='relu', input_shape=(1, xt_scaled.shape[1])),
        Dropout(0.2),
        Dense(2)
    ])
    model.compile(optimizer='adam', loss='mse')
    model.fit(X_lstm, yt_scaled, epochs=10, batch_size=32, verbose=0)
    
    preds_scaled = model.predict(xtest_scaled.reshape(1000, 1, xt_scaled.shape[1]), verbose=0)
    preds = scaler_y.inverse_transform(preds_scaled)
    
    return r2_score(y_test.values[:1000, 0], preds[:, 0])

# ============================================================================
# 4. MAIN EXPERIMENT
# ============================================================================

def run_full_comparison():
    X_train, X_test, y_train, y_test = prepare_comparison_data('IndianWeatherRepository.csv')
    
    # 1. Our Proposed Stacked Model
    print("Training Proposed Stacked Ensemble...")
    base = [('xgb', XGBRegressor(n_estimators=100)), ('lgbm', LGBMRegressor(n_estimators=100, verbose=-1))]
    stack = MultiOutputRegressor(StackingRegressor(estimators=base, final_estimator=Ridge()))
    stack.fit(X_train, y_train)
    stack_r2 = r2_score(y_test['temp'], stack.predict(X_test)[:, 0])
    
    # 2. Run Baselines
    arima_r2 = run_arima_baseline(y_train, y_test)
    lstm_r2 = run_lstm_baseline(X_train, X_test, y_train, y_test)
    
    # Final Comparison Table (For your paper!)
    print("\n" + "="*30)
    print("FINAL PERFORMANCE COMPARISON")
    print("="*30)
    print(f"{'Model':<20} | {'Temp R2 Score':<15}")
    print("-" * 38)
    print(f"{'ARIMA (Classical)':<20} | {arima_r2:.4f}")
    print(f"{'LSTM (Deep Learning)':<20} | {lstm_r2:.4f}")
    print(f"{'Proposed Stacked':<20} | {stack_r2:.4f}")
    print("="*30)

if __name__ == "__main__":
    run_full_comparison()