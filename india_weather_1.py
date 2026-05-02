import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from catboost import CatBoostRegressor
from sklearn.ensemble import HistGradientBoostingRegressor, StackingRegressor
from sklearn.multioutput import MultiOutputRegressor
from sklearn.linear_model import Ridge
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import r2_score

# ============================================================================
# 1. DATA PREPARATION (Fixes "Object" Errors & Preps Multi-Target)
# ============================================================================

def prepare_data(filepath):
    df = pd.read_csv(filepath)
    
    # Standardize column names
    temp_col = [c for c in df.columns if 'temp' in c.lower()][0]
    hum_col = [c for c in df.columns if 'hum' in c.lower()][0]
    df = df.rename(columns={temp_col: 'temp', hum_col: 'hum'})
    
    # Encode Categorical Text (Condition, Wind Direction, etc.)
    le = LabelEncoder()
    categorical_features = ['condition_text', 'wind_direction', 'moon_phase', 'region']
    for col in categorical_features:
        if col in df.columns:
            df[col] = le.fit_transform(df[col].astype(str))

    # Convert Time Strings to Numerical Hours
    time_cols = ['sunrise', 'sunset']
    for col in time_cols:
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], format='%I:%M %p', errors='coerce').dt.hour
            df[col] = df[col].fillna(df[col].mean())

    # Create Lags (Crucial for R2 > 0.85)
    for col in ['temp', 'hum']:
        df[f'{col}_lag1'] = df[col].shift(1)
        df[f'{col}_diff'] = df[col].diff()

    # Drop non-numeric / ID columns
    df = df.select_dtypes(include=[np.number]).dropna().reset_index(drop=True)
    
    X = df.drop(columns=['temp', 'hum'])
    y = df[['temp', 'hum']]
    return X, y

# ============================================================================
# 2. BUILDING THE STACK (Wrapped in MultiOutputRegressor)
# ============================================================================

def get_stacking_model():
    # Define our 4 Gradient Boosting Algorithms
    base_models = [
        ('xgb', XGBRegressor(n_estimators=200, max_depth=5, learning_rate=0.05)),
        ('lgbm', LGBMRegressor(n_estimators=200, verbose=-1)),
        ('cat', CatBoostRegressor(iterations=200, verbose=0)),
        ('hgb', HistGradientBoostingRegressor(max_iter=200))
    ]
    
    # Create the Stacking Regressor (The Level-1 Meta-Learner is Ridge)
    stack = StackingRegressor(
        estimators=base_models,
        final_estimator=Ridge()
    )
    
    # IMPORTANT: Wrap the stack in MultiOutputRegressor to handle [Temp, Hum]
    return MultiOutputRegressor(stack)

# ============================================================================
# 3. TRAINING & VISUALIZATION
# ============================================================================

def run_experiment():
    X, y = prepare_data('IndianWeatherRepository.csv')
    
    # Split (Temporal)
    split = int(0.8 * len(X))
    X_train, X_test = X[:split], X[split:]
    y_train, y_test = y[:split], y[split:]
    
    print(f"Training Multi-Output Stacked Ensemble on {X_train.shape[1]} features...")
    model = get_stacking_model()
    model.fit(X_train, y_train)
    
    # Predictions
    preds = model.predict(X_test)
    
    # Metrics
    r2_temp = r2_score(y_test['temp'], preds[:, 0])
    r2_hum = r2_score(y_test['hum'], preds[:, 1])
    
    print(f"\nTarget Achievement:\nTemp R2: {r2_temp:.4f}\nHum R2: {r2_hum:.4f}")

    # Visualization
    
    plt.figure(figsize=(15, 5))
    
    plt.subplot(1, 2, 1)
    plt.plot(y_test['temp'].values[:100], label='Actual Temp', color='blue')
    plt.plot(preds[:100, 0], label='Predicted Temp', linestyle='--', color='red')
    plt.title(f"Temperature Prediction (R2: {r2_temp:.3f})")
    plt.legend()
    
    plt.subplot(1, 2, 2)
    plt.plot(y_test['hum'].values[:100], label='Actual Hum', color='green')
    plt.plot(preds[:100, 1], label='Predicted Hum', linestyle='--', color='orange')
    plt.title(f"Humidity Prediction (R2: {r2_hum:.3f})")
    plt.legend()
    
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    run_experiment()