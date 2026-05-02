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
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error


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
    # Increase font sizes slightly
    plt.rcParams.update({'font.size': 12, 'axes.titlesize': 14, 'axes.labelsize': 12,
                         'xtick.labelsize': 11, 'ytick.labelsize': 11})

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

    # ----- Metrics Calculation -----
    # Temperature
    r2_temp = r2_score(y_test['temp'], preds[:, 0])
    mae_temp = mean_absolute_error(y_test['temp'], preds[:, 0])
    rmse_temp = np.sqrt(mean_squared_error(y_test['temp'], preds[:, 0]))

    # Humidity
    r2_hum = r2_score(y_test['hum'], preds[:, 1])
    mae_hum = mean_absolute_error(y_test['hum'], preds[:, 1])
    rmse_hum = np.sqrt(mean_squared_error(y_test['hum'], preds[:, 1]))

    # Print metrics
    print("\n" + "=" * 50)
    print("EVALUATION METRICS")
    print("=" * 50)
    print(f"Temperature  -> R²: {r2_temp:.4f}, MAE: {mae_temp:.4f}, RMSE: {rmse_temp:.4f}")
    print(f"Humidity     -> R²: {r2_hum:.4f}, MAE: {mae_hum:.4f}, RMSE: {rmse_hum:.4f}")
    print("=" * 50 + "\n")

    # ===== FIGURE 1: Temperature =====
    fig1, axes1 = plt.subplots(1, 2, figsize=(14, 5))
    fig1.suptitle('Temperature: Model Performance', fontsize=16)

    # Left: Time series (first 200 points)
    n_points = 200
    ax = axes1[0]
    ax.plot(y_test['temp'].values[:n_points], label='Actual', color='blue', linewidth=1.5)
    ax.plot(preds[:n_points, 0], label='Predicted', linestyle='--', color='red', linewidth=1.0)
    ax.set_xlabel('Time Step')
    ax.set_ylabel('Temperature')
    ax.set_title(f'Time Series (first {n_points} points)')
    ax.legend()
    ax.grid(True, linestyle=':', alpha=0.6)

    # Right: Scatter (Observed vs Predicted)
    ax = axes1[1]
    ax.scatter(y_test['temp'], preds[:, 0], alpha=0.5, edgecolors='k', linewidth=0.5)
    min_val = min(y_test['temp'].min(), preds[:, 0].min())
    max_val = max(y_test['temp'].max(), preds[:, 0].max())
    ax.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=0.8, label='Ideal')
    ax.set_xlabel('Observed Temperature')
    ax.set_ylabel('Predicted Temperature')
    ax.set_title('Observed vs Predicted')
    ax.legend()
    ax.grid(True, linestyle=':', alpha=0.6)
    ax.set_aspect('equal', adjustable='box')
    plt.tight_layout()

    # ===== FIGURE 2: Humidity =====
    fig2, axes2 = plt.subplots(1, 2, figsize=(14, 5))
    fig2.suptitle('Relative Humidity: Model Performance', fontsize=16)

    # Left: Time series (first 200 points)
    ax = axes2[0]
    ax.plot(y_test['hum'].values[:n_points], label='Actual', color='green', linewidth=1.5)
    ax.plot(preds[:n_points, 1], label='Predicted', linestyle='--', color='orange', linewidth=1.0)
    ax.set_xlabel('Time Step')
    ax.set_ylabel('Humidity')
    ax.set_title(f'Time Series (first {n_points} points)')
    ax.legend()
    ax.grid(True, linestyle=':', alpha=0.6)

    # Right: Scatter (Observed vs Predicted)
    ax = axes2[1]
    ax.scatter(y_test['hum'], preds[:, 1], alpha=0.5, edgecolors='k', linewidth=0.5)
    min_val = min(y_test['hum'].min(), preds[:, 1].min())
    max_val = max(y_test['hum'].max(), preds[:, 1].max())
    ax.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=0.8, label='Ideal')
    ax.set_xlabel('Observed Humidity')
    ax.set_ylabel('Predicted Humidity')
    ax.set_title('Observed vs Predicted')
    ax.legend()
    ax.grid(True, linestyle=':', alpha=0.6)
    ax.set_aspect('equal', adjustable='box')
    plt.tight_layout()

    # ===== FIGURE 3: Summary Bar Chart (Metrics) =====
    fig3, ax = plt.subplots(figsize=(8, 6))
    fig3.suptitle('Stacked Ensemble Performance: Temperature vs Humidity', fontsize=16)

    metrics = ['R²', 'MAE', 'RMSE']
    x = np.arange(len(metrics))  # label locations
    width = 0.35  # width of bars

    # Temperature values
    temp_vals = [r2_temp, mae_temp, rmse_temp]
    # Humidity values
    hum_vals = [r2_hum, mae_hum, rmse_hum]

    bars1 = ax.bar(x - width / 2, temp_vals, width, label='Temperature', color='blue', alpha=0.8)
    bars2 = ax.bar(x + width / 2, hum_vals, width, label='Humidity', color='green', alpha=0.8)

    # Add value labels on top of bars
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            ax.annotate(f'{height:.4f}',
                        xy=(bar.get_x() + bar.get_width() / 2, height),
                        xytext=(0, 3),  # 3 points vertical offset
                        textcoords="offset points",
                        ha='center', va='bottom', fontsize=10)

    ax.set_xlabel('Metric')
    ax.set_ylabel('Score / Error Magnitude')
    ax.set_xticks(x)
    ax.set_xticklabels(metrics)
    ax.legend()
    ax.grid(True, axis='y', linestyle=':', alpha=0.6)  # thin dashed grid on y-axis

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    run_experiment()