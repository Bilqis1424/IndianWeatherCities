"""
================================================================================
FINAL COMPLETE CODE WITH IMPROVED LSTM AND FIXED SHAP
================================================================================
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import re
import warnings
import shap
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from sklearn.ensemble import StackingRegressor
from sklearn.multioutput import MultiOutputRegressor
from sklearn.linear_model import Ridge
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from catboost import CatBoostRegressor
from statsmodels.tsa.arima.model import ARIMA
from scipy import stats
from pandas.plotting import autocorrelation_plot
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, LSTM, Dense, Dropout, Bidirectional
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.optimizers import Adam
import matplotlib.patches as mpatches

warnings.filterwarnings('ignore')
tf.random.set_seed(42)
np.random.seed(42)

# ============================================================================
# 1. DATA PREPARATION (NO LEAKAGE)
# ============================================================================

def sanitize_column_names(df):
    df.columns = [re.sub(r'[^A-Za-z0-9_]+', '_', str(col)).strip('_') for col in df.columns]
    return df

def prepare_comparison_data(filepath):
    df = pd.read_csv(filepath, low_memory=False)
    df = df.loc[:, ~df.columns.str.contains('^Unnamed')]
    
    # Rename columns
    df = df.rename(columns={
        'timezone': 'city',
        'last_updated_epoch': 'region',
        'wind_mph': 'datetime',
        'temperature_celsius': 'temp',
        'humidity': 'hum'
    })
    
    # Parse datetime
    df['datetime'] = pd.to_datetime(df['datetime'], format='%m/%d/%Y %H:%M', errors='coerce')
    for col in ['temp', 'hum']:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    
    # Drop rows with missing essential data
    df = df.dropna(subset=['temp', 'hum', 'datetime', 'city', 'region'])
    
    # Sort by city and datetime
    df = df.sort_values(['city', 'datetime']).reset_index(drop=True)
    
    # Global temporal split (75/25)
    split_idx = int(len(df) * 0.75)
    train_df = df.iloc[:split_idx].copy()
    test_df = df.iloc[split_idx:].copy()
    
    # Feature engineering (no leakage, separate per split)
    def engineer_features(d):
        d = d.copy()
        d['temp_lag1'] = d.groupby('city')['temp'].shift(1)
        d['hum_lag1'] = d.groupby('city')['hum'].shift(1)
        d['hour_sin'] = np.sin(2 * np.pi * d['datetime'].dt.hour / 24)
        d['hour_cos'] = np.cos(2 * np.pi * d['datetime'].dt.hour / 24)
        d['month_sin'] = np.sin(2 * np.pi * d['datetime'].dt.month / 12)
        d['month_cos'] = np.cos(2 * np.pi * d['datetime'].dt.month / 12)
        return d
    
    train_df = engineer_features(train_df)
    test_df = engineer_features(test_df)
    
    # Categorical encoding: fit on train only
    cat_cols = ['city', 'region', 'condition_text']
    for col in cat_cols:
        if col in train_df.columns:
            unique_vals = train_df[col].astype(str).unique()
            mapping = {val: i for i, val in enumerate(unique_vals)}
            train_df[col] = train_df[col].astype(str).map(mapping)
            test_df[col] = test_df[col].astype(str).map(mapping).fillna(-1).astype(int)
    
    # Drop rows with NaN (from lags)
    train_df = train_df.dropna().reset_index(drop=True)
    test_df = test_df.dropna().reset_index(drop=True)
    
    # Combine for numeric selection
    full_df = pd.concat([train_df, test_df], axis=0)
    full_df = sanitize_column_names(full_df)
    full_df = full_df.select_dtypes(include=[np.number])
    
    X = full_df.drop(columns=['temp', 'hum'])
    y = full_df[['temp', 'hum']]
    
    return (X.iloc[:len(train_df)], X.iloc[len(train_df):],
            y.iloc[:len(train_df)], y.iloc[len(train_df):])


# ============================================================================
# 2. ARIMA BASELINE
# ============================================================================
def run_arima_baseline(y_train, y_test, variable='temp', samples=100):
    print(f"Running ARIMA baseline for {variable}...")
    history = list(y_train[variable].values[-200:])
    test_vals = y_test[variable].values[:samples]
    predictions = []
    for t in range(len(test_vals)):
        model = ARIMA(history, order=(5,1,0))  # improved order
        res = model.fit()
        predictions.append(res.forecast()[0])
        history.append(test_vals[t])
    r2 = r2_score(test_vals, predictions)
    print(f"  ARIMA {variable} R²: {r2:.4f}")
    return np.array(predictions), r2


# ============================================================================
# 3. IMPROVED LSTM BASELINE (Bidirectional, more layers, better scaling)
# ============================================================================
def run_lstm_baseline(X_train, X_test, y_train, y_test, n_steps=48):
    print("Running LSTM baseline (improved architecture)...")
    # Scale the target variables
    scaler = MinMaxScaler()
    train_data = np.column_stack([y_train['temp'].values, y_train['hum'].values])
    test_data = np.column_stack([y_test['temp'].values, y_test['hum'].values])
    
    # Scale the data
    train_scaled = scaler.fit_transform(train_data)
    test_scaled = scaler.transform(test_data)
    
    # Create sequences
    X_seq, y_seq = [], []
    for i in range(len(train_scaled) - n_steps):
        X_seq.append(train_scaled[i:i+n_steps, :])  # shape (n_steps, 2)
        y_seq.append(train_scaled[i+n_steps, :])    # shape (2,)
    X_seq = np.array(X_seq)
    y_seq = np.array(y_seq)
    
    if len(X_seq) < 50:
        print("  Not enough data for LSTM sequences.")
        return np.array([]), np.array([]), 0.0, 0.0
    
    # Build improved LSTM model
    inputs = Input(shape=(n_steps, 2))
    x = Bidirectional(LSTM(128, activation='relu', return_sequences=True))(inputs)
    x = Dropout(0.3)(x)
    x = Bidirectional(LSTM(64, activation='relu', return_sequences=True))(x)
    x = Dropout(0.3)(x)
    x = LSTM(32, activation='relu')(x)
    x = Dropout(0.2)(x)
    x = Dense(16, activation='relu')(x)
    outputs = Dense(2)(x)
    model = Model(inputs=inputs, outputs=outputs)
    model.compile(optimizer=Adam(learning_rate=0.001), loss='mse', metrics=['mae'])
    
    # Train with validation and learning rate reduction
    val_size = int(0.2 * len(X_seq))
    X_val, y_val = X_seq[-val_size:], y_seq[-val_size:]
    X_tr, y_tr = X_seq[:-val_size], y_seq[:-val_size]
    early_stop = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, min_lr=0.00001)
    model.fit(X_tr, y_tr, validation_data=(X_val, y_val),
              epochs=100, batch_size=64, callbacks=[early_stop, reduce_lr], verbose=0)
    
    # Predict on test
    test_preds = []
    for i in range(len(test_scaled) - n_steps):
        seq = test_scaled[i:i+n_steps].reshape(1, n_steps, 2)
        pred_scaled = model.predict(seq, verbose=0)[0]
        test_preds.append(pred_scaled)
    if not test_preds:
        print("  No test predictions generated.")
        return np.array([]), np.array([]), 0.0, 0.0
    
    y_pred_scaled = np.array(test_preds)
    y_pred = scaler.inverse_transform(y_pred_scaled)
    y_true_aligned = y_test.iloc[n_steps:].reset_index(drop=True)
    if len(y_true_aligned) != len(y_pred):
        print("  Length mismatch.")
        return np.array([]), np.array([]), 0.0, 0.0
    
    r2_temp = r2_score(y_true_aligned['temp'], y_pred[:,0])
    r2_hum = r2_score(y_true_aligned['hum'], y_pred[:,1])
    print(f"  LSTM -> Temp R²: {r2_temp:.4f}, Hum R²: {r2_hum:.4f}")
    return y_pred[:,0], y_pred[:,1], r2_temp, r2_hum


# ============================================================================
# 4. STACKED ENSEMBLE
# ============================================================================
def run_stacked_ensemble(X_train, X_test, y_train, y_test):
    print("Training Proposed Stacked Ensemble...")
    base_models = [
        ('xgb', XGBRegressor(n_estimators=300, max_depth=6, learning_rate=0.03,
                             random_state=42, verbosity=0)),
        ('lgbm', LGBMRegressor(n_estimators=300, num_leaves=31, learning_rate=0.03,
                               random_state=42, verbose=-1)),
        ('cat', CatBoostRegressor(iterations=300, depth=6, learning_rate=0.03,
                                  random_seed=42, verbose=0))
    ]
    stack = StackingRegressor(estimators=base_models, final_estimator=Ridge(alpha=1.0), cv=5)
    multi_output_stack = MultiOutputRegressor(stack, n_jobs=-1)
    multi_output_stack.fit(X_train, y_train)
    y_pred = multi_output_stack.predict(X_test)
    r2_temp = r2_score(y_test['temp'], y_pred[:,0])
    r2_hum = r2_score(y_test['hum'], y_pred[:,1])
    print(f"  Stacked Ensemble -> Temp R²: {r2_temp:.4f}, Hum R²: {r2_hum:.4f}")
    return y_pred[:,0], y_pred[:,1], r2_temp, r2_hum, multi_output_stack


# ============================================================================
# 5. FIXED SHAP ANALYSIS (extract XGBoost from stacking)
# ============================================================================
def shap_analysis(stack_model, X_sample, feature_names):
    # stack_model is MultiOutputRegressor; first output (temp) contains a StackingRegressor
    stacking_for_temp = stack_model.estimators_[0]  # StackingRegressor
    # Extract the XGBoost base learner from the stacking regressor
    xgb_model = None
    for name, estimator in stacking_for_temp.estimators_:
        if name == 'xgb':
            xgb_model = estimator
            break
    if xgb_model is None:
        print("Could not extract XGBoost model from stacking. Using first base learner.")
        xgb_model = stacking_for_temp.estimators_[0][1]
    
    explainer = shap.TreeExplainer(xgb_model)
    shap_values = explainer.shap_values(X_sample)
    plt.figure(figsize=(10,6))
    shap.summary_plot(shap_values, X_sample, feature_names=feature_names, show=False)
    plt.title("SHAP Feature Importance (Stacked Ensemble – XGBoost Component)")
    plt.tight_layout()
    plt.savefig('shap_summary.png', dpi=150)
    plt.close()
    mean_abs = np.abs(shap_values).mean(axis=0)
    imp = pd.DataFrame({'feature': feature_names, 'importance': mean_abs})
    imp = imp.sort_values('importance', ascending=False).head(15)
    plt.figure(figsize=(10,5))
    plt.barh(imp['feature'], imp['importance'])
    plt.xlabel('Mean |SHAP value|')
    plt.title('Top 15 Features (XGBoost from Stacking)')
    plt.gca().invert_yaxis()
    plt.tight_layout()
    plt.savefig('shap_bar.png', dpi=150)
    plt.close()
    print("SHAP plots saved.")
    return imp


# ============================================================================
# 6. RESIDUAL DIAGNOSTICS
# ============================================================================
def plot_residuals(y_true, y_pred, target_name):
    residuals = y_true - y_pred
    fig, axes = plt.subplots(2,2, figsize=(12,10))
    fig.suptitle(f'Residual Diagnostics: {target_name}')
    axes[0,0].scatter(y_pred, residuals, alpha=0.5, s=10)
    axes[0,0].axhline(0, color='r', linestyle='--')
    axes[0,0].set_xlabel('Predicted'); axes[0,0].set_ylabel('Residuals')
    axes[0,0].set_title('Residuals vs Predicted')
    stats.probplot(residuals, dist="norm", plot=axes[0,1])
    axes[0,1].set_title('Q-Q Plot')
    axes[1,0].hist(residuals, bins=50, edgecolor='black', alpha=0.7)
    axes[1,0].set_xlabel('Residuals'); axes[1,0].set_ylabel('Frequency')
    axes[1,0].set_title('Distribution')
    autocorrelation_plot(pd.Series(residuals), ax=axes[1,1])
    axes[1,1].set_title('Autocorrelation')
    axes[1,1].set_xlim([0,40])
    plt.tight_layout()
    plt.savefig(f'residuals_{target_name}.png', dpi=150)
    plt.close()
    print(f"Residual plot saved: residuals_{target_name}.png")


# ============================================================================
# 7. MULTI-STEP FORECASTING (STACKED)
# ============================================================================
def recursive_forecast(model, initial_features, n_steps=48, feature_names=None):
    preds = []
    curr = initial_features.copy()
    if feature_names and 'temp_lag1' in feature_names:
        temp_lag_idx = feature_names.index('temp_lag1')
        hum_lag_idx = feature_names.index('hum_lag1')
    else:
        temp_lag_idx = hum_lag_idx = None
    for _ in range(n_steps):
        next_pred = model.predict(curr.reshape(1, -1))[0]
        preds.append(next_pred)
        if temp_lag_idx is not None:
            curr[temp_lag_idx] = next_pred[0]
            curr[hum_lag_idx] = next_pred[1]
        else:
            break
    return np.array(preds)

def evaluate_multi_step(model, X_test, y_test, horizons=[1,6,12,24,48]):
    feat_names = X_test.columns.tolist()
    results = {'temp': {}, 'hum': {}}
    for h in horizons:
        temp_errors, hum_errors = [], []
        max_start = len(X_test) - h
        if max_start <= 0:
            continue
        for i in range(max_start):
            init = X_test.iloc[i].values
            true_temp = y_test.iloc[i+1:i+h+1]['temp'].values
            true_hum = y_test.iloc[i+1:i+h+1]['hum'].values
            pred = recursive_forecast(model, init, h, feat_names)
            if len(pred) != h:
                continue
            temp_errors.append(np.mean(np.abs(true_temp - pred[:,0])))
            hum_errors.append(np.mean(np.abs(true_hum - pred[:,1])))
        if temp_errors:
            results['temp'][h] = np.mean(temp_errors)
            results['hum'][h] = np.mean(hum_errors)
    print("\nMulti-step MAE (Stacked Ensemble):")
    for h in sorted(results['temp'].keys()):
        print(f"  Horizon {h}h: Temp MAE = {results['temp'][h]:.4f}, Hum MAE = {results['hum'][h]:.4f}")
    return results


# ============================================================================
# 8. MAIN EXPERIMENT
# ============================================================================
def run_full_comparison():
    print("Loading and preparing data (no leakage)...")
    X_train, X_test, y_train, y_test = prepare_comparison_data('IndianWeatherRepository.csv')
    print(f"Training: {X_train.shape}, Test: {X_test.shape}")
    
    # ---- Stacked Ensemble ----
    stack_pred_temp, stack_pred_hum, stack_r2_temp, stack_r2_hum, stack_model = run_stacked_ensemble(
        X_train, X_test, y_train, y_test)
    
    # ---- ARIMA ----
    arima_pred_temp, arima_r2_temp = run_arima_baseline(y_train, y_test, 'temp', 100)
    arima_pred_hum, arima_r2_hum = run_arima_baseline(y_train, y_test, 'hum', 100)
    
    # ---- Improved LSTM ----
    lstm_pred_temp, lstm_pred_hum, lstm_r2_temp, lstm_r2_hum = run_lstm_baseline(
        X_train, X_test, y_train, y_test, n_steps=48)
    
    # ---- 1. Bar charts for both variables (UPDATED: title, colors, value labels, fonts, spacing, legend) ----
    models = ['ARIMA', 'LSTM', 'Proposed Stacked']
    temp_r2 = [arima_r2_temp, lstm_r2_temp, stack_r2_temp]
    hum_r2 = [arima_r2_hum, lstm_r2_hum, stack_r2_hum]
    
    # MAE / RMSE for temperature
    mae_temp = [
        mean_absolute_error(y_test['temp'][:len(arima_pred_temp)], arima_pred_temp) if len(arima_pred_temp)>0 else np.nan,
        mean_absolute_error(y_test['temp'][:len(lstm_pred_temp)], lstm_pred_temp) if len(lstm_pred_temp)>0 else np.nan,
        mean_absolute_error(y_test['temp'], stack_pred_temp)
    ]
    rmse_temp = [
        np.sqrt(mean_squared_error(y_test['temp'][:len(arima_pred_temp)], arima_pred_temp)) if len(arima_pred_temp)>0 else np.nan,
        np.sqrt(mean_squared_error(y_test['temp'][:len(lstm_pred_temp)], lstm_pred_temp)) if len(lstm_pred_temp)>0 else np.nan,
        np.sqrt(mean_squared_error(y_test['temp'], stack_pred_temp))
    ]
    # MAE / RMSE for humidity
    mae_hum = [
        mean_absolute_error(y_test['hum'][:len(arima_pred_hum)], arima_pred_hum) if len(arima_pred_hum)>0 else np.nan,
        mean_absolute_error(y_test['hum'][:len(lstm_pred_hum)], lstm_pred_hum) if len(lstm_pred_hum)>0 else np.nan,
        mean_absolute_error(y_test['hum'], stack_pred_hum)
    ]
    rmse_hum = [
        np.sqrt(mean_squared_error(y_test['hum'][:len(arima_pred_hum)], arima_pred_hum)) if len(arima_pred_hum)>0 else np.nan,
        np.sqrt(mean_squared_error(y_test['hum'][:len(lstm_pred_hum)], lstm_pred_hum)) if len(lstm_pred_hum)>0 else np.nan,
        np.sqrt(mean_squared_error(y_test['hum'], stack_pred_hum))
    ]
    
    # Define custom colors for models
    model_colors = {'ARIMA': 'grey', 'LSTM': 'orange', 'Proposed Stacked': 'darkgreen'}
    palette = [model_colors[m] for m in models]
    
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    # Shortened main title
    fig.suptitle('Model Performance Comparison for Temperature and Relative Humidity Forecasting',
                 fontsize=16, y=0.98)
    # Adjust spacing to prevent clutter
    plt.subplots_adjust(top=0.92)  # extra padding for main title
    plt.tight_layout(pad=3.0)      # increased padding between subplots
    
    # Helper function to add value labels with consistent decimals and larger font
    def add_value_labels(ax, fmt='{:.4f}', fontsize=9):
        for container in ax.containers:
            ax.bar_label(container, fmt=fmt, label_type='edge', fontsize=fontsize, padding=3)
    
    # Customize font sizes for all subplots
    for ax_row in axes:
        for ax in ax_row:
            ax.title.set_fontsize(14)
            ax.xaxis.label.set_fontsize(12)
            ax.yaxis.label.set_fontsize(12)
            ax.tick_params(labelsize=10)
    
    # Temperature R²
    sns.barplot(x=models, y=temp_r2, ax=axes[0,0], palette=palette)
    axes[0,0].set_title('Temperature R²')
    axes[0,0].set_ylabel('R² Score')
    add_value_labels(axes[0,0], fmt='{:.4f}')
    
    # Temperature MAE
    sns.barplot(x=models, y=mae_temp, ax=axes[0,1], palette=palette)
    axes[0,1].set_title('Temperature MAE')
    axes[0,1].set_ylabel('Mean Absolute Error')
    add_value_labels(axes[0,1], fmt='{:.4f}')
    
    # Temperature RMSE
    sns.barplot(x=models, y=rmse_temp, ax=axes[0,2], palette=palette)
    axes[0,2].set_title('Temperature RMSE')
    axes[0,2].set_ylabel('Root Mean Square Error')
    add_value_labels(axes[0,2], fmt='{:.4f}')
    
    # Humidity R²
    sns.barplot(x=models, y=hum_r2, ax=axes[1,0], palette=palette)
    axes[1,0].set_title('Humidity R²')
    axes[1,0].set_ylabel('R² Score')
    add_value_labels(axes[1,0], fmt='{:.4f}')
    
    # Humidity MAE
    sns.barplot(x=models, y=mae_hum, ax=axes[1,1], palette=palette)
    axes[1,1].set_title('Humidity MAE')
    axes[1,1].set_ylabel('Mean Absolute Error')
    add_value_labels(axes[1,1], fmt='{:.4f}')
    
    # Humidity RMSE
    sns.barplot(x=models, y=rmse_hum, ax=axes[1,2], palette=palette)
    axes[1,2].set_title('Humidity RMSE')
    axes[1,2].set_ylabel('Root Mean Square Error')
    add_value_labels(axes[1,2], fmt='{:.4f}')
    
    # Add legend
    legend_patches = [mpatches.Patch(color=model_colors[m], label=m) for m in models]
    fig.legend(handles=legend_patches, loc='lower center', ncol=3,
               fontsize=12, bbox_to_anchor=(0.5, -0.02), frameon=False)
    
    # Save and show
    plt.savefig('baseline_comparison.png', dpi=150, bbox_inches='tight')
    plt.show()
    
    # ---- 2. Scatter plots ----
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    fig.suptitle('Actual vs Predicted Scatter Plots', fontsize=16)
    # Temperature
    axes[0,0].scatter(y_test['temp'][:len(arima_pred_temp)], arima_pred_temp, alpha=0.5, color='blue')
    axes[0,0].plot([y_test['temp'].min(), y_test['temp'].max()],
                   [y_test['temp'].min(), y_test['temp'].max()], 'r--')
    axes[0,0].set_title(f'ARIMA (Temp) R²={arima_r2_temp:.3f}')
    axes[0,1].scatter(y_test['temp'][:len(lstm_pred_temp)], lstm_pred_temp, alpha=0.5, color='green')
    axes[0,1].plot([y_test['temp'].min(), y_test['temp'].max()],
                   [y_test['temp'].min(), y_test['temp'].max()], 'r--')
    axes[0,1].set_title(f'LSTM (Temp) R²={lstm_r2_temp:.3f}')
    axes[0,2].scatter(y_test['temp'], stack_pred_temp, alpha=0.5, color='purple')
    axes[0,2].plot([y_test['temp'].min(), y_test['temp'].max()],
                   [y_test['temp'].min(), y_test['temp'].max()], 'r--')
    axes[0,2].set_title(f'Stacked (Temp) R²={stack_r2_temp:.3f}')
    # Humidity
    axes[1,0].scatter(y_test['hum'][:len(arima_pred_hum)], arima_pred_hum, alpha=0.5, color='blue')
    axes[1,0].plot([y_test['hum'].min(), y_test['hum'].max()],
                   [y_test['hum'].min(), y_test['hum'].max()], 'r--')
    axes[1,0].set_title(f'ARIMA (Hum) R²={arima_r2_hum:.3f}')
    axes[1,1].scatter(y_test['hum'][:len(lstm_pred_hum)], lstm_pred_hum, alpha=0.5, color='green')
    axes[1,1].plot([y_test['hum'].min(), y_test['hum'].max()],
                   [y_test['hum'].min(), y_test['hum'].max()], 'r--')
    axes[1,1].set_title(f'LSTM (Hum) R²={lstm_r2_hum:.3f}')
    axes[1,2].scatter(y_test['hum'], stack_pred_hum, alpha=0.5, color='purple')
    axes[1,2].plot([y_test['hum'].min(), y_test['hum'].max()],
                   [y_test['hum'].min(), y_test['hum'].max()], 'r--')
    axes[1,2].set_title(f'Stacked (Hum) R²={stack_r2_hum:.3f}')
    plt.tight_layout()
    plt.savefig('scatter_plots.png', dpi=150)
    plt.show()
    
    # ---- 3. Time series plots (48 hours) ----
    n_plot = min(48, len(y_test))
    fig, axes = plt.subplots(1, 2, figsize=(16, 5))
    axes[0].plot(y_test['temp'].values[:n_plot], label='Actual', color='black', linewidth=2)
    axes[0].plot(stack_pred_temp[:n_plot], label='Stacked Forecast', color='red', linestyle='--')
    axes[0].set_title('48‑Hour Temperature Forecast')
    axes[0].set_xlabel('Hours'); axes[0].set_ylabel('°C'); axes[0].legend(); axes[0].grid(True, alpha=0.3)
    axes[1].plot(y_test['hum'].values[:n_plot], label='Actual', color='black', linewidth=2)
    axes[1].plot(stack_pred_hum[:n_plot], label='Stacked Forecast', color='blue', linestyle='--')
    axes[1].set_title('48‑Hour Humidity Forecast')
    axes[1].set_xlabel('Hours'); axes[1].set_ylabel('%'); axes[1].legend(); axes[1].grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('48hr_forecast.png', dpi=150)
    plt.show()
    
    # ---- 4. SHAP for stacked model (fixed extraction) ----
    print("Generating SHAP explanations...")
    X_sample = X_test.sample(min(100, len(X_test)), random_state=42)
    shap_importance = shap_analysis(stack_model, X_sample, X_test.columns.tolist())
    shap_importance.to_csv('shap_importance.csv', index=False)
    
    # ---- 5. Residual diagnostics ----
    plot_residuals(y_test['temp'], stack_pred_temp, 'temperature')
    plot_residuals(y_test['hum'], stack_pred_hum, 'humidity')
    
    # ---- 6. Multi-step forecasting ----
    multi_step = evaluate_multi_step(stack_model, X_test, y_test, horizons=[1,6,12,24,48])
    if multi_step['temp']:
        df_multi = pd.DataFrame({'Horizon': list(multi_step['temp'].keys()),
                                 'Temp_MAE': list(multi_step['temp'].values()),
                                 'Hum_MAE': list(multi_step['hum'].values())})
        df_multi.to_csv('multi_step_results.csv', index=False)
    
    # ---- 7. Final performance table ----
    print("\n" + "="*60)
    print("FINAL PERFORMANCE COMPARISON")
    print("="*60)
    print(f"{'Model':<20} | {'Temp R²':<10} | {'Hum R²':<10} | {'Temp MAE':<10} | {'Hum MAE':<10}")
    print("-"*68)
    print(f"{'ARIMA':<20} | {arima_r2_temp:.4f}    | {arima_r2_hum:.4f}    | {mae_temp[0]:.4f}    | {mae_hum[0]:.4f}")
    print(f"{'LSTM':<20} | {lstm_r2_temp:.4f}    | {lstm_r2_hum:.4f}    | {mae_temp[1]:.4f}    | {mae_hum[1]:.4f}")
    print(f"{'Proposed Stacked':<20} | {stack_r2_temp:.4f}    | {stack_r2_hum:.4f}    | {mae_temp[2]:.4f}    | {mae_hum[2]:.4f}")
    print("="*60)
    
    summary = pd.DataFrame({
        'Model': models,
        'Temp_R2': temp_r2,
        'Hum_R2': hum_r2,
        'Temp_MAE': mae_temp,
        'Hum_MAE': mae_hum,
        'Temp_RMSE': rmse_temp,
        'Hum_RMSE': rmse_hum
    })
    summary.to_csv('performance_summary.csv', index=False)
    print("\nAll outputs saved.")


if __name__ == "__main__":
    run_full_comparison()
