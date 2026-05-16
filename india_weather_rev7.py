"""
================================================================================
FINAL CODE – ARIMA, LSTM, STACKED ENSEMBLE ONLY
With improved bar chart, 48h forecasts & residual diagnostics for all 3 models
================================================================================
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import re
import warnings
import shap
from sklearn.preprocessing import MinMaxScaler
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
    
    df = df.rename(columns={
        'timezone': 'city',
        'last_updated_epoch': 'region',
        'wind_mph': 'datetime',
        'temperature_celsius': 'temp',
        'humidity': 'hum'
    })
    
    df['datetime'] = pd.to_datetime(df['datetime'], format='%m/%d/%Y %H:%M', errors='coerce')
    for col in ['temp', 'hum']:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    
    df = df.dropna(subset=['temp', 'hum', 'datetime', 'city', 'region'])
    df = df.sort_values(['city', 'datetime']).reset_index(drop=True)
    
    split_idx = int(len(df) * 0.75)
    train_df = df.iloc[:split_idx].copy()
    test_df = df.iloc[split_idx:].copy()
    
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
    
    cat_cols = ['city', 'region', 'condition_text']
    for col in cat_cols:
        if col in train_df.columns:
            unique_vals = train_df[col].astype(str).unique()
            mapping = {val: i for i, val in enumerate(unique_vals)}
            train_df[col] = train_df[col].astype(str).map(mapping)
            test_df[col] = test_df[col].astype(str).map(mapping).fillna(-1).astype(int)
    
    train_df = train_df.dropna().reset_index(drop=True)
    test_df = test_df.dropna().reset_index(drop=True)
    
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
        model = ARIMA(history, order=(5,1,0))
        res = model.fit()
        predictions.append(res.forecast()[0])
        history.append(test_vals[t])
    r2 = r2_score(test_vals, predictions)
    print(f"  ARIMA {variable} R²: {r2:.4f}")
    return np.array(predictions), r2


# ============================================================================
# 3. IMPROVED LSTM BASELINE
# ============================================================================
def run_lstm_baseline(X_train, X_test, y_train, y_test, n_steps=48):
    print("Running LSTM baseline (improved architecture)...")
    scaler = MinMaxScaler()
    train_data = np.column_stack([y_train['temp'].values, y_train['hum'].values])
    test_data = np.column_stack([y_test['temp'].values, y_test['hum'].values])
    
    train_scaled = scaler.fit_transform(train_data)
    test_scaled = scaler.transform(test_data)
    
    X_seq, y_seq = [], []
    for i in range(len(train_scaled) - n_steps):
        X_seq.append(train_scaled[i:i+n_steps, :])
        y_seq.append(train_scaled[i+n_steps, :])
    X_seq = np.array(X_seq)
    y_seq = np.array(y_seq)
    
    if len(X_seq) < 50:
        print("  Not enough data for LSTM sequences.")
        return np.array([]), np.array([]), 0.0, 0.0
    
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
    
    val_size = int(0.2 * len(X_seq))
    X_val, y_val = X_seq[-val_size:], y_seq[-val_size:]
    X_tr, y_tr = X_seq[:-val_size], y_seq[:-val_size]
    early_stop = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, min_lr=0.00001)
    model.fit(X_tr, y_tr, validation_data=(X_val, y_val),
              epochs=100, batch_size=64, callbacks=[early_stop, reduce_lr], verbose=0)
    
    test_preds = []
    for i in range(len(test_scaled) - n_steps):
        seq = test_scaled[i:i+n_steps].reshape(1, n_steps, 2)
        pred_scaled = model.predict(seq, verbose=0)[0]
        test_preds.append(pred_scaled)
    if not test_preds:
        return np.array([]), np.array([]), 0.0, 0.0
    
    y_pred_scaled = np.array(test_preds)
    y_pred = scaler.inverse_transform(y_pred_scaled)
    y_true_aligned = y_test.iloc[n_steps:].reset_index(drop=True)
    if len(y_true_aligned) != len(y_pred):
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
# 5. SHAP ANALYSIS (extract XGBoost from stacking)
# ============================================================================
def shap_analysis(stack_model, X_sample, feature_names):
    stacking_for_temp = stack_model.estimators_[0]
    xgb_model = None
    for name, estimator in stacking_for_temp.estimators_:
        if name == 'xgb':
            xgb_model = estimator
            break
    if xgb_model is None:
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
    return imp


# ============================================================================
# 6. RESIDUAL DIAGNOSTICS (for a single model)
# ============================================================================
def plot_residuals_single(y_true, y_pred, model_name, target_name, axs=None):
    residuals = y_true - y_pred
    if axs is None:
        fig, axs = plt.subplots(2,2, figsize=(10,8))
        fig.suptitle(f'{model_name} - {target_name} Residuals')
        single = True
    else:
        single = False
    
    axs[0,0].scatter(y_pred, residuals, alpha=0.5, s=10)
    axs[0,0].axhline(0, color='r', linestyle='--')
    axs[0,0].set_xlabel('Predicted'); axs[0,0].set_ylabel('Residuals')
    axs[0,0].set_title('Residuals vs Predicted')
    
    stats.probplot(residuals, dist="norm", plot=axs[0,1])
    axs[0,1].set_title('Q-Q Plot')
    
    axs[1,0].hist(residuals, bins=30, edgecolor='black', alpha=0.7)
    axs[1,0].set_xlabel('Residuals'); axs[1,0].set_ylabel('Frequency')
    axs[1,0].set_title('Distribution')
    
    autocorrelation_plot(pd.Series(residuals), ax=axs[1,1])
    axs[1,1].set_title('Autocorrelation')
    axs[1,1].set_xlim([0,40])
    
    if single:
        plt.tight_layout()
        plt.savefig(f'residuals_{model_name}_{target_name}.png', dpi=150)
        plt.close()


# ============================================================================
# 7. MULTI-STEP FORECASTING (STACKED only)
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
    print("Loading and preparing data...")
    X_train, X_test, y_train, y_test = prepare_comparison_data('IndianWeatherRepository.csv')
    print(f"Training: {X_train.shape}, Test: {X_test.shape}")
    
    # ----- Train all three models -----
    # Stacked Ensemble
    stack_pred_temp, stack_pred_hum, stack_r2_temp, stack_r2_hum, stack_model = run_stacked_ensemble(
        X_train, X_test, y_train, y_test)
    
    # ARIMA
    arima_pred_temp, arima_r2_temp = run_arima_baseline(y_train, y_test, 'temp', 100)
    arima_pred_hum, arima_r2_hum = run_arima_baseline(y_train, y_test, 'hum', 100)
    
    # LSTM
    lstm_pred_temp, lstm_pred_hum, lstm_r2_temp, lstm_r2_hum = run_lstm_baseline(
        X_train, X_test, y_train, y_test, n_steps=48)
    
    # ----- Collect predictions and names -----
    model_names = ['ARIMA', 'LSTM', 'Stacked']
    temp_preds = [arima_pred_temp, lstm_pred_temp, stack_pred_temp]
    hum_preds  = [arima_pred_hum,  lstm_pred_hum,  stack_pred_hum]
    temp_r2_list = [arima_r2_temp, lstm_r2_temp, stack_r2_temp]
    hum_r2_list  = [arima_r2_hum,  lstm_r2_hum,  stack_r2_hum]
    colors = ['grey', 'orange', 'darkgreen']
    
    # ----- 48‑hour forecast comparison (all three models) -----
    n_plot = min(48, len(y_test))
    # Find minimum prediction length among models
    min_len_temp = min([len(p) for p in temp_preds if len(p)>0])
    min_len_hum = min([len(p) for p in hum_preds if len(p)>0])
    n_plot_temp = min(n_plot, min_len_temp)
    n_plot_hum = min(n_plot, min_len_hum)
    
    fig, axes = plt.subplots(2, 3, figsize=(18, 8))
    fig.suptitle('48‑Hour Forecast Comparison (Actual vs Predicted)', fontsize=16)
    
    # Temperature row (row 0)
    for i, (name, preds, col) in enumerate(zip(model_names, temp_preds, colors)):
        if len(preds) >= n_plot_temp:
            axes[0,i].plot(y_test['temp'].values[:n_plot_temp], label='Actual', color='black', linewidth=1.5)
            axes[0,i].plot(preds[:n_plot_temp], label=f'{name}', color=col, linestyle='--', linewidth=1.5)
            axes[0,i].set_title(f'{name} (Temperature)', fontsize=10)
        else:
            axes[0,i].text(0.5, 0.5, 'No predictions', ha='center', transform=axes[0,i].transAxes)
            axes[0,i].set_title(f'{name} (insufficient data)', fontsize=10)
        axes[0,i].grid(True, alpha=0.3)
        axes[0,i].legend(fontsize=8)
    
    # Humidity row (row 1)
    for i, (name, preds, col) in enumerate(zip(model_names, hum_preds, colors)):
        if len(preds) >= n_plot_hum:
            axes[1,i].plot(y_test['hum'].values[:n_plot_hum], label='Actual', color='black', linewidth=1.5)
            axes[1,i].plot(preds[:n_plot_hum], label=f'{name}', color=col, linestyle='--', linewidth=1.5)
            axes[1,i].set_title(f'{name} (Humidity)', fontsize=10)
        else:
            axes[1,i].text(0.5, 0.5, 'No predictions', ha='center', transform=axes[1,i].transAxes)
            axes[1,i].set_title(f'{name} (insufficient data)', fontsize=10)
        axes[1,i].grid(True, alpha=0.3)
        axes[1,i].legend(fontsize=8)
    
    plt.tight_layout()
    plt.savefig('48hr_forecast_all_models.png', dpi=150)
    plt.show()
    
    # ----- Residual diagnostics for all models (Temperature) -----
    temp_common_len = min([len(y_test)] + [len(p) for p in temp_preds if len(p)>0])
    fig_temp, axes_temp = plt.subplots(3, 4, figsize=(14, 12))
    fig_temp.suptitle('Temperature Residual Diagnostics – All Models', fontsize=16, y=0.98)
    
    for i, (name, preds, col) in enumerate(zip(model_names, temp_preds, colors)):
        if len(preds) < temp_common_len:
            for j in range(4):
                axes_temp[i,j].text(0.5, 0.5, 'Insufficient predictions', ha='center', transform=axes_temp[i,j].transAxes)
        else:
            y_true_align = y_test['temp'].values[:temp_common_len]
            pred_align = preds[:temp_common_len]
            residuals = y_true_align - pred_align
            axes_temp[i,0].scatter(pred_align, residuals, alpha=0.5, s=10, color=col)
            axes_temp[i,0].axhline(0, color='r', linestyle='--')
            axes_temp[i,0].set_ylabel(name)
            axes_temp[i,0].set_title('Residuals vs Predicted' if i==0 else '')
            stats.probplot(residuals, dist="norm", plot=axes_temp[i,1])
            axes_temp[i,1].set_title('Q-Q Plot' if i==0 else '')
            axes_temp[i,2].hist(residuals, bins=30, edgecolor='black', alpha=0.7, color=col)
            axes_temp[i,2].set_title('Histogram' if i==0 else '')
            autocorrelation_plot(pd.Series(residuals), ax=axes_temp[i,3])
            axes_temp[i,3].set_title('Autocorrelation' if i==0 else '')
            axes_temp[i,3].set_xlim([0,40])
    
    plt.tight_layout()
    plt.savefig('residuals_temperature_all_models.png', dpi=150)
    plt.show()
    
    # ----- Residual diagnostics for all models (Humidity) -----
    hum_common_len = min([len(y_test)] + [len(p) for p in hum_preds if len(p)>0])
    fig_hum, axes_hum = plt.subplots(3, 4, figsize=(14, 12))
    fig_hum.suptitle('Humidity Residual Diagnostics – All Models', fontsize=16, y=0.98)
    
    for i, (name, preds, col) in enumerate(zip(model_names, hum_preds, colors)):
        if len(preds) < hum_common_len:
            for j in range(4):
                axes_hum[i,j].text(0.5, 0.5, 'Insufficient predictions', ha='center', transform=axes_hum[i,j].transAxes)
        else:
            y_true_align = y_test['hum'].values[:hum_common_len]
            pred_align = preds[:hum_common_len]
            residuals = y_true_align - pred_align
            axes_hum[i,0].scatter(pred_align, residuals, alpha=0.5, s=10, color=col)
            axes_hum[i,0].axhline(0, color='r', linestyle='--')
            axes_hum[i,0].set_ylabel(name)
            axes_hum[i,0].set_title('Residuals vs Predicted' if i==0 else '')
            stats.probplot(residuals, dist="norm", plot=axes_hum[i,1])
            axes_hum[i,1].set_title('Q-Q Plot' if i==0 else '')
            axes_hum[i,2].hist(residuals, bins=30, edgecolor='black', alpha=0.7, color=col)
            axes_hum[i,2].set_title('Histogram' if i==0 else '')
            autocorrelation_plot(pd.Series(residuals), ax=axes_hum[i,3])
            axes_hum[i,3].set_title('Autocorrelation' if i==0 else '')
            axes_hum[i,3].set_xlim([0,40])
    
    plt.tight_layout()
    plt.savefig('residuals_humidity_all_models.png', dpi=150)
    plt.show()
    
    # ----- 1. Bar charts (improved: larger figsize, rotated labels, legend right, row spacing) -----
    models = ['ARIMA', 'LSTM', 'Prop. Stacked']
    temp_r2 = [arima_r2_temp, lstm_r2_temp, stack_r2_temp]
    hum_r2   = [arima_r2_hum,   lstm_r2_hum,   stack_r2_hum]
    
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
    
    model_colors = {'ARIMA': 'grey', 'LSTM': 'orange', 'Prop. Stacked': 'darkgreen'}
    palette = [model_colors[m] for m in models]
    
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))  # increased width for rotated labels
    fig.suptitle('Model Performance Comparison for Temperature and Relative Humidity Forecasting',
                 fontsize=16, y=0.98)
    
    # Increase vertical spacing between rows
    plt.subplots_adjust(top=0.92, hspace=0.4, bottom=0.1)
    plt.tight_layout(pad=3.0, h_pad=5.0)
    
    def add_value_labels(ax, fmt='{:.4f}', fontsize=9):
        for container in ax.containers:
            ax.bar_label(container, fmt=fmt, label_type='edge', fontsize=fontsize, padding=3)
    
    # Rotate x-axis labels and set font properties
    for ax_row in axes:
        for ax in ax_row:
            ax.title.set_fontsize(14)
            ax.xaxis.label.set_fontsize(12)
            ax.yaxis.label.set_fontsize(12)
            ax.tick_params(labelsize=10)
            ax.set_xticklabels(models, rotation=45, ha='right')
    
    # Temperature R²
    sns.barplot(x=models, y=temp_r2, ax=axes[0,0], palette=palette)
    axes[0,0].set_title('Temperature R²', pad=12)
    axes[0,0].set_ylabel('R² Score')
    add_value_labels(axes[0,0])
    
    # Temperature MAE
    sns.barplot(x=models, y=mae_temp, ax=axes[0,1], palette=palette)
    axes[0,1].set_title('Temperature MAE', pad=12)
    axes[0,1].set_ylabel('Mean Absolute Error')
    add_value_labels(axes[0,1])
    
    # Temperature RMSE
    sns.barplot(x=models, y=rmse_temp, ax=axes[0,2], palette=palette)
    axes[0,2].set_title('Temperature RMSE', pad=12)
    axes[0,2].set_ylabel('Root Mean Square Error')
    add_value_labels(axes[0,2])
    
    # Humidity R²
    sns.barplot(x=models, y=hum_r2, ax=axes[1,0], palette=palette)
    axes[1,0].set_title('Humidity R²', pad=12)
    axes[1,0].set_ylabel('R² Score')
    add_value_labels(axes[1,0])
    
    # Humidity MAE
    sns.barplot(x=models, y=mae_hum, ax=axes[1,1], palette=palette)
    axes[1,1].set_title('Humidity MAE', pad=12)
    axes[1,1].set_ylabel('Mean Absolute Error')
    add_value_labels(axes[1,1])
    
    # Humidity RMSE
    sns.barplot(x=models, y=rmse_hum, ax=axes[1,2], palette=palette)
    axes[1,2].set_title('Humidity RMSE', pad=12)
    axes[1,2].set_ylabel('Root Mean Square Error')
    add_value_labels(axes[1,2])
    
    # Legend placed on the right side
    legend_patches = [mpatches.Patch(color=model_colors[m], label=m) for m in models]
    fig.legend(handles=legend_patches, loc='center right', bbox_to_anchor=(0.98, 0.5),
               fontsize=10, frameon=False)
    
    plt.savefig('baseline_comparison.png', dpi=150, bbox_inches='tight')
    plt.show()
    
    # ---- 2. Scatter plots (three models) ----
    fig, axes = plt.subplots(2, 3, figsize=(18, 8))
    fig.suptitle('Actual vs Predicted Scatter Plots', fontsize=16)
    for i, (name, pred_temp, pred_hum, col) in enumerate(zip(model_names, temp_preds, hum_preds, colors)):
        if len(pred_temp) > 0:
            axes[0,i].scatter(y_test['temp'][:len(pred_temp)], pred_temp, alpha=0.5, color=col)
            axes[0,i].plot([y_test['temp'].min(), y_test['temp'].max()],
                           [y_test['temp'].min(), y_test['temp'].max()], 'r--')
            axes[0,i].set_title(f'{name} (Temp) R²={temp_r2[i]:.3f}')
        if len(pred_hum) > 0:
            axes[1,i].scatter(y_test['hum'][:len(pred_hum)], pred_hum, alpha=0.5, color=col)
            axes[1,i].plot([y_test['hum'].min(), y_test['hum'].max()],
                           [y_test['hum'].min(), y_test['hum'].max()], 'r--')
            axes[1,i].set_title(f'{name} (Hum) R²={hum_r2[i]:.3f}')
    plt.tight_layout()
    plt.savefig('scatter_plots.png', dpi=150)
    plt.show()
    
    # ---- 3. SHAP (stacked only) ----
    print("Generating SHAP explanations...")
    X_sample = X_test.sample(min(100, len(X_test)), random_state=42)
    shap_importance = shap_analysis(stack_model, X_sample, X_test.columns.tolist())
    shap_importance.to_csv('shap_importance.csv', index=False)
    
    # ---- 4. Multi-step forecasting (stacked only) ----
    evaluate_multi_step(stack_model, X_test, y_test, horizons=[1,6,12,24,48])
    
    # ---- 5. Final performance table (three models) ----
    print("\n" + "="*60)
    print("FINAL PERFORMANCE COMPARISON")
    print("="*60)
    print(f"{'Model':<20} | {'Temp R²':<10} | {'Hum R²':<10} | {'Temp MAE':<10} | {'Hum MAE':<10} | {'Temp RMSE':<10} | {'Hum RMSE':<10}")
    print("-"*100)
    print(f"{'ARIMA':<20} | {arima_r2_temp:.4f}    | {arima_r2_hum:.4f}    | {mae_temp[0]:.4f}    | {mae_hum[0]:.4f}    | {rmse_temp[0]:.4f}     | {rmse_hum[0]:.4f}")
    print(f"{'LSTM':<20} | {lstm_r2_temp:.4f}    | {lstm_r2_hum:.4f}    | {mae_temp[1]:.4f}    | {mae_hum[1]:.4f}    | {rmse_temp[1]:.4f}     | {rmse_hum[1]:.4f}")
    print(f"{'Proposed Stacked':<20} | {stack_r2_temp:.4f}    | {stack_r2_hum:.4f}    | {mae_temp[2]:.4f}    | {mae_hum[2]:.4f}    | {rmse_temp[2]:.4f}     | {rmse_hum[2]:.4f}")
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