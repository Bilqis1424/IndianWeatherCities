import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from catboost import CatBoostRegressor
from sklearn.ensemble import HistGradientBoostingRegressor, StackingRegressor
from sklearn.multioutput import MultiOutputRegressor
from sklearn.linear_model import Ridge
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

# Set aesthetic style for high-quality figures
sns.set_theme(style="whitegrid", palette="muted")
plt.rcParams['font.sans-serif'] = 'Arial'
plt.rcParams['figure.figsize'] = (12, 6)

def prepare_and_analyze(filepath):
    df = pd.read_csv(filepath)
    
    # Standardize column names
    temp_col = [c for c in df.columns if 'temp' in c.lower()][0]
    hum_col = [c for c in df.columns if 'hum' in c.lower()][0]
    df = df.rename(columns={temp_col: 'temp', hum_col: 'hum'})
    
    # 1. PREPROCESSING INSIGHTS: Correlation Heatmap
    
    plt.figure(figsize=(10, 8))
    corr = df.select_dtypes(include=[np.number]).corr()
    sns.heatmap(corr, cmap='RdYlBu', annot=False)
    plt.title("Insight: Feature Correlation Matrix", fontsize=14, fontweight='bold')
    plt.show()

    # Feature Engineering
    le = LabelEncoder()
    cat_cols = ['condition_text', 'wind_direction', 'location_name', 'region']
    for col in cat_cols:
        if col in df.columns:
            df[col] = le.fit_transform(df[col].astype(str))

    for col in ['temp', 'hum']:
        df[f'{col}_lag1'] = df.groupby('location_name')[col].shift(1)

    df = df.dropna().reset_index(drop=True)
    X = df.select_dtypes(include=[np.number]).drop(columns=['temp', 'hum'])
    y = df[['temp', 'hum']]
    return X, y, df

def run_experiment():
    X, y, df_full = prepare_and_analyze('IndianWeatherRepository.csv')
    
    # Temporal Split
    split = int(0.8 * len(X))
    X_train, X_test = X[:split], X[split:]
    y_train, y_test = y[:split], y[split:]
    
    # Define Stacked Ensemble
    base_models = [
        ('xgb', XGBRegressor(n_estimators=100, max_depth=5)),
        ('lgbm', LGBMRegressor(n_estimators=100, verbose=-1)),
        ('cat', CatBoostRegressor(iterations=100, verbose=0)),
        ('hgb', HistGradientBoostingRegressor(max_iter=100))
    ]
    stack = StackingRegressor(estimators=base_models, final_estimator=Ridge())
    model = MultiOutputRegressor(stack)

    # Training
    print("\n[INFO] Training Multi-Output Stacked Ensemble...")
    model.fit(X_train, y_train)
    preds = model.predict(X_test)

    # ============================================================================
    # 4. RESULTS REPORTING (MAE, RMSE, R2)
    # ============================================================================
    results = []
    print("\n" + "="*50)
    print(f"{'METRIC REPORT':^50}")
    print("="*50)
    
    for i, target in enumerate(['Temperature', 'Humidity']):
        y_true = y_test.iloc[:, i]
        y_pred = preds[:, i]
        
        r2 = r2_score(y_true, y_pred)
        mae = mean_absolute_error(y_true, y_pred)
        rmse = np.sqrt(mean_squared_error(y_true, y_pred))
        
        results.append({'Target': target, 'R2': r2, 'MAE': mae, 'RMSE': rmse})
        
        print(f"Results for {target.upper()}:")
        print(f"  - R-Squared (R2):     {r2:.6f}")
        print(f"  - Mean Absolute Error (MAE):  {mae:.6f}")
        print(f"  - Root Mean Square Error (RMSE): {rmse:.6f}")
        print("-" * 50)

    # ============================================================================
    # 5. FINAL VISUALIZATIONS
    # ============================================================================
    
    # --- Aesthetic Metric Barchart ---
    metrics_df = pd.DataFrame(results).melt(id_vars='Target', var_name='Metric', value_name='Score')
    plt.figure(figsize=(12, 6))
    ax = sns.barplot(data=metrics_df, x='Metric', y='Score', hue='Target', palette='viridis', edgecolor='black')
    plt.yscale('log')
    for container in ax.containers:
        ax.bar_label(container, fmt='%.4f', padding=3, fontweight='bold')
    plt.title('Performance Metrics (R2, MAE, RMSE)', fontsize=16, fontweight='bold')
    plt.show()

    # --- R2 Scatter Proof ---
    
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    for i, target in enumerate(['temp', 'hum']):
        axes[i].scatter(y_test[target], preds[:, i], alpha=0.3, s=15, color='#2c3e50')
        axes[i].plot([y_test[target].min(), y_test[target].max()], 
                     [y_test[target].min(), y_test[target].max()], 'r--', lw=2)
        axes[i].set_title(f'{target.capitalize()} R2 Proof')
    plt.show()

    # --- 48-Hour Forecast ---
    
    plt.figure(figsize=(15, 8))
    for i, (target, color) in enumerate([('Temperature', 'crimson'), ('Humidity', 'dodgerblue')]):
        plt.subplot(2, 1, i+1)
        plt.plot(y_test.iloc[-48:, i].values, 'o-', label='Actual Data', color=color, alpha=0.6)
        plt.plot(preds[-48:, i], 'x--', label='Ensemble Forecast', color='black')
        plt.title(f'48-Hour Final Test Window: {target}')
        plt.legend()
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    run_experiment()