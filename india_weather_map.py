import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import warnings
import geopandas as gpd
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from catboost import CatBoostRegressor
from sklearn.ensemble import HistGradientBoostingRegressor, StackingRegressor
from sklearn.multioutput import MultiOutputRegressor
from sklearn.linear_model import Ridge
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

# Setup
warnings.filterwarnings('ignore')

# ============================================================================
# 1. GEOSPATIAL VISUALIZATION (Fixing Deprecation Error)
# ============================================================================
def plot_research_area(df):
    print("--- Generating Research Area Map ---")
    locations = df[['location_name', 'latitude', 'longitude']].drop_duplicates()
    
    try:
        # Using a direct URL for the world map to avoid the deprecated datasets error
        world_url = gpd.read_file('./ne_110m_admin_0_countries/ne_110m_admin_0_countries.shp')
        world = gpd.read_file(world_url)
        # Filter for India (Note: case sensitivity depends on the source, check for 'India')
        india = world[world['name'].str.contains('India', na=False)]
        
        if india.empty:
            # Fallback if the geojson uses 'ISO_A3' or different naming
            india = world[world['id'] == 'IND']

        fig, ax = plt.subplots(figsize=(10, 12))
        india.plot(ax=ax, color='#f2f2f2', edgecolor='#333333', linewidth=1)
        
        ax.scatter(locations['longitude'], locations['latitude'], 
                   c='#d62728', s=20, alpha=0.7, edgecolors='white', 
                   label='Meteorological Stations')
        
        ax.set_title('Geographical Distribution of Research Area: India', fontsize=14)
        ax.set_xlabel('Longitude')
        ax.set_ylabel('Latitude')
        ax.legend()
        plt.grid(True, linestyle='--', alpha=0.5)
        plt.savefig('research_map_india.png', dpi=300)
        print("Map successfully saved as 'research_map_india.png'")
        plt.show()
        
    except Exception as e:
        print(f"Map generation skipped due to connection/library error: {e}")
        print("Ensure you have an internet connection to download the base map.")


# ============================================================================
# 2. DATA PREPARATION & FEATURE ENGINEERING
# ============================================================================
def prepare_data(filepath):
    df = pd.read_csv(filepath)
    
    # 1. Map Plotting
    plot_research_area(df)

    # 2. Standardize target columns
    temp_col = [c for c in df.columns if 'temp' in c.lower()][0]
    hum_col = [c for c in df.columns if 'hum' in c.lower()][0]
    df = df.rename(columns={temp_col: 'temp', hum_col: 'hum'})
    
    # 3. Categorical Encoding
    le = LabelEncoder()
    for col in ['condition_text', 'wind_direction', 'region']:
        if col in df.columns:
            df[col] = le.fit_transform(df[col].astype(str))

    # 4. Cyclical Time Features
    df['last_updated'] = pd.to_datetime(df['last_updated'])
    df['hour_sin'] = np.sin(2 * np.pi * df['last_updated'].dt.hour / 24)
    df['hour_cos'] = np.cos(2 * np.pi * df['last_updated'].dt.hour / 24)

    # 5. Lagged Features (Crucial for Stacking performance)
    df = df.sort_values(['location_name', 'last_updated'])
    for col in ['temp', 'hum']:
        df[f'{col}_lag1'] = df.groupby('location_name')[col].shift(1)
    
    df = df.dropna().reset_index(drop=True)
    
    # Select features for training
    features = ['latitude', 'longitude', 'hour_sin', 'hour_cos', 'temp_lag1', 'hum_lag1', 'region']
    X = df[features]
    y = df[['temp', 'hum']]
    
    return X, y, df, features

# ============================================================================
# 3. MULTI-OUTPUT STACKED ENSEMBLE
# ============================================================================
def get_stacking_model():
    base_models = [
        ('xgb', XGBRegressor(n_estimators=100, learning_rate=0.05)),
        ('lgbm', LGBMRegressor(n_estimators=100, verbose=-1)),
        ('cat', CatBoostRegressor(iterations=100, verbose=0)),
        ('hgb', HistGradientBoostingRegressor(max_iter=100))
    ]
    
    stack = StackingRegressor(
        estimators=base_models,
        final_estimator=Ridge()
    )
    return MultiOutputRegressor(stack)

# ============================================================================
# 4. EXPERIMENT EXECUTION
# ============================================================================
def run_experiment():
    if not os.path.exists('IndianWeatherRepository.csv'):
        print("Dataset not found!")
        return

    X, y, df_full, feature_names = prepare_data('IndianWeatherRepository.csv')
    
    # Split
    split = int(0.85 * len(X))
    X_train, X_test = X[:split], X[split:]
    y_train, y_test = y[:split], y[split:]
    
    # Train
    print("\n--- Training Stacked Ensemble (XGB+LGBM+CAT+HGB) ---")
    model = get_stacking_model()
    model.fit(X_train, y_train)
    
    # Predict & Evaluate
    preds = model.predict(X_test)
    
    # =========================================
    # METRICS BAR CHART
    # =========================================
    metrics = {'Temp': {}, 'Hum': {}}
    for i, label in enumerate(['Temp', 'Hum']):
        metrics[label]['R2'] = r2_score(y_test.iloc[:, i], preds[:, i])
        metrics[label]['MAE'] = mean_absolute_error(y_test.iloc[:, i], preds[:, i])
        metrics[label]['RMSE'] = np.sqrt(mean_squared_error(y_test.iloc[:, i], preds[:, i]))

    fig, ax = plt.subplots(1, 3, figsize=(15, 5))
    m_list = ['R2', 'MAE', 'RMSE']
    for i, m in enumerate(m_list):
        vals = [metrics['Temp'][m], metrics['Hum'][m]]
        ax[i].bar(['Temperature', 'Humidity'], vals, color=['#d62728', '#1f77b4'])
        ax[i].set_title(f'Comparison: {m}')
        for j, v in enumerate(vals):
            ax[i].text(j, v, f'{v:.3f}', ha='center', va='bottom', fontweight='bold')
    
    plt.suptitle('Stacked Ensemble Model Performance Metrics')
    plt.tight_layout()
    plt.show()

    # =========================================
    # 48-HOUR FUTURE FORECAST (Adilabad)
    # =========================================
    print("\n--- Generating 48-Hour Future Forecast ---")
    target_loc = "Adilabad" 
    loc_data = df_full[df_full['location_name'] == target_loc].iloc[-1:]
    
    current_features = loc_data[feature_names].values
    future_preds = []
    
    for i in range(48):
        # Predict [Temp, Hum]
        pred = model.predict(current_features)[0]
        future_preds.append(pred)
        
        # Update features for next step:
        # Features are: [lat, lon, h_sin, h_cos, temp_lag1, hum_lag1, region]
        new_features = current_features.copy()
        new_features[0, 4] = pred[0] # Update temp_lag1 with current prediction
        new_features[0, 5] = pred[1] # Update hum_lag1 with current prediction
        
        # Simple hour increment for cyclical features (approximation)
        new_features[0, 2] = np.sin(2 * np.pi * (i % 24) / 24) 
        new_features[0, 3] = np.cos(2 * np.pi * (i % 24) / 24)
        
        current_features = new_features

    future_preds = np.array(future_preds)
    
    plt.figure(figsize=(12, 6))
    plt.plot(future_preds[:, 0], label='Forecasted Temp', color='red', marker='o', markersize=3)
    plt.plot(future_preds[:, 1], label='Forecasted Hum', color='blue', marker='x', markersize=3)
    plt.title(f'48-Hour Recursive Forecast for {target_loc}')
    plt.xlabel('Hours Ahead')
    plt.ylabel('Value')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.show()

if __name__ == "__main__":
    run_experiment()