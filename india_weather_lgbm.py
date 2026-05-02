import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import lightgbm as lgb
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
import folium
from folium.plugins import MarkerCluster
import warnings
warnings.filterwarnings('ignore')

# ============================================================================
# MANUAL COORDINATES (Indian cities)
# ============================================================================
MANUAL_COORDS = {
    'Ashoknagar': [24.57, 77.72],
    'Raisen': [23.33, 77.80],
    'Chhindwara': [22.07, 78.93],
    'Betul': [21.86, 77.93],
    'Hoshangabad': [22.75, 77.72],
    'Sehore': [23.20, 77.08],
    'Jabalpur': [23.17, 79.95],
    'Narsimhapur': [22.95, 79.20],
    'Panna': [24.72, 80.20],
    'Ujjain': [23.18, 75.77],
    'Rewa': [24.53, 81.30],
    'Dindori': [22.95, 81.08],
    'Balaghat': [21.78, 80.25],
    'Barwani': [22.03, 74.90],
    'Kolkata': [22.5726, 88.3639],
    'Mumbai': [19.0760, 72.8777],
    'Delhi': [28.6139, 77.2090],
    'Chennai': [13.0827, 80.2707],
    'Bengaluru': [12.9716, 77.5946],
    'Hyderabad': [17.3850, 78.4867],
    'Ahmedabad': [23.0225, 72.5714],
    'Pune': [18.5204, 73.8567],
    'Jaipur': [26.9124, 75.7873],
    'Lucknow': [26.8467, 80.9462],
    'Nagpur': [21.1458, 79.0882],
    'Indore': [22.7196, 75.8577],
    'Bhopal': [23.2599, 77.4126],
    'Visakhapatnam': [17.6868, 83.2185],
    'Patna': [25.5941, 85.1376],
    'Vadodara': [22.3072, 73.1812],
    'Guwahati': [26.1445, 91.7362],
    'Chandigarh': [30.7333, 76.7794],
    'Srinagar': [34.0837, 74.7973],
    'Thiruvananthapuram': [8.5241, 76.9366]
}

# ============================================================================
# 1. DATA LOADING & PREPROCESSING (target = temperature_celsius)
# ============================================================================
def load_and_prepare_data(filepath):
    df = pd.read_csv(filepath)
    print("Available columns:", list(df.columns))
    
    # ----- Extract city from 'timezone' -----
    if 'timezone' not in df.columns:
        raise ValueError("No 'timezone' column found.")
    
    def extract_city(tz):
        if pd.isna(tz):
            return None
        tz_str = str(tz).strip()
        if '/' in tz_str:
            return tz_str.split('/')[-1]
        else:
            return tz_str
    
    df['city'] = df['timezone'].apply(extract_city)
    print(f"Unique cities: {df['city'].nunique()}")
    print("Sample cities:", df['city'].dropna().head(5).tolist())
    
    # ----- Coordinates from manual dict -----
    df['latitude'] = df['city'].apply(lambda x: MANUAL_COORDS.get(x, [None, None])[0])
    df['longitude'] = df['city'].apply(lambda x: MANUAL_COORDS.get(x, [None, None])[1])
    found = df['latitude'].notna().sum()
    print(f"Coordinates found for {found} out of {len(df)} rows.")
    
    metadata = df[['city', 'latitude', 'longitude']].copy()
    
    # ----- Time features -----
    if 'last_updated' in df.columns:
        df['last_updated'] = pd.to_datetime(df['last_updated'])
        df['hour'] = df['last_updated'].dt.hour
        df['dayofyear'] = df['last_updated'].dt.dayofyear
        df['hour_sin'] = np.sin(2 * np.pi * df['hour'] / 24)
        df['hour_cos'] = np.cos(2 * np.pi * df['hour'] / 24)
        df['doy_sin'] = np.sin(2 * np.pi * df['dayofyear'] / 365)
        df['doy_cos'] = np.cos(2 * np.pi * df['dayofyear'] / 365)
        df.drop(['last_updated', 'hour', 'dayofyear'], axis=1, inplace=True)
    
    # ----- Encode categoricals -----
    for col in ['condition_text', 'wind_direction']:
        if col in df.columns:
            df[col] = LabelEncoder().fit_transform(df[col].astype(str))
    
    # ----- Target: temperature_celsius -----
    target_col = 'temperature_celsius'
    if target_col not in df.columns:
        raise ValueError("temperature_celsius column not found.")
    
    df[target_col] = pd.to_numeric(df[target_col], errors='coerce')
    
    # Keep only numeric columns, drop rows where target is NaN
    numeric_df = df.select_dtypes(include=[np.number]).dropna(subset=[target_col])
    print(f"Rows after dropping NaNs: {len(numeric_df)}")
    
    # Align metadata with numeric_df rows
    valid_indices = numeric_df.index
    metadata_aligned = metadata.loc[valid_indices].reset_index(drop=True)
    
    X = numeric_df.drop(columns=[target_col])
    y = numeric_df[target_col]
    
    return X, y, target_col, metadata_aligned, 'latitude', 'longitude', 'city'

# ============================================================================
# 2. FEATURE ENGINEERING (lags and rolling stats for temperature)
# ============================================================================
def add_temporal_features(X, y, max_lag=24, windows=[3,6,12,24]):
    X_new = X.copy()
    for lag in [1, 3, 6, 12, 24]:
        X_new[f'temp_lag_{lag}'] = y.shift(lag).fillna(0)
    for w in windows:
        X_new[f'temp_rm_{w}'] = y.rolling(w, min_periods=1).mean().fillna(0)
        X_new[f'temp_std_{w}'] = y.rolling(w, min_periods=1).std().fillna(0)
    # Remove first max_lag rows (shift introduces zeros)
    X_new = X_new.iloc[max_lag:].reset_index(drop=True)
    y_new = y.iloc[max_lag:].reset_index(drop=True)
    return X_new, y_new

# ============================================================================
# 3. TRAIN LIGHTGBM REGRESSOR
# ============================================================================
def train_lightgbm(X_train, y_train, X_val, y_val):
    train_data = lgb.Dataset(X_train, label=y_train)
    val_data = lgb.Dataset(X_val, label=y_val, reference=train_data)
    params = {
        'objective': 'regression',
        'metric': 'rmse',
        'boosting_type': 'gbdt',
        'num_leaves': 31,
        'learning_rate': 0.05,
        'feature_fraction': 0.8,
        'bagging_fraction': 0.8,
        'bagging_freq': 5,
        'verbose': -1,
        'random_state': 42,
        'n_jobs': -1
    }
    model = lgb.train(params, train_data, valid_sets=[val_data],
                      num_boost_round=1000,
                      callbacks=[lgb.early_stopping(50), lgb.log_evaluation(100)])
    return model

# ============================================================================
# 4. STATIC PLOTS (separate figures)
# ============================================================================
def plot_scatter(y_true, y_pred, target_name):
    plt.figure(figsize=(8, 6))
    plt.scatter(y_true, y_pred, alpha=0.3, s=10, color='royalblue')
    plt.plot([y_true.min(), y_true.max()], [y_true.min(), y_true.max()], 'r--', lw=2)
    plt.xlabel('Actual')
    plt.ylabel('Predicted')
    plt.title(f'{target_name}: Actual vs Predicted')
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.show()

def plot_residuals(y_true, y_pred, target_name):
    residuals = y_true - y_pred
    plt.figure(figsize=(8, 6))
    plt.scatter(y_pred, residuals, alpha=0.3, s=10, color='darkorange')
    plt.axhline(y=0, color='r', linestyle='--', lw=2)
    plt.xlabel('Predicted')
    plt.ylabel('Residual')
    plt.title(f'{target_name}: Residuals vs Predicted')
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.show()

def plot_residual_histogram(y_true, y_pred, target_name):
    residuals = y_true - y_pred
    plt.figure(figsize=(8, 6))
    plt.hist(residuals, bins=50, edgecolor='black', alpha=0.7, color='purple')
    plt.xlabel('Residual')
    plt.ylabel('Frequency')
    plt.title(f'{target_name}: Residual Distribution')
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.show()

def plot_time_series(y_true, y_pred, target_name, n_samples=500):
    plt.figure(figsize=(12, 5))
    plt.plot(y_true[:n_samples], label='Actual', alpha=0.7, lw=1)
    plt.plot(y_pred[:n_samples], label='Predicted', alpha=0.7, lw=1)
    plt.xlabel('Time step')
    plt.ylabel(target_name)
    plt.title(f'{target_name}: Time Series (first {n_samples} points)')
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.show()

def plot_feature_importance(model, X_val, target_name, top_n=15):
    importance = model.feature_importance(importance_type='gain')
    feat_names = X_val.columns
    idx = np.argsort(importance)[-top_n:]
    plt.figure(figsize=(10, 6))
    plt.barh(range(len(idx)), importance[idx], color='teal')
    plt.yticks(range(len(idx)), [feat_names[i] for i in idx])
    plt.xlabel('Gain')
    plt.title(f'{target_name}: Top {top_n} Feature Importance')
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.show()

def plot_metrics_table(y_true, y_pred, target_name):
    r2 = r2_score(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae = mean_absolute_error(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(6, 3))
    ax.axis('off')
    ax.text(0.1, 0.6, f'R² = {r2:.4f}\nRMSE = {rmse:.4f}\nMAE = {mae:.4f}',
            fontsize=16, transform=ax.transAxes)
    ax.set_title(f'{target_name}: Performance Metrics')
    plt.tight_layout()
    plt.show()

# ============================================================================
# 5. INTERACTIVE MAP
# ============================================================================
def create_region_map(region_summary, target_name="Temperature (°C)"):
    region_summary = region_summary.dropna(subset=['latitude', 'longitude']).copy()
    if region_summary.empty:
        print("No valid coordinates found to create the map.")
        return None
    
    center_lat = region_summary['latitude'].mean()
    center_lon = region_summary['longitude'].mean()
    m = folium.Map(location=[center_lat, center_lon], zoom_start=6, tiles='OpenStreetMap')
    marker_cluster = MarkerCluster().add_to(m)
    
    for _, row in region_summary.iterrows():
        popup_text = f"""
        <b>{row['region']}</b><br>
        Actual Temperature: {row['actual_mean']:.1f} °C<br>
        Predicted Temperature: {row['pred_mean']:.1f} °C<br>
        Absolute Error: {row['error']:.1f} °C<br>
        Number of samples: {row['count']}
        """
        if row['error'] < 2:
            color = 'green'
        elif row['error'] < 4:
            color = 'orange'
        else:
            color = 'red'
        folium.Marker(
            location=[row['latitude'], row['longitude']],
            popup=folium.Popup(popup_text, max_width=300),
            icon=folium.Icon(color=color, icon='info-sign')
        ).add_to(marker_cluster)
    
    title_html = f'<h3 align="center" style="font-family:Arial;">{target_name} – City-wise Actual vs Predicted</h3>'
    m.get_root().html.add_child(folium.Element(title_html))
    return m

# ============================================================================
# 6. MAIN
# ============================================================================
def main():
    filepath = 'IndianWeatherRepository.csv'  # change to your file
    print("Loading data...")
    X, y, target_name, metadata, lat_col, lon_col, region_col = load_and_prepare_data(filepath)
    print(f"Original shape: X={X.shape}, y={y.shape}")
    
    if len(X) == 0:
        print("No valid data after preprocessing. Exiting.")
        return
    
    print("Adding temporal features...")
    X, y = add_temporal_features(X, y, max_lag=24, windows=[3,6,12,24])
    print(f"After feature engineering: X={X.shape}, y={y.shape}")
    
    if len(X) == 0:
        print("Not enough data after feature engineering. Exiting.")
        return
    
    # --- CRITICAL FIX: Align metadata exactly with the rows that remain in X ---
    # After add_temporal_features, we dropped the first 24 rows.
    # So we take the last len(X) rows of the original metadata.
    metadata = metadata.iloc[-len(X):].reset_index(drop=True)
    
    # Temporal split (80% train, 20% validation) - no shuffling
    split_idx = int(len(X) * 0.8)
    X_train, X_val = X.iloc[:split_idx], X.iloc[split_idx:]
    y_train, y_val = y.iloc[:split_idx], y.iloc[split_idx:]
    metadata_train = metadata.iloc[:split_idx]
    metadata_val = metadata.iloc[split_idx:]
    
    print(f"Train size: {X_train.shape[0]}, Validation size: {X_val.shape[0]}")
    print(f"Metadata validation size: {len(metadata_val)}")  # Should equal len(y_val)
    
    # Train model
    model = train_lightgbm(X_train, y_train, X_val, y_val)
    
    # Predict
    y_pred = model.predict(X_val)
    
    # Static plots
    plot_scatter(y_val, y_pred, target_name)
    plot_residuals(y_val, y_pred, target_name)
    plot_residual_histogram(y_val, y_pred, target_name)
    plot_time_series(y_val, y_pred, target_name, n_samples=500)
    plot_feature_importance(model, X_val, target_name)
    plot_metrics_table(y_val, y_pred, target_name)
    
    # Region (city) summary for map
    val_df = pd.DataFrame({
        'region': metadata_val[region_col],
        'actual': y_val,
        'predicted': y_pred,
        'latitude': metadata_val[lat_col],
        'longitude': metadata_val[lon_col]
    })
    
    region_summary = val_df.groupby('region').agg(
        actual_mean=('actual', 'mean'),
        pred_mean=('predicted', 'mean'),
        count=('actual', 'count'),
        latitude=('latitude', 'first'),
        longitude=('longitude', 'first')
    ).reset_index()
    region_summary['error'] = np.abs(region_summary['actual_mean'] - region_summary['pred_mean'])
    
    print(f"\nNumber of cities with coordinates: {region_summary['latitude'].notna().sum()}")
    print("Top 5 cities by prediction error:")
    print(region_summary.nlargest(5, 'error')[['region', 'actual_mean', 'pred_mean', 'error']])
    
    # Interactive map
    print("\nGenerating map...")
    m = create_region_map(region_summary, target_name=target_name)
    if m:
        m.save("temperature_region_map.html")
        print("Map saved as 'temperature_region_map.html'")
    else:
        print("Map generation failed.")
    
    try:
        from IPython.display import display, HTML
        display(HTML('<iframe src="temperature_region_map.html" width="100%" height="600"></iframe>'))
    except:
        pass

if __name__ == "__main__":
    main()