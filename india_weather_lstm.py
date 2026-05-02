import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import RobustScaler, LabelEncoder
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

# ============================================================================
# 1. ADVANCED PREPROCESSING (Robust to Outliers)
# ============================================================================

def prepare_data_v4(filepath, seq_length=24): # Increased lookback to 24 hours
    df = pd.read_csv(filepath)
    
    # Standardize column names
    temp_col = [c for c in df.columns if 'temp' in c.lower()][0]
    hum_col = [c for c in df.columns if 'hum' in c.lower()][0]
    df = df.rename(columns={temp_col: 'temp', hum_col: 'hum'})

    # Feature Engineering: Rolling Averages (Helps the model see trends)
    df['temp_roll_mean'] = df['temp'].rolling(window=3).mean().fillna(method='bfill')
    df['hum_roll_mean'] = df['hum'].rolling(window=3).mean().fillna(method='bfill')

    # Cyclical Time
    if 'last_updated' in df.columns:
        df['hour'] = pd.to_datetime(df['last_updated']).dt.hour
        df['hour_sin'] = np.sin(2 * np.pi * df['hour']/24)
        df['hour_cos'] = np.cos(2 * np.pi * df['hour']/24)

    le = LabelEncoder()
    for col in ['condition_text', 'wind_direction', 'region']:
        if col in df.columns:
            df[col] = le.fit_transform(df[col].astype(str))

    df = df.select_dtypes(include=[np.number]).dropna().reset_index(drop=True)

    X_raw = df.drop(columns=['temp', 'hum']).values
    y_raw = df[['temp', 'hum']].values

    # RobustScaler is better for weather data with extreme spikes
    scaler_x, scaler_y = RobustScaler(), RobustScaler()
    X_scaled = scaler_x.fit_transform(X_raw)
    y_scaled = scaler_y.fit_transform(y_raw)

    X_seq, y_seq = [], []
    for i in range(len(X_scaled) - seq_length):
        X_seq.append(X_scaled[i : i + seq_length])
        y_seq.append(y_scaled[i + seq_length])
    
    return np.array(X_seq), np.array(y_seq), scaler_y

# ============================================================================
# 2. BIDIRECTIONAL LSTM ARCHITECTURE
# ============================================================================

class WeatherBiLSTM(nn.Module):
    def __init__(self, input_size):
        super(WeatherBiLSTM, self).__init__()
        # Bidirectional = True doubles the hidden_size in the output
        self.lstm = nn.LSTM(input_size, 128, num_layers=2, batch_first=True, 
                            dropout=0.2, bidirectional=True)
        
        self.fc = nn.Sequential(
            nn.Linear(128 * 2, 64), # 128 * 2 because of bidirectionality
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, 2)
        )

    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        # We take the last time step's output
        out = self.fc(lstm_out[:, -1, :])
        return out

# ============================================================================
# 3. TRAINING & HISTORY
# ============================================================================

def run_experiment():
    SEQ_LEN = 24 # Full day context
    EPOCHS = 100
    X, y, scaler_y = prepare_data_v4('IndianWeatherRepository.csv', SEQ_LEN)
    
    split = int(0.8 * len(X))
    train_loader = DataLoader(TensorDataset(torch.tensor(X[:split], dtype=torch.float32), 
                                            torch.tensor(y[:split], dtype=torch.float32)), 
                              batch_size=32, shuffle=True)
    
    X_test_t = torch.tensor(X[split:], dtype=torch.float32)
    y_test_raw = scaler_y.inverse_transform(y[split:])

    model = WeatherBiLSTM(X.shape[2])
    optimizer = optim.AdamW(model.parameters(), lr=0.0003, weight_decay=1e-4)
    criterion = nn.SmoothL1Loss() # Hybrid between MSE and MAE
    
    history = []
    print("Starting Deep Learning Optimization...")

    for epoch in range(EPOCHS):
        model.train()
        epoch_loss = 0
        for b_x, b_y in train_loader:
            optimizer.zero_grad()
            loss = criterion(model(b_x), b_y)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
        
        avg_loss = epoch_loss/len(train_loader)
        history.append(avg_loss)
        if (epoch+1) % 10 == 0:
            print(f"Epoch {epoch+1} | Loss: {avg_loss:.6f}")

    # Results
    model.eval()
    with torch.no_grad():
        preds = scaler_y.inverse_transform(model(X_test_t).numpy())

    t_r2 = r2_score(y_test_raw[:, 0], preds[:, 0])
    h_r2 = r2_score(y_test_raw[:, 1], preds[:, 1])

    # --- VISUALS ---
    plt.figure(figsize=(10, 4))
    plt.plot(history, color='teal', lw=2)
    plt.title("Training History (Convergence)")
    plt.ylabel("Loss")
    plt.show()

    fig, ax = plt.subplots(1, 2, figsize=(14, 5))
    ax[0].scatter(y_test_raw[:, 0], preds[:, 0], color='blue', alpha=0.2)
    ax[0].set_title(f"Temp R²: {t_r2:.4f}")
    ax[1].scatter(y_test_raw[:, 1], preds[:, 1], color='green', alpha=0.2)
    ax[1].set_title(f"Hum R²: {h_r2:.4f}")
    plt.show()

if __name__ == "__main__":
    run_experiment()