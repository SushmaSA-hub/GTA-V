import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler

# IMPORTANT: Use correct separator
df = pd.read_csv("tripinfos_clean.csv", sep=";")

# Feature set (X)
features = [
    "tripinfo_routeLength",
    "tripinfo_waitingTime",
    "tripinfo_timeLoss",
    "tripinfo_stopTime",
    "tripinfo_speedFactor",
    "tripinfo_departDelay"
]

# Target (Y)
target = "tripinfo_duration"

X = df[features].values
y = df[target].values.reshape(-1, 1)

# Normalize
scaler_X = MinMaxScaler()
scaler_y = MinMaxScaler()

X_scaled = scaler_X.fit_transform(X)
y_scaled = scaler_y.fit_transform(y)

# Create sequences
def create_sequences(X, y, time_steps=5):
    Xs, ys = [], []
    for i in range(len(X) - time_steps):
        Xs.append(X[i:i+time_steps])
        ys.append(y[i+time_steps])
    return np.array(Xs), np.array(ys)

TIME_STEPS = 5
X_seq, y_seq = create_sequences(X_scaled, y_scaled, TIME_STEPS)

split = int(0.8 * len(X_seq))
X_train, X_test = X_seq[:split], X_seq[split:]
y_train, y_test = y_seq[:split], y_seq[split:]

print("Training shape:", X_train.shape)
print("Testing shape :", X_test.shape)
