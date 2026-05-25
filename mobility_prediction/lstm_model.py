import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from tensorflow.keras.callbacks import EarlyStopping

# ------------------------------
# 1️⃣ Load Data (IMPORTANT: sep=";")
# ------------------------------
df = pd.read_csv("tripinfos_clean.csv", sep=";")

# Ensure numeric columns
df = df.apply(pd.to_numeric, errors='ignore')

# ------------------------------
# 2️⃣ Define Features and Target
# ------------------------------
features = [
    "tripinfo_routeLength",
    "tripinfo_waitingTime",
    "tripinfo_timeLoss",
    "tripinfo_stopTime",
    "tripinfo_speedFactor",
    "tripinfo_departDelay"
]

target = "tripinfo_duration"

# Check if columns exist
for col in features + [target]:
    if col not in df.columns:
        raise ValueError(f"Column missing: {col}")

# Extract values
X = df[features].astype(float).values
y = df[target].astype(float).values.reshape(-1, 1)

# ------------------------------
# 3️⃣ Normalize Data
# ------------------------------
scaler_X = MinMaxScaler()
scaler_y = MinMaxScaler()

X_scaled = scaler_X.fit_transform(X)
y_scaled = scaler_y.fit_transform(y)

# ------------------------------
# 4️⃣ Create Time Sequences
# ------------------------------
def create_sequences(X, y, time_steps=5):
    Xs, ys = [], []
    for i in range(len(X) - time_steps):
        Xs.append(X[i:i+time_steps])
        ys.append(y[i+time_steps])
    return np.array(Xs), np.array(ys)

TIME_STEPS = 5
X_seq, y_seq = create_sequences(X_scaled, y_scaled, TIME_STEPS)

# ------------------------------
# 5️⃣ Train-Test Split
# ------------------------------
split = int(0.8 * len(X_seq))
X_train, X_test = X_seq[:split], X_seq[split:]
y_train, y_test = y_seq[:split], y_seq[split:]

print("Training shape:", X_train.shape)
print("Testing shape :", X_test.shape)

# ------------------------------
# 6️⃣ Build LSTM Model
# ------------------------------
model = Sequential([
    LSTM(64, return_sequences=True,
         input_shape=(X_train.shape[1], X_train.shape[2])),
    LSTM(32),
    Dense(1)
])

model.compile(
    optimizer="adam",
    loss="mse"
)

model.summary()

# ------------------------------
# 7️⃣ Train Model
# ------------------------------
early_stop = EarlyStopping(
    monitor="val_loss",
    patience=5,
    restore_best_weights=True
)

history = model.fit(
    X_train, y_train,
    epochs=20,
    batch_size=64,
    validation_split=0.1,
    callbacks=[early_stop],
    verbose=1
)

# ------------------------------
# 8️⃣ Make Predictions
# ------------------------------
y_pred_scaled = model.predict(X_test)

# Inverse transform
y_test_inv = scaler_y.inverse_transform(y_test)
y_pred_inv = scaler_y.inverse_transform(y_pred_scaled)

# ------------------------------
# 9️⃣ Evaluation Metrics
# ------------------------------
rmse = np.sqrt(mean_squared_error(y_test_inv, y_pred_inv))
mae = mean_absolute_error(y_test_inv, y_pred_inv)

print("\nFinal Evaluation Results")
print("-------------------------")
print("RMSE:", rmse)
print("MAE :", mae)
