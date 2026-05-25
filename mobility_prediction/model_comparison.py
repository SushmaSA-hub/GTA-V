import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from catboost import CatBoostRegressor
from sklearn.preprocessing import MinMaxScaler
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import GRU, Dense
from tensorflow.keras.callbacks import EarlyStopping

# ------------------------------
# Load Data
# ------------------------------
df = pd.read_csv("tripinfos_clean.csv", sep=";")


features = [
    "tripinfo_routeLength",
    "tripinfo_waitingTime",
    "tripinfo_stopTime",
    "tripinfo_speedFactor",
    "tripinfo_departDelay"
]

target = "tripinfo_duration"

X = df[features].astype(float)
y = df[target].astype(float)

# ------------------------------
# Train-Test Split
# ------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# ------------------------------
# Define Models
# ------------------------------
models = {
    "Decision Tree": DecisionTreeRegressor(random_state=42),
    "Random Forest": RandomForestRegressor(n_estimators=200, random_state=42, n_jobs=-1),
    "XGBoost": XGBRegressor(n_estimators=200, random_state=42, n_jobs=-1),
    "LightGBM": LGBMRegressor(n_estimators=200, random_state=42),
    "CatBoost": CatBoostRegressor(iterations=200, verbose=0, random_state=42)
}

results = []

# ------------------------------
# Train Classical Models
# ------------------------------
for name, model in models.items():
    print(f"\nTraining {name}...")
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    results.append([name, rmse, mae, r2])

# ------------------------------
# GRU Model (Deep Learning)
# ------------------------------
print("\nTraining GRU...")

scaler_X = MinMaxScaler()
scaler_y = MinMaxScaler()

X_scaled = scaler_X.fit_transform(X)
y_scaled = scaler_y.fit_transform(y.values.reshape(-1, 1))

def create_sequences(X, y, time_steps=5):
    Xs, ys = [], []
    for i in range(len(X) - time_steps):
        Xs.append(X[i:i+time_steps])
        ys.append(y[i+time_steps])
    return np.array(Xs), np.array(ys)

TIME_STEPS = 5
X_seq, y_seq = create_sequences(X_scaled, y_scaled, TIME_STEPS)

split = int(0.8 * len(X_seq))
X_train_gru, X_test_gru = X_seq[:split], X_seq[split:]
y_train_gru, y_test_gru = y_seq[:split], y_seq[split:]

gru_model = Sequential([
    GRU(64, return_sequences=True, input_shape=(TIME_STEPS, len(features))),
    GRU(32),
    Dense(1)
])

gru_model.compile(optimizer="adam", loss="mse")

early_stop = EarlyStopping(patience=5, restore_best_weights=True)

gru_model.fit(
    X_train_gru, y_train_gru,
    epochs=20,
    batch_size=64,
    validation_split=0.1,
    callbacks=[early_stop],
    verbose=0
)

y_pred_gru_scaled = gru_model.predict(X_test_gru)
y_test_gru_inv = scaler_y.inverse_transform(y_test_gru)
y_pred_gru_inv = scaler_y.inverse_transform(y_pred_gru_scaled)

rmse_gru = np.sqrt(mean_squared_error(y_test_gru_inv, y_pred_gru_inv))
mae_gru = mean_absolute_error(y_test_gru_inv, y_pred_gru_inv)
r2_gru = r2_score(y_test_gru_inv, y_pred_gru_inv)

results.append(["GRU", rmse_gru, mae_gru, r2_gru])

# ------------------------------
# Display Results
# ------------------------------
results_df = pd.DataFrame(
    results,
    columns=["Model", "RMSE", "MAE", "R2"]
)

print("\nFinal Model Comparison (Without tripinfo_timeLoss)")
print("----------------------------------------------------")
print(results_df.sort_values(by="RMSE"))
