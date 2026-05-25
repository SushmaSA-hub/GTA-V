import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# ------------------------------
# 1️⃣ Load Data
# ------------------------------
df = pd.read_csv("tripinfos_clean.csv", sep=";")

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

X = df[features].astype(float)
y = df[target].astype(float)

# ------------------------------
# 3️⃣ Train-Test Split (80–20)
# ------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

print("Training samples:", X_train.shape[0])
print("Testing samples :", X_test.shape[0])

# ------------------------------
# 4️⃣ Build Random Forest Model
# ------------------------------
rf = RandomForestRegressor(
    n_estimators=200,
    max_depth=None,
    random_state=42,
    n_jobs=-1
)

rf.fit(X_train, y_train)

# ------------------------------
# 5️⃣ Predictions
# ------------------------------
y_pred = rf.predict(X_test)

# ------------------------------
# 6️⃣ Evaluation Metrics
# ------------------------------
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print("\nRandom Forest Results")
print("----------------------")
print("RMSE:", rmse)
print("MAE :", mae)
print("R²  :", r2)

# ------------------------------
# 7️⃣ Feature Importance
# ------------------------------
importances = rf.feature_importances_

print("\nFeature Importance:")
for feature, importance in zip(features, importances):
    print(f"{feature}: {importance:.4f}")
