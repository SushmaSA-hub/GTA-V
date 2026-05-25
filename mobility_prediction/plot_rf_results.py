import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split

# Load dataset
df = pd.read_csv("tripinfos_clean.csv", sep=";")

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

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Train model
rf = RandomForestRegressor(
    n_estimators=200,
    random_state=42,
    n_jobs=-1
)

rf.fit(X_train, y_train)

# Predict
y_pred = rf.predict(X_test)

# Plot
plt.figure()
plt.scatter(y_test, y_pred, alpha=0.5)
plt.xlabel("Actual Travel Time (seconds)")
plt.ylabel("Predicted Travel Time (seconds)")
plt.title("Actual vs Predicted Travel Time (Random Forest)")

# Perfect prediction reference line
min_val = min(y_test.min(), y_pred.min())
max_val = max(y_test.max(), y_pred.max())
plt.plot([min_val, max_val], [min_val, max_val])

plt.show()
