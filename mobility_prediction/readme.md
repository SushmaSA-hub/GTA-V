# Mobility Prediction Using Machine Learning and Deep Learning

## 📌 Project Overview

This project focuses on predicting vehicle travel time (mobility prediction) using traffic simulation data generated from SUMO (Simulation of Urban Mobility). Multiple machine learning and deep learning models were evaluated and compared to determine the most suitable approach for trip-level mobility prediction.

The objective is to analyze how different models perform on structured traffic data and to understand the impact of feature selection on prediction accuracy.

---

## 📊 Dataset Description

The dataset was generated using SUMO traffic simulation and converted to CSV format.

### Target Variable:
- `tripinfo_duration` → Travel time (seconds)

### Features Used:

- `tripinfo_routeLength`
- `tripinfo_waitingTime`
- `tripinfo_timeLoss`
- `tripinfo_stopTime`
- `tripinfo_speedFactor`
- `tripinfo_departDelay`



---

## ⚙️ Environment Setup

### 1️⃣ Create Virtual Environment

```bash
python3 -m venv mobility_env
source mobility_env/bin/activate
