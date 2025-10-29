import numpy as np
from sklearn.linear_model import LinearRegression
import joblib

# Dummy training data
# [temperature, humidity, soil_moisture] → watering time (hours)
X = np.array([
    [25, 60, 30],
    [30, 40, 20],
    [35, 30, 15],
    [20, 80, 50],
    [28, 55, 35]
])
y = np.array([12, 8, 5, 24, 10])

model = LinearRegression()
model.fit(X, y)

joblib.dump(model, "watering_model.pkl")
print("✅ Watering model saved as watering_model.pkl")
