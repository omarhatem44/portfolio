import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.linear_model import Ridge
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score

# Load dataset
path = "C:\\Users\\Smart\\OneDrive\\Desktop\\ML\\datasets\\gym\\gym_members_exercise_tracking.csv"
data = pd.read_csv(path)

data = data.apply(pd.to_numeric, errors='coerce')  
data = data.fillna(0)  

# Extract features
features = ["Age", "Gender", "Weight (kg)", "Height (m)", "BMI", "Experience_Level", "Workout_Frequency (days/week)", 
            "Water_Intake (liters)", "Fat_Percentage", "Max_BPM", "Avg_BPM", "Resting_BPM", "Session_Duration (hours)", "Workout_Type"]
X = data[features]
y = data["Calories_Burned"]

# Standardize features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Polynomial Features (degree 3)
poly = PolynomialFeatures(degree=3, include_bias=False)
X_poly = poly.fit_transform(X_scaled)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X_poly, y, test_size=0.2, random_state=42)

# Ridge Regression (L2 Regularization)
lambda_ = 0.05
ridge = Ridge(alpha=lambda_)
ridge.fit(X_train, y_train)

# Predictions
y_pred = ridge.predict(X_test)

# Compute Cost (Mean Squared Error)
cost = np.mean((y_pred - y_test) ** 2) / 2
print(f"Final Cost: {cost}")

r2 = r2_score(y_test, y_pred)
print(f"R² Score: {r2}")


# Plot Convergence
plt.plot(y_test.values, y_pred, 'bo')
plt.xlabel("Actual Calories Burned")
plt.ylabel("Predicted Calories Burned")
plt.title("Actual vs Predicted Calories Burned")
plt.show()
