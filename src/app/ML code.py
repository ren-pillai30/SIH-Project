import pandas as pd
import numpy as np
import time
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import xgboost as xgb # Using XGBoost for potentially higher accuracy

# Load the dataset (make sure the path is correct)
# For GitHub Codespaces, ensure 'crop_production.csv' is in the same folder
df = pd.read_csv(r"C:\Users\Aryan\Downloads\crop_production.csv")

print("Dataset Preview:")
print(df.head())

# --- Data Cleaning and Preprocessing ---
# Keep only valid rows
df = df[(df["Area"] > 0) & (df["Production"] > 0)]

# Calculate yield (our target variable)
df["Yield"] = df["Production"] / df["Area"]

# Handle infinite / missing values that might result from division
df = df.replace([np.inf, -np.inf], np.nan)
df = df.dropna(subset=["Yield"])

# Remove extreme outliers to help the model generalize better
df = df[df["Yield"] <= 25]

# --- 1. Feature Engineering (Highly Recommended) ---
# To significantly improve accuracy, this is where you would merge external datasets.
# For example:
# weather_df = pd.read_csv("district_weather.csv")
# soil_df = pd.read_csv("district_soil_data.csv")
# df = pd.merge(df, weather_df, on=["District_Name", "Crop_Year"])
# df = pd.merge(df, soil_df, on=["District_Name"])
# ----------------------------------------------------

# One-hot encode categorical variables to convert them to numbers
categorical = ["State_Name", "District_Name", "Season", "Crop"]
# ADD any new features (like 'Rainfall', 'Temperature') to the list below
numeric_features = ["Crop_Year", "Area"] 
df_encoded = pd.get_dummies(df[categorical + numeric_features])

# Define features (X) and target (y)
X = df_encoded
y = df["Yield"]

# --- 2. Improved Data Splitting ---
# Using a random train-test split is more robust than manual slicing.
# This prevents bias and gives a more reliable measure of model performance.
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

print(f"Train set size: {X_train.shape}")
print(f"Test set size: {X_test.shape}")

# --- 3. Use a More Advanced Model (XGBoost) ---
# XGBoost is a powerful gradient boosting algorithm that often outperforms Random Forest.
model = xgb.XGBRegressor(
    n_estimators=1000,      # Number of trees to build
    learning_rate=0.05,     # How much to shrink the contribution of each tree
    early_stopping_rounds=50, # Stop training if performance doesn't improve for 50 rounds
    random_state=42
)

print("\nStarting XGBoost model training...")
start_time = time.time()
# The eval_set allows the model to use early stopping
model.fit(X_train, y_train, eval_set=[(X_test, y_test)], verbose=False)
print(f"Training completed in {time.time() - start_time:.2f} seconds.")

# --- 4. Evaluate the Model ---
y_pred = model.predict(X_test)

rmse = np.sqrt(mean_squared_error(y_test, y_pred))
r2 = r2_score(y_test, y_pred)

print("\nModel Performance:")
print(f"RMSE: {rmse:.2f}")
print(f"R² Score: {r2:.2f}")

# --- Example Prediction ---
sample = pd.DataFrame({
    "State_Name": ["Andaman and Nicobar Islands"],
    "District_Name": ["NICOBARS"],
    "Crop_Year": [2000],
    "Season": ["Whole Year "],
    "Crop": ["Banana"],
    "Area": [176.0]
})

sample_encoded = pd.get_dummies(sample)
sample_encoded = sample_encoded.reindex(columns=X.columns, fill_value=0)

pred = model.predict(sample_encoded)
print(f"\nPredicted Yield for sample input (tons/ha): {pred[0]:.2f}")

# --- Visualizations ---
# (A) Plot Predicted vs Actual
plt.figure(figsize=(6, 6))
plt.scatter(y_test, y_pred, alpha=0.6, color="blue", edgecolors="k")
plt.xlabel("Actual Yield (tons/ha)")
plt.ylabel("Predicted Yield (tons/ha)")
plt.title("Predicted vs Actual Crop Yield")
plt.plot([y.min(), y.max()], [y.min(), y.max()], "r--", lw=2)
plt.tight_layout()
plt.show()

# (B) Feature Importance
# Get feature importances and plot the top 10
importances = model.feature_importances_
indices = np.argsort(importances)[-10:]
top_features = X.columns[indices]

plt.figure(figsize=(8, 5))
plt.barh(range(len(indices)), importances[indices], align="center", color="green")
plt.yticks(range(len(indices)), top_features)
plt.xlabel("Feature Importance")
plt.title("Top 10 Important Features")
plt.tight_layout()
plt.show()