import pandas as pd
import numpy as np
import time
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt
from google.colab import files

df = pd.read_csv("crop_production.csv")

print("Dataset Preview:")
print(df.head())

# Keep only valid rows
df = df[(df["Area"] > 0) & (df["Production"] > 0)]

# Calculate yield
df["Yield"] = df["Production"] / df["Area"]

# Handle infinite / missing values
df = df.replace([np.inf, -np.inf], np.nan)
df = df.dropna(subset=["Yield"])

# Remove extreme outliers
df = df[df["Yield"] <= 25]

# One-hot encode categorical variables
categorical = ["State_Name", "District_Name", "Season", "Crop"]
df_encoded = pd.get_dummies(df[categorical + ["Crop_Year", "Area"]])

# Features and target
X = df_encoded
y = df["Yield"]

# Define custom train/test sets by slicing indices
train_idx = list(range(0, 5000)) + list(range(10000, len(df)))
test_idx = list(range(5000, 10000))

X_train = X.iloc[train_idx]
y_train = y.iloc[train_idx]

X_test = X.iloc[test_idx]
y_test = y.iloc[test_idx]

print("Train set size:", X_train.shape, "Test set size:", X_test.shape)

model = RandomForestRegressor(n_estimators=100, random_state=42)

print("Starting training on custom train set...")
start_time = time.time()
model.fit(X_train, y_train)
print(f"Training completed in {time.time() - start_time:.2f} seconds.")

y_pred = model.predict(X_test)

rmse = np.sqrt(mean_squared_error(y_test, y_pred))
r2 = r2_score(y_test, y_pred)

print("\nCustom Test Model Performance:")
print(f"RMSE: {rmse:.2f}")
print(f"R² Score: {r2:.2f}")

# Example input
sample = pd.DataFrame({
    "State_Name": ["Andaman and Nicobar Islands"],
    "District_Name": ["NICOBARS"],
    "Crop_Year": [2000],
    "Season": ["Whole Year "],
    "Crop": ["Banana"],
    "Area": [176.0]
})

# Encode sample same as training data
sample_encoded = pd.get_dummies(sample)
sample_encoded = sample_encoded.reindex(columns=X.columns, fill_value=0)

# Predict
pred = model.predict(sample_encoded)
print("\nPredicted Yield for sample input (tons/ha):", pred[0])

plt.figure(figsize=(6,6))
plt.scatter(y_test, y_pred, alpha=0.6, color="blue", edgecolors="k")
plt.xlabel("Actual Yield (tons/ha)")
plt.ylabel("Predicted Yield (tons/ha)")
plt.title("Predicted vs Actual Crop Yield (Custom Test Set)")
plt.plot([y.min(), y.max()], [y.min(), y.max()], "r--")
plt.tight_layout()
plt.show()

importances = model.feature_importances_
indices = importances.argsort()[-10:]

plt.figure(figsize=(8,5))
plt.barh(range(len(indices)), importances[indices], align="center", color="green")
plt.yticks(range(len(indices)), [X.columns[i] for i in indices])
plt.xlabel("Feature Importance")
plt.title("Top 10 Important Features")
plt.tight_layout()
plt.show()

print("Data after preprocessing:", df.shape)
