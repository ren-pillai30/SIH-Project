# SIH-Project
An AI-powered model to predict crop yield based on historical agricultural data. This project uses machine learning to forecast production and identify key factors for optimizing harvest.

## 🎯 Project Goal

The primary objective is to build a reliable prediction model that can:
1.  **Forecast Crop Yield:** Accurately predict the yield (tons per hectare) for a given crop in a specific location and year.
2.  **Identify Key Factors:** Analyze which features (e.g., area, crop type, district) have the most significant impact on yield.
3.  **Enable Optimization:** Provide a foundation for a recommendation system that can suggest strategies to improve agricultural output.

---

## ✨ Key Features

- **Data Preprocessing & Cleaning:** Robust scripts to handle missing values and prepare the dataset for modeling.
- **Random Forest Model:** Utilizes a `RandomForestRegressor` for its high accuracy and ability to handle complex, non-linear relationships.
- **Yield Prediction:** A trained model capable of predicting yield based on inputs like `State`, `District`, `Crop`, `Season`, and `Area`.
- **Feature Importance Analysis:** Visualizations to show the most influential factors affecting crop yield.
- **Performance Evaluation:** The model is evaluated using metrics like **Root Mean Squared Error (RMSE)** and **R² Score**.

---

## 🛠️ Tech Stack

- **Language:** Python 3.x
- **Libraries:**
  - **Data Manipulation:** Pandas, NumPy
  - **Machine Learning:** Scikit-learn
  - **Visualization:** Matplotlib, Seaborn

---

## 🚀 Getting Started

### Prerequisites
- Python 3.8 or higher
- Pip package manager


