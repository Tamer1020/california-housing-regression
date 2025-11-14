# California Housing Regression – End-to-End ML Project

This project demonstrates a complete Machine Learning workflow using the **California Housing dataset**.  
We apply several regression techniques and evaluate their performance, then select and save the best model.

---

##  Project Overview

The goal of this project is to **predict median house prices** based on features such as:

- Median income  
- House age  
- Average rooms  
- Average bedrooms  
- Population  
- Occupancy  
- Latitude & Longitude  

The project includes:

- Data loading & preprocessing  
- Exploratory Data Analysis (EDA)  
- Correlation & heatmaps  
- Residual analysis & diagnostic plots  
- Multiple Linear Regression  
- Ridge Regression  
- Lasso Regression  
- ElasticNet Regression (Grid Search over alpha & l1_ratio)  
- Model comparison using R² and error metrics  
- Saving the best model (`ElasticNet`) and the scaler for deployment  

---

##  Models Tested

| Model                | Best R² Score (approx.) |
|----------------------|-------------------------|
| Linear Regression    | ~0.593                  |
| Ridge Regression     | ~0.595                  |
| Lasso Regression     | ~0.59 (over-penalized)  |
| ElasticNet Regression| **~0.596 (Best)**       |

The best ElasticNet configuration found:

- `alpha = 0.01`  
- `l1_ratio = 0.7`  
- `max_iter = 5000`

This combination gave the best balance between bias/variance and handled multicollinearity in the features.

---

##  Files in this Repository

- `Multiple_Linear_Regression_California_Housing.ipynb`  
  → Main notebook with EDA, training, evaluation, and plots.

- `best_elasticnet_model.pkl`  
  → Saved ElasticNet model with the best hyperparameters.

- `scaler.pkl`  
  → Fitted `StandardScaler` used to normalize the features before training.

- `README.md`  
  → Project documentation (this file).

---

##  How to Use the Saved Model

```python
import pickle
import numpy as np

# Load model
with open("best_elasticnet_model.pkl", "rb") as f:
    model = pickle.load(f)

# Load scaler
with open("scaler.pkl", "rb") as f:
    scaler = pickle.load(f)

# Example input (replace with real values)
# [MedInc, HouseAge, AveRooms, AveBedrms, Population, AveOccup, Latitude, Longitude]
x = np.array([[5.0, 30.0, 6.0, 1.0, 1500, 3.0, 34.5, -118.5]])

# Scale data
x_scaled = scaler.transform(x)

# Predict
prediction = model.predict(x_scaled)
print("Predicted price:", prediction[0])
