# California Housing Regression

A structured machine-learning regression project using the California Housing dataset. The repository covers exploratory analysis, feature scaling, regularized linear models, hyperparameter selection, residual analysis, model comparison, and serialization of the selected model for reuse.

## Objective

Predict median house values from eight numerical features:

- median income
- house age
- average rooms
- average bedrooms
- population
- average occupancy
- latitude
- longitude

## Workflow

- load and inspect the dataset
- explore correlations and feature distributions
- standardize model inputs
- train Linear, Ridge, Lasso, and ElasticNet regression models
- tune ElasticNet across `alpha` and `l1_ratio`
- compare model performance using R² and error diagnostics
- inspect residual plots
- save the selected model and fitted scaler

## Results

| Model | R² score (approx.) |
|---|---:|
| Linear Regression | 0.593 |
| Ridge Regression | 0.595 |
| Lasso Regression | 0.590 |
| ElasticNet Regression | **0.596** |

Selected ElasticNet configuration:

| Parameter | Value |
|---|---:|
| `alpha` | 0.01 |
| `l1_ratio` | 0.7 |
| `max_iter` | 5000 |

These results belong to this repository's recorded experiment and should be interpreted as a baseline for the selected preprocessing and validation setup.

## Repository Contents

- `california_housing_regression.ipynb` — analysis, training, evaluation, and diagnostic plots
- `best_elasticnet_model.pkl` — serialized ElasticNet model
- `scaler.pkl` — fitted `StandardScaler`
- `README.md` — project documentation

## Reusing the Saved Model

```python
import pickle
import numpy as np

with open("best_elasticnet_model.pkl", "rb") as model_file:
    model = pickle.load(model_file)

with open("scaler.pkl", "rb") as scaler_file:
    scaler = pickle.load(scaler_file)

# Feature order:
# [MedInc, HouseAge, AveRooms, AveBedrms,
#  Population, AveOccup, Latitude, Longitude]
features = np.array([[5.0, 30.0, 6.0, 1.0, 1500, 3.0, 34.5, -118.5]])

scaled_features = scaler.transform(features)
prediction = model.predict(scaled_features)

print("Predicted median house value:", prediction[0])
```

## Scope

This is a compact structured-data ML project. Its purpose is to demonstrate a reproducible regression workflow and comparison of regularized linear models; it is not presented as a production housing-valuation service.
