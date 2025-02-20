import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV
import joblib
import os

# Load the normalized training data
X_train = pd.read_csv("data/processed/X_train_scaled.csv")
y_train = pd.read_csv("data/processed/y_train.csv").squeeze()  # Convert DataFrame to Series

# Define the model and parameters to test
model = RandomForestRegressor(random_state=42)
param_grid = {
    'n_estimators': [100, 200],  
    'max_depth': [None, 10, 15, 20], 
    'min_samples_split': [ 4, 50]  
}

# Set up GridSearch with cross-validation
grid_search = GridSearchCV(model, param_grid, cv=5, scoring='r2', n_jobs=-1)
grid_search.fit(X_train, y_train)

# Print and save the best parameters
best_params = grid_search.best_params_
print("Best parameters:", best_params)

os.makedirs("models", exist_ok=True)
joblib.dump(best_params, "models/best_params.pkl")

