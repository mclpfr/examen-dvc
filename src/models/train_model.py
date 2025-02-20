import pandas as pd
from sklearn.ensemble import RandomForestRegressor
import joblib
import os

# Load the normalized training data
X_train = pd.read_csv("data/processed/X_train_scaled.csv")
y_train = pd.read_csv("data/processed/y_train.csv").squeeze()

# Load the best parameters found by GridSearch
best_params = joblib.load("models/best_params.pkl")

# Create and train the model using the best parameters
model = RandomForestRegressor(random_state=42, **best_params)
model.fit(X_train, y_train)

# Save the trained model
joblib.dump(model, "models/trained_model.pkl")

