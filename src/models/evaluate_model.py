import pandas as pd
from sklearn.metrics import mean_squared_error, r2_score
import joblib
import os
import json

# Load the normalized test data
X_test = pd.read_csv("data/processed/X_test_scaled.csv")
y_test = pd.read_csv("data/processed/y_test.csv").squeeze()

# Load the trained model
model = joblib.load("models/trained_model.pkl")

# Make predictions on the test set
predictions = model.predict(X_test)

# Calculate evaluation metrics
mse = mean_squared_error(y_test, predictions)
r2 = r2_score(y_test, predictions)

metrics = {
    "mse": mse,
    "r2": r2
}
print("Metrics:", metrics)

# Save predictions to a CSV file
results = pd.DataFrame({
    "y_test": y_test,
    "predictions": predictions
})
os.makedirs("data", exist_ok=True)
results.to_csv("data/predictions.csv", index=False)

# Save the scores to a JSON file
os.makedirs("metrics", exist_ok=True)
with open("metrics/scores.json", "w") as f:
    json.dump(metrics, f, indent=4)

