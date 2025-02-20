import pandas as pd
from sklearn.preprocessing import StandardScaler
import os

# Load the training and test data
X_train = pd.read_csv("data/processed/X_train.csv")
X_test = pd.read_csv("data/processed/X_test.csv")

# Initialize the scaler and apply it
scaler = StandardScaler()

# Remove date column if exist
if 'date' in X_train.columns:
    X_train = X_train.drop('date', axis=1)
    X_test = X_test.drop('date', axis=1)

X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Convert the arrays back to DataFrames to retain column names
X_train_scaled = pd.DataFrame(X_train_scaled, columns=X_train.columns)
X_test_scaled = pd.DataFrame(X_test_scaled, columns=X_test.columns)

# Save the scaled datasets
X_train_scaled.to_csv("data/processed/X_train_scaled.csv", index=False)
X_test_scaled.to_csv("data/processed/X_test_scaled.csv", index=False)
