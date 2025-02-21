import pandas as pd
from sklearn.model_selection import train_test_split
import os

# Load the dataset raw.csv
df = pd.read_csv("data/raw/raw.csv")

# Separate features and target variable
X = df.iloc[:, :-1]  # All columns except the last one
y = df.iloc[:, -1]   # Last column (silica_concentrate)

# Split the data into training and test sets 
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

os.makedirs("data/processed", exist_ok=True)

# Save the datasets
X_train.to_csv("data/processed/X_train.csv", index=False)
X_test.to_csv("data/processed/X_test.csv", index=False)
y_train.to_csv("data/processed/y_train.csv", index=False)
y_test.to_csv("data/processed/y_test.csv", index=False)

