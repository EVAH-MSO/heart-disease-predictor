"""
Download the UCI Heart Disease dataset (Cleveland).
"""
import pandas as pd
import urllib.request
import os

URL = "https://archive.ics.uci.edu/ml/machine-learning-databases/heart-disease/processed.cleveland.data"

COLUMNS = [
    "age", "sex", "cp", "trestbps", "chol", "fbs", "restecg",
    "thalach", "exang", "oldpeak", "slope", "ca", "thal", "target"
]

os.makedirs("data", exist_ok=True)

print("📥 Downloading dataset...")
urllib.request.urlretrieve(URL, "data/heart.csv")

df = pd.read_csv("data/heart.csv", names=COLUMNS, na_values="?")

print(f"✅ Downloaded: {df.shape[0]} rows, {df.shape[1]} columns")
print(df.head())
print(f"\nMissing values:\n{df.isnull().sum()}")

df.to_csv("data/heart_clean.csv", index=False)
print("\n✅ Saved clean version to data/heart_clean.csv")
