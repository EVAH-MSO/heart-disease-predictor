"""
Explore the heart disease dataset.
"""
import pandas as pd

df = pd.read_csv("data/heart_clean.csv")

print("=" * 60)
print("HEART DISEASE DATASET — COLUMN MEANINGS")
print("=" * 60)

meanings = {
    "age":      "Age in years",
    "sex":      "Sex (1 = male, 0 = female)",
    "cp":       "Chest pain type (0-3)",
    "trestbps": "Resting blood pressure (mm Hg)",
    "chol":     "Serum cholesterol (mg/dl)",
    "fbs":      "Fasting blood sugar > 120 mg/dl (1 = true)",
    "restecg":  "Resting ECG results (0-2)",
    "thalach":  "Max heart rate achieved",
    "exang":    "Exercise-induced angina (1 = yes)",
    "oldpeak":  "ST depression induced by exercise",
    "slope":    "Slope of peak exercise ST segment (0-2)",
    "ca":       "Number of major vessels colored by fluoroscopy (0-3)",
    "thal":     "Thalassemia (1 = normal, 2 = fixed defect, 3 = reversible)",
    "target":   "🎯 0 = no disease, 1-4 = disease"
}

for col, meaning in meanings.items():
    print(f"  {col:10s} → {meaning}")

print("\n" + "=" * 60)
print("TARGET DISTRIBUTION")
print("=" * 60)
print(df["target"].value_counts().sort_index())

print("\n" + "=" * 60)
print("BASIC STATISTICS")
print("=" * 60)
print(df.describe())

print("\n" + "=" * 60)
print("CORRELATION WITH TARGET")
print("=" * 60)
df["target_binary"] = (df["target"] > 0).astype(int)
correlations = df.corr()["target_binary"].sort_values(ascending=False)
print(correlations)
