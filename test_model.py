"""
Test the saved model with sample patient data.
"""
import joblib
import pandas as pd

model = joblib.load("model/heart_model.pkl")
features = joblib.load("model/feature_names.pkl")

print(f"✅ Model loaded")
print(f"✅ Features: {features}\n")

patient = {
    "age": 55, "sex": 1, "cp": 0, "trestbps": 140, "chol": 250,
    "fbs": 0, "restecg": 1, "thalach": 150, "exang": 1,
    "oldpeak": 2.0, "slope": 1, "ca": 1, "thal": 3
}

df = pd.DataFrame([patient])[features]

prediction = model.predict(df)[0]
probability = model.predict_proba(df)[0]

print("=" * 50)
print("PATIENT:")
print("=" * 50)
for k, v in patient.items():
    print(f"  {k:10s}: {v}")

print("\n" + "=" * 50)
print("PREDICTION:")
print("=" * 50)
print(f"  Result: {'⚠️  Heart Disease Likely' if prediction == 1 else '✅ No Heart Disease'}")
print(f"  Confidence: {probability[prediction] * 100:.1f}%")
print(f"  Disease probability: {probability[1] * 100:.1f}%")

