"""
Load the trained model and provide a predict() function.
"""
import joblib
import pandas as pd
import os

MODEL_PATH = "model/heart_model.pkl"
FEATURES_PATH = "model/feature_names.pkl"

if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(
        f"Model not found at {MODEL_PATH}. Run 'python train_model.py' first."
    )

_model = joblib.load(MODEL_PATH)
_features = joblib.load(FEATURES_PATH)


def predict(patient_data: dict) -> dict:
    """Predict heart disease risk."""
    df = pd.DataFrame([patient_data])[_features]

    prediction = int(_model.predict(df)[0])
    probability = float(_model.predict_proba(df)[0][1])

    if probability < 0.3:
        risk = "Low"
    elif probability < 0.6:
        risk = "Moderate"
    else:
        risk = "High"

    return {
        "prediction": prediction,
        "probability": round(probability * 100, 2),
        "risk_level": risk,
        "has_disease": prediction == 1,
    }


if __name__ == "__main__":
    sample = {
        "age": 55, "sex": 1, "cp": 0, "trestbps": 140, "chol": 250,
        "fbs": 0, "restecg": 1, "thalach": 150, "exang": 1,
        "oldpeak": 2.0, "slope": 1, "ca": 1, "thal": 3
    }
    print(predict(sample))
