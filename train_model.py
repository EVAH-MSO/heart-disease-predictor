"""
Train a heart disease prediction model.
"""
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, confusion_matrix, classification_report
)
import joblib
import os

# ============ LOAD ============
print("📥 Loading data...")
df = pd.read_csv("data/heart_clean.csv")
df = df.dropna()
print(f"✅ After dropping NaN: {df.shape[0]} rows")

# ============ PREPARE ============
df["target"] = (df["target"] > 0).astype(int)

X = df.drop("target", axis=1)
y = df["target"]

print(f"\n📊 Features: {list(X.columns)}")
print(f"🎯 Target distribution:\n{y.value_counts()}")

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
print(f"\n✂️  Train: {X_train.shape} | Test: {X_test.shape}")

# ============ TRAIN ============
print("\n🤖 Training models...")

models = {
    "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42),
    "Random Forest": RandomForestClassifier(
        n_estimators=200, max_depth=10, random_state=42, n_jobs=-1
    ),
}

best_model = None
best_score = 0
best_name = ""

for name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]

    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred)
    rec = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    auc = roc_auc_score(y_test, y_proba)

    print(f"\n📈 {name}:")
    print(f"   Accuracy : {acc:.3f}")
    print(f"   Precision: {prec:.3f}")
    print(f"   Recall   : {rec:.3f}")
    print(f"   F1 Score : {f1:.3f}")
    print(f"   ROC-AUC  : {auc:.3f}")

    if auc > best_score:
        best_score = auc
        best_model = model
        best_name = name

print(f"\n🏆 Best model: {best_name} (ROC-AUC = {best_score:.3f})")

# ============ REPORT ============
print("\n" + "=" * 60)
print("CLASSIFICATION REPORT")
print("=" * 60)
y_pred_best = best_model.predict(X_test)
print(classification_report(y_test, y_pred_best, target_names=["No Disease", "Disease"]))

print("Confusion Matrix:")
cm = confusion_matrix(y_test, y_pred_best)
print(f"                 Predicted No  Predicted Yes")
print(f"Actual No       {cm[0][0]:>10}     {cm[0][1]:>10}")
print(f"Actual Yes      {cm[1][0]:>10}     {cm[1][1]:>10}")

if hasattr(best_model, "feature_importances_"):
    print("\n" + "=" * 60)
    print("FEATURE IMPORTANCE")
    print("=" * 60)
    importances = pd.DataFrame({
        "feature": X.columns,
        "importance": best_model.feature_importances_
    }).sort_values("importance", ascending=False)
    print(importances.to_string(index=False))

# ============ SAVE ============
os.makedirs("model", exist_ok=True)
joblib.dump(best_model, "model/heart_model.pkl")
joblib.dump(list(X.columns), "model/feature_names.pkl")

print("\n✅ Model saved to model/heart_model.pkl")
print("✅ Feature names saved to model/feature_names.pkl")
