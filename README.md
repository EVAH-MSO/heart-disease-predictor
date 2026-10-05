# 🏥 Heart Disease Predictor

An ML-powered web app that predicts heart disease risk from patient health metrics.

> ⚠️ **Disclaimer:** For educational purposes only. Not medical advice.

## 📊 Model Performance
- **Accuracy:** ~88%
- **ROC-AUC:** ~0.94
- **Algorithm:** Logistic Regression / Random Forest

## 🛠️ Tech Stack
- Python, scikit-learn, Pandas
- Flask (web + API)
- Docker (upcoming)
- GitHub Actions (upcoming)

## 🚀 Run Locally

\`\`\`bash
# Setup
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt

# Get data & train
python download_data.py
python train_model.py

# Run app
python app.py
\`\`\`

Visit: http://localhost:5000

## 🔌 API Usage

\`\`\`bash
curl -X POST http://localhost:5000/api/predict \\
  -H "Content-Type: application/json" \\
  -d '{"age":55,"sex":1,"cp":0,"trestbps":140,"chol":250,"fbs":0,"restecg":1,"thalach":150,"exang":1,"oldpeak":2.0,"slope":1,"ca":1,"thal":3}'
\`\`\`

## 📁 Project Structure

- \`templates/\` + \`static/\` → Frontend
- \`app.py\` + \`predict.py\` → API + ML logic
- \`train_model.py\` → Model training
- \`model/\` → Saved model artifacts
