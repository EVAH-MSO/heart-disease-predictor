"""
Flask web app for heart disease prediction.
"""
import os

from flask import Flask, render_template, request, jsonify
from predict import predict

app = Flask(__name__)


@app.route("/")
def home():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def make_prediction():
    try:
        data = {
            "age":      float(request.form["age"]),
            "sex":      int(request.form["sex"]),
            "cp":       int(request.form["cp"]),
            "trestbps": float(request.form["trestbps"]),
            "chol":     float(request.form["chol"]),
            "fbs":      int(request.form["fbs"]),
            "restecg":  int(request.form["restecg"]),
            "thalach":  float(request.form["thalach"]),
            "exang":    int(request.form["exang"]),
            "oldpeak":  float(request.form["oldpeak"]),
            "slope":    int(request.form["slope"]),
            "ca":       int(request.form["ca"]),
            "thal":     int(request.form["thal"]),
        }

        result = predict(data)
        return render_template("result.html", result=result, patient=data)

    except Exception as e:
        return render_template("result.html", error=str(e))


@app.route("/api/predict", methods=["POST"])
def api_predict():
    try:
        data = request.get_json()
        result = predict(data)
        return jsonify(result), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 400


@app.route("/health")
def health():
    return jsonify({"status": "healthy"}), 200

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    debug = os.environ.get("FLASK_DEBUG", "false").lower() == "true"
    app.run(host="0.0.0.0", port=port, debug=debug)
