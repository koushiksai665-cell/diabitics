"""
app.py
Flask backend for the AI-Based Diabetes Predictor.
Loads diabetes_ensemble.pkl and exposes a single POST /predict endpoint
that diabetes_ai_v3.html (or any frontend) can call.
"""
import joblib
import numpy as np
from flask import Flask, request, jsonify
from flask_cors import CORS

app = Flask(__name__)
CORS(app)  # allows the HTML file (different origin) to call this API

bundle = joblib.load("diabetes_ensemble.pkl")
rf_model = bundle["rf_model"]
gb_model = bundle["gb_model"]
scaler = bundle["scaler"]
FEATURES = bundle["feature_order"]
W_RF, W_GB = bundle["weights"]["rf"], bundle["weights"]["gb"]


def categorize(prob):
    if prob < 0.35:
        return "Non-Diabetic"
    elif prob < 0.65:
        return "At Risk"
    else:
        return "Diabetic"


@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json()

    # Validate all required fields are present
    missing = [f for f in FEATURES if f not in data]
    if missing:
        return jsonify({"error": f"Missing fields: {missing}"}), 400

    # Build feature vector in the exact training order
    row = np.array([[data[f] for f in FEATURES]])
    row_scaled = scaler.transform(row)

    rf_prob = rf_model.predict_proba(row_scaled)[0][1]
    gb_prob = gb_model.predict_proba(row_scaled)[0][1]
    final_prob = W_RF * rf_prob + W_GB * gb_prob

    return jsonify({
        "probability": round(float(final_prob), 4),
        "risk_category": categorize(final_prob),
        "rf_probability": round(float(rf_prob), 4),
        "gb_probability": round(float(gb_prob), 4)
    })


@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "ok"})


if __name__ == "__main__":
    import os
    port = int(os.environ.get("PORT", 5000))
    # debug=True only for local dev — turn off before deploying
    app.run(host="0.0.0.0", port=port, debug=True)
