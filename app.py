"""
Predictive Pulse - Flask API Backend
"""
import json, os, sys
sys.path.insert(0, os.path.dirname(__file__))

from flask import Flask, request, jsonify, render_template, send_from_directory
from ml_engine import predict, simulate_trend, get_recommendations, load_models, train_models, FEATURE_NAMES, RISK_LABELS

app = Flask(__name__, template_folder="templates", static_folder="static")

# Load models once at startup
_models = None

def get_models():
    global _models
    if _models is None:
        model_dir = os.path.join(os.path.dirname(__file__), "models")
        if not os.path.exists(os.path.join(model_dir, "scaler.pkl")):
            print("📦 Training models for first run...")
            train_models()
        _models = load_models()
    return _models


def validate_inputs(data):
    required = {
        "age":           (18, 90),
        "weight_kg":     (40, 150),
        "height_cm":     (140, 210),
        "heart_rate":    (45, 120),
        "stress_level":  (1, 10),
        "sleep_hours":   (3, 12),
        "exercise_days": (0, 7),
        "salt_intake":   (1, 10),
        "smoking":       (0, 1),
        "diabetes":      (0, 1),
        "alcohol_units": (0, 30),
    }
    inputs = {}
    errors = []
    for field, (lo, hi) in required.items():
        val = data.get(field)
        if val is None:
            errors.append(f"Missing field: {field}")
            continue
        try:
            val = float(val)
        except (TypeError, ValueError):
            errors.append(f"Invalid value for {field}")
            continue
        if not (lo <= val <= hi):
            errors.append(f"{field} must be between {lo} and {hi}")
            continue
        inputs[field] = val
    return inputs, errors


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/api/predict", methods=["POST"])
def api_predict():
    data = request.get_json(force=True) or {}
    inputs, errors = validate_inputs(data)
    if errors:
        return jsonify({"error": errors}), 400
    try:
        models  = get_models()
        result  = predict(inputs, models)
        recs    = get_recommendations(inputs, result)
        trend   = simulate_trend(inputs, days=data.get("trend_days", 30), models=models)
        return jsonify({
            "prediction": result,
            "recommendations": recs,
            "trend": trend,
            "inputs": inputs,
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/health")
def health():
    return jsonify({"status": "ok", "models_loaded": _models is not None})


if __name__ == "__main__":
    print("🚀 Starting Predictive Pulse API...")
    get_models()
    app.run(debug=False, port=5050)
