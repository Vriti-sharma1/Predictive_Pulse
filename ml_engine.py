"""
Predictive Pulse - ML Engine
Trains models for BP prediction and risk classification using synthetic clinical data.
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor, RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, accuracy_score
import joblib
import os

MODEL_DIR = os.path.join(os.path.dirname(__file__), "models")

FEATURE_NAMES = [
    "age", "weight_kg", "height_cm", "heart_rate",
    "stress_level", "sleep_hours", "exercise_days",
    "salt_intake", "smoking", "diabetes", "alcohol_units"
]

RISK_LABELS = ["Optimal", "Elevated", "Stage 1 HTN", "Stage 2 HTN", "Hypertensive Crisis"]


def generate_synthetic_data(n=8000, seed=42):
    np.random.seed(seed)
    age         = np.random.normal(45, 15, n).clip(18, 90)
    weight      = np.random.normal(75, 15, n).clip(40, 150)
    height      = np.random.normal(168, 10, n).clip(140, 210)
    heart_rate  = np.random.normal(72, 12, n).clip(45, 120)
    stress      = np.random.uniform(1, 10, n)
    sleep       = np.random.normal(7, 1.5, n).clip(3, 12)
    exercise    = np.random.randint(0, 8, n).astype(float)
    salt        = np.random.uniform(1, 10, n)
    smoking     = np.random.binomial(1, 0.2, n).astype(float)
    diabetes    = np.random.binomial(1, 0.1, n).astype(float)
    alcohol     = np.random.uniform(0, 14, n)

    bmi = weight / (height / 100) ** 2

    # Physiologically-grounded BP formula
    systolic = (
        110
        + (age - 30) * 0.5
        + (bmi - 22) * 1.2
        + (heart_rate - 70) * 0.3
        + stress * 1.9
        - sleep * 1.3
        - exercise * 1.0
        + salt * 2.8
        + smoking * 9
        + diabetes * 11
        + alcohol * 0.6
        + np.random.normal(0, 4, n)
    )
    diastolic = (
        70
        + (age - 30) * 0.25
        + (bmi - 22) * 0.65
        + (heart_rate - 70) * 0.12
        + stress * 0.9
        - sleep * 0.65
        - exercise * 0.5
        + salt * 1.4
        + smoking * 4.5
        + diabetes * 5.5
        + alcohol * 0.3
        + np.random.normal(0, 3, n)
    )
    systolic  = systolic.clip(85, 200)
    diastolic = diastolic.clip(55, 130)

    def risk_class(s, d):
        if s < 120 and d < 80:   return 0
        if s < 130 and d < 80:   return 1
        if s < 140 or d < 90:    return 2
        if s < 180 and d < 120:  return 3
        return 4

    risk = np.array([risk_class(s, d) for s, d in zip(systolic, diastolic)])

    df = pd.DataFrame({
        "age": age, "weight_kg": weight, "height_cm": height,
        "heart_rate": heart_rate, "stress_level": stress,
        "sleep_hours": sleep, "exercise_days": exercise,
        "salt_intake": salt, "smoking": smoking,
        "diabetes": diabetes, "alcohol_units": alcohol,
        "systolic": systolic, "diastolic": diastolic, "risk_class": risk
    })
    return df


def train_models(verbose=True):
    os.makedirs(MODEL_DIR, exist_ok=True)
    df = generate_synthetic_data()
    X = df[FEATURE_NAMES].values
    y_sys  = df["systolic"].values
    y_dia  = df["diastolic"].values
    y_risk = df["risk_class"].values

    X_train, X_test, ys_train, ys_test, yd_train, yd_test, yr_train, yr_test = \
        train_test_split(X, y_sys, y_dia, y_risk, test_size=0.2, random_state=42)

    scaler = StandardScaler().fit(X_train)
    Xtr, Xte = scaler.transform(X_train), scaler.transform(X_test)

    # Systolic model
    sys_model = GradientBoostingRegressor(n_estimators=200, max_depth=5, learning_rate=0.08, random_state=42)
    sys_model.fit(Xtr, ys_train)
    sys_mae = mean_absolute_error(ys_test, sys_model.predict(Xte))

    # Diastolic model
    dia_model = GradientBoostingRegressor(n_estimators=200, max_depth=5, learning_rate=0.08, random_state=42)
    dia_model.fit(Xtr, yd_train)
    dia_mae = mean_absolute_error(yd_test, dia_model.predict(Xte))

    # Risk classifier
    risk_model = RandomForestClassifier(n_estimators=200, max_depth=8, random_state=42)
    risk_model.fit(Xtr, yr_train)
    risk_acc = accuracy_score(yr_test, risk_model.predict(Xte))

    joblib.dump(scaler,     os.path.join(MODEL_DIR, "scaler.pkl"))
    joblib.dump(sys_model,  os.path.join(MODEL_DIR, "systolic_model.pkl"))
    joblib.dump(dia_model,  os.path.join(MODEL_DIR, "diastolic_model.pkl"))
    joblib.dump(risk_model, os.path.join(MODEL_DIR, "risk_model.pkl"))

    if verbose:
        print(f"✅ Systolic MAE:   {sys_mae:.2f} mmHg")
        print(f"✅ Diastolic MAE:  {dia_mae:.2f} mmHg")
        print(f"✅ Risk Accuracy:  {risk_acc*100:.1f}%")

    return {"systolic_mae": sys_mae, "diastolic_mae": dia_mae, "risk_accuracy": risk_acc}


def load_models():
    scaler     = joblib.load(os.path.join(MODEL_DIR, "scaler.pkl"))
    sys_model  = joblib.load(os.path.join(MODEL_DIR, "systolic_model.pkl"))
    dia_model  = joblib.load(os.path.join(MODEL_DIR, "diastolic_model.pkl"))
    risk_model = joblib.load(os.path.join(MODEL_DIR, "risk_model.pkl"))
    return scaler, sys_model, dia_model, risk_model


def predict(inputs: dict, models=None):
    if models is None:
        models = load_models()
    scaler, sys_model, dia_model, risk_model = models

    row = np.array([[inputs[f] for f in FEATURE_NAMES]])
    X_scaled = scaler.transform(row)

    systolic  = float(sys_model.predict(X_scaled)[0])
    diastolic = float(dia_model.predict(X_scaled)[0])
    risk_idx  = int(risk_model.predict(X_scaled)[0])
    risk_probs = risk_model.predict_proba(X_scaled)[0].tolist()

    systolic  = round(max(85, min(200, systolic)))
    diastolic = round(max(55, min(130, diastolic)))

    # Feature importances for explanation
    sys_imp  = sys_model.feature_importances_
    feat_imp = {FEATURE_NAMES[i]: round(float(sys_imp[i]) * 100, 1) for i in range(len(FEATURE_NAMES))}

    return {
        "systolic":     systolic,
        "diastolic":    diastolic,
        "risk_index":   risk_idx,
        "risk_label":   RISK_LABELS[risk_idx],
        "risk_probs":   {RISK_LABELS[i]: round(p * 100, 1) for i, p in enumerate(risk_probs)},
        "feature_importance": feat_imp,
        "pulse_pressure": systolic - diastolic,
        "map":          round(diastolic + (systolic - diastolic) / 3),
        "bmi":          round(inputs["weight_kg"] / (inputs["height_cm"] / 100) ** 2, 1),
    }


def simulate_trend(inputs: dict, days=30, models=None):
    """Simulate BP trend over N days with small random daily variation."""
    if models is None:
        models = load_models()
    base = predict(inputs, models)
    trend = []
    rng = np.random.default_rng(99)
    for d in range(days):
        noise_s = rng.normal(0, 3.5)
        noise_d = rng.normal(0, 2.0)
        trend.append({
            "day":       d + 1,
            "label":     f"Day {d+1}",
            "systolic":  int(round(max(85, min(200, base["systolic"] + noise_s + d * 0.1 * np.sin(d * 0.4))))),
            "diastolic": int(round(max(55, min(130, base["diastolic"] + noise_d)))),
        })
    return trend


def get_recommendations(inputs: dict, prediction: dict) -> list:
    recs = []
    bmi = prediction["bmi"]
    risk_idx = prediction["risk_index"]

    if risk_idx >= 3:
        recs.append({"urgency": "critical", "icon": "🏥", "title": "Seek Medical Attention",
                     "detail": "Your predicted BP is in a dangerous range. Please consult a doctor immediately.", "category": "medical"})
    if inputs["stress_level"] > 6:
        recs.append({"urgency": "high", "icon": "🧘", "title": "Reduce Stress",
                     "detail": "High stress raises cortisol, which constricts blood vessels. Try daily meditation, deep breathing (4-7-8 technique), or yoga.", "category": "lifestyle"})
    if inputs["sleep_hours"] < 7:
        recs.append({"urgency": "high", "icon": "😴", "title": "Improve Sleep Quality",
                     "detail": f"You're getting ~{inputs['sleep_hours']}h sleep. Aim for 7–9h. Sleep deprivation raises BP by activating the sympathetic nervous system.", "category": "lifestyle"})
    if inputs["exercise_days"] < 3:
        recs.append({"urgency": "medium", "icon": "🏃", "title": "Exercise Regularly",
                     "detail": "30 minutes of moderate aerobic exercise (brisk walk, cycling, swimming) 5× per week can lower systolic BP by 4–9 mmHg.", "category": "activity"})
    if inputs["salt_intake"] > 5:
        recs.append({"urgency": "high", "icon": "🧂", "title": "Cut Salt Intake",
                     "detail": "High sodium intake causes water retention and raises BP. Target <2300mg/day (about 1 tsp). Avoid processed foods.", "category": "diet"})
    if inputs["smoking"]:
        recs.append({"urgency": "critical", "icon": "🚭", "title": "Quit Smoking",
                     "detail": "Each cigarette elevates BP for 30+ minutes and damages artery walls. Quitting reduces CV risk by 50% within 1 year.", "category": "lifestyle"})
    if inputs["alcohol_units"] > 7:
        recs.append({"urgency": "medium", "icon": "🍷", "title": "Reduce Alcohol",
                     "detail": f"Consuming {inputs['alcohol_units']} units/week exceeds safe limits. Heavy drinking raises BP. Limit to <7 units/week.", "category": "diet"})
    if bmi > 27:
        recs.append({"urgency": "medium", "icon": "⚖️", "title": "Weight Management",
                     "detail": f"BMI {bmi} — losing just 5 kg can lower systolic BP by 4–5 mmHg. Focus on caloric deficit + exercise.", "category": "diet"})
    if inputs["heart_rate"] > 85:
        recs.append({"urgency": "medium", "icon": "❤️", "title": "Lower Resting Heart Rate",
                     "detail": f"Resting HR of {inputs['heart_rate']} bpm is elevated. Regular aerobic exercise and stress reduction can help.", "category": "activity"})
    if inputs["diabetes"]:
        recs.append({"urgency": "high", "icon": "🩸", "title": "Manage Blood Sugar",
                     "detail": "Diabetes significantly amplifies hypertension risk. Work closely with your doctor to monitor HbA1c and BP together.", "category": "medical"})
    if not recs:
        recs.append({"urgency": "low", "icon": "✅", "title": "Excellent Health Profile!",
                     "detail": "Your lifestyle inputs are well within healthy ranges. Keep up the great work and schedule annual check-ups.", "category": "general"})
    return recs


if __name__ == "__main__":
    print("🔬 Training Predictive Pulse ML models...")
    metrics = train_models()
    print("\n📊 Sample prediction:")
    sample = {"age": 45, "weight_kg": 82, "height_cm": 175, "heart_rate": 78,
              "stress_level": 6, "sleep_hours": 6, "exercise_days": 2,
              "salt_intake": 6, "smoking": 0, "diabetes": 0, "alcohol_units": 5}
    result = predict(sample)
    print(f"  BP: {result['systolic']}/{result['diastolic']} mmHg — {result['risk_label']}")
