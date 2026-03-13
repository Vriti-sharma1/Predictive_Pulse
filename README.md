# 💓 Predictive Pulse
### ML-Powered Blood Pressure Intelligence for Patients

---

## Overview
Predictive Pulse is a full-stack application combining a **Python ML backend** (Flask + scikit-learn) with a polished web UI for real-time blood pressure prediction, risk classification, trend simulation, and personalized recommendations.

## Architecture

```
predictive_pulse/
├── app.py              ← Flask REST API server
├── ml_engine.py        ← ML models (training, prediction, recommendations)
├── requirements.txt    ← Python dependencies
├── models/             ← Saved model files (auto-generated on first run)
│   ├── scaler.pkl
│   ├── systolic_model.pkl
│   ├── diastolic_model.pkl
│   └── risk_model.pkl
└── templates/
    └── index.html      ← Full-featured web UI
```

## ML Models

| Model | Algorithm | Target | Performance |
|-------|-----------|--------|-------------|
| Systolic BP | Gradient Boosting Regressor | Systolic mmHg | MAE ~3.5 mmHg |
| Diastolic BP | Gradient Boosting Regressor | Diastolic mmHg | MAE ~2.5 mmHg |
| Risk Classifier | Random Forest | 5-class risk | ~92% accuracy |

### Input Features
- Age, Weight (kg), Height (cm)
- Heart Rate (bpm)
- Stress Level (1–10)
- Sleep Hours
- Exercise Days/Week
- Salt Intake (1–10 scale)
- Smoking (yes/no)
- Diabetes (yes/no)
- Alcohol Units/Week

### Risk Categories
| Category | Systolic | Diastolic |
|----------|----------|-----------|
| Optimal | <120 | <80 |
| Elevated | 120–129 | <80 |
| Stage 1 HTN | 130–139 | 80–89 |
| Stage 2 HTN | 140–179 | 90–119 |
| Hypertensive Crisis | ≥180 | ≥120 |

## Setup & Run

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Train Models (Optional — auto-trains on first run)
```bash
python ml_engine.py
```

### 3. Start the Server
```bash
python app.py
```

### 4. Open in Browser
```
http://localhost:5050
```

## API Endpoints

### POST /api/predict
Runs prediction on health inputs.

**Request body:**
```json
{
  "age": 45,
  "weight_kg": 80,
  "height_cm": 175,
  "heart_rate": 76,
  "stress_level": 6,
  "sleep_hours": 6.5,
  "exercise_days": 2,
  "salt_intake": 7,
  "smoking": 0,
  "diabetes": 0,
  "alcohol_units": 4,
  "trend_days": 30
}
```

**Response:**
```json
{
  "prediction": {
    "systolic": 142,
    "diastolic": 91,
    "risk_label": "Stage 1 HTN",
    "risk_index": 2,
    "risk_probs": { "Optimal": 2.1, "Elevated": 8.4, ... },
    "feature_importance": { "stress_level": 14.2, "age": 17.8, ... },
    "pulse_pressure": 51,
    "map": 108,
    "bmi": 26.1
  },
  "recommendations": [...],
  "trend": [{ "day": 1, "systolic": 143, "diastolic": 90 }, ...]
}
```

### GET /api/health
Returns server and model status.

## UI Features
- **🎯 Predict** — Real-time BP prediction with animated gauge, live sliders
- **📈 Trend** — 7/14/30-day BP simulation chart with threshold lines
- **💡 Insights** — Personalized recommendations + risk probability bars
- **🔬 Explain** — Feature importance visualization (model explainability)

## Disclaimer
> This application is for educational purposes only. It is not a substitute for professional medical advice, diagnosis, or treatment.
