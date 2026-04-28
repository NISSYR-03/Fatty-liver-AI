"""
backend/models/predictor.py
Loads trained XGBoost + SHAP and exposes predict().
"""

import os, json
import numpy as np
import joblib

MODEL_DIR = os.path.join(os.path.dirname(__file__),
                         "..", "..", "ml_training", "saved_models")

BASE_FEATURES = ["age","gender","bmi","alt","ast",
                 "bilirubin","albumin","triglycerides","glucose"]

ALL_FEATURES  = BASE_FEATURES + ["ast_alt_ratio","bmi_category","age_group"]

NORMAL = {          # (low, high, unit)
    "alt":           (7,   56,  "U/L"),
    "ast":           (10,  40,  "U/L"),
    "bilirubin":     (0.2, 1.2, "mg/dL"),
    "albumin":       (3.5, 5.0, "g/dL"),
    "triglycerides": (0,   150, "mg/dL"),
    "glucose":       (70,  100, "mg/dL"),
    "bmi":           (18.5,24.9,"kg/m²"),
}


def _engineer(data: dict) -> dict:
    """Compute the 3 derived features the training pipeline added."""
    d = dict(data)
    bmi = float(d.get("bmi", 0))
    age = float(d.get("age", 0))
    ast = float(d.get("ast", 0))
    alt = float(d.get("alt", 0))

    d["ast_alt_ratio"] = ast / (alt + 1e-5)

    # bmi_category: bins [0,18.5,25,30,100] → labels [0,1,2,3]
    if bmi <= 18.5:   d["bmi_category"] = 0
    elif bmi <= 25:   d["bmi_category"] = 1
    elif bmi <= 30:   d["bmi_category"] = 2
    else:             d["bmi_category"] = 3

    # age_group: bins [0,30,50,70,120] → labels [0,1,2,3]
    if age <= 30:     d["age_group"] = 0
    elif age <= 50:   d["age_group"] = 1
    elif age <= 70:   d["age_group"] = 2
    else:             d["age_group"] = 3

    return d

RECS = {
    "Low":
        "🟢 Great news — your markers suggest low fatty liver risk. "
        "Stay active (150 min/week), eat a Mediterranean-style diet, "
        "limit alcohol, and get an annual liver panel.",
    "Medium":
        "🟡 Moderate risk detected. Adopt a low-fat, high-fibre diet, "
        "exercise 30 min × 5 days/week, reduce sugar intake, and visit "
        "your physician for a follow-up within 3 months.",
    "High":
        "🔴 High risk — please consult a hepatologist promptly. "
        "Get an abdominal ultrasound + full liver function test. "
        "Avoid alcohol completely, pursue medically supervised weight loss, "
        "and monitor blood glucose closely.",
}


class Predictor:
    def __init__(self):
        self.model     = None
        self.scaler    = None
        self.explainer = None
        self.meta      = {}
        self._load()

    def _load(self):
        try:
            self.model  = joblib.load(f"{MODEL_DIR}/xgboost_model.pkl")
            self.scaler = joblib.load(f"{MODEL_DIR}/scaler.pkl")
        except FileNotFoundError as e:
            print(f"⚠️  Model missing ({e}). Run ml_training/train_model.py")
            return

        try:
            self.explainer = joblib.load(f"{MODEL_DIR}/shap_explainer.pkl")
        except Exception as e:
            self.explainer = None
            print(f"⚠️  SHAP explainer failed to load: {e}")

        mp = f"{MODEL_DIR}/model_meta.json"
        if os.path.exists(mp):
            self.meta = json.load(open(mp))
        print("✅ ML model loaded.")

    def is_ready(self):
        return self.model is not None

    def predict(self, data: dict) -> dict:
        if not self.is_ready():
            raise RuntimeError("Model not loaded. Train first.")

        # Engineer the 3 derived features to match training pipeline
        enriched = _engineer(data)
        vals = [float(enriched[f]) for f in ALL_FEATURES]
        X    = np.array(vals).reshape(1, -1)
        Xs   = self.scaler.transform(X)

        cls    = int(self.model.predict(Xs)[0])
        probas = self.model.predict_proba(Xs)[0].tolist()
        label  = ["Low","Medium","High"][cls]

        # SHAP contributions are optional when explainer loading fails
        shap_d = {}
        if self.explainer is not None:
            sv = self.explainer.shap_values(Xs)
            if isinstance(sv, list):
                row = sv[cls][0]
            else:
                row = sv[cls][0] if sv.ndim == 3 else sv[0]
            shap_d = {ALL_FEATURES[i]: round(float(row[i]),4) for i in range(len(ALL_FEATURES))}

        # Clinical flags
        flags = []
        for feat,(lo,hi,unit) in NORMAL.items():
            v = float(data.get(feat,0))
            if v < lo:
                flags.append({"field":feat,"status":"LOW",
                               "value":v,"normal":f"{lo}–{hi} {unit}"})
            elif v > hi:
                flags.append({"field":feat,"status":"HIGH",
                               "value":v,"normal":f"{lo}–{hi} {unit}"})

        return {
            "risk_label":         label,
            "risk_index":         cls,
            "probabilities":      probas,
            "probability_labels": ["Low","Medium","High"],
            "shap_contributions": shap_d,
            "clinical_flags":     flags,
            "recommendation":     RECS[label],
            "input_values":       dict(zip(BASE_FEATURES, [float(data[f]) for f in BASE_FEATURES])),
            "model_accuracy":     self.meta.get("accuracy","N/A"),
        }
