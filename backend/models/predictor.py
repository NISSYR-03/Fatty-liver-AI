"""
backend/models/predictor.py
Loads trained XGBoost + SHAP and exposes predict().
"""

import os, json, sys
import numpy as np
import joblib
import shap

# ─────────────────────────────────────────────────────────────────────────────
#  ManualEnsemble Class
#  This must be present for joblib to unpickle the ensemble model.
#  We inject it into __main__ to match the environment where it was saved.
# ─────────────────────────────────────────────────────────────────────────────
class ManualEnsemble:
    """Weighted soft-voting over pre-fitted classifiers."""
    def __init__(self, models, weights=None):
        self.models  = models
        self.weights = weights or [1.0] * len(models)

    def predict_proba(self, X):
        probas = np.array([m.predict_proba(X) for m in self.models])
        w      = np.array(self.weights)[:, None, None]
        return (probas * w).sum(axis=0) / w.sum()

    def predict(self, X):
        return np.argmax(self.predict_proba(X), axis=1)

    def fit(self, X, y):
        return self

import __main__
__main__.ManualEnsemble = ManualEnsemble
# ─────────────────────────────────────────────────────────────────────────────

MODEL_DIR = os.path.join(os.path.dirname(__file__),
                         "..", "..", "ml_training", "saved_models")

BASE_FEATURES = ["age","gender","bmi","alt","ast",
                 "bilirubin","albumin","triglycerides","glucose"]

ALL_FEATURES  = BASE_FEATURES + [
    "ast_alt_ratio", "albumin_bilirubin",
    "bmi_category", "age_group",
    "liver_enzymes", "metabolic_score", "female_risk",
    "bmi_sq", "alt_sq", "bmi_glucose", "age_bmi",
    "tri_glucose", "alt_albumin",
    "log_alt", "log_ast", "log_tri", "log_bili"
]

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
    """Compute derived features the training pipeline added."""
    d = dict(data)
    bmi = float(d.get("bmi", 0))
    age = float(d.get("age", 0))
    ast = float(d.get("ast", 0))
    alt = float(d.get("alt", 0))
    albumin = float(d.get("albumin", 0))
    bilirubin = float(d.get("bilirubin", 0))
    glucose = float(d.get("glucose", 0))
    triglycerides = float(d.get("triglycerides", 0))
    gender = float(d.get("gender", 0))

    # Original features
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

    # New engineered features
    d["liver_enzymes"] = (ast + alt) / 2
    d["metabolic_score"] = bmi * (glucose / 100) * (triglycerides / 150)
    d["albumin_bilirubin"] = albumin / (bilirubin + 0.1)
    d["female_risk"] = (1 - gender) * bmi

    # Extended features for 26-feature model
    d["bmi_sq"]      = bmi ** 2
    d["alt_sq"]      = alt ** 2
    d["bmi_glucose"] = bmi * glucose
    d["age_bmi"]     = age * bmi
    d["tri_glucose"] = triglycerides * glucose
    d["alt_albumin"] = alt / (albumin + 1e-5)
    d["log_alt"]     = np.log1p(alt)
    d["log_ast"]     = np.log1p(ast)
    d["log_tri"]     = np.log1p(triglycerides)
    d["log_bili"]    = np.log1p(bilirubin)

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


class ManualEnsemble:
    """Weighted soft-voting over pre-fitted classifiers."""
    def __init__(self, models, weights=None):
        self.models  = models
        self.weights = weights or [1.0] * len(models)

    def predict_proba(self, X):
        probas = np.array([m.predict_proba(X) for m in self.models])
        w      = np.array(self.weights)[:, None, None]
        return (probas * w).sum(axis=0) / w.sum()

    def predict(self, X):
        return np.argmax(self.predict_proba(X), axis=1)

    def fit(self, X, y):   # sklearn compat stub
        return self


class Predictor:
    def __init__(self):
        self.model     = None
        self.scaler    = None
        self.explainer = None
        self.meta      = {}
        self._load()

    def _load(self):
        try:
            # On Render, we only load the XGBoost model to save memory
            if os.environ.get("RENDER"):
                self.model = joblib.load(f"{MODEL_DIR}/xgboost_model.pkl")
                # If it's an ensemble, take the first model (XGBoost)
                if hasattr(self.model, "models"):
                    self.model = self.model.models[0]
            else:
                self.model = joblib.load(f"{MODEL_DIR}/xgboost_model.pkl")
            
            self.scaler = joblib.load(f"{MODEL_DIR}/scaler.pkl")
        except FileNotFoundError as e:
            print(f"⚠️  Model missing ({e}). Run ml_training/train_model.py")
            return

        # Build SHAP explainer lazily at prediction time to avoid version-specific
        # pickling issues across SHAP/XGBoost releases.
        self.explainer = None

        mp = f"{MODEL_DIR}/model_meta.json"
        if os.path.exists(mp):
            self.meta = json.load(open(mp))
        print("✅ ML model loaded.")

    def is_ready(self):
        return self.model is not None

    def _build_explainer(self):
        if self.explainer is not None:
            return
        try:
            if hasattr(self.model, "get_booster"):
                booster = self.model.get_booster()
                self.explainer = shap.TreeExplainer(booster, feature_names=ALL_FEATURES)
            if self.explainer is None:
                try:
                    self.explainer = shap.TreeExplainer(self.model, feature_names=ALL_FEATURES)
                except Exception as e:
                    print(f"⚠️ Explainer error: {e}")
                    self.explainer = None
        except Exception:
            try:
                self.explainer = shap.Explainer(self.model)
            except Exception:
                self.explainer = None

    def _extract_shap(self, shap_values, cls_index):
        try:
            if isinstance(shap_values, list):
                row = np.asarray(shap_values[cls_index])[0]
            elif hasattr(shap_values, "values"):
                vals = np.asarray(shap_values.values)
                if vals.ndim == 3:
                    row = vals[cls_index][0]
                elif vals.ndim == 2:
                    row = vals[0]
                else:
                    row = vals.reshape(-1)[:len(ALL_FEATURES)]
            else:
                vals = np.asarray(shap_values)
                if vals.ndim == 3:
                    row = vals[cls_index][0]
                elif vals.ndim == 2:
                    row = vals[0]
                else:
                    row = vals.reshape(-1)[:len(ALL_FEATURES)]

            return {ALL_FEATURES[i]: round(float(row[i]), 4)
                    for i in range(min(len(row), len(ALL_FEATURES)))}
        except Exception:
            return {}

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

        # Skip SHAP on Render to prevent memory exhaustion and timeouts
        if os.environ.get("RENDER"):
            shap_d = {}
        else:
            try:
                self._build_explainer()
                if self.explainer is not None:
                    if hasattr(self.explainer, "shap_values"):
                        sv = self.explainer.shap_values(Xs)
                    else:
                        sv = self.explainer(Xs)
                    shap_d = self._extract_shap(sv, cls)
            except Exception:
                shap_d = {}

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
