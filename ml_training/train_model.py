"""
train_model.py  ─  HepatoAI  ≥ 90 % accuracy pipeline
=======================================================
Techniques used
───────────────
1.  Richer feature engineering  (ratios, polynomials, log-transforms)
2.  Optuna hyper-param search   (fast: 15 trials, 300-tree proxy models)
3.  Voting Ensemble             (XGBoost + LightGBM + Random Forest)
4.  SMOTE balanced oversampling (train only, no leakage)
5.  Sample-weight bias correction
6.  Early stopping              (anti-overfitting)
7.  Stratified K-Fold CV        (robust evaluation)
"""

import numpy as np
import pandas as pd
import joblib, shap, json, os, warnings

from imblearn.over_sampling import SMOTE
from sklearn.model_selection import (train_test_split, StratifiedKFold,
                                     cross_val_score)
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (classification_report, accuracy_score,
                             roc_auc_score, confusion_matrix)
from sklearn.ensemble import RandomForestClassifier
from sklearn.utils.class_weight import compute_sample_weight

from xgboost import XGBClassifier

try:
    import lightgbm as lgb
    HAS_LGB = True
except ImportError:
    HAS_LGB = False
    print("⚠️  LightGBM not installed – using XGBoost + RF ensemble only")

try:
    import optuna
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    HAS_OPTUNA = True
except ImportError:
    HAS_OPTUNA = False
    print("⚠️  Optuna not installed – using default hyperparameters")

warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────────────────────────────────────
FEATURES = ["age", "gender", "bmi", "alt", "ast",
            "bilirubin", "albumin", "triglycerides", "glucose"]

ENGINEERED_FEATURES = [
    "ast_alt_ratio", "albumin_bilirubin",
    "bmi_category", "age_group",
    "liver_enzymes", "metabolic_score", "female_risk",
    "bmi_sq", "alt_sq", "bmi_glucose", "age_bmi",
    "tri_glucose", "alt_albumin",
    "log_alt", "log_ast", "log_tri", "log_bili",
]

DIR = "saved_models"


# ─────────────────────────────────────────────────────────────────────────────
# Soft-voting wrapper – must be at MODULE LEVEL to allow joblib pickling
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

    def fit(self, X, y):   # sklearn compat stub
        return self


# ─────────────────────────────────────────────────────────────────────────────
def feature_engineering(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["ast_alt_ratio"]     = df["ast"] / (df["alt"] + 1e-5)
    df["albumin_bilirubin"] = df["albumin"] / (df["bilirubin"] + 0.1)
    df["bmi_category"] = pd.cut(df["bmi"],
                                bins=[0, 18.5, 25, 30, 100],
                                labels=[0, 1, 2, 3]).astype(int)
    df["age_group"] = pd.cut(df["age"],
                             bins=[0, 30, 50, 70, 120],
                             labels=[0, 1, 2, 3]).astype(int)
    df["liver_enzymes"]   = (df["alt"] + df["ast"]) / 2
    df["metabolic_score"] = df["bmi"] * (df["glucose"] / 100) * (df["triglycerides"] / 150)
    df["female_risk"]     = (1 - df["gender"]) * df["bmi"]
    df["bmi_sq"]          = df["bmi"] ** 2
    df["alt_sq"]          = df["alt"] ** 2
    df["bmi_glucose"]     = df["bmi"] * df["glucose"]
    df["age_bmi"]         = df["age"] * df["bmi"]
    df["tri_glucose"]     = df["triglycerides"] * df["glucose"]
    df["alt_albumin"]     = df["alt"] / (df["albumin"] + 1e-5)
    df["log_alt"]         = np.log1p(df["alt"])
    df["log_ast"]         = np.log1p(df["ast"])
    df["log_tri"]         = np.log1p(df["triglycerides"])
    df["log_bili"]        = np.log1p(df["bilirubin"])
    return df


# ─────────────────────────────────────────────────────────────────────────────
# Optuna objective – uses small n_estimators proxy for speed
# ─────────────────────────────────────────────────────────────────────────────
def _xgb_objective(trial, X_tr, y_tr):
    params = {
        "objective":        "multi:softprob",
        "num_class":        3,
        "eval_metric":      "mlogloss",
        "tree_method":      "hist",
        "n_jobs":           -1,
        "random_state":     42,
        "n_estimators":     300,          # ← proxy; fast CV
        "max_depth":        trial.suggest_int("max_depth", 4, 9),
        "learning_rate":    trial.suggest_float("learning_rate", 0.01, 0.15, log=True),
        "subsample":        trial.suggest_float("subsample", 0.65, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
        "min_child_weight": trial.suggest_int("min_child_weight", 1, 8),
        "gamma":            trial.suggest_float("gamma", 0.0, 0.4),
        "reg_lambda":       trial.suggest_float("reg_lambda", 0.5, 6.0),
        "reg_alpha":        trial.suggest_float("reg_alpha", 0.0, 2.0),
    }
    model = XGBClassifier(**params)
    cv    = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
    scores = cross_val_score(model, X_tr, y_tr, cv=cv,
                             scoring="accuracy", n_jobs=-1)
    return scores.mean()


# ─────────────────────────────────────────────────────────────────────────────
def train():
    csv = os.path.join(DIR, "fatty_liver_dataset.csv")
    if not os.path.exists(csv):
        raise FileNotFoundError("Run generate_dataset.py first!")

    print("📥 Loading dataset …")
    df = pd.read_csv(csv)
    df = feature_engineering(df)
    feature_cols = FEATURES + ENGINEERED_FEATURES

    X = df[feature_cols].values
    y = df["label"].values

    print(f"   {X.shape[0]} samples  |  {X.shape[1]} features")
    u, c = np.unique(y, return_counts=True)
    print(f"   Class dist: {dict(zip(u, c))}")

    # ── Split ──────────────────────────────────────────────────────────────────
    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=0.20, stratify=y, random_state=42)

    # ── Scale ──────────────────────────────────────────────────────────────────
    scaler = StandardScaler()
    X_tr_s = scaler.fit_transform(X_tr)
    X_te_s = scaler.transform(X_te)

    # ── SMOTE ─────────────────────────────────────────────────────────────────
    print("⚖️  SMOTE …")
    smote = SMOTE(k_neighbors=5, random_state=42)
    X_tr_s, y_tr = smote.fit_resample(X_tr_s, y_tr)
    print(f"   → {X_tr_s.shape[0]} samples after SMOTE")

    # ── Sample weights ─────────────────────────────────────────────────────────
    sw = compute_sample_weight("balanced", y_tr)

    # ── Optuna (fast: 15 trials, 3-fold CV, 300-tree proxy) ───────────────────
    if HAS_OPTUNA:
        print("\n🔍 Optuna search (15 trials × 3-fold CV) …")
        study = optuna.create_study(
            direction="maximize",
            sampler=optuna.samplers.TPESampler(seed=42),
        )
        study.optimize(
            lambda trial: _xgb_objective(trial, X_tr_s, y_tr),
            n_trials=15,
            show_progress_bar=False,
        )
        best = study.best_params
        print(f"   Best proxy CV: {study.best_value:.4f}")
        print(f"   Params: {best}")
        # Final model uses many more trees
        xgb_params = {
            "objective":             "multi:softprob",
            "num_class":             3,
            "eval_metric":           "mlogloss",
            "tree_method":           "hist",
            "n_jobs":                -1,
            "random_state":          42,
            "n_estimators":          800,
            "early_stopping_rounds": 50,
            **best,
        }
    else:
        xgb_params = {
            "objective":             "multi:softprob",
            "num_class":             3,
            "n_estimators":          800,
            "max_depth":             7,
            "learning_rate":         0.03,
            "subsample":             0.85,
            "colsample_bytree":      0.80,
            "min_child_weight":      2,
            "gamma":                 0.05,
            "reg_lambda":            3.0,
            "reg_alpha":             0.5,
            "eval_metric":           "mlogloss",
            "tree_method":           "hist",
            "n_jobs":                -1,
            "random_state":          42,
            "early_stopping_rounds": 50,
        }

    # ── Validation split for early stopping ────────────────────────────────────
    X_fin, X_val, y_fin, y_val = train_test_split(
        X_tr_s, y_tr, test_size=0.15, stratify=y_tr, random_state=42)
    sw_fin = compute_sample_weight("balanced", y_fin)

    # ── Train XGBoost ──────────────────────────────────────────────────────────
    print("\n🚀 Training XGBoost …")
    xgb_model = XGBClassifier(**xgb_params)
    xgb_model.fit(X_fin, y_fin,
                  sample_weight=sw_fin,
                  eval_set=[(X_val, y_val)],
                  verbose=100)

    # ── Train LightGBM ─────────────────────────────────────────────────────────
    estimators = [("xgb", xgb_model)]
    if HAS_LGB:
        print("\n🚀 Training LightGBM …")
        lgb_model = lgb.LGBMClassifier(
            objective="multiclass",
            num_class=3,
            n_estimators=600,
            max_depth=7,
            learning_rate=0.03,
            subsample=0.85,
            colsample_bytree=0.80,
            reg_lambda=3.0,
            reg_alpha=0.5,
            min_child_samples=10,
            random_state=42,
            n_jobs=-1,
            verbose=-1,
        )
        lgb_model.fit(X_fin, y_fin, sample_weight=sw_fin)
        estimators.append(("lgb", lgb_model))

    # ── Train Random Forest ────────────────────────────────────────────────────
    print("🌲 Training Random Forest …")
    rf_model = RandomForestClassifier(
        n_estimators=400,
        max_depth=20,
        min_samples_leaf=2,
        min_samples_split=4,
        max_features="sqrt",
        class_weight="balanced",
        n_jobs=-1,
        random_state=42,
    )
    rf_model.fit(X_fin, y_fin, sample_weight=sw_fin)
    estimators.append(("rf", rf_model))

    # ── Manual soft-voting (avoids refit issues with early_stopping_rounds) ──────
    print("\n🏆 Building manual soft-voting ensemble …")

    models_list  = [m for _, m in estimators]
    # Give XGBoost slightly higher weight as it was optuna-tuned
    weights_list = [1.4] + [1.0] * (len(estimators) - 1)
    ensemble     = ManualEnsemble(models_list, weights_list)

    # ── Evaluate ───────────────────────────────────────────────────────────────
    y_pred  = ensemble.predict(X_te_s)
    y_proba = ensemble.predict_proba(X_te_s)

    acc = accuracy_score(y_te, y_pred)
    auc = roc_auc_score(y_te, y_proba, multi_class="ovr")

    print(f"\n{'='*55}")
    print(f"  ✅ Test Accuracy : {acc*100:.2f}%")
    print(f"  ✅ AUC-ROC       : {auc:.4f}")
    print(f"{'='*55}")
    print("\n" + classification_report(
        y_te, y_pred, target_names=["Low", "Medium", "High"]))
    print("Confusion Matrix:")
    print(confusion_matrix(y_te, y_pred))

    # ── Overfitting check ──────────────────────────────────────────────────────
    train_acc = accuracy_score(y_fin, ensemble.predict(X_fin))
    gap = train_acc - acc
    print(f"\n   Train Acc: {train_acc*100:.2f}%  |  Test Acc: {acc*100:.2f}%  |  Gap: {gap*100:.1f}%")
    if gap < 0.05:
        print("   ✅ No significant overfitting (gap < 5%)")
    else:
        print(f"   ⚠️  Overfitting gap = {gap*100:.1f}%")

    # ── Cross-validation (XGBoost, no early stop) ─────────────────────────────
    print("\n🔄 5-fold CV on XGBoost (quick check) …")
    cv_params = {k: v for k, v in xgb_params.items()
                 if k != "early_stopping_rounds"}
    cv_params["n_estimators"] = 400
    cv_xgb = XGBClassifier(**cv_params)
    cv_scores = cross_val_score(
        cv_xgb, X_tr_s, y_tr,
        cv=StratifiedKFold(n_splits=5, shuffle=True, random_state=42),
        scoring="accuracy", n_jobs=-1,
    )
    print(f"   XGBoost CV: {cv_scores.mean()*100:.2f}% ± {cv_scores.std()*100:.2f}%")

    # ── SHAP ──────────────────────────────────────────────────────────────────
    explainer = None
    try:
        print("\n🔍 SHAP explainer …")
        explainer = shap.TreeExplainer(xgb_model)
        sv = explainer.shap_values(X_te_s[:20], check_additivity=False)
        print(f"   SHAP shape: {np.array(sv).shape}")
    except Exception as e:
        print(f"⚠️  SHAP failed: {e}")

    # ── Save ───────────────────────────────────────────────────────────────────
    os.makedirs(DIR, exist_ok=True)
    joblib.dump(ensemble,  f"{DIR}/xgboost_model.pkl")
    joblib.dump(scaler,    f"{DIR}/scaler.pkl")
    if explainer:
        joblib.dump(explainer, f"{DIR}/shap_explainer.pkl")

    meta = {
        "features": feature_cols,
        "classes":  ["Low", "Medium", "High"],
        "accuracy": round(float(acc), 4),
        "auc":      round(float(auc), 4),
        "cv_mean":  round(float(cv_scores.mean()), 4),
        "cv_std":   round(float(cv_scores.std()), 4),
        "ensemble": [e[0] for e in estimators],
    }
    with open(f"{DIR}/model_meta.json", "w") as f:
        json.dump(meta, f, indent=2)

    print(f"\n✅ Saved to ./{DIR}/")
    print(f"   Ensemble: {meta['ensemble']}")
    return acc


if __name__ == "__main__":
    train()