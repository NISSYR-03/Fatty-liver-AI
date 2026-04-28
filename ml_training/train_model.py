import numpy as np
import pandas as pd
import joblib, shap, json, os, warnings

from imblearn.over_sampling import SMOTE
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, accuracy_score, roc_auc_score

from xgboost import XGBClassifier

warnings.filterwarnings("ignore")

FEATURES = ["age","gender","bmi","alt","ast","bilirubin","albumin","triglycerides","glucose"]
DIR = "saved_models"


def feature_engineering(df):
    # 🔥 Add powerful medical features
    df["ast_alt_ratio"] = df["ast"] / (df["alt"] + 1e-5)
    df["bmi_category"] = pd.cut(df["bmi"], bins=[0,18.5,25,30,100], labels=[0,1,2,3]).astype(int)
    df["age_group"] = pd.cut(df["age"], bins=[0,30,50,70,120], labels=[0,1,2,3]).astype(int)
    return df


def train():
    csv = os.path.join(DIR, "fatty_liver_dataset.csv")
    if not os.path.exists(csv):
        raise FileNotFoundError("Run generate_dataset.py first!")

    df = pd.read_csv(csv)

    # 🔥 Feature Engineering
    df = feature_engineering(df)

    feature_cols = FEATURES + ["ast_alt_ratio","bmi_category","age_group"]

    X = df[feature_cols].values
    y = df["label"].values

    # ✅ SPLIT FIRST (NO LEAKAGE)
    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42)

    # ✅ SCALE
    scaler = StandardScaler()
    X_tr_s = scaler.fit_transform(X_tr)
    X_te_s = scaler.transform(X_te)

    # ✅ APPLY SMOTE ONLY ON TRAIN
    smote = SMOTE(random_state=42)
    X_tr_s, y_tr = smote.fit_resample(X_tr_s, y_tr)

    # 🔥 IMPROVED MODEL
    model = XGBClassifier(
        n_estimators=700,
        max_depth=7,
        learning_rate=0.03,
        subsample=0.85,
        colsample_bytree=0.85,
        min_child_weight=2,
        gamma=0.05,
        reg_lambda=1,
        reg_alpha=0.3,
        eval_metric="mlogloss",
        random_state=42,
        n_jobs=-1
    )

    model.fit(X_tr_s, y_tr)

    # 🔥 EVALUATION
    y_pred = model.predict(X_te_s)
    y_proba = model.predict_proba(X_te_s)

    acc = accuracy_score(y_te, y_pred)
    auc = roc_auc_score(y_te, y_proba, multi_class="ovr")

    print(f"\n📊 Accuracy : {acc:.4f}")
    print(f"📊 AUC-ROC  : {auc:.4f}")
    print("\n" + classification_report(
        y_te, y_pred, target_names=["Low","Medium","High"]
    ))

    # 🔥 CROSS VALIDATION
    cv = cross_val_score(
        model, X_tr_s, y_tr,
        cv=StratifiedKFold(5),
        scoring="accuracy"
    )
    print(f"CV  : {cv.mean():.4f} ± {cv.std():.4f}")

    # 🔥 SHAP
    print("\n🔍 Computing SHAP explainer …")
    explainer = shap.TreeExplainer(model)
    sv = explainer.shap_values(X_te_s[:20])
    print(f"   SHAP shape: {np.array(sv).shape}")

    # 💾 SAVE
    os.makedirs(DIR, exist_ok=True)
    joblib.dump(model, f"{DIR}/xgboost_model.pkl")
    joblib.dump(scaler, f"{DIR}/scaler.pkl")
    joblib.dump(explainer, f"{DIR}/shap_explainer.pkl")

    meta = {
        "features": feature_cols,
        "classes": ["Low","Medium","High"],
        "accuracy": round(float(acc), 4),
        "auc": round(float(auc), 4),
        "cv_mean": round(float(cv.mean()), 4)
    }

    with open(f"{DIR}/model_meta.json","w") as f:
        json.dump(meta, f, indent=2)

    print(f"\n✅ All files saved to ./{DIR}/")


if __name__ == "__main__":
    train()