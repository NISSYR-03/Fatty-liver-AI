"""
STEP 1 — generate_dataset.py
Run: python generate_dataset.py
Creates: saved_models/fatty_liver_dataset.csv

IMPROVED: Larger dataset (20k), tighter class boundaries,
          stronger feature-label relationships, minimal noise.
"""

import numpy as np
import pandas as pd
import os

np.random.seed(42)
N = 20000   # Double the data for better generalisation


def generate():
    age           = np.random.randint(18, 80, N)
    gender        = np.random.choice([0, 1], N)          # 0=Female 1=Male
    bmi           = np.random.normal(26, 4.5, N).clip(15, 55)
    alt           = np.random.normal(28, 14, N).clip(5, 300)
    ast           = np.random.normal(24, 11, N).clip(5, 300)
    bilirubin     = np.random.normal(0.9, 0.45, N).clip(0.1, 6)
    albumin       = np.random.normal(4.1, 0.45, N).clip(1.5, 5.5)
    triglycerides = np.random.normal(140, 65, N).clip(30, 700)
    glucose       = np.random.normal(98, 22, N).clip(55, 400)

    # ── Latent risk score ─────────────────────────────────────────────────────
    # Stronger coefficients → cleaner class separation
    # Very low noise → less ambiguity at decision boundaries
    risk = (
        0.08  * (bmi - 24)           +   # obesity drives risk
        0.018 * (alt - 25)           +   # liver enzymes
        0.012 * (ast - 22)           +
        0.30  * bilirubin            +   # bilirubin is strong marker
        0.008 * (triglycerides - 100)+   # metabolic syndrome
        0.006 * (glucose - 90)       +
        0.018 * age                  +   # age effect
        0.35  * gender               -   # males at higher risk
        0.55  * (albumin - 4)        +   # low albumin = worse
        np.random.normal(0, 0.15, N)     # reduced noise
    )

    # Tighter, non-overlapping bins
    label = pd.cut(
        risk,
        bins=[-np.inf, 0.75, 2.1, np.inf],
        labels=[0, 1, 2]
    ).astype(int)

    df = pd.DataFrame({
        "age": age,
        "gender": gender,
        "bmi": bmi.round(1),
        "alt": alt.round(1),
        "ast": ast.round(1),
        "bilirubin": bilirubin.round(2),
        "albumin": albumin.round(2),
        "triglycerides": triglycerides.round(1),
        "glucose": glucose.round(1),
        "label": label,
    })

    os.makedirs("saved_models", exist_ok=True)
    path = "saved_models/fatty_liver_dataset.csv"
    df.to_csv(path, index=False)

    c = df["label"].value_counts().sort_index()
    print(f"✅ Dataset saved → {path}  (N={N})")
    print(f"   Low(0)={c.get(0,0)}  Medium(1)={c.get(1,0)}  High(2)={c.get(2,0)}")


if __name__ == "__main__":
    generate()
