"""
STEP 1 — generate_dataset.py
Run: python generate_dataset.py
Creates: saved_models/fatty_liver_dataset.csv
"""

import numpy as np
import pandas as pd
import os

np.random.seed(42)
N = 10000

def generate():
    age          = np.random.randint(18, 75, N)
    gender       = np.random.choice([0, 1], N)        # 0=Female 1=Male
    bmi          = np.random.normal(26, 5, N).clip(15, 55)
    alt          = np.random.normal(28, 15, N).clip(5, 250)
    ast          = np.random.normal(24, 12, N).clip(5, 250)
    bilirubin    = np.random.normal(0.9, 0.5, N).clip(0.1, 6)
    albumin      = np.random.normal(4.1, 0.5, N).clip(1.5, 5.5)
    triglycerides= np.random.normal(140, 70, N).clip(30, 700)
    glucose      = np.random.normal(98, 25, N).clip(55, 350)

    # Latent risk score — mirrors real clinical weighting with stronger separation
    risk = (
        0.05  * (bmi - 24) +
        0.012 * (alt - 25) +
        0.008 * (ast - 22) +
        0.22  * bilirubin  +
        0.006 * (triglycerides - 100) +
        0.005 * (glucose - 90) +
        0.015 * age +
        0.32  * gender -
        0.42  * (albumin - 4) +
        np.random.normal(0, 0.25, N)  # Reduced noise for better separation
    )

    label = pd.cut(risk, bins=[-np.inf, 0.8, 2.0, np.inf],
                   labels=[0, 1, 2]).astype(int)

    df = pd.DataFrame({
        "age": age, "gender": gender, "bmi": bmi.round(1),
        "alt": alt.round(1), "ast": ast.round(1),
        "bilirubin": bilirubin.round(2), "albumin": albumin.round(2),
        "triglycerides": triglycerides.round(1),
        "glucose": glucose.round(1), "label": label
    })

    os.makedirs("saved_models", exist_ok=True)
    path = "saved_models/fatty_liver_dataset.csv"
    df.to_csv(path, index=False)
    c = df["label"].value_counts().sort_index()
    print(f"✅ Dataset saved → {path}")
    print(f"   Low(0)={c.get(0,0)}  Medium(1)={c.get(1,0)}  High(2)={c.get(2,0)}")

if __name__ == "__main__":
    generate()
