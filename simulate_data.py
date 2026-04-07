# simulate_data.py
import numpy as np
import pandas as pd
import os

np.random.seed(42)
os.makedirs("data", exist_ok=True)

FEATURES = ["ax", "ay", "az", "gx", "gy", "gz", "sma", "smg", "resultant"]

def generate_samples(n_total=2000, fall_ratio=0.2):
    n_falls = int(n_total * fall_ratio)
    n_adl   = n_total - n_falls

    # Normal activity
    adl = np.random.randn(n_adl, 9) * [0.3,0.3,0.3, 0.2,0.2,0.2, 0.2,0.2,0.3]
    adl[:,2] += 1.0  # gravity on z-axis

    # Falls - higher magnitude
    falls = np.random.randn(n_falls, 9) * [2.5,2.5,2.5, 1.5,1.5,1.5, 2.0,1.5,2.5]
    falls[:,2] += 0.5

    X = np.vstack([adl, falls])
    y = np.array([0]*n_adl + [1]*n_falls)

    df = pd.DataFrame(X, columns=FEATURES)
    df["label"] = y
    return df.sample(frac=1, random_state=42).reset_index(drop=True)

# Train / test split
df = generate_samples(2000)
split = int(len(df) * 0.8)
train_df = df.iloc[:split]
test_df  = df.iloc[split:]

# Drift data - shift the distribution slightly
drift_df = generate_samples(500)
drift_df[FEATURES] += np.random.randn(*drift_df[FEATURES].shape) * 0.5

train_df.to_csv("data/train.csv", index=False)
test_df.to_csv("data/test.csv",   index=False)
drift_df.to_csv("data/drift.csv", index=False)

print(f"Generated {len(df)} samples ({df['label'].sum()} falls)")
print("Saved: data/train.csv, data/test.csv, data/drift.csv")