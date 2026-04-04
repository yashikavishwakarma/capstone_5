import numpy as np
import pandas as pd
import os

os.makedirs("data", exist_ok=True)
os.makedirs("model", exist_ok=True)
os.makedirs("reports", exist_ok=True)

np.random.seed(42)
n_total = 2000
n_falls = 400

def make_samples(n, is_fall):
    scale = 3.5 if is_fall else 1.0
    data = np.random.randn(n, 35) * scale
    return data

fall_data = make_samples(n_falls, True)
adl_data  = make_samples(n_total - n_falls, False)

X = np.vstack([fall_data, adl_data])
y = np.array([1]*n_falls + [0]*(n_total - n_falls))

cols = [f"feature_{i}" for i in range(35)]
df = pd.DataFrame(X, columns=cols)
df["label"] = y
df = df.sample(frac=1, random_state=42).reset_index(drop=True)

split = int(0.8 * len(df))
df.iloc[:split].to_csv("data/train.csv", index=False)
df.iloc[split:].to_csv("data/test.csv",  index=False)
df.iloc[split:].to_csv("data/drift.csv", index=False)

print(f"Generated {len(df)} samples ({n_falls} falls)")
print("Saved: data/train.csv, data/test.csv, data/drift.csv")
