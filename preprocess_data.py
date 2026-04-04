"""
preprocess_data.py - Correctly handles MobiFall v2.0 format
- Comments start with # and @DATA marks start of data
- Acc and gyro are in SEPARATE files, matched by suffix
- Comma-separated values: timestamp, x, y, z
"""

import os, glob
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

DATASET_ROOT = "MobiFall_Dataset_v2.0"
FALL_CODES   = {"FOL", "FKL", "BSC", "SDL"}
ADL_CODES    = {"STD", "WAL", "JOG", "JUM", "STU", "STN", "SCH", "CSI", "CSO"}
WINDOW_SIZE  = 100
STEP_SIZE    = 50


def read_sensor_file(filepath):
    """Read a single acc or gyro file. Returns DataFrame with timestamp, x, y, z."""
    rows = []
    data_started = False
    with open(filepath, "r") as f:
        for line in f:
            line = line.strip()
            if line == "@DATA":
                data_started = True
                continue
            if not data_started or not line or line.startswith("#"):
                continue
            parts = [p.strip() for p in line.split(",")]
            if len(parts) < 4:
                continue
            try:
                rows.append([float(p) for p in parts[:4]])
            except ValueError:
                continue
    if not rows:
        return None
    return pd.DataFrame(rows, columns=["timestamp", "x", "y", "z"])


def load_trial(acc_path, gyro_path):
    """
    Merge acc and gyro files for one trial.
    Returns DataFrame with ax,ay,az,gx,gy,gz aligned by index.
    """
    acc  = read_sensor_file(acc_path)
    gyro = read_sensor_file(gyro_path)
    if acc is None or gyro is None:
        return None

    # Trim to same length and merge by position (timestamps may differ slightly)
    n = min(len(acc), len(gyro))
    acc  = acc.iloc[:n].reset_index(drop=True)
    gyro = gyro.iloc[:n].reset_index(drop=True)

    df = pd.DataFrame({
        "ax": acc["x"],  "ay": acc["y"],  "az": acc["z"],
        "gx": gyro["x"], "gy": gyro["y"], "gz": gyro["z"],
    })
    return df if len(df) >= WINDOW_SIZE else None


def extract_features(window):
    feats = {}
    for col in ["ax", "ay", "az", "gx", "gy", "gz"]:
        s = window[col].values
        feats[f"{col}_mean"]  = np.mean(s)
        feats[f"{col}_std"]   = np.std(s)
        feats[f"{col}_max"]   = np.max(s)
        feats[f"{col}_min"]   = np.min(s)
        feats[f"{col}_range"] = np.max(s) - np.min(s)

    feats["sma_acc"] = (np.sum(np.abs(window["ax"])) +
                        np.sum(np.abs(window["ay"])) +
                        np.sum(np.abs(window["az"]))) / len(window)
    feats["sma_gyr"] = (np.sum(np.abs(window["gx"])) +
                        np.sum(np.abs(window["gy"])) +
                        np.sum(np.abs(window["gz"]))) / len(window)

    resultant = np.sqrt(window["ax"]**2 + window["ay"]**2 + window["az"]**2)
    feats["resultant_mean"] = np.mean(resultant)
    feats["resultant_max"]  = np.max(resultant)
    feats["resultant_std"]  = np.std(resultant)
    return feats


def windows_from_df(df, label):
    rows = []
    for start in range(0, len(df) - WINDOW_SIZE, STEP_SIZE):
        w = df.iloc[start: start + WINDOW_SIZE]
        f = extract_features(w)
        f["label"] = label
        rows.append(f)
    return rows


def scan_activity_folder(act_path, label):
    """
    Match _acc_ and _gyro_ files by their trial suffix e.g. FOL_acc_1_1 + FOL_gyro_1_1
    """
    rows = []
    acc_files = sorted(glob.glob(os.path.join(act_path, "*_acc_*.txt")))

    for acc_path in acc_files:
        gyro_path = acc_path.replace("_acc_", "_gyro_")
        if not os.path.exists(gyro_path):
            continue
        df = load_trial(acc_path, gyro_path)
        if df is not None:
            rows.extend(windows_from_df(df, label))
    return rows


def scan_subject(sub_dir):
    rows = []
    for root, dirs, files in os.walk(sub_dir):
        folder_name = os.path.basename(root).upper()
        if folder_name in FALL_CODES:
            label = 1
        elif folder_name in ADL_CODES:
            label = 0
        else:
            continue
        rows.extend(scan_activity_folder(root, label))
    return rows


def build_dataset():
    if not os.path.exists(DATASET_ROOT):
        print(f"❌ Folder '{DATASET_ROOT}' not found.")
        return

    subject_dirs = sorted(glob.glob(os.path.join(DATASET_ROOT, "sub*")))
    print(f"📂 Found {len(subject_dirs)} subjects\n")

    all_rows = []
    for sub_dir in subject_dirs:
        sub_name = os.path.basename(sub_dir)
        rows = scan_subject(sub_dir)
        all_rows.extend(rows)
        print(f"   ✅ {sub_name} → {len(rows)} windows")

    if not all_rows:
        print("\n❌ No data extracted.")
        return

    full_df = pd.DataFrame(all_rows)
    print(f"\n📊 Total windows : {len(full_df)}")
    print(f"   Falls  (1)    : {int(full_df['label'].sum())}")
    print(f"   ADL    (0)    : {int((full_df['label']==0).sum())}")
    print(f"   Features      : {full_df.shape[1] - 1}")

    train_df, temp_df = train_test_split(
        full_df, test_size=0.3, random_state=42, stratify=full_df["label"])
    test_df, drift_df = train_test_split(
        temp_df, test_size=0.5, random_state=42, stratify=temp_df["label"])

    sensor_cols = [c for c in drift_df.columns if c != "label"]
    drift_df = drift_df.copy()
    drift_df[sensor_cols] += np.random.normal(0.3, 0.4, size=drift_df[sensor_cols].shape)

    os.makedirs("data", exist_ok=True)
    train_df.to_csv("data/train.csv", index=False)
    test_df.to_csv("data/test.csv",   index=False)
    drift_df.to_csv("data/drift.csv", index=False)

    print(f"\n✅ data/train.csv  → {len(train_df)} windows")
    print(f"✅ data/test.csv   → {len(test_df)} windows")
    print(f"✅ data/drift.csv  → {len(drift_df)} windows")
    print(f"\n🚀 Now run: python train.py --n_estimators=100 --max_depth=10")


if __name__ == "__main__":
    build_dataset()
