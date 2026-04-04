"""
train.py
Trains a Random Forest classifier on fall detection sensor data.
Logs all experiments to MLflow: params, metrics, confusion matrix, model artifact.
"""

import mlflow
import mlflow.sklearn
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os, json, argparse

from sklearn.ensemble        import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics         import (accuracy_score, f1_score,
                                     precision_score, recall_score,
                                     confusion_matrix, classification_report)


# ── Helpers ───────────────────────────────────────────────────────────────────

def load_data(path="data/train.csv"):
    df = pd.read_csv(path)
    df = df.loc[:, ~df.columns.str.contains("^Unnamed")]
    X  = df.drop("label", axis=1)
    y  = df["label"]
    return X, y


def plot_confusion_matrix(cm, save_path="confusion_matrix.png"):
    fig, ax = plt.subplots(figsize=(5, 4))
    im = ax.imshow(cm, cmap="Blues")
    plt.colorbar(im, ax=ax)
    ax.set_xticks([0,1]); ax.set_yticks([0,1])
    ax.set_xticklabels(["No Fall","Fall"])
    ax.set_yticklabels(["No Fall","Fall"])
    ax.set_xlabel("Predicted"); ax.set_ylabel("Actual")
    ax.set_title("Confusion Matrix")
    for i in range(2):
        for j in range(2):
            ax.text(j, i, str(cm[i,j]), ha="center", va="center",
                    color="white" if cm[i,j] > cm.max()/2 else "black", fontsize=14)
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
    return save_path


# ── Main ──────────────────────────────────────────────────────────────────────

def train(n_estimators=100, max_depth=10, min_samples_split=2):

    mlflow.set_experiment("fall-detection-cp5")

    X, y = load_data("data/train.csv")
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    with mlflow.start_run():

        # ── Params
        params = dict(n_estimators=n_estimators,
                      max_depth=max_depth,
                      min_samples_split=min_samples_split)
        mlflow.log_params(params)

        # ── Train
        clf = RandomForestClassifier(**params, random_state=42, n_jobs=-1)
        clf.fit(X_train, y_train)

        # ── Evaluate
        y_pred = clf.predict(X_val)
        metrics = {
            "accuracy":  round(accuracy_score(y_val, y_pred),  4),
            "f1_score":  round(f1_score(y_val, y_pred),        4),
            "precision": round(precision_score(y_val, y_pred), 4),
            "recall":    round(recall_score(y_val, y_pred),    4),
        }
        mlflow.log_metrics(metrics)

        # ── Confusion matrix artifact
        cm      = confusion_matrix(y_val, y_pred)
        cm_path = plot_confusion_matrix(cm)
        mlflow.log_artifact(cm_path)
        os.remove(cm_path)

        # ── Classification report
        report = classification_report(y_val, y_pred,
                                       target_names=["No Fall","Fall"])
        with open("classification_report.txt","w") as f:
            f.write(report)
        mlflow.log_artifact("classification_report.txt")
        os.remove("classification_report.txt")

        # ── Log model
        mlflow.sklearn.log_model(clf, "random_forest_model")

        # ── Save locally for Streamlit
        os.makedirs("model", exist_ok=True)
        import pickle
        with open("model/model.pkl","wb") as f:
            pickle.dump(clf, f)
        with open("model/metrics.json","w") as f:
            json.dump(metrics, f)

        print("\n📊 Training Results:")
        for k, v in metrics.items():
            print(f"   {k:<12} {v:.4f}")
        print(f"\n✅ Run logged to MLflow | model saved → model/model.pkl")
        return metrics


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--n_estimators",    type=int, default=100)
    parser.add_argument("--max_depth",       type=int, default=10)
    parser.add_argument("--min_samples_split", type=int, default=2)
    args = parser.parse_args()

    train(args.n_estimators, args.max_depth, args.min_samples_split)
