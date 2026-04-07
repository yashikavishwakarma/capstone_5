# monitor.py
import pandas as pd
from evidently.report import Report
from evidently.metric_preset import DataDriftPreset
import os

# Load data
train_df = pd.read_csv("data/train.csv")
drift_df = pd.read_csv("data/drift.csv")

# Drop label column for drift analysis
train_df = train_df.drop(columns=["label"])
drift_df = drift_df.drop(columns=["label"])

# Create report
report = Report(metrics=[DataDriftPreset()])
report.run(reference_data=train_df, current_data=drift_df)

# Save report
os.makedirs("reports", exist_ok=True)
report.save_html("reports/drift_report.html")
print("✅ Drift report saved to reports/drift_report.html")