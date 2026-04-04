"""
monitor.py
Uses Evidently AI to detect data drift between reference (train)
and current (drift-injected) sensor data. Saves an HTML report.
"""

import pandas as pd
import os

from evidently.report          import Report
from evidently.metric_preset   import DataDriftPreset, DataQualityPreset
from evidently.metrics         import DatasetDriftMetric


def run_drift_report(ref_path="data/train.csv",
                     cur_path="data/drift.csv",
                     out_dir="reports"):

    os.makedirs(out_dir, exist_ok=True)

    ref = pd.read_csv(ref_path).drop("label", axis=1)
    cur = pd.read_csv(cur_path).drop("label", axis=1)

    # Trim to same size for cleaner comparison
    n = min(len(ref), len(cur))
    ref = ref.sample(n, random_state=42)
    cur = cur.sample(n, random_state=42)

    report = Report(metrics=[
        DatasetDriftMetric(),
        DataDriftPreset(),
        DataQualityPreset(),
    ])

    report.run(reference_data=ref, current_data=cur)

    out_path = os.path.join(out_dir, "drift_report.html")
    report.save_html(out_path)
    print(f"✅ Drift report saved → {out_path}")
    return out_path


if __name__ == "__main__":
    run_drift_report()
