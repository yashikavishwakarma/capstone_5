# 🩺 Fall Detection — MLOps Pipeline (CP5)

> **Capstone Project 5**: DevOps for AI Deployment  
> Built on top of CP1 (TinyML wearable), CP2 (R analytics), CP3 (Power BI dashboards)

---

## 📐 Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                      CP5 MLOps Pipeline                         │
│                                                                 │
│  MobiFall / Simulated          GitHub Actions CI/CD             │
│  Sensor Data (accel+gyro)  ──► Auto retrain on push            │
│          │                          │                           │
│          ▼                          ▼                           │
│   simulate_data.py          Quality gate check                  │
│   (train / test / drift)    (accuracy ≥ 80%)                   │
│          │                                                      │
│          ▼                                                      │
│      train.py ──────────► MLflow Experiment Tracking            │
│   Random Forest             - params, metrics, artifacts        │
│   Classifier                - model registry                    │
│          │                                                      │
│          ▼                                                      │
│    model/model.pkl                                              │
│          │                                                      │
│          ├──────────────► monitor.py (Evidently AI)             │
│          │                 - data drift detection               │
│          │                 - HTML drift report                  │
│          │                                                      │
│          └──────────────► dashboard.py (Streamlit)              │
│                            - live fall prediction               │
│                            - MLflow run history                 │
│                            - embedded drift report              │
└─────────────────────────────────────────────────────────────────┘
```

---

## 🔗 Connection to Previous Capstones

| Capstone | Technology | CP5 Connection |
|---|---|---|
| **CP1** | TinyML / Arduino wearable | Wearable generates accel/gyro → fed to our prediction pipeline |
| **CP2** | R — statistical analysis | Feature engineering informed by R exploratory analysis |
| **CP3** | Power BI dashboards | Streamlit dashboard extends caregiver view to real-time MLOps |
| **CP5** | MLflow + Evidently + Streamlit | Production deployment + monitoring layer |

---

## 🚀 Quick Start

### 1. Install dependencies
```bash
pip install -r requirements.txt
```

### 2. Generate synthetic sensor data
```bash
python simulate_data.py
```

### 3. Train the model & log to MLflow
```bash
python train.py --n_estimators=100 --max_depth=10
```

### 4. View MLflow experiment tracker
```bash
mlflow ui
# Open http://localhost:5000
```

### 5. Run drift detection report
```bash
python monitor.py
# Report saved to reports/drift_report.html
```

### 6. Launch Streamlit dashboard
```bash
streamlit run dashboard.py
# Open http://localhost:8501
```

---

## 📁 Project Structure

```
cp5-mlops/
├── simulate_data.py          # Generates train / test / drift CSV data
├── train.py                  # Random Forest training + MLflow logging
├── monitor.py                # Evidently AI drift report
├── dashboard.py              # Streamlit demo dashboard
├── requirements.txt
├── data/
│   ├── train.csv
│   ├── test.csv
│   └── drift.csv             # Drift-injected data for monitoring demo
├── model/
│   ├── model.pkl             # Saved trained model
│   └── metrics.json          # Latest run metrics
├── reports/
│   └── drift_report.html     # Evidently drift report
└── .github/
    └── workflows/
        └── retrain.yml       # GitHub Actions CI/CD pipeline
```

---

## 📊 MLOps Features Demonstrated

| Feature | Tool | What it shows |
|---|---|---|
| Experiment tracking | MLflow | Logs params, metrics, confusion matrix per run |
| Model registry | MLflow | Versioned model storage |
| Data drift detection | Evidently AI | Flags sensor drift between training and production data |
| CI/CD pipeline | GitHub Actions | Auto-retrain + quality gate on every push |
| Live inference UI | Streamlit | Real-time fall prediction with confidence score |

---

## 📈 Dataset

- **Synthetic data**: Generated to mirror MobiFall v2.0 characteristics
- **MobiFall Dataset**: https://www.kaggle.com/datasets/kmknation/mobifall-dataset-v20
- **Features**: `ax, ay, az` (accelerometer), `gx, gy, gz` (gyroscope), `sma, smg, resultant`
- **Labels**: `0 = Normal Activity (ADL)`, `1 = Fall`

---

## 👥 Team

Capstone Project 5 — [Your Names Here]  
Program: [Your Program Name]
