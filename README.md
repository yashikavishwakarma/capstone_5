# 🩺 Fall Detection — MLOps Pipeline (CP5)

[![Fall Detection — Retrain & Validate](https://github.com/HARMANXPAL/cp5-fall-detection-mlops/actions/workflows/retrain.yml/badge.svg)](https://github.com/HARMANXPAL/cp5-fall-detection-mlops/actions/workflows/retrain.yml)
![Python](https://img.shields.io/badge/Python-3.10%2B-blue)
![Docker](https://img.shields.io/badge/Docker-Compose-2496ED?logo=docker)
![MLflow](https://img.shields.io/badge/MLflow-2.13-orange)
![Streamlit](https://img.shields.io/badge/Streamlit-1.32%2B-red)

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
│          ├──────────────► app.py (FastAPI)                      │
│          │                 - REST API for predictions           │
│          │                 - /predict, /health endpoints        │
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
| **CP5** | MLflow + Evidently + Streamlit + FastAPI | Production deployment + monitoring layer |

---

## 🚀 Quick Start

### 🐳 Option 1 — Docker Compose (Recommended)

Runs all 3 services with one command. No Python setup needed.

**Prerequisites:** Install [Docker Desktop](https://www.docker.com/products/docker-desktop)

```bash
git clone https://github.com/yashikavishwakarma/capstone_5
cd capstone_5
docker compose up
```

Then open in your browser:

| Service | URL |
|---|---|
| 🎨 Streamlit Dashboard | http://localhost:8501 |
| ⚡ FastAPI Docs | http://localhost:8000/docs |
| 📊 MLflow UI | http://localhost:5000 |

> **Note:** First run trains the model automatically (~2-3 mins). Subsequent runs start instantly.

---

### 💻 Option 2 — Run Locally

**Prerequisites:** Python 3.10+, Git

**Step 1 — Clone the repo**
```bash
git clone https://github.com/yashikavishwakarma/capstone_5
cd capstone_5
```

**Step 2 — Create virtual environment**
```bash
python -m venv .venv
.venv\Scripts\activate        # Windows
source .venv/bin/activate     # Mac/Linux
```

**Step 3 — Install dependencies**
```bash
python -m pip install setuptools==68.0.0
python -m pip install -r requirements.txt
```

**Step 4 — Generate data & train model**
```bash
python simulate_data.py
python train.py --n_estimators=100 --max_depth=10
python monitor.py
```

**Step 5 — Run all services** (open 3 separate terminals)
```bash
# Terminal 1 - Dashboard
streamlit run dashboard.py

# Terminal 2 - API
uvicorn app:app --host 0.0.0.0 --port 8000

# Terminal 3 - MLflow UI
mlflow ui
```

---

## 📁 Project Structure

```
cp5/
├── simulate_data.py          # Generates train / test / drift CSV data
├── train.py                  # Random Forest training + MLflow logging
├── monitor.py                # Evidently AI drift report
├── app.py                    # FastAPI REST API for predictions
├── dashboard.py              # Streamlit MLOps dashboard
├── preprocess_data.py        # Data preprocessing utilities
├── requirements.txt
├── Dockerfile                # Docker image for API + Dashboard
├── Dockerfile.mlflow         # Docker image for MLflow server
├── docker-compose.yml        # Runs all 3 services together
├── retrain.yml               # GitHub Actions CI/CD pipeline
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
| REST API | FastAPI | `/predict`, `/health` endpoints for integration |
| Containerization | Docker Compose | All services run in isolated containers |

---

## 🎯 Model Performance

| Metric | Score |
|---|---|
| Accuracy | **99.69%** |
| F1 Score | **99.19%** |
| Precision | **100.00%** |
| Recall | **98.39%** |

Model: Random Forest Classifier (n_estimators=100, max_depth=10)

---

## 📈 Dataset

- **Synthetic data**: Generated to mirror MobiFall v2.0 characteristics
- **MobiFall Dataset**: https://www.kaggle.com/datasets/kmknation/mobifall-dataset-v20
- **Features**: `ax, ay, az` (accelerometer), `gx, gy, gz` (gyroscope), `sma, smg, resultant`
- **Labels**: `0 = Normal Activity (ADL)`, `1 = Fall`

---

## 🐳 Docker Hub

Pull and run the dashboard directly:
```bash
docker run -p 8501:8501 yashikaa0000/fall-detection-cp5
```

---

## 👥 Team

Capstone Project 5 — Yashika, Harman, Amaan
Program: Artificial Intelligence and Machine Learning