# dashboard.py
import streamlit as st
import pandas as pd
import numpy as np
import pickle, json, os, subprocess, sys

st.set_page_config(page_title="Fall Detection MLOps", page_icon="🩺", layout="wide")

st.markdown("""
<style>
    .fall-alert { background:#f38ba8; color:#1e1e2e; padding:16px;
                  border-radius:10px; text-align:center; font-size:1.3rem; font-weight:700; }
    .safe-alert { background:#a6e3a1; color:#1e1e2e; padding:16px;
                  border-radius:10px; text-align:center; font-size:1.3rem; font-weight:700; }
</style>
""", unsafe_allow_html=True)

@st.cache_resource
def load_model():
    if not os.path.exists("model/model.pkl"):
        return None
    with open("model/model.pkl","rb") as f:
        return pickle.load(f)

def load_metrics():
    if not os.path.exists("model/metrics.json"):
        return {}
    with open("model/metrics.json") as f:
        return json.load(f)

def load_mlflow_runs():
    try:
        import mlflow
        client = mlflow.tracking.MlflowClient()
        exp    = client.get_experiment_by_name("fall-detection-cp5")
        if exp is None:
            return pd.DataFrame()
        runs = client.search_runs(
            experiment_ids=[exp.experiment_id],
            order_by=["start_time DESC"],
            max_results=10
        )
        rows = []
        for r in runs:
            rows.append({
                "Run ID":     r.info.run_id[:8],
                "Status":     r.info.status,
                "Accuracy":   r.data.metrics.get("accuracy",  "-"),
                "F1 Score":   r.data.metrics.get("f1_score",  "-"),
                "Precision":  r.data.metrics.get("precision", "-"),
                "Recall":     r.data.metrics.get("recall",    "-"),
                "n_estimators": r.data.params.get("n_estimators", "-"),
                "max_depth":    r.data.params.get("max_depth",    "-"),
            })
        return pd.DataFrame(rows)
    except Exception:
        return pd.DataFrame()

# ── Header
st.title("🩺 Fall Detection — MLOps Dashboard")
st.caption("Capstone Project 5 | DevOps for AI | MLflow + Evidently + Streamlit")
st.divider()

# ── Sidebar
with st.sidebar:
    st.header("⚙️ Pipeline Controls")

    st.subheader("Train Model")
    n_est = st.slider("n_estimators", 50, 300, 100, 50)
    depth = st.slider("max_depth", 3, 20, 10, 1)

    if st.button("🚀 Train + Log to MLflow", use_container_width=True):
        with st.spinner("Training..."):
            result = subprocess.run(
                [sys.executable, "train.py",
                 f"--n_estimators={n_est}",
                 f"--max_depth={depth}"],
                capture_output=True, text=True
            )
        if result.returncode == 0:
            st.success("Model trained & logged!")
            st.cache_resource.clear()
        else:
            st.error(result.stderr[-500:])

    if st.button("📊 Run Drift Report", use_container_width=True):
        with st.spinner("Running Evidently..."):
            subprocess.run([sys.executable, "monitor.py"], check=True)
        st.success("Drift report ready!")

# ── Tabs
tab1, tab2, tab3 = st.tabs(["🔮 Live Prediction", "📈 MLflow Experiments", "🌊 Drift Monitor"])

FEATURES = ["ax", "ay", "az", "gx", "gy", "gz", "sma", "smg", "resultant"]

# ══ TAB 1
with tab1:
    model   = load_model()
    metrics = load_metrics()

    if model is None:
        st.warning("No trained model found. Click Train in the sidebar.")
    else:
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Accuracy",  f"{metrics.get('accuracy',0):.2%}")
        c2.metric("F1 Score",  f"{metrics.get('f1_score',0):.2%}")
        c3.metric("Precision", f"{metrics.get('precision',0):.2%}")
        c4.metric("Recall",    f"{metrics.get('recall',0):.2%}")

        st.divider()
        st.subheader("🎚️ Enter Sensor Readings")

        # Presets
        p1, p2, p3 = st.columns(3)
        preset = None
        if p1.button("💥 Simulate Fall"):    preset = "fall"
        if p2.button("🚶 Simulate Walking"): preset = "walk"
        if p3.button("🪑 Simulate Sitting"): preset = "sit"

        defaults = {
            "fall": [2.8, -1.6, 3.2, 85.0, 72.0, 63.0, 4.5, 3.8, 4.6],
            "walk": [0.2,  0.1, 1.0,  6.0,  5.0,  4.0, 0.4, 0.3, 1.0],
            "sit":  [0.05,0.05, 1.0,  1.0,  1.0,  1.0, 0.1, 0.1, 1.0],
        }
        vals = defaults.get(preset, [0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.5, 0.3, 1.0])

        col_a, col_b, col_c = st.columns(3)
        with col_a:
            st.markdown("**Accelerometer**")
            ax = st.slider("ax", -5.0, 5.0, float(vals[0]), 0.1)
            ay = st.slider("ay", -5.0, 5.0, float(vals[1]), 0.1)
            az = st.slider("az", -5.0, 5.0, float(vals[2]), 0.1)
        with col_b:
            st.markdown("**Gyroscope**")
            gx = st.slider("gx", -200.0, 200.0, float(vals[3]), 1.0)
            gy = st.slider("gy", -200.0, 200.0, float(vals[4]), 1.0)
            gz = st.slider("gz", -200.0, 200.0, float(vals[5]), 1.0)
        with col_c:
            st.markdown("**Derived**")
            sma       = st.slider("SMA",       0.0, 10.0, float(vals[6]), 0.1)
            smg       = st.slider("SMG",       0.0, 10.0, float(vals[7]), 0.1)
            resultant = st.slider("Resultant", 0.0, 10.0, float(vals[8]), 0.1)

        features = pd.DataFrame(
            [[ax, ay, az, gx, gy, gz, sma, smg, resultant]],
            columns=FEATURES
        )

        pred  = model.predict(features)[0]
        proba = model.predict_proba(features)[0]

        st.divider()
        if pred == 1:
            st.markdown(f'<div class="fall-alert">🚨 FALL DETECTED — Confidence: {proba[1]:.1%}</div>',
                        unsafe_allow_html=True)
        else:
            st.markdown(f'<div class="safe-alert">✅ NORMAL ACTIVITY — Confidence: {proba[0]:.1%}</div>',
                        unsafe_allow_html=True)
        st.progress(float(proba[1]), text=f"Fall probability: {proba[1]:.1%}")

# ══ TAB 2
with tab2:
    st.subheader("📋 Experiment Runs")
    runs_df = load_mlflow_runs()
    if runs_df.empty:
        st.info("No MLflow runs found yet. Train at least one model from the sidebar.")
    else:
        st.dataframe(runs_df, use_container_width=True, hide_index=True)
    st.divider()
    st.info("💡 Run `mlflow ui` in terminal to open full MLflow tracker at http://localhost:5000")

# ══ TAB 3
with tab3:
    st.subheader("🌊 Data Drift Report")
    report_path = "reports/drift_report.html"
    if not os.path.exists(report_path):
        st.info("No drift report yet. Click Run Drift Report in the sidebar.")
    else:
        with open(report_path, "r", encoding="utf-8") as f:
            html = f.read()
        st.components.v1.html(html, height=800, scrolling=True)