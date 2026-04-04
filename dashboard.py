"""
dashboard.py
Streamlit dashboard for CP5 Fall Detection MLOps demo.
Shows: live prediction, model metrics, MLflow run history, drift report.

Run with:  streamlit run dashboard.py
"""

import streamlit as st
import pandas as pd
import numpy as np
import pickle, json, os, subprocess, sys

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Fall Detection MLOps",
    page_icon="🩺",
    layout="wide"
)

# ── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
    .metric-card {
        background: #1e1e2e;
        border-radius: 12px;
        padding: 20px;
        text-align: center;
        border: 1px solid #313244;
    }
    .metric-value { font-size: 2rem; font-weight: 700; color: #cba6f7; }
    .metric-label { color: #a6adc8; font-size: 0.9rem; margin-top: 4px; }
    .fall-alert   { background:#f38ba8; color:#1e1e2e; padding:16px;
                    border-radius:10px; text-align:center; font-size:1.3rem; font-weight:700; }
    .safe-alert   { background:#a6e3a1; color:#1e1e2e; padding:16px;
                    border-radius:10px; text-align:center; font-size:1.3rem; font-weight:700; }
</style>
""", unsafe_allow_html=True)


# ── Helpers ───────────────────────────────────────────────────────────────────

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
                "Run ID":    r.info.run_id[:8],
                "Status":    r.info.status,
                "Accuracy":  r.data.metrics.get("accuracy",  "-"),
                "F1 Score":  r.data.metrics.get("f1_score",  "-"),
                "Precision": r.data.metrics.get("precision", "-"),
                "Recall":    r.data.metrics.get("recall",    "-"),
                "n_estimators": r.data.params.get("n_estimators", "-"),
                "max_depth":    r.data.params.get("max_depth",    "-"),
            })
        return pd.DataFrame(rows)
    except Exception:
        return pd.DataFrame()


# ── Header ────────────────────────────────────────────────────────────────────
st.title("🩺 Fall Detection — MLOps Dashboard")
st.caption("Capstone Project 5 | DevOps for AI Deployment | MLflow + Evidently + Streamlit")
st.divider()


# ── Sidebar: Run pipeline ─────────────────────────────────────────────────────
with st.sidebar:
    st.header("⚙️ Pipeline Controls")

    if st.button("🔄 Generate Data", use_container_width=True):
        with st.spinner("Generating sensor data..."):
            subprocess.run([sys.executable, "preprocess_data.py"], check=True)
        st.success("Data generated!")

    st.subheader("Train Model")
    n_est  = st.slider("n_estimators", 50, 300, 100, 50)
    depth  = st.slider("max_depth",     3,  20,  10,  1)

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


# ── Tabs ──────────────────────────────────────────────────────────────────────
tab1, tab2, tab3 = st.tabs(["🔮 Live Prediction", "📈 MLflow Experiments", "🌊 Drift Monitor"])


# ══ TAB 1: Live Prediction ════════════════════════════════════════════════════
with tab1:
    model   = load_model()
    metrics = load_metrics()

    if model is None:
        st.warning("⚠️ No trained model found. Click **Train + Log to MLflow** in the sidebar.")
    else:
        # Metric cards
        c1, c2, c3, c4 = st.columns(4)
        for col, (name, key) in zip(
            [c1,c2,c3,c4],
            [("Accuracy","accuracy"),("F1 Score","f1_score"),
             ("Precision","precision"),("Recall","recall")]
        ):
            val = metrics.get(key, "—")
            val_str = f"{val:.2%}" if isinstance(val, float) else val
            col.markdown(f"""
            <div class="metric-card">
                <div class="metric-value">{val_str}</div>
                <div class="metric-label">{name}</div>
            </div>""", unsafe_allow_html=True)

        st.divider()
        st.subheader("🎚️ Enter Sensor Readings")

        col_a, col_b = st.columns(2)
        with col_a:
            st.markdown("**Accelerometer (g)**")
            ax = st.slider("ax", -5.0, 5.0, 0.1, 0.1)
            ay = st.slider("ay", -5.0, 5.0, 0.1, 0.1)
            az = st.slider("az", -5.0, 5.0, 1.0, 0.1)

        with col_b:
            st.markdown("**Gyroscope (°/s)**")
            gx = st.slider("gx", -200.0, 200.0, 5.0, 1.0)
            gy = st.slider("gy", -200.0, 200.0, 5.0, 1.0)
            gz = st.slider("gz", -200.0, 200.0, 5.0, 1.0)

        # Quick presets
        st.markdown("**Quick Presets:**")
        p1, p2, p3 = st.columns(3)
        use_fall   = p1.button("💥 Simulate Fall")
        use_walk   = p2.button("🚶 Simulate Walking")
        use_sit    = p3.button("🪑 Simulate Sitting")

        if use_fall:
            ax,ay,az = 2.8,-1.6,3.2
            gx,gy,gz = 85.0,72.0,63.0
        elif use_walk:
            ax,ay,az = 0.2, 0.1, 1.0
            gx,gy,gz = 6.0, 5.0, 4.0
        elif use_sit:
            ax,ay,az = 0.05,0.05,1.0
            gx,gy,gz = 1.0, 1.0, 1.0

        # Simulate a 100-sample window around the slider values
        # This matches the 35-feature extraction used during training
        np.random.seed(0)
        noise_scale = 0.05
        win = pd.DataFrame({
            "ax": np.random.normal(ax, noise_scale, 100),
            "ay": np.random.normal(ay, noise_scale, 100),
            "az": np.random.normal(az, noise_scale, 100),
            "gx": np.random.normal(gx, noise_scale, 100),
            "gy": np.random.normal(gy, noise_scale, 100),
            "gz": np.random.normal(gz, noise_scale, 100),
        })
        def extract_features_dash(w):
            f = {}
            for col in ["ax","ay","az","gx","gy","gz"]:
                s = w[col].values
                f[f"{col}_mean"]  = np.mean(s)
                f[f"{col}_std"]   = np.std(s)
                f[f"{col}_max"]   = np.max(s)
                f[f"{col}_min"]   = np.min(s)
                f[f"{col}_range"] = np.max(s) - np.min(s)
            f["sma_acc"] = (np.sum(np.abs(w["ax"]))+np.sum(np.abs(w["ay"]))+np.sum(np.abs(w["az"])))/len(w)
            f["sma_gyr"] = (np.sum(np.abs(w["gx"]))+np.sum(np.abs(w["gy"]))+np.sum(np.abs(w["gz"])))/len(w)
            res = np.sqrt(w["ax"]**2+w["ay"]**2+w["az"]**2)
            f["resultant_mean"] = np.mean(res)
            f["resultant_max"]  = np.max(res)
            f["resultant_std"]  = np.std(res)
            return f
        feat_dict = extract_features_dash(win)
        features  = pd.DataFrame([feat_dict])
        pred     = model.predict(features)[0]
        proba    = model.predict_proba(features)[0]

        st.divider()
        if pred == 1:
            st.markdown(f'<div class="fall-alert">🚨 FALL DETECTED — Confidence: {proba[1]:.1%}</div>',
                        unsafe_allow_html=True)
        else:
            st.markdown(f'<div class="safe-alert">✅ NORMAL ACTIVITY — Confidence: {proba[0]:.1%}</div>',
                        unsafe_allow_html=True)

        st.progress(float(proba[1]), text=f"Fall probability: {proba[1]:.1%}")


# ══ TAB 2: MLflow Experiments ═════════════════════════════════════════════════
with tab2:
    st.subheader("📋 Experiment Runs")
    runs_df = load_mlflow_runs()

    if runs_df.empty:
        st.info("No MLflow runs found yet. Train at least one model from the sidebar.")
    else:
        st.dataframe(runs_df, use_container_width=True, hide_index=True)

        if "Accuracy" in runs_df.columns and len(runs_df) > 1:
            st.subheader("Accuracy Across Runs")
            chart_df = runs_df[["Run ID","Accuracy","F1 Score"]].copy()
            chart_df = chart_df[chart_df["Accuracy"] != "-"]
            chart_df["Accuracy"] = chart_df["Accuracy"].astype(float)
            chart_df["F1 Score"] = chart_df["F1 Score"].astype(float)
            chart_df = chart_df.set_index("Run ID")
            st.bar_chart(chart_df)

    st.divider()
    st.info("💡 Run `mlflow ui` in your terminal to open the full MLflow experiment tracker at http://localhost:5000")


# ══ TAB 3: Drift Monitor ══════════════════════════════════════════════════════
with tab3:
    st.subheader("🌊 Data Drift Report")
    report_path = "reports/drift_report.html"

    if not os.path.exists(report_path):
        st.info("No drift report yet. Click **Run Drift Report** in the sidebar.")
    else:
        with open(report_path,"r",encoding="utf-8") as f:
            html = f.read()
        st.components.v1.html(html, height=800, scrolling=True)
