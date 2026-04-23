import streamlit as st
import joblib
import numpy as np
import logging
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report

# ──────────────────────────────────────────────
# Logging Configuration
# ──────────────────────────────────────────────
logging.basicConfig(
    filename='app.log',
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

# ──────────────────────────────────────────────
# Page Configuration
# ──────────────────────────────────────────────
st.set_page_config(
    page_title="AgroNova - Crop Variety Classifier",
    page_icon="🌿",
    layout="centered"
)

# ──────────────────────────────────────────────
# Header Section
# ──────────────────────────────────────────────
st.markdown("## 🌿 AgroNova — Crop Variety Classifier")
st.markdown("`Model Version: v1.0`")
st.markdown(
    "An **MLOps-powered** crop classification tool. "
    "Enter leaf measurements to predict the crop variety, "
    "or explore model performance and feature insights below."
)
st.markdown("---")

# ──────────────────────────────────────────────
# Load Trained Model
# ──────────────────────────────────────────────
MODEL_PATH = "model_v1.pkl"
try:
    model = joblib.load(MODEL_PATH)
except FileNotFoundError:
    st.error(f"⚠️ Model file `{MODEL_PATH}` not found. Please run `train.py` first.")
    st.stop()

# ──────────────────────────────────────────────
# Constants
# ──────────────────────────────────────────────
CROP_VARIETIES = {
    0: "Crop Variety A (Setosa)",
    1: "Crop Variety B (Versicolor)",
    2: "Crop Variety C (Virginica)"
}

FEATURE_NAMES = ["Leaf Length", "Leaf Width", "Petal Length", "Petal Width"]

CLASS_LABELS = ["Crop Variety A", "Crop Variety B", "Crop Variety C"]

# ──────────────────────────────────────────────
# Pre-compute Performance Metrics (cached)
# ──────────────────────────────────────────────
@st.cache_data
def get_metrics():
    iris = load_iris()
    X, y = iris.data, iris.target
    _, X_test, _, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    report_dict = classification_report(
        y_test, y_pred,
        target_names=CLASS_LABELS,
        output_dict=True
    )
    return accuracy, report_dict

accuracy, report_dict = get_metrics()

# ──────────────────────────────────────────────
# Tabs
# ──────────────────────────────────────────────
tab_predict, tab_performance, tab_insights = st.tabs([
    "🌱  Predict",
    "📊  Performance",
    "🔍  Insights"
])

# ══════════════════════════════════════════════
# TAB 1 — PREDICT
# ══════════════════════════════════════════════
with tab_predict:
    st.markdown("### 🌱 Crop Variety Prediction")
    st.markdown("Select a sample crop for a quick demo or enter custom leaf measurements below.")
    st.markdown("---")

    # ── Sample Selector ──────────────────────
    demo_options = [
        "Custom Input",
        "Crop Variety A (Setosa)",
        "Crop Variety B (Versicolor)",
        "Crop Variety C (Virginica)"
    ]
    selected_demo = st.selectbox("🔎 Quick Demo — Select a Sample Crop", demo_options)

    # Set defaults based on selection
    DEFAULTS = {
        "Crop Variety A (Setosa)":    (5.1, 3.5, 1.4, 0.2),
        "Crop Variety B (Versicolor)":(6.0, 2.8, 4.5, 1.3),
        "Crop Variety C (Virginica)": (6.5, 3.0, 5.5, 2.0),
        "Custom Input":               (5.1, 3.5, 1.4, 0.2),
    }
    def_sl, def_sw, def_pl, def_pw = DEFAULTS[selected_demo]

    st.markdown(" ")

    # ── Input Card ───────────────────────────
    with st.container():
        st.markdown("#### 📋 Enter Crop Measurements")
        col1, col2 = st.columns(2, gap="large")

        with col1:
            sepal_length = st.number_input(
                "🌿 Leaf Length (cm)", min_value=0.0, value=def_sl, format="%.2f"
            )
            sepal_width = st.number_input(
                "🌿 Leaf Width (cm)", min_value=0.0, value=def_sw, format="%.2f"
            )
        with col2:
            petal_length = st.number_input(
                "🌸 Petal Length (cm)", min_value=0.0, value=def_pl, format="%.2f"
            )
            petal_width = st.number_input(
                "🌸 Petal Width (cm)", min_value=0.0, value=def_pw, format="%.2f"
            )

    st.markdown(" ")

    # ── Action Buttons ────────────────────────
    col_btn1, col_btn2, _ = st.columns([1, 1, 2])
    with col_btn1:
        predict_clicked = st.button("🔍 Predict Variety", use_container_width=True)
    with col_btn2:
        if st.button("🔄 Reset", use_container_width=True):
            st.rerun()

    st.markdown("---")

    # ── Prediction Result ─────────────────────
    if predict_clicked:
        inputs = [sepal_length, sepal_width, petal_length, petal_width]
        if any(v < 0 for v in inputs):
            st.error("⚠️ All input values must be positive numbers.")
            logging.warning("User attempted to predict with negative inputs.")
        else:
            input_data = np.array([inputs])
            try:
                prediction   = model.predict(input_data)[0]
                probabilities = model.predict_proba(input_data)[0]
                confidence   = max(probabilities) * 100
                predicted_crop = CROP_VARIETIES.get(prediction, "Unknown")

                st.success(f"🌱 **Predicted Crop Variety:** {predicted_crop}")
                st.markdown(" ")

                m1, m2 = st.columns(2)
                m1.metric(label="Confidence Score", value=f"{confidence:.2f}%")
                m2.metric(label="Predicted Class",  value=f"Class {prediction}")

                logging.info(
                    f"Prediction: Inputs={input_data.tolist()} → "
                    f"Output={predicted_crop} (Confidence: {confidence:.2f}%)"
                )
            except Exception as e:
                st.error(f"Prediction error: {str(e)}")
                logging.error(f"Prediction error: {str(e)}")

# ══════════════════════════════════════════════
# TAB 2 — PERFORMANCE
# ══════════════════════════════════════════════
with tab_performance:
    st.markdown("### 📊 Model Performance Metrics")
    st.markdown("Evaluated on the **held-out test set** (20% of the Iris dataset, `random_state=42`).")
    st.markdown("---")

    # ── Accuracy Metric ───────────────────────
    st.markdown("#### Overall Accuracy")
    col_acc, col_pad = st.columns([1, 2])
    col_acc.metric(label="Accuracy Score", value=f"{accuracy:.4f}", delta="Test Set")

    st.markdown(" ")
    st.markdown("---")

    # ── Classification Report Table ───────────
    st.markdown("#### 📋 Classification Report")
    st.caption("Precision, Recall, and F1-Score per crop variety.")

    # Build a clean DataFrame from the report dict
    rows = []
    for label in CLASS_LABELS:
        r = report_dict[label]
        rows.append({
            "Crop Variety":  label,
            "Precision":     round(r["precision"], 3),
            "Recall":        round(r["recall"], 3),
            "F1-Score":      round(r["f1-score"], 3),
            "Support":       int(r["support"])
        })

    report_df = pd.DataFrame(rows).set_index("Crop Variety")
    st.dataframe(report_df, use_container_width=True)

    st.markdown(" ")

    # ── Overall Averages ──────────────────────
    st.markdown("#### 📈 Weighted Averages")
    wa = report_dict["weighted avg"]
    c1, c2, c3 = st.columns(3)
    c1.metric("Precision", f"{wa['precision']:.3f}")
    c2.metric("Recall",    f"{wa['recall']:.3f}")
    c3.metric("F1-Score",  f"{wa['f1-score']:.3f}")

# ══════════════════════════════════════════════
# TAB 3 — INSIGHTS
# ══════════════════════════════════════════════
with tab_insights:
    st.markdown("### 🔍 Feature Importance Analysis")
    st.markdown("Understanding which leaf measurements **influence predictions the most**.")
    st.markdown("---")

    if hasattr(model, "feature_importances_"):
        importances = model.feature_importances_

        # ── Summary Metrics ───────────────────
        st.markdown("#### 📌 Importance Scores")
        cols = st.columns(4)
        for col, name, imp in zip(cols, FEATURE_NAMES, importances):
            col.metric(label=name, value=f"{imp:.4f}")

        st.markdown(" ")

        # ── Bar Chart ─────────────────────────
        st.markdown("#### 📊 Feature Importance Chart")
        st.caption("💡 Higher values indicate greater influence on the prediction outcome.")

        chart_data = pd.DataFrame(
            {"Importance": importances},
            index=FEATURE_NAMES
        )
        st.bar_chart(chart_data, use_container_width=True)

        st.markdown("---")

        # ── Key Insight ───────────────────────
        top_feature = FEATURE_NAMES[int(np.argmax(importances))]
        st.info(
            f"🏆 **Top Feature:** `{top_feature}` has the highest importance score "
            f"of `{max(importances):.4f}`, making it the strongest predictor in this model."
        )
    else:
        st.info("The current model does not support feature importances.")
