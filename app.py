import streamlit as st
import joblib
import numpy as np
import pandas as pd
import logging
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report

# ──────────────────────────────────────────────
# Logging
# ──────────────────────────────────────────────
logging.basicConfig(
    filename="app.log",
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)

# ──────────────────────────────────────────────
# Page Config
# ──────────────────────────────────────────────
st.set_page_config(
    page_title="AgroNova – Crop Variety Classifier",
    page_icon="🌿",
    layout="centered"
)

# ──────────────────────────────────────────────
# Minimal Custom Styling (no frameworks)
# ──────────────────────────────────────────────
st.markdown("""
<style>
    /* Card-style containers */
    .card {
        background-color: #f9fafb;
        border: 1px solid #e0e0e0;
        border-radius: 12px;
        padding: 24px 28px;
        margin-bottom: 16px;
    }
    /* Result highlight card */
    .result-card {
        background-color: #e8f5e9;
        border: 1px solid #a5d6a7;
        border-radius: 12px;
        padding: 20px 24px;
        margin-top: 16px;
    }
    /* Footer */
    .footer {
        text-align: center;
        color: #9e9e9e;
        font-size: 13px;
        padding-top: 32px;
        padding-bottom: 8px;
    }
    /* Tab spacing fix */
    div[data-baseweb="tab-list"] {
        gap: 8px;
    }
    /* Subtle divider */
    hr { border-top: 1px solid #e0e0e0; }
</style>
""", unsafe_allow_html=True)

# ──────────────────────────────────────────────
# Constants
# ──────────────────────────────────────────────
MODEL_PATH = "model_v1.pkl"
CROP_VARIETIES = {
    0: "Crop Variety A (Setosa)",
    1: "Crop Variety B (Versicolor)",
    2: "Crop Variety C (Virginica)"
}
FEATURE_NAMES  = ["Leaf Length", "Leaf Width", "Petal Length", "Petal Width"]
CLASS_LABELS   = ["Crop Variety A", "Crop Variety B", "Crop Variety C"]
SAMPLE_PRESETS = {
    "Custom Input":                (5.1, 3.5, 1.4, 0.2),
    "Crop Variety A (Setosa)":     (5.1, 3.5, 1.4, 0.2),
    "Crop Variety B (Versicolor)": (6.0, 2.8, 4.5, 1.3),
    "Crop Variety C (Virginica)":  (6.5, 3.0, 5.5, 2.0),
}

# ──────────────────────────────────────────────
# Load Model
# ──────────────────────────────────────────────
try:
    model = joblib.load(MODEL_PATH)
except FileNotFoundError:
    st.error(f"⚠️ Model file `{MODEL_PATH}` not found. Please run `train.py` first.")
    st.stop()

# ──────────────────────────────────────────────
# Pre-compute Metrics (cached – runs once)
# ──────────────────────────────────────────────
@st.cache_data
def get_performance_metrics():
    iris = load_iris()
    X, y = iris.data, iris.target
    _, X_test, _, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    y_pred   = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    report   = classification_report(
        y_test, y_pred, target_names=CLASS_LABELS, output_dict=True
    )
    return accuracy, report

accuracy, report_dict = get_performance_metrics()

# ──────────────────────────────────────────────
# Pre-compute Feature Importances (static – from model)
# ──────────────────────────────────────────────
importances      = model.feature_importances_
top_feature_idx  = int(np.argmax(importances))
top_feature_name = FEATURE_NAMES[top_feature_idx]

# ──────────────────────────────────────────────
# App Header
# ──────────────────────────────────────────────
st.markdown("## 🌿 AgroNova")
st.markdown("`MLOps Project  ·  Model v1.0  ·  Random Forest Classifier`")
st.markdown(
    "An end-to-end MLOps demo that classifies crop varieties from leaf measurements. "
    "Explore predictions, model performance, and feature insights below."
)
st.markdown("---")

# ──────────────────────────────────────────────
# Tabs
# ──────────────────────────────────────────────
tab_predict, tab_performance, tab_insights = st.tabs([
    "🌱  Predict",
    "📊  Performance",
    "🔍  Insights",
])

# ══════════════════════════════════════════════
# TAB 1 ─ PREDICT
# ══════════════════════════════════════════════
with tab_predict:

    st.markdown("### 🌱 Crop Variety Prediction")
    st.caption("Select a preset sample or enter your own measurements to classify a crop variety.")
    st.markdown(" ")

    # ── Preset Selector ──────────────────────
    selected_demo = st.selectbox(
        "🔎 Load a Sample Preset",
        list(SAMPLE_PRESETS.keys())
    )
    def_sl, def_sw, def_pl, def_pw = SAMPLE_PRESETS[selected_demo]

    st.markdown(" ")

    # ── Input Card ───────────────────────────
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown("**📋 Leaf & Petal Measurements**")
    col1, col2 = st.columns(2, gap="large")
    with col1:
        sepal_length = st.number_input("🌿 Leaf Length (cm)", min_value=0.0, value=def_sl, format="%.2f", key="sl")
        sepal_width  = st.number_input("🌿 Leaf Width (cm)",  min_value=0.0, value=def_sw, format="%.2f", key="sw")
    with col2:
        petal_length = st.number_input("🌸 Petal Length (cm)", min_value=0.0, value=def_pl, format="%.2f", key="pl")
        petal_width  = st.number_input("🌸 Petal Width (cm)",  min_value=0.0, value=def_pw, format="%.2f", key="pw")
    st.markdown('</div>', unsafe_allow_html=True)

    # ── Action Buttons ────────────────────────
    btn1, btn2, _ = st.columns([1.2, 1, 2])
    with btn1:
        predict_clicked = st.button("🔍 Predict Variety", use_container_width=True)
    with btn2:
        if st.button("🔄 Reset", use_container_width=True):
            st.rerun()

    # ── Prediction Result ─────────────────────
    if predict_clicked:
        inputs = [sepal_length, sepal_width, petal_length, petal_width]
        if any(v < 0 for v in inputs):
            st.error("⚠️ All measurements must be positive values.")
            logging.warning("Negative inputs submitted.")
        else:
            input_array = np.array([inputs])
            try:
                prediction    = model.predict(input_array)[0]
                proba         = model.predict_proba(input_array)[0]
                confidence    = max(proba) * 100
                predicted_crop = CROP_VARIETIES.get(prediction, "Unknown")

                # Result card
                st.markdown('<div class="result-card">', unsafe_allow_html=True)
                st.success(f"**🌱 Predicted Variety:** {predicted_crop}")
                st.markdown('</div>', unsafe_allow_html=True)

                # Confidence metrics
                st.markdown(" ")
                m1, m2, m3 = st.columns(3)
                m1.metric("Confidence",      f"{confidence:.1f}%")
                m2.metric("Predicted Class", f"Class {prediction}")
                m3.metric("Total Classes",   "3")

                # ── Dynamic Insight based on input ───────
                st.markdown("---")
                st.markdown("**📌 Prediction Insight**")

                # Compare user's petal_length against training median (~3.75)
                if petal_length > 3.75:
                    st.info(
                        f"📌 Your input has a **high Petal Length ({petal_length:.2f} cm)**, "
                        f"which strongly influenced this prediction — "
                        f"Petal Length is the model's most impactful feature."
                    )
                elif petal_length < 2.0:
                    st.info(
                        f"📌 Your input has a **low Petal Length ({petal_length:.2f} cm)**, "
                        f"which is a strong indicator of Setosa-type varieties."
                    )
                else:
                    st.info(
                        f"📌 Your Petal Length ({petal_length:.2f} cm) falls in a **transition zone** — "
                        f"the model relied on multiple features to make this prediction."
                    )

                logging.info(
                    f"Prediction: {input_array.tolist()} → {predicted_crop} "
                    f"(Confidence: {confidence:.1f}%)"
                )

            except Exception as e:
                st.error(f"Prediction error: {str(e)}")
                logging.error(f"Prediction error: {str(e)}")

# ══════════════════════════════════════════════
# TAB 2 ─ PERFORMANCE
# ══════════════════════════════════════════════
with tab_performance:

    st.markdown("### 📊 Model Performance Metrics")
    st.caption(
        "Evaluated on a held-out test set — 20% of the Iris dataset (`random_state=42`). "
        "Metrics are computed once at startup."
    )
    st.markdown("---")

    # ── Accuracy ──────────────────────────────
    st.markdown("#### Overall Accuracy")
    acc_col, _ = st.columns([1, 2])
    acc_col.metric("Test Set Accuracy", f"{accuracy:.4f}", delta="Held-out set")

    st.markdown(" ")
    st.markdown("---")

    # ── Classification Report Table ───────────
    st.markdown("#### 📋 Classification Report")
    st.caption("Per-class breakdown of Precision, Recall, and F1-Score.")

    rows = []
    for label in CLASS_LABELS:
        r = report_dict[label]
        rows.append({
            "Crop Variety": label,
            "Precision":    round(r["precision"], 3),
            "Recall":       round(r["recall"],    3),
            "F1-Score":     round(r["f1-score"],  3),
            "Support":      int(r["support"]),
        })
    report_df = pd.DataFrame(rows).set_index("Crop Variety")
    st.dataframe(report_df, use_container_width=True)

    st.markdown(" ")
    st.markdown("---")

    # ── Weighted Averages ─────────────────────
    st.markdown("#### 📈 Weighted Averages")
    st.caption("Aggregated scores weighted by class support.")
    wa = report_dict["weighted avg"]
    c1, c2, c3 = st.columns(3)
    c1.metric("Precision", f"{wa['precision']:.3f}")
    c2.metric("Recall",    f"{wa['recall']:.3f}")
    c3.metric("F1-Score",  f"{wa['f1-score']:.3f}")

# ══════════════════════════════════════════════
# TAB 3 ─ INSIGHTS
# ══════════════════════════════════════════════
with tab_insights:

    st.markdown("### 🔍 Feature Importance Analysis")
    st.caption(
        "Feature importance shows the **global influence** of each measurement "
        "on the model's predictions — derived directly from the trained Random Forest."
    )
    st.markdown("---")

    # ── Static Global Insight ─────────────────
    st.info(
        f"🔥 Based on model behavior, **{top_feature_name}** is the most influential feature "
        f"with an importance score of `{importances[top_feature_idx]:.4f}`."
    )

    st.markdown(" ")

    # ── Importance Score Cards ────────────────
    st.markdown("#### 📌 Importance Scores per Feature")
    cols = st.columns(4)
    for col, name, score in zip(cols, FEATURE_NAMES, importances):
        label = f"⭐ {name}" if name == top_feature_name else name
        col.metric(label=label, value=f"{score:.4f}")

    st.markdown(" ")
    st.markdown("---")

    # ── Bar Chart ─────────────────────────────
    st.markdown("#### 📊 Feature Importance Chart")
    st.caption("Higher bars = greater influence on prediction outcomes.")

    chart_df = pd.DataFrame(
        {"Importance Score": importances},
        index=FEATURE_NAMES
    )
    st.bar_chart(chart_df, use_container_width=True)

    st.markdown("---")

    # ── Interpretation Guide ──────────────────
    st.markdown("#### 💡 How to Read This")
    g1, g2 = st.columns(2)
    g1.markdown("**High Importance**\nFeature strongly separates crop classes. Small changes have big impact.")
    g2.markdown("**Low Importance**\nFeature contributes less to class separation. Model relies on it less.")

# ──────────────────────────────────────────────
# Footer
# ──────────────────────────────────────────────
st.markdown("---")
st.markdown(
    '<div class="footer">AgroNova &nbsp;|&nbsp; MLOps Project &nbsp;|&nbsp; '
    'Built with Streamlit &amp; scikit-learn</div>',
    unsafe_allow_html=True
)
