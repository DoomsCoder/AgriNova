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
    page_title="AgroNova – MLOps Dashboard",
    page_icon="🌿",
    layout="centered",
    initial_sidebar_state="collapsed"
)

# ──────────────────────────────────────────────
# Premium Custom CSS (SaaS-style)
# ──────────────────────────────────────────────
st.markdown("""
<style>
    /* App background */
    .stApp {
        background-color: #f5f7f9;
        font-family: 'Inter', sans-serif;
    }

    /* Centered Header */
    .header-container {
        text-align: center;
        padding: 2rem 1rem;
        margin-bottom: 2rem;
    }
    .header-title {
        font-size: 3rem;
        font-weight: 800;
        color: #1e293b;
        margin-bottom: 0.5rem;
    }
    .header-subtitle {
        font-size: 1.1rem;
        font-weight: 600;
        color: #10b981;
        letter-spacing: 1px;
        text-transform: uppercase;
        margin-bottom: 1rem;
    }
    .header-desc {
        font-size: 1rem;
        color: #64748b;
        max-width: 600px;
        margin: 0 auto;
        line-height: 1.5;
    }
    
    /* Result Card (Green Theme) */
    .success-card {
        background: linear-gradient(135deg, #ecfdf5 0%, #d1fae5 100%);
        border-radius: 12px;
        padding: 24px;
        box-shadow: 0 10px 15px -3px rgba(16, 185, 129, 0.1), 0 4px 6px -2px rgba(16, 185, 129, 0.05);
        margin-top: 24px;
        border: 1px solid #a7f3d0;
        text-align: center;
    }
    .success-card h3 {
        color: #065f46;
        margin: 0;
        font-size: 1.5rem;
        font-weight: 700;
    }

    /* Top Feature Highlight Card */
    .highlight-card {
        background-color: #fffbeb;
        border-left: 4px solid #f59e0b;
        border-radius: 8px;
        padding: 20px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.02);
        margin-bottom: 24px;
    }

    /* Buttons */
    .stButton>button {
        border-radius: 8px !important;
        font-weight: 600 !important;
        padding: 0.5rem 1rem !important;
        transition: all 0.2s ease !important;
        border: none !important;
        box-shadow: 0 2px 4px rgba(0,0,0,0.05) !important;
    }
    .stButton>button:hover {
        transform: translateY(-1px) !important;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1) !important;
    }
    /* Primary Predict Button styling override */
    div.row-widget.stButton:first-of-type > button {
        background-color: #10b981 !important;
        color: white !important;
    }
    div.row-widget.stButton:first-of-type > button:hover {
        background-color: #059669 !important;
    }

    /* Inputs */
    .stNumberInput>div>div>input {
        border-radius: 8px !important;
        background-color: #ffffff !important;
        border: 1px solid #cbd5e1 !important;
    }
    
    /* Selectbox */
    .stSelectbox>div>div>div {
        border-radius: 8px !important;
    }

    /* Tabs styling */
    .stTabs [data-baseweb="tab-list"] {
        justify-content: center;
        gap: 40px; /* Padding between tabs */
        margin-bottom: 24px;
        margin-top: 16px;
    }
    .stTabs [data-baseweb="tab"] {
        padding-top: 12px;
        padding-bottom: 12px;
        font-weight: 600;
    }

    /* Footer */
    .footer {
        text-align: center;
        color: #94a3b8;
        font-size: 0.875rem;
        padding-top: 40px;
        padding-bottom: 20px;
        margin-top: 40px;
        border-top: 1px solid #e2e8f0;
    }

    /* Metrics */
    [data-testid="stMetricValue"] {
        font-weight: 700 !important;
        color: #0f172a !important;
    }
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
# Pre-compute Metrics (cached)
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
    
    # Compute class averages for comparison chart
    df = pd.DataFrame(X, columns=FEATURE_NAMES)
    df['target'] = y
    averages = df.groupby('target').mean()
    averages.index = CLASS_LABELS
    
    return accuracy, report, averages

accuracy, report_dict, class_averages = get_performance_metrics()

importances      = model.feature_importances_
top_feature_idx  = int(np.argmax(importances))
top_feature_name = FEATURE_NAMES[top_feature_idx]

# ──────────────────────────────────────────────
# App Header (Centered, Modern)
# ──────────────────────────────────────────────
st.markdown("""
<div class="header-container">
    <div class="header-title">🌿 AgroNova</div>
    <div class="header-subtitle">MLOps Dashboard • Random Forest • v1.0</div>
    <div class="header-desc">
        A premium machine learning dashboard for crop variety classification based on leaf measurements. Experience seamless, dynamic inference with built-in model observability.
    </div>
</div>
""", unsafe_allow_html=True)

# ──────────────────────────────────────────────
# Tabs
# ──────────────────────────────────────────────
tab_predict, tab_performance, tab_insights = st.tabs([
    "🌱 Predict",
    "📊 Performance",
    "🔍 Insights",
])

# ══════════════════════════════════════════════
# TAB 1 ─ PREDICT
# ══════════════════════════════════════════════
with tab_predict:

    st.subheader("📋 Measurement Input")
    st.markdown("Enter the crop measurements below to predict the variety.")
    
    selected_demo = st.selectbox(
        "🔎 Quick Demo (Presets)",
        list(SAMPLE_PRESETS.keys())
    )
    def_sl, def_sw, def_pl, def_pw = SAMPLE_PRESETS[selected_demo]

    st.markdown("<br>", unsafe_allow_html=True)

    col1, col2 = st.columns(2, gap="large")
    with col1:
        sepal_length = st.number_input("🌿 Leaf Length (cm)", min_value=0.0, value=def_sl, format="%.2f", key="sl")
        sepal_width  = st.number_input("🌿 Leaf Width (cm)",  min_value=0.0, value=def_sw, format="%.2f", key="sw")
    with col2:
        petal_length = st.number_input("🌸 Petal Length (cm)", min_value=0.0, value=def_pl, format="%.2f", key="pl")
        petal_width  = st.number_input("🌸 Petal Width (cm)",  min_value=0.0, value=def_pw, format="%.2f", key="pw")
    
    st.markdown("<br>", unsafe_allow_html=True)
    
    # Buttons
    btn1, btn2 = st.columns([1, 1])
    with btn1:
        predict_clicked = st.button("✨ Predict Variety", use_container_width=True)
    with btn2:
        if st.button("🔄 Reset Form", use_container_width=True):
            st.rerun()

    # ── Prediction Result ──
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

                # Result card (Green highlight)
                st.markdown(f"""
                <div class="success-card">
                    <div style="font-size: 0.9rem; color: #047857; text-transform: uppercase; font-weight: 700; letter-spacing: 1px; margin-bottom: 8px;">Model Prediction</div>
                    <h3>{predicted_crop}</h3>
                </div>
                """, unsafe_allow_html=True)

                st.markdown("<br>", unsafe_allow_html=True)
                
                # Metrics Card
                st.subheader("🎯 Prediction Metrics")
                m1, m2 = st.columns(2)
                m1.metric("Confidence Score", f"{confidence:.1f}%")
                m2.metric("Predicted Class ID", f"Class {prediction}")
                
                # Dynamic insight based on input
                st.markdown("---")
                if petal_length > 3.75:
                    st.info(
                        f"📌 **Insight:** Your input has a high Petal Length ({petal_length:.2f} cm), "
                        f"which strongly influenced this prediction. Petal Length is the model's most impactful feature."
                    )
                elif petal_length < 2.0:
                    st.info(
                        f"📌 **Insight:** Your input has a low Petal Length ({petal_length:.2f} cm), "
                        f"which is a strong indicator of Setosa-type varieties."
                    )
                else:
                    st.info(
                        f"📌 **Insight:** Your Petal Length ({petal_length:.2f} cm) falls in a transition zone. "
                        f"The model relied on multiple features to make this prediction."
                    )

                logging.info(f"Prediction: {input_array.tolist()} → {predicted_crop} (Confidence: {confidence:.1f}%)")

                st.markdown("---")
                st.subheader("📊 Your Input vs. Class Average")
                st.markdown(f"Comparing your custom measurements against the typical **{predicted_crop}**.")
                
                # Comparison Chart
                predicted_class_label = CLASS_LABELS[prediction]
                comparison_df = pd.DataFrame({
                    "Your Input": input_array[0],
                    f"Average {predicted_class_label}": class_averages.loc[predicted_class_label].values
                }, index=FEATURE_NAMES)
                
                st.bar_chart(comparison_df, use_container_width=True)

            except Exception as e:
                st.error(f"Prediction error: {str(e)}")
                logging.error(f"Prediction error: {str(e)}")

# ══════════════════════════════════════════════
# TAB 2 ─ PERFORMANCE
# ══════════════════════════════════════════════
with tab_performance:

    st.subheader("📈 Overall Accuracy")
    st.info("The model was tested on 20% of the dataset that it had never seen before. An accuracy of 1.00 (100%) means it guessed every single crop correctly in the test set. This is common for this specific dataset because the crop varieties are very distinct!")
    st.metric("Test Set Accuracy", f"{accuracy:.4f}", delta="Held-out dataset")

    st.markdown("---")

    st.subheader("📋 Classification Report")
    st.markdown("This table breaks down the model's performance for each specific crop variety:")
    st.markdown("- **Precision:** When the model predicted a variety, how often was it right?")
    st.markdown("- **Recall:** Out of all the actual crops of a variety, how many did the model find?")
    st.markdown("- **F1-Score:** The balance between Precision and Recall. 1.0 means perfect!")
    
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

    st.markdown("---")

    st.subheader("⚖️ Weighted Averages")
    wa = report_dict["weighted avg"]
    c1, c2, c3 = st.columns(3)
    c1.metric("Precision Avg", f"{wa['precision']:.3f}")
    c2.metric("Recall Avg",    f"{wa['recall']:.3f}")
    c3.metric("F1-Score Avg",  f"{wa['f1-score']:.3f}")

# ══════════════════════════════════════════════
# TAB 3 ─ INSIGHTS
# ══════════════════════════════════════════════
with tab_insights:

    # ── Static Global Insight ─────────────────
    st.markdown(f"""
    <div class="highlight-card">
        <div style="font-weight: 700; color: #b45309; margin-bottom: 4px;">🔥 Top Influencing Feature</div>
        <div style="color: #92400e;">Based on Random Forest model behavior, <b>{top_feature_name}</b> is the most influential feature with a global importance score of <code>{importances[top_feature_idx]:.4f}</code>.</div>
    </div>
    """, unsafe_allow_html=True)

    st.subheader("📌 Feature Importance Scores")
    st.markdown("Feature importance tells us **which measurements the AI looks at the most** when making a decision. A higher score means the AI relies heavily on that specific measurement.")
    cols = st.columns(4)
    for col, name, score in zip(cols, FEATURE_NAMES, importances):
        label = f"⭐ {name}" if name == top_feature_name else name
        col.metric(label=label, value=f"{score:.4f}")

    st.markdown("---")

    st.subheader("📊 Visual Distribution")
    st.caption("Higher bars indicate greater influence on prediction outcomes.")

    chart_df = pd.DataFrame(
        {"Importance Score": importances},
        index=FEATURE_NAMES
    )
    st.bar_chart(chart_df, use_container_width=True)

# ──────────────────────────────────────────────
# Footer
# ──────────────────────────────────────────────
st.markdown(
    '<div class="footer">AgroNova • MLOps Project • Vedant Kakade</div>',
    unsafe_allow_html=True
)

