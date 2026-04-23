import streamlit as st
import joblib
import numpy as np
import logging

# Configure logging
logging.basicConfig(
    filename='app.log',
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

# Page configuration
st.set_page_config(
    page_title="AgroNova - Crop Variety Classifier",
    page_icon="🌿",
    layout="centered"
)

# App Title and Description
st.title("🌿 AgroNova")
st.markdown("**Model Version: v1.0**")
st.markdown("""
### Crop Variety Classifier
Welcome to AgroNova! This simple MLOps tool helps classify crop varieties based on measurements. 
Enter the parameters below to get a prediction.
""")

# Load the trained model
MODEL_PATH = "model_v1.pkl"
try:
    model = joblib.load(MODEL_PATH)
except FileNotFoundError:
    st.error(f"Model file '{MODEL_PATH}' not found. Please run train.py first.")
    st.stop()

# Class names mapping (Iris classes mapped to Crop themes for the demo)
CROP_VARIETIES = {
    0: "Crop Variety A (Setosa)",
    1: "Crop Variety B (Versicolor)",
    2: "Crop Variety C (Virginica)"
}

st.header("Input Crop Parameters")

# Demo Selectbox
demo_options = [
    "Custom Input", 
    "Crop Variety A (Setosa)", 
    "Crop Variety B (Versicolor)", 
    "Crop Variety C (Virginica)"
]
selected_demo = st.selectbox("Select Sample Crop (for demo)", demo_options)

# Set default values based on selection
if selected_demo == "Crop Variety A (Setosa)":
    def_sl, def_sw, def_pl, def_pw = 5.1, 3.5, 1.4, 0.2
elif selected_demo == "Crop Variety B (Versicolor)":
    def_sl, def_sw, def_pl, def_pw = 6.0, 2.8, 4.5, 1.3
elif selected_demo == "Crop Variety C (Virginica)":
    def_sl, def_sw, def_pl, def_pw = 6.5, 3.0, 5.5, 2.0
else:
    # Custom Input default values
    def_sl, def_sw, def_pl, def_pw = 5.1, 3.5, 1.4, 0.2

# Input features with validation
col1, col2 = st.columns(2)

with col1:
    sepal_length = st.number_input("Leaf Length (cm)", min_value=0.0, value=def_sl, format="%.2f")
    sepal_width = st.number_input("Leaf Width (cm)", min_value=0.0, value=def_sw, format="%.2f")

with col2:
    petal_length = st.number_input("Petal Length (cm)", min_value=0.0, value=def_pl, format="%.2f")
    petal_width = st.number_input("Petal Width (cm)", min_value=0.0, value=def_pw, format="%.2f")

# Buttons
col_btn1, col_btn2 = st.columns([1, 1])

with col_btn1:
    predict_clicked = st.button("Predict Variety")

with col_btn2:
    if st.button("Reset Inputs"):
        st.rerun()

# Validation logic
if predict_clicked:
    if any(v < 0 for v in [sepal_length, sepal_width, petal_length, petal_width]):
        st.error("Error: All input values must be positive numbers.")
        logging.warning("User attempted to predict with negative inputs.")
    else:
        # Prepare input for prediction
        input_data = np.array([[sepal_length, sepal_width, petal_length, petal_width]])
        
        # Predict
        try:
            prediction = model.predict(input_data)[0]
            probabilities = model.predict_proba(input_data)[0]
            confidence = max(probabilities) * 100
            
            predicted_crop = CROP_VARIETIES.get(prediction, "Unknown")
            
            st.success(f"🌱 The predicted crop variety is: **{predicted_crop}**")
            st.info(f"📊 Prediction Confidence: **{confidence:.2f}%**")
            
            # Log the successful prediction
            logging.info(f"Prediction successful: Inputs={input_data.tolist()} -> Output={predicted_crop} (Confidence: {confidence:.2f}%)")
        except Exception as e:
            st.error(f"An error occurred during prediction: {str(e)}")
            logging.error(f"Prediction error: {str(e)}")
