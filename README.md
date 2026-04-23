# AgroNova MLOps Project

Welcome to the **AgroNova** project! This is a simple, exam-friendly MLOps application that features a machine learning pipeline using Python, Scikit-Learn, Streamlit, Docker, and GitHub Actions.

## Project Overview

AgroNova is a "Crop Variety Classifier". It predicts the variety of a crop (Class A, B, or C) based on input parameters such as Leaf Length, Leaf Width, Petal Length, and Petal Width.

### Key Features
- **Machine Learning Model**: Uses a pre-trained Random Forest Classifier.
- **Model Versioning**: The model is saved and versioned as `model_v1.pkl`.
- **Streamlit Web App**: A clean and minimal web interface with a Gray + Green aesthetic.
- **Input Validation & Confidence Scoring**: Checks for positive inputs and displays the model's confidence probability.
- **Application Logging**: Prediction inputs, outputs, and errors are logged locally in `app.log`.
- **CI/CD Pipeline**: A GitHub Actions workflow automates the testing and build steps.
- **Containerization**: Packaged using Docker for easy deployment anywhere.

## How to Run Locally

### Prerequisites
Make sure you have Python 3.9+ installed.

1. **Install Dependencies**
   ```bash
   pip install -r requirements.txt
   ```

2. **Train the Model (Optional)**
   The model `model_v1.pkl` is already generated, but you can retrain it by running:
   ```bash
   python train.py
   ```

3. **Run the Streamlit App**
   Start the application locally using Streamlit:
   ```bash
   python -m streamlit run app.py
   ```
   Open your browser and navigate to `http://localhost:8501`.

## Docker Usage

To build and run the application using Docker:

1. **Build the Image**
   ```bash
   docker build -t agronova-app:latest .
   ```

2. **Run the Container**
   ```bash
   docker run -p 8501:8501 agronova-app:latest
   ```
