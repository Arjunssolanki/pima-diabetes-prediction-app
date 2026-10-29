import os
import subprocess
import yaml
import joblib
import pandas as pd
import numpy as np
import streamlit as st

st.set_page_config(
    page_title="Pima Diabetes MLOps Dashboard",
    page_icon="🩺",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Auto-train models if missing
REQUIRED_MODELS = [
    "models/randomforest_model.joblib",
    "models/supportvectormachine_model.joblib",
    "models/xgboost_model.joblib"
]

if any(not os.path.exists(m) for m in REQUIRED_MODELS):
    st.info("⚙️ Initializing: Model binaries missing. Automatically running `src/train.py`...")
    try:
        subprocess.run(["python", "src/train.py"], check=True, capture_output=True)
        st.success("✅ Model training completed successfully!")
    except subprocess.CalledProcessError as e:
        st.error(f"Training failed: {e}")

def load_config():
    if os.path.exists("config.yaml"):
        with open("config.yaml", "r") as file:
            return yaml.safe_load(file)
    return None

config = load_config()

# Sidebar Configuration
st.sidebar.markdown("## ⚙️ Configuration & Options")
data_source = st.sidebar.radio("Select Data Source:", ["Default Dataset", "Upload Custom CSV"])

df_data = None
if data_source == "Default Dataset":
    processed_path = config["data"]["processed_path"] if config and "data" in config else "data/processed/diabetes_processed.csv"
    if os.path.exists(processed_path):
        df_data = pd.read_csv(processed_path)
    else:
        st.sidebar.warning("Processed dataset not found.")
else:
    uploaded_file = st.sidebar.file_uploader("Upload your input CSV file", type=["csv"])
    if uploaded_file is not None:
        df_data = pd.read_csv(uploaded_file)

# Main Navigation Tabs
tab_analysis, tab_model_sel, tab_inference = st.tabs([
    "📊 Data & QQ-Plot Analysis", 
    "🤖 Model Performance & Selection", 
    "🩺 Patient Clinical Inference"
])

with tab_analysis:
    st.header("Exploratory Data Analysis & Transformations")
    st.markdown("Inspect the distribution, skewness, and processed structure of the Pima Indians Diabetes dataset.")
    if df_data is not None:
        st.dataframe(df_data.head(10), use_container_width=True)
        st.subheader("Dataset Summary Statistics")
        st.dataframe(df_data.describe(), use_container_width=True)
    else:
        st.info("Please ensure the default dataset is generated or upload a custom CSV via the sidebar.")

with tab_model_sel:
    st.header("Model Selection & Architecture")
    st.markdown("Select an optimized model binary loaded directly from the training pipeline (`models/` directory):")

    model_options = {
        "Random Forest": "randomforest_model.joblib",
        "Support Vector Machine": "supportvectormachine_model.joblib",
        "XGBoost": "xgboost_model.joblib"
    }

    selected_model_name = st.selectbox("Select Active Model:", list(model_options.keys()))
    filename = model_options[selected_model_name]
    model_path = os.path.join("models", filename)

    col1, col2 = st.columns([1, 1])

    with col1:
        st.subheader("Model Status")
        if os.path.exists(model_path):
            st.success(f"✅ Model binary for {selected_model_name} loaded successfully from `{model_path}`.")
        else:
            st.error(f"❌ Model binary for {selected_model_name} not found.")

    with col2:
        st.subheader("Pipeline Details")
        st.info(
            "**Pipeline Components:**\n\n"
            "- **Preprocessing:** Box-Cox / Yeo-Johnson Power Transformations\n"
            "- **Feature Engineering:** Interaction Terms (`Glucose_Insulin_Ratio`, `Age_BMI_Factor`)\n"
            "- **Scaling:** StandardScaler\n"
            "- **Imbalance Handling:** Synthetic Minority Over-sampling Technique (SMOTE)\n"
            "- **Optimization:** Hyperparameter tuning via GridSearchCV"
        )

with tab_inference:
    st.header("Patient Clinical Inference Engine")
    st.markdown("Enter patient biometric and clinical indicators below to evaluate diabetes risk in real-time.")

    if os.path.exists(model_path):
        pipeline = joblib.load(model_path)

        with st.form("inference_form"):
            col_a, col_b, col_c = st.columns(3)

            with col_a:
                pregnancies = st.number_input("Pregnancies", min_value=0, max_value=20, value=1)
                glucose = st.number_input("Glucose Level", min_value=0.0, max_value=300.0, value=120.0)
                blood_pressure = st.number_input("Blood Pressure (mmHg)", min_value=0.0, max_value=200.0, value=70.0)

            with col_b:
                skin_thickness = st.number_input("Skin Thickness (mm)", min_value=0.0, max_value=100.0, value=20.0)
                insulin = st.number_input("Insulin Level (mu U/ml)", min_value=0.0, max_value=900.0, value=85.0)
                bmi = st.number_input("BMI (Weight in kg/(height in m)^2)", min_value=0.0, max_value=70.0, value=25.0)

            with col_c:
                dpf = st.number_input("Diabetes Pedigree Function", min_value=0.0, max_value=3.0, value=0.5)
                age = st.number_input("Age (Years)", min_value=1, max_value=120, value=30)

            submitted = st.form_submit_button("Run Clinical Prediction")

        if submitted:
            input_data = pd.DataFrame([{
                "Pregnancies": pregnancies,
                "Glucose": glucose,
                "BloodPressure": blood_pressure,
                "SkinThickness": skin_thickness,
                "Insulin": insulin,
                "BMI": bmi,
                "DiabetesPedigreeFunction": dpf,
                "Age": age
            }])

            # Compute interaction terms
            input_data["Glucose_Insulin_Ratio"] = input_data["Glucose"] * (input_data["Insulin"] + 1)
            input_data["Age_BMI_Factor"] = input_data["Age"] * input_data["BMI"]

            prediction = pipeline.predict(input_data)[0]
            probability = pipeline.predict_proba(input_data)[0][1]

            st.markdown("---")
            st.subheader("Prediction Results")
            
            res_col1, res_col2 = st.columns(2)
            with res_col1:
                if prediction == 1:
                    st.error(f"⚠️ **Result:** Positive for Diabetes Risk (Model: {selected_model_name})")
                else:
                    st.success(f"🟢 **Result:** Negative for Diabetes Risk (Model: {selected_model_name})")
            
            with res_col2:
                st.metric(label="Estimated Probability of Diabetes", value=f"{probability * 100:.2f}%")
    else:
        st.warning("Model binary not available.")

st.markdown("---")
st.markdown("Pima Diabetes Prediction App | Production MLOps Infrastructure")