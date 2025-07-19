import streamlit as st
import pandas as pd
import numpy as np
import shap
import pickle
import matplotlib.pyplot as plt

# Set page config
st.set_page_config(page_title="🏦 Loan Approval Predictor", layout="centered")
st.title("🏦 Loan Approval Predictor")
st.markdown("This app predicts whether a loan will be **Approved** or **Rejected** based on applicant details, and shows which features influenced the decision using SHAP.")

# Load model
try:
    with open('best_rf_model.pkl', 'rb') as model_file:
        model = pickle.load(model_file)
    st.success("✅ Model loaded successfully!")
except FileNotFoundError:
    st.error("❌ Model file not found. Make sure 'best_rf_model.pkl' is in the app directory.")
    st.stop()

# Input fields
st.subheader("📋 Applicant Details")
no_of_dependents = st.number_input("Number of Dependents", min_value=0, step=1)
education = st.selectbox("Education", ['Graduate', 'Not Graduate'])
self_employed = st.selectbox("Self Employed", ['Yes', 'No'])
income_annum = st.number_input("Annual Income", min_value=0)
loan_amount = st.number_input("Loan Amount", min_value=0)
loan_term = st.number_input("Loan Term (in months)", min_value=1)
cibil_score = st.number_input("CIBIL Score (300-900)", min_value=300, max_value=900)
residential_assets_value = st.number_input("Residential Assets Value", min_value=0)
commercial_assets_value = st.number_input("Commercial Assets Value", min_value=0)
luxury_assets_value = st.number_input("Luxury Assets Value", min_value=0)
bank_asset_value = st.number_input("Bank Asset Value", min_value=0)

# Convert to dataframe
input_df = pd.DataFrame({
    'no_of_dependents': [no_of_dependents],
    'education': [1 if education == 'Graduate' else 0],
    'self_employed': [1 if self_employed == 'Yes' else 0],
    'income_annum': [income_annum],
    'loan_amount': [loan_amount],
    'loan_term': [loan_term],
    'cibil_score': [cibil_score],
    'residential_assets_value': [residential_assets_value],
    'commercial_assets_value': [commercial_assets_value],
    'luxury_assets_value': [luxury_assets_value],
    'bank_asset_value': [bank_asset_value]
})

# Predict
if st.button("🔮 Predict Loan Approval"):
    prediction = model.predict(input_df)[0]
    prediction_proba = model.predict_proba(input_df)[0]

    status = "✅ Loan Approved" if prediction == 1 else "❌ Loan Rejected"
    confidence = round(prediction_proba[prediction] * 100, 1)

    st.subheader("📌 Prediction Result")
    st.markdown(f"**{status}** with confidence of **{confidence}%**")

    # SHAP Explanation
    st.subheader("🔍 Explanation (SHAP)")
    try:
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(input_df)

        # Use class 0 if binary classification (most common for approval/rejection)
        class_index = 0 if len(shap_values) == 1 else prediction

        # Display force plot (static image fallback)
        shap.initjs()
        st.set_option('deprecation.showPyplotGlobalUse', False)
        shap.force_plot(
            base_value=explainer.expected_value[class_index] if isinstance(explainer.expected_value, (list, np.ndarray)) else explainer.expected_value,
            shap_values=shap_values[class_index][0],
            features=input_df.iloc[0],
            matplotlib=True,
            show=False
        )
        plt.tight_layout()
        st.pyplot(bbox_inches='tight')
    except Exception as e:
        st.error(f"SHAP explanation failed: {e}")
