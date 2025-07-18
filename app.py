import streamlit as st
import pandas as pd
import numpy as np
import pickle
import shap
import matplotlib.pyplot as plt

# Load the trained model
with open('best_rf_model.pkl', 'rb') as model_file:
    model = pickle.load(model_file)

# Load the SHAP explainer
with open('shap_explainer.pkl', 'rb') as explainer_file:
    explainer = pickle.load(explainer_file)

# App config
st.set_page_config(page_title="Loan Approval Predictor", page_icon="🏦")
st.title("🏦 Loan Approval Predictor")
st.markdown("This app predicts whether a loan will be **Approved** or **Rejected** based on applicant data.")

# User inputs
no_of_dependents = st.number_input("👨‍👩‍👧 Number of Dependents", min_value=0)
education = st.selectbox("🎓 Education Level", ["Graduate", "Not Graduate"])
self_employed = st.selectbox("💼 Self Employed?", ["Yes", "No"])
income_annum = st.number_input("📥 Annual Income", min_value=0)
loan_amount = st.number_input("💰 Loan Amount", min_value=0)
loan_term = st.number_input("⏳ Loan Term (in months)", min_value=1)
cibil_score = st.slider("📊 CIBIL Score", 300, 900, step=1)
residential_assets_value = st.number_input("🏠 Residential Asset Value", min_value=0)
commercial_assets_value = st.number_input("🏢 Commercial Asset Value", min_value=0)
luxury_assets_value = st.number_input("💎 Luxury Asset Value", min_value=0)
bank_asset_value = st.number_input("🏦 Bank Asset Value", min_value=0)

# Encode categorical variables
education_encoded = 1 if education == "Graduate" else 0
self_employed_encoded = 1 if self_employed == "Yes" else 0

# Create input DataFrame with correct column order
input_data = pd.DataFrame([[
    no_of_dependents,
    education_encoded,
    self_employed_encoded,
    income_annum,
    loan_amount,
    loan_term,
    cibil_score,
    residential_assets_value,
    commercial_assets_value,
    luxury_assets_value,
    bank_asset_value
]], columns=[
    'no_of_dependents',
    'education',
    'self_employed',
    'income_annum',
    'loan_amount',
    'loan_term',
    'cibil_score',
    'residential_assets_value',
    'commercial_assets_value',
    'luxury_assets_value',
    'bank_asset_value'
])

# Prediction
if st.button("🔍 Predict Loan Approval"):
    prediction = model.predict(input_data)[0]
    confidence = model.predict_proba(input_data)[0][int(prediction)] * 100

    if prediction == 1:
        st.success(f"✅ Loan Approved with {confidence:.2f}% confidence.")
    else:
        st.error(f"❌ Loan Rejected with {confidence:.2f}% confidence.")

    # SHAP explanation
    st.markdown("### 🔍 SHAP Explanation (Feature Influence)")

    shap_values = explainer.shap_values(input_data)

    # For binary classification (TreeExplainer returns a list)
    if isinstance(shap_values, list):
        shap_values_instance = shap_values[1][0]  # for class 1
        expected_value = explainer.expected_value[1]
    else:
        shap_values_instance = shap_values[0]
        expected_value = explainer.expected_value

    # Waterfall plot (matplotlib)
    st.set_option('deprecation.showPyplotGlobalUse', False)
    shap.plots._waterfall.waterfall_legacy(expected_value, shap_values_instance, input_data.iloc[0])
    st.pyplot(bbox_inches='tight')
